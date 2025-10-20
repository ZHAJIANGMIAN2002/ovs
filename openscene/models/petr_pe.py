# Copyright (c) 2025
# Minimal PETR-like PE head (shell). Identity for now.

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class TeacherPETRPEHead(nn.Module):
    """Minimal shell for PETR-style 3D PE + FPE on teacher side.

    Current behavior: identity mapping (returns feat_3d unchanged).
    Later we will add: world coord normalization -> PETR 3D sine PE ->
    two-layer projection -> optional SE gating -> residual add.
    """

    def __init__(self, embed_dim: int = 512, use_fpe: bool = True, lambda_pe: float = 1.0,
                 pc_range: Optional[list] = None, debug: bool = False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_fpe = use_fpe
        # Make PE residual scale learnable (driven by distillation loss)
        self.lambda_pe = nn.Parameter(torch.tensor(float(lambda_pe), dtype=torch.float32))
        self.debug = debug
        self._last_debug: Optional[dict] = None
        # Fixed pc_range = [x_min, y_min, z_min, x_max, y_max, z_max]; if None, fallback to min-max per batch
        if pc_range is not None and len(pc_range) == 6:
            self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float32), persistent=False)
        else:
            self.pc_range = None

        # PETR-style pos2posemb3d produces 3*num_pos_feats dims. Follow PETR: num_pos_feats = C//2
        self.num_pos_feats = max(1, embed_dim // 2)
        self._pe_out_dim = 3 * self.num_pos_feats

        # Two-layer projection to C (similar to PETR query_embedding)
        self.pe_fc1 = nn.Linear(self._pe_out_dim, embed_dim)
        self.pe_act = nn.ReLU(inplace=True)
        self.pe_fc2 = nn.Linear(embed_dim, embed_dim)

        # PETRv2-like FPE (content-sensitive): LN(feat_3d + pe_proj) -> MLP -> residual gate
        hid = max(1, embed_dim // 4)
        self.fpe_ln = nn.LayerNorm(embed_dim)
        self.fpe_fc1 = nn.Linear(embed_dim, hid)
        self.fpe_fc2 = nn.Linear(hid, embed_dim)
        self.fpe_act = nn.ReLU(inplace=True)
        # residual gate scale, learnable; initialized to 0.1 to avoid saturation
        self.fpe_alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.out_ln = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Conservative init: make injection start near-identity
        nn.init.zeros_(self.pe_fc1.weight)
        nn.init.zeros_(self.pe_fc1.bias)
        nn.init.zeros_(self.pe_fc2.weight)
        nn.init.zeros_(self.pe_fc2.bias)

        nn.init.zeros_(self.fpe_fc1.weight)
        nn.init.zeros_(self.fpe_fc1.bias)
        nn.init.zeros_(self.fpe_fc2.weight)
        nn.init.zeros_(self.fpe_fc2.bias)

        # LayerNorm defaults are fine

    @staticmethod
    def pos2posemb3d(pos: Tensor, num_pos_feats: int = 256, temperature: int = 10000) -> Tensor:
        """PETR-style 3D sine/cosine positional embedding.
        Args:
            pos: (N, 3), values typically in [0,1]
        Returns:
            (N, 3*num_pos_feats)
        """
        scale = 2 * math.pi
        pos = pos * scale
        device = pos.device
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        px = pos[:, 0:1] / dim_t
        py = pos[:, 1:2] / dim_t
        pz = pos[:, 2:3] / dim_t
        # interleave sin/cos then flatten
        def _emb(p: Tensor) -> Tensor:
            sin = torch.sin(p[:, 0::2])
            cos = torch.cos(p[:, 1::2])
            return torch.stack((sin, cos), dim=-1).flatten(-2)
        emb_x = _emb(px)
        emb_y = _emb(py)
        emb_z = _emb(pz)
        # PETR orders (y, x, z)
        return torch.cat((emb_y, emb_x, emb_z), dim=-1)

    @torch.no_grad()
    def _normalize_world_coords(self, coords: Tensor, voxel_size: float, device: torch.device) -> Tensor:
        """Per-sample min-max normalize xyz to [0,1].
        Args:
            coords: (N, 4) int tensor [batch, x, y, z]
        Returns:
            pos: (N, 3) float in [0,1]
        """
        coords = coords.to(device)
        xyz = coords[:, 1:4].to(device=device, dtype=torch.float32) * float(voxel_size)
        # Fixed pc_range if provided
        if self.pc_range is not None:
            pr = self.pc_range.to(device)
            # pc_range order: [xmin, ymin, zmin, xmax, ymax, zmax]
            mn = pr[0:3]
            mx = pr[3:6]
            rng = torch.clamp(mx - mn, min=1e-6)
            pos = torch.clamp((xyz - mn) / rng, 0.0, 1.0)
            return pos
        # Fallback: per-batch-id min-max
        batch_ids = coords[:, 0]
        pos = torch.empty_like(xyz)
        unique_batches = torch.unique(batch_ids)
        eps = 1e-6
        for b in unique_batches:
            mask = (batch_ids == b)
            if mask.any():
                xyz_b = xyz[mask]
                mn = xyz_b.min(dim=0)[0]
                mx = xyz_b.max(dim=0)[0]
                rng = torch.clamp(mx - mn, min=eps)
                pos[mask] = torch.clamp((xyz_b - mn) / rng, 0.0, 1.0)
        return pos

    def forward(self, coords: Tensor, voxel_size: float, feat_3d: Tensor) -> Tensor:
        """PETR 3D PE + PETRv2 FPE on per-point fused features.

        Args:
            coords: (N, 4) int tensor, [batch, x, y, z] voxel indices
            voxel_size: float voxel size
            feat_3d: (N, C) fused teacher features
        Returns:
            (N, C) enriched teacher features
        """
        if feat_3d.numel() == 0:
            return feat_3d

        device = feat_3d.device
        # 1) normalize xyz -> [0,1]
        pos = self._normalize_world_coords(coords, voxel_size, device)

        # 2) PETR 3D PE (sine/cos) -> 1.5C -> 2-layer proj -> C
        pe_raw = self.pos2posemb3d(pos, num_pos_feats=self.num_pos_feats)
        pe_proj = self.pe_fc2(self.pe_act(self.pe_fc1(pe_raw)))

        # 3) residual add
        h = feat_3d + (self.lambda_pe * pe_proj)

        # 4) PETRv2 FPE (content-sensitive residual gate)
        if self.use_fpe:
            gate_in = self.fpe_ln(feat_3d + pe_proj)
            gate_raw = self.fpe_fc2(self.fpe_act(self.fpe_fc1(gate_in)))
            g = 1.0 + self.fpe_alpha * torch.tanh(gate_raw)
            h = h * g
        else:
            g = None

        # 5) output norm for stability
        out = self.out_ln(h)

        # 6) debug monitors (lightweight)
        if self.debug and self.training:
            with torch.no_grad():
                N = pos.size(0)
                s = min(1024, N)
                idx = torch.randint(0, N, (s,), device=pos.device)
                pos_s = pos[idx]
                # pos range
                pos_min = pos_s.min(dim=0)[0]
                pos_max = pos_s.max(dim=0)[0]
                # injection magnitude ratio
                eps = 1e-8
                feat_s = feat_3d[idx]
                pe_s = pe_proj[idx]
                ratio = (pe_s.norm(dim=1) / (feat_s.norm(dim=1) + eps)).clamp(max=1e6)
                # gate stats
                if g is not None:
                    g_s = g[idx]
                    g_flat = g_s.flatten()
                else:
                    g_flat = torch.tensor([], device=pos.device)

                def _p(t: Tensor, q: float) -> Tensor:
                    if t.numel() == 0:
                        return torch.tensor(0., device=t.device)
                    k = max(0, int(q * (t.numel() - 1)))
                    vals, _ = torch.sort(t)
                    return vals[k]

                self._last_debug = {
                    'pos_min': pos_min.detach().cpu(),
                    'pos_max': pos_max.detach().cpu(),
                    'ratio_mean': ratio.mean().item(),
                    'ratio_p10': _p(ratio, 0.10).item(),
                    'ratio_p90': _p(ratio, 0.90).item(),
                    'gate_mean': (g_flat.mean().item() if g_flat.numel() else None),
                    'gate_p10': (_p(g_flat, 0.10).item() if g_flat.numel() else None),
                    'gate_p90': (_p(g_flat, 0.90).item() if g_flat.numel() else None),
                }

        return out
