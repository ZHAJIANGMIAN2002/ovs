"""View-Specific 3D Positional Encoding (VS3D-PE) module.

OPTIMIZED VERSION: Uses pre-computed Fourier features from offline processing.
- No online Fourier encoding (done offline in convert_pe_to_fourier.py)
- Only learnable MLP + view fusion remain
"""

import torch
import torch.nn as nn


class VS3DPEEncoder(nn.Module):
    """View-Specific 3D Positional Encoding Encoder (Optimized).
    
    Uses pre-computed Fourier features from offline processing.
    Only applies learnable MLP and view fusion.
    
    Args:
        input_dim: Input Fourier feature dimension (default: 63 = 3 + 3*2*10)
        hidden_dim: MLP hidden dimension (default: 128)
        output_dim: Output PE feature dimension (default: 64)
        fusion: Fusion strategy - 'mean' or 'attention' (default: 'mean')
    """
    
    def __init__(self, input_dim=63, hidden_dim=128, output_dim=64, fusion='mean', keep_ratio=1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion = fusion
        self.keep_ratio = keep_ratio  # View-drop: 1.0=no drop, <1.0=randomly drop views during training
        
        # Learnable MLP: Fourier[63] → hidden[128] → output[64] (2-layer, matches PETR/PETRv2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        # Stabilization: LayerNorm on fused PE and learnable scaling α (init 0.1)
        self.pe_norm = nn.LayerNorm(output_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        
        # Optional: Attention-based fusion
        if fusion == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim // 4, 1)
            )
    
    def forward(self, fourier_pe_batch, view_counts_batch):
        """Compute VS3D-PE from pre-computed Fourier features.
        
        Args:
            fourier_pe_batch: Tuple of tensors, each [total_views_i, 63] fp16
            view_counts_batch: [N_total] tensor, number of views per point
        
        Returns:
            [N_total, output_dim] tensor, fused PE features for each point
        """
        device = view_counts_batch.device
        N_total = view_counts_batch.shape[0]
        
        # Concatenate all Fourier features from batch (DataLoader pre-converts to fp32; we ensure device)
        fourier_all = torch.cat([f if (isinstance(f, torch.Tensor) and f.device == device) else f.to(device)
                                 for f in fourier_pe_batch], dim=0)  # [total_views, 63]
        
        # Ensure float32 for MLP
        if fourier_all.dtype != torch.float32:
            fourier_all = fourier_all.float()
        
        # Apply MLP to all views at once
        pe_all_views = self.mlp(fourier_all)  # [total_views, output_dim]
        
        # ===== VIEW-DROP (training only) =====
        # Only apply if keep_ratio < 1.0 AND avg views per point >= 10 (for valid points only)
        num_valid_points = (view_counts_batch > 0).sum().item()
        avg_views_per_point = fourier_all.size(0) / float(num_valid_points) if num_valid_points > 0 else 0.0
        apply_view_drop = self.training and self.keep_ratio < 1.0 and avg_views_per_point >= 10.0
        
        if apply_view_drop:
            # Randomly drop views per point (keep_ratio fraction)
            point_ids_full = torch.repeat_interleave(
                torch.arange(N_total, device=device, dtype=torch.long),
                view_counts_batch
            )  # [total_views]
            keep_mask = torch.rand(pe_all_views.size(0), device=device) < self.keep_ratio
            pe_all_views = pe_all_views[keep_mask]
            point_ids = point_ids_full[keep_mask]
            # Update view counts after drop
            view_counts_batch = torch.bincount(point_ids, minlength=N_total).to(view_counts_batch.dtype)
        else:
            # No drop: use all views
            point_ids = torch.repeat_interleave(
                torch.arange(N_total, device=device, dtype=torch.long),
                view_counts_batch
            )  # [total_views]
        
        total_views = pe_all_views.size(0)
        # Sanity check: ensure grouping and src lengths match
        if point_ids.numel() != total_views:
            raise RuntimeError(f"VS3DPEEncoder: sum(view_counts)={point_ids.numel()} != total_views={total_views}")
        
        if self.fusion == 'mean':
            # Vectorized mean aggregation using index_add_ (more stable backward)
            pe_final = torch.zeros(N_total, self.output_dim, device=device, dtype=pe_all_views.dtype)
            pe_final.index_add_(0, point_ids, pe_all_views)
            
            # Divide by view counts (avoid division by zero)
            view_counts_safe = view_counts_batch.clamp(min=1).unsqueeze(1).to(pe_final.dtype)  # [N_total, 1]
            pe_final = pe_final / view_counts_safe
            
        elif self.fusion == 'attention':
            # Compute attention weights for all views
            attn_logits = self.attention(pe_all_views).squeeze(-1)  # [total_views]
            
            # Softmax per point group (vectorized)
            # Step 1: Compute max per point for numerical stability
            max_per_point = torch.zeros(N_total, device=device, dtype=attn_logits.dtype)
            max_per_point.scatter_reduce_(
                dim=0,
                index=point_ids,
                src=attn_logits,
                reduce='amax',
                include_self=False
            )
            
            # Step 2: Subtract max and exp
            attn_exp = torch.exp(attn_logits - max_per_point[point_ids])  # [total_views]
            
            # Step 3: Sum exp per point
            exp_sum_per_point = torch.zeros(N_total, device=device, dtype=attn_exp.dtype)
            exp_sum_per_point.index_add_(0, point_ids, attn_exp)
            
            # Step 4: Normalize to get weights
            attn_weights = attn_exp / (exp_sum_per_point[point_ids] + 1e-8)  # [total_views]
            
            # Step 5: Weighted sum
            weighted_views = pe_all_views * attn_weights.unsqueeze(1)  # [total_views, output_dim]
            pe_final = torch.zeros(N_total, self.output_dim, device=device, dtype=weighted_views.dtype)
            pe_final.index_add_(0, point_ids, weighted_views)
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion}")
        
        # Normalize and scale (cast to float for LN if needed)
        if pe_final.dtype != torch.float32:
            pe_final = pe_final.float()
        pe_final = self.pe_norm(pe_final) * self.alpha
        return pe_final  # [N_total, output_dim]


class PECatLinear(nn.Module):
    """DITR-style mid-layer fusion: concat(x, PE_proj) -> Linear -> residual.

    - Projects PE to feature dim, optional LayerNorm for stabilization
    - Concatenates with backbone features and linearly projects back to C
    - Adds residual with learnable scaling alpha
    """

    def __init__(self, feat_dim: int, pe_dim: int, use_layernorm: bool = True, alpha_init: float = 0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.pe_dim = pe_dim
        self.pe_proj = nn.Linear(pe_dim, feat_dim)
        self.pe_ln = nn.LayerNorm(feat_dim) if use_layernorm else nn.Identity()
        self.fuse = nn.Linear(feat_dim * 2, feat_dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        # Kaiming init for stability
        nn.init.kaiming_normal_(self.pe_proj.weight, nonlinearity='linear')
        nn.init.zeros_(self.pe_proj.bias)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor, pe_full: torch.Tensor) -> torch.Tensor:
        # x: [N, C], pe_full: [N, pe_dim]
        if pe_full.numel() == 0:
            return x
        pe = self.pe_proj(pe_full)
        pe = self.pe_ln(pe)
        fused = torch.cat([x, pe], dim=1)
        fused = self.fuse(fused)
        out = x + self.alpha * fused
        return out


# Backward compatibility: keep FourierPositionalEncoding for reference/testing
class FourierPositionalEncoding(nn.Module):
    """NeRF-style sinusoidal positional encoding (DEPRECATED - now done offline)."""
    
    def __init__(self, num_frequencies=10, include_input=True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.register_buffer(
            'freq_bands',
            2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        )
    
    def forward(self, coords):
        """[N, 3] -> [N, 3 + 3*2*L] if include_input else [N, 3*2*L]"""
        import math
        if self.freq_bands.device != coords.device:
            self.freq_bands = self.freq_bands.to(coords.device)
        coords_scaled = coords.unsqueeze(-1) * self.freq_bands.unsqueeze(0).unsqueeze(0)
        sin_enc = torch.sin(coords_scaled * math.pi)
        cos_enc = torch.cos(coords_scaled * math.pi)
        fourier_enc = torch.cat([sin_enc, cos_enc], dim=-1).reshape(coords.size(0), -1)
        if self.include_input:
            return torch.cat([coords, fourier_enc], dim=-1)
        else:
            return fourier_enc
