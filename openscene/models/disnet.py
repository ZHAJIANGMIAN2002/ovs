'''3D model for distillation.'''

from collections import OrderedDict
from models.mink_unet import mink_unet
import torch
from torch import nn
import importlib.util
import os

# Import the PTV3 Adapter
from models.ptv3_adapter import PTV3Adapter

# Import VS3D-PE module
from models.vs3d_pe import VS3DPEEncoder, PECatLinear

# Import SparseTensor for PE voxelization alignment
from MinkowskiEngine import SparseTensor


def state_dict_remove_moudle(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


class DisNet(nn.Module):
    '''3D Sparse UNet for Distillation.'''
    def __init__(self, cfg=None):
        super(DisNet, self).__init__()
        if not hasattr(cfg, 'feature_2d_extractor'):
            cfg.feature_2d_extractor = 'openseg'
        if 'lseg' in cfg.feature_2d_extractor:
            last_dim = 512
        elif 'openseg' in cfg.feature_2d_extractor:
            last_dim = 768
        else:
            raise NotImplementedError

        # --- Dynamic Backbone Selection ---
        arch = cfg.arch_3d
        if arch == 'PTV3Adapter':
            # Dynamically load the PTV3 configuration file
            spec = importlib.util.spec_from_file_location("ptv3_config", cfg.ptv3_config)
            ptv3_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ptv3_config_module)
            ptv3_cfg = ptv3_config_module.model

            # Respect in_channels from ptv3 config (e.g., 3 for RGB or 6 for RGB+Normal)
            requested_in_channels = ptv3_cfg.get('in_channels', None)
            if requested_in_channels is None and hasattr(cfg, 'in_channels'):
                # Allow overriding via YAML top-level (for visibility in config)
                requested_in_channels = cfg.in_channels
                ptv3_cfg['in_channels'] = requested_in_channels
            
            # Store original in_channels for potential weight transfer
            original_in_channels = requested_in_channels
            
            # VS3D-PE injection mode: 'mid' (default, DITR-style) or 'input'
            use_pe = getattr(cfg, 'use_vs3d_pe', False)
            inject_mode = 'mid'
            if use_pe:
                pe_cfg_tmp = getattr(cfg, 'vs3d_pe', {})
                if isinstance(pe_cfg_tmp, dict):
                    inject_mode = pe_cfg_tmp.get('inject', 'mid')
                else:
                    inject_mode = getattr(pe_cfg_tmp, 'inject', 'mid')

            # Always use mid-layer injection: keep pretrained input channels intact
            requested_in_channels = original_in_channels
            ptv3_cfg['in_channels'] = requested_in_channels
            
            self.net3d = PTV3Adapter(
                ptv3_cfg=ptv3_cfg,
                in_channels=requested_in_channels,
                voxel_size=cfg.voxel_size
            )
            
            # No input-channel expansion when using mid-layer injection
            self.original_in_channels = original_in_channels
            self.use_pe_for_init = False
            
            # The output of PTV3 may not match `last_dim`. Add a projection head.
            # From the PTV3 config: dec_channels=(64, 64, 128, 256)
            # The unpooling starts from enc_channels[-1] (512) -> dec_channels[-1] (256) -> ... -> dec_channels[0] (64)
            ptv3_out_dim = ptv3_cfg['dec_channels'][0]

            # Build a configurable projection head. Default remains a single Linear layer
            # for backward compatibility. If cfg.proj_head is provided and type=="mlp",
            # we use a 2-layer MLP: [LayerNorm] -> Linear(in, hidden) -> GELU -> [Dropout] -> Linear(hidden, out)
            self.projection_head = self._build_projection_head(
                input_dim=ptv3_out_dim,
                output_dim=last_dim,
                cfg=cfg
            )
            
        else:
            # Original MinkowskiNet for 3D point clouds
            # Keep input channels intact; VS3D-PE uses mid-layer injection only for PTv3
            mink_in_channels = 3
            self.net3d = mink_unet(in_channels=mink_in_channels, out_channels=last_dim, D=3, arch=cfg.arch_3d)
            self.projection_head = None

        # --- VS3D-PE Integration ---
        self.use_vs3d_pe = getattr(cfg, 'use_vs3d_pe', False)
        if self.use_vs3d_pe:
            pe_cfg = getattr(cfg, 'vs3d_pe', {})
            
            # Parse PE config
            if isinstance(pe_cfg, dict):
                num_freq = pe_cfg.get('num_frequencies', 10)
                hidden_dim = pe_cfg.get('hidden_dim', 64)
                pe_output_dim = pe_cfg.get('output_dim', 32)
                fusion_mode = pe_cfg.get('fusion', 'mean')
            else:
                num_freq = getattr(pe_cfg, 'num_frequencies', 10)
                hidden_dim = getattr(pe_cfg, 'hidden_dim', 64)
                pe_output_dim = getattr(pe_cfg, 'output_dim', 32)
                fusion_mode = getattr(pe_cfg, 'fusion', 'mean')
            
            # Fixed to mid-layer injection
            self.pe_inject_mode = 'mid'

            # Create PE encoder (optimized: uses pre-computed Fourier features)
            keep_ratio = pe_cfg.get('keep_ratio', 1.0) if isinstance(pe_cfg, dict) else getattr(pe_cfg, 'keep_ratio', 1.0)
            self.pe_encoder = VS3DPEEncoder(
                input_dim=63,  # Pre-computed Fourier features (3 + 3*2*10)
                hidden_dim=hidden_dim,
                output_dim=64,  # Output 64-dim directly (will concat with RGB's 3 dims)
                fusion=fusion_mode,
                keep_ratio=keep_ratio
            )

            # Mid-layer injectors (DITR-style Cat+Linear)
            if arch == 'PTV3Adapter':
                ptv3_out_dim = ptv3_cfg['dec_channels'][0]
                # Optional: per-stage injectors for DITR "all blocks"
                inject_all = False
                inject_stages = None
                if isinstance(pe_cfg, dict):
                    inject_all = bool(pe_cfg.get('inject_all_blocks', False))
                    inject_stages = pe_cfg.get('inject_stages', None)
                else:
                    inject_all = bool(getattr(pe_cfg, 'inject_all_blocks', False))
                    inject_stages = getattr(pe_cfg, 'inject_stages', None)
                self.inject_all_blocks = inject_all
                if self.inject_all_blocks:
                    stage_dims = list(ptv3_cfg['dec_channels'])
                    from torch import nn as _nn
                    self.pe_injectors_by_stage = _nn.ModuleDict()
                    for s, dim in enumerate(stage_dims):
                        if inject_stages is not None and isinstance(inject_stages, (list, tuple)) and (s not in inject_stages):
                            continue
                        self.pe_injectors_by_stage[f'dec{s}'] = PECatLinear(feat_dim=dim, pe_dim=64, use_layernorm=True, alpha_init=0.1)
                    # Do not keep single last-block injector to avoid DDP unused params
                    self.pe_injector = None
                else:
                    # Single last-block injector only
                    self.pe_injectors_by_stage = None
                    self.pe_injector = PECatLinear(feat_dim=ptv3_out_dim, pe_dim=64, use_layernorm=True, alpha_init=0.1)
            else:
                # MinkUNet: simple output-layer injection via Cat+Linear
                self.inject_all_blocks = False
                self.pe_injectors_by_stage = None
                # feat_dim equals Mink output channels (last_dim)
                self.pe_injector_mink = PECatLinear(feat_dim=last_dim, pe_dim=64, use_layernorm=True, alpha_init=0.1)
        else:
            self.pe_encoder = None
            self.pe_inject_mode = 'none'
            self.pe_injector = None
            self.pe_injector_mink = None


    # Removed input-channel expansion/weight surgery: mid-layer injection only
    
    def forward(self, sparse_3d, pe_data=None):
        '''Forward method with optional VS3D-PE.
        
        Args:
            sparse_3d: SparseTensor with features [N_vox, 3] (after voxelization)
            pe_data: Optional tuple of (fourier_pe_batch, view_counts_voxelized, mask_voxelized)
                     - fourier_pe_batch: packed Fourier features for all visible voxelized points
                     - view_counts_voxelized: [N_vox] view counts per voxelized point
                     - mask_voxelized: [N_vox] bool, which voxelized points have PE data
        '''
        pe_feat_full = None
        if self.use_vs3d_pe and pe_data is not None:
            fourier_pe_batch, view_counts_voxelized, mask_voxelized = pe_data

            # Sanity align lengths with current SparseTensor
            N_vox = sparse_3d.F.shape[0]
            if mask_voxelized.dtype != torch.bool:
                mask_voxelized = mask_voxelized.bool()
            if view_counts_voxelized.numel() != N_vox:
                if view_counts_voxelized.numel() == int(mask_voxelized.sum().item()):
                    vc_full = torch.zeros(N_vox, dtype=view_counts_voxelized.dtype, device=view_counts_voxelized.device)
                    vc_full[mask_voxelized] = view_counts_voxelized
                    view_counts_voxelized = vc_full
                else:
                    raise RuntimeError(f"[DisNet] view_counts size mismatch: vc={view_counts_voxelized.numel()} vs N_vox={N_vox}, mask_sum={int(mask_voxelized.sum().item())}")

            # Compute PE only for visible points
            vc_nonzero = view_counts_voxelized[mask_voxelized]
            pe_feat_masked = self.pe_encoder(fourier_pe_batch, vc_nonzero)

            # Full-size PE tensor for injection
            pe_feat_full = torch.zeros(N_vox, 64, device=sparse_3d.F.device, dtype=torch.float32)
            pe_feat_full[mask_voxelized] = pe_feat_masked

            # Pass PE to PTv3 backbone for internal injection
            if isinstance(self.net3d, PTV3Adapter) and self.pe_inject_mode == 'mid':
                # Attach one-shot PE features
                setattr(self.net3d.backbone, '_pe_full', pe_feat_full)
                if self.inject_all_blocks and self.pe_injectors_by_stage is not None:
                    # Register per-stage injectors on backbone
                    setattr(self.net3d.backbone, '_pe_injectors_by_stage', self.pe_injectors_by_stage)
                    # Ensure no last-block injector is used simultaneously
                    if hasattr(self.net3d.backbone, '_pe_injector_last'):
                        setattr(self.net3d.backbone, '_pe_injector_last', None)
                else:
                    # Single last-block injector
                    setattr(self.net3d.backbone, '_pe_injector_last', self.pe_injector)
        
        # Forward through backbone (expects in_channels=3 or 67 depending on use_vs3d_pe)
        x = self.net3d(sparse_3d)

        # If using MinkUNet with VS3D-PE, inject at output layer
        if (self.use_vs3d_pe 
            and not isinstance(self.net3d, PTV3Adapter)
            and pe_feat_full is not None 
            and hasattr(self, 'pe_injector_mink') 
            and self.pe_injector_mink is not None):
            # Align lengths if needed (defensive)
            if pe_feat_full.shape[0] != x.shape[0]:
                min_len = min(pe_feat_full.shape[0], x.shape[0])
                x = x[:min_len]
                pe_feat_full = pe_feat_full[:min_len]
            x = self.pe_injector_mink(x, pe_feat_full)

        # Note: mid-layer injection已在PTv3内部完成，如未开启all_blocks且未设置内部注入器，可在此处作为后备，但当前默认不再在此处注入
        if self.projection_head:
            x = self.projection_head(x)
        return x

    def _build_projection_head(self, input_dim: int, output_dim: int, cfg):
        """Create projection head module based on config.

        Accepted cfg fields (all optional):
          - proj_head.type: "linear" | "mlp" (default: "linear" for backward compatibility)
          - proj_head.hidden_dim: int (default: 256 when type=="mlp")
          - proj_head.dropout: float in [0,1] (default: 0.1 when type=="mlp")
          - proj_head.use_ln: bool (default: True when type=="mlp")
          - proj_head.pre_expand_dim: int (optional). If > input_dim, insert a Linear(input_dim->pre_expand_dim)
          - proj_head.pre_expand_act: bool (default: False). If True, add GELU after pre-expand
        """
        proj_cfg = getattr(cfg, 'proj_head', None)

        def _get(container, key, default):
            if container is None:
                return default
            if isinstance(container, dict):
                return container.get(key, default)
            return getattr(container, key, default)

        proj_type = _get(proj_cfg, 'type', 'linear')
        if isinstance(proj_type, str) and proj_type.lower() == 'mlp':
            hidden_dim = int(_get(proj_cfg, 'hidden_dim', 256))
            dropout_p = float(_get(proj_cfg, 'dropout', 0.1))
            use_ln = bool(_get(proj_cfg, 'use_ln', True))
            pre_expand_dim = _get(proj_cfg, 'pre_expand_dim', None)
            pre_expand_act = bool(_get(proj_cfg, 'pre_expand_act', False))

            layers = []
            if use_ln:
                layers.append(nn.LayerNorm(input_dim))
            # Optional pre-expansion: Linear(input_dim -> pre_expand_dim)
            effective_in = input_dim
            if pre_expand_dim is not None:
                try:
                    ped = int(pre_expand_dim)
                except Exception:
                    ped = None
                if ped is not None and ped > input_dim:
                    layers.append(nn.Linear(input_dim, ped))
                    if pre_expand_act:
                        layers.append(nn.GELU())
                    effective_in = ped

            layers.append(nn.Linear(effective_in, hidden_dim))
            layers.append(nn.GELU())
            if dropout_p and dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            layers.append(nn.Linear(hidden_dim, output_dim))
            return nn.Sequential(*layers)

        # default: single Linear to keep previous behavior
        return nn.Linear(input_dim, output_dim)