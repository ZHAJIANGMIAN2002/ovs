'''3D model for distillation.'''

from collections import OrderedDict
from models.mink_unet import mink_unet
import torch
from torch import nn
import importlib.util

# Import the PTV3 Adapter
from models.ptv3_adapter import PTV3Adapter

# Import VS3D-PE module
from models.vs3d_pe import VS3DPEEncoder

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
            
            # CRITICAL: If VS3D-PE is enabled, expand input channels BEFORE creating PTV3Adapter
            use_pe = getattr(cfg, 'use_vs3d_pe', False)
            if use_pe:
                pe_cfg = getattr(cfg, 'vs3d_pe', {})
                pe_output_dim = pe_cfg.get('output_dim', 32) if isinstance(pe_cfg, dict) else getattr(pe_cfg, 'output_dim', 32)
                pe_projected_dim = 64  # PE is projected to 64 dims
                requested_in_channels = original_in_channels + pe_projected_dim  # 3 + 64 = 67
                ptv3_cfg['in_channels'] = requested_in_channels
                print(f"VS3D-PE enabled: Expanding PTv3 input channels to {requested_in_channels} (RGB + PE)")
            
            self.net3d = PTV3Adapter(
                ptv3_cfg=ptv3_cfg,
                in_channels=requested_in_channels,
                voxel_size=cfg.voxel_size
            )
            
            # If PE enabled and pretrained weights loaded, expand first layer weights intelligently
            self.original_in_channels = original_in_channels
            self.use_pe_for_init = use_pe
            
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
            # If VS3D-PE enabled, expand input channels
            mink_in_channels = 3
            if getattr(cfg, 'use_vs3d_pe', False):
                mink_in_channels = 67  # 3 (RGB) + 64 (PE)
                print(f"VS3D-PE enabled: Expanding MinkUNet input channels to {mink_in_channels}")
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
            
            # Create PE encoder (outputs 64-dim directly, no extra projection needed)
            self.pe_encoder = VS3DPEEncoder(
                num_frequencies=num_freq,
                hidden_dim=hidden_dim,
                output_dim=64,  # Output 64-dim directly (will concat with RGB's 3 dims)
                fusion=fusion_mode
            )
        else:
            self.pe_encoder = None


    def expand_pretrained_weights_for_pe(self, pretrained_state_dict):
        """Intelligently expand pretrained weights to accommodate PE channels.
        
        For the first layer that takes [in_channels, ...] input:
        - Keep the first 3 channels from pretrained (RGB)
        - Initialize the remaining 64 channels (PE) with small random values
        
        This preserves learned RGB features while adding learnable PE features.
        """
        if not self.use_pe_for_init:
            return pretrained_state_dict
        
        print(f"\n{'='*70}")
        print("Expanding pretrained weights for VS3D-PE...")
        print(f"{'='*70}")
        
        new_state_dict = {}
        for key, value in pretrained_state_dict.items():
            # Find first layer weights based on actual checkpoint structure:
            # For PTv3: 'module.net3d.backbone.embedding.stem.conv.weight' → shape (32, 5, 5, 5, 3)
            # For MinkUNet: 'module.net3d.conv0p1s1.kernel' → shape (125, 3, 32)
            
            is_first_layer = False
            
            # PTv3: embedding.stem.conv.weight
            if 'embedding' in key and 'stem' in key and 'conv.weight' in key:
                is_first_layer = True
            
            # MinkUNet: conv0p1s1.kernel
            if 'conv0' in key and 'kernel' in key:
                is_first_layer = True
            
            if is_first_layer and value.dim() >= 2:
                # Check if this has in_channels dimension = 3
                if value.shape[-1] == self.original_in_channels or value.shape[1] == self.original_in_channels or value.shape[0] == self.original_in_channels:
                    # Determine format based on checkpoint:
                    # PTv3: (32, 5, 5, 5, 3) → in_channels at dim=-1
                    # MinkUNet: (125, 3, 32) → in_channels at dim=1
                    
                    new_shape = list(value.shape)
                    
                    if value.shape[-1] == self.original_in_channels:  # PTv3: (..., 3)
                        in_channels_dim = -1
                        new_shape[-1] = self.original_in_channels + 64  # 3 → 67
                    elif value.shape[1] == self.original_in_channels:  # MinkUNet: (K, 3, C)
                        in_channels_dim = 1
                        new_shape[1] = self.original_in_channels + 64
                    else:  # (3, ...) format
                        in_channels_dim = 0
                        new_shape[0] = self.original_in_channels + 64
                    
                    # Create expanded weight tensor
                    new_weight = torch.zeros(new_shape, dtype=value.dtype, device=value.device)
                    
                    # Copy pretrained RGB channels and initialize PE channels
                    if in_channels_dim == -1:  # PTv3: (..., 3) → (..., 67)
                        new_weight[..., :self.original_in_channels] = value
                        torch.nn.init.kaiming_normal_(new_weight[..., self.original_in_channels:], mode='fan_out')
                        new_weight[..., self.original_in_channels:] *= 0.1
                    elif in_channels_dim == 1:  # MinkUNet: (K, 3, C) → (K, 67, C)
                        new_weight[:, :self.original_in_channels, :] = value
                        torch.nn.init.kaiming_normal_(new_weight[:, self.original_in_channels:, :], mode='fan_out')
                        new_weight[:, self.original_in_channels:, :] *= 0.1
                    else:  # (3, ...) → (67, ...)
                        new_weight[:self.original_in_channels, ...] = value
                        torch.nn.init.kaiming_normal_(new_weight[self.original_in_channels:, ...], mode='fan_out')
                        new_weight[self.original_in_channels:, ...] *= 0.1
                    
                    new_state_dict[key] = new_weight
                    print(f"✅ Expanded {key}")
                    print(f"   Shape: {tuple(value.shape)} → {tuple(new_weight.shape)}")
                    print(f"   RGB channels [0:{self.original_in_channels}]: from pretrained ✅")
                    print(f"   PE channels [{self.original_in_channels}:67]: random init ×0.1 🆕")
                else:
                    new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        
        print(f"{'='*70}\n")
        return new_state_dict
    
    def forward(self, sparse_3d, pe_data=None):
        '''Forward method with optional VS3D-PE.
        
        Args:
            sparse_3d: SparseTensor with features [N_vox, 3] (after voxelization)
            pe_data: Optional tuple of (coords_world_masked, pe_metadata_masked, mask)
        '''
        if self.use_vs3d_pe and pe_data is not None:
            coords_world_masked, pe_metadata_masked, mask = pe_data
            
            # Compute PE for visible (masked) points
            pe_feat_masked = self.pe_encoder(coords_world_masked, pe_metadata_masked)  # [N_masked, 64]
            
            # Expand PE to full size (zeros for non-visible points)
            N_vox = sparse_3d.F.shape[0]
            pe_feat_full = torch.zeros(N_vox, 64, device=sparse_3d.F.device, dtype=sparse_3d.F.dtype)
            pe_feat_full[mask] = pe_feat_masked  # Fill visible points with PE
            
            # Concatenate RGB + PE → [N_vox, 67]
            sparse_3d._F = torch.cat([sparse_3d.F, pe_feat_full], dim=1)
        
        # Forward through backbone (expects in_channels=3 or 67 depending on use_vs3d_pe)
        x = self.net3d(sparse_3d)
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