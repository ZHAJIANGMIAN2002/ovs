'''3D model for distillation.'''

from collections import OrderedDict
from models.mink_unet import mink_unet
from torch import nn
import importlib.util

# Import the PTV3 Adapter
from models.ptv3_adapter import PTV3Adapter


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
            self.net3d = PTV3Adapter(
                ptv3_cfg=ptv3_cfg,
                in_channels=requested_in_channels,
                voxel_size=cfg.voxel_size
            )
            
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
            self.net3d = mink_unet(in_channels=3, out_channels=last_dim, D=3, arch=cfg.arch_3d)
            self.projection_head = None


    def forward(self, sparse_3d):
        '''Forward method.'''
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