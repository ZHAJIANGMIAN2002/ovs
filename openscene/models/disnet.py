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
            self.projection_head = nn.Linear(ptv3_out_dim, last_dim)
            
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