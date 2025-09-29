import torch
import torch.nn as nn
import numpy as np
from pointcept.models.builder import build_model
from pointcept.models.utils.structure import Point


class PTV3Adapter(nn.Module):
    def __init__(self, ptv3_cfg=None, in_channels=None, voxel_size=0.02):
        super().__init__()
        self.point_max = ptv3_cfg.get("point_max", 1e9)
        backbone_cfg = ptv3_cfg.copy()
        
        # Explicitly override in_channels if provided.
        if in_channels is not None:
            backbone_cfg['in_channels'] = in_channels
            
        backbone_cfg.pop("point_max", None)
        self.backbone = build_model(backbone_cfg)
        self.voxel_size = voxel_size

    def forward(self, x):
        coords = x.C
        feats = x.F
        device = feats.device
        num_points = coords.shape[0]

        if num_points <= self.point_max:
            # No sampling needed, process all points
            sampled_coords, sampled_feats = coords, feats
            idx_crop = None
        else:
            # SphereCrop sampling
            float_coords_for_crop = coords[:, 1:].float() * self.voxel_size
            center_idx = torch.randint(0, num_points, (1,)).item()
            center = float_coords_for_crop[center_idx]
            dist2 = torch.sum((float_coords_for_crop - center).pow(2), dim=1)
            idx_crop = torch.argsort(dist2)[:int(self.point_max)]
            
            sampled_coords = coords[idx_crop]
            sampled_feats = feats[idx_crop]

        # Convert voxel coordinates to float coordinates for PTV3 input
        grid_coord = sampled_coords[:, 1:]
        float_coords = grid_coord.float() * self.voxel_size

        # a batch of one scene
        batch_indices = sampled_coords[:, 0].long()
        # ptv3 need offsets (int64 expected by Pointcept utils)
        bincount = torch.bincount(batch_indices)
        offsets = torch.cumsum(bincount, dim=0).long()

        data_dict = {
            "coord": float_coords,
            "grid_coord": grid_coord,
            "feat": sampled_feats,
            "offset": offsets,
            "batch": batch_indices,
        }
        point_out = self.backbone(data_dict)
        processed_features = point_out.feat

        if idx_crop is None:
            # No sampling was performed, return features directly
            return processed_features
        else:
            # Sampling was performed, scatter features back to a full tensor
            full_features = torch.zeros(num_points, processed_features.shape[1],
                                        dtype=processed_features.dtype, device=device)
            full_features[idx_crop] = processed_features
            return full_features 