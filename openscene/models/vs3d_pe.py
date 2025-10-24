"""View-Specific 3D Positional Encoding (VS3D-PE) module.

This module implements geometric positional encoding that transforms each 3D point
into multiple camera-local coordinate systems and encodes them using Fourier features.

Inspired by:
- PETR/PETRv2: 3D PE for multi-view 3D detection
- CAPE: Camera-centric encoding for cross-scene generalization
- NeRF: Fourier positional encoding for high-frequency details
"""

import torch
import torch.nn as nn
import math


class FourierPositionalEncoding(nn.Module):
    """NeRF-style sinusoidal positional encoding.
    
    Transforms 3D coordinates into high-dimensional Fourier features:
    [x, y, z] -> [x, y, z, sin(2^0*π*x), cos(2^0*π*x), ..., sin(2^(L-1)*π*z), cos(2^(L-1)*π*z)]
    
    Args:
        num_frequencies: Number of frequency bands (L). Default: 10
        include_input: Whether to concatenate original coords. Default: True
    """
    
    def __init__(self, num_frequencies=10, include_input=True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        
        # Frequency bands: [2^0, 2^1, ..., 2^(L-1)]
        self.register_buffer(
            'freq_bands',
            2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        )
    
    def forward(self, coords):
        """
        Args:
            coords: [N, 3] in camera local coordinates
        
        Returns:
            [N, 3 + 3*2*L] if include_input else [N, 3*2*L]
        """
        # Ensure freq_bands is on same device as coords
        if self.freq_bands.device != coords.device:
            self.freq_bands = self.freq_bands.to(coords.device)
        
        # coords: [N, 3]
        # freq_bands: [L]
        # coords_scaled: [N, 3, L]
        coords_scaled = coords.unsqueeze(-1) * self.freq_bands.unsqueeze(0).unsqueeze(0)
        
        # Apply sin and cos: [N, 3, L] -> [N, 3, 2*L]
        sin_enc = torch.sin(coords_scaled * math.pi)
        cos_enc = torch.cos(coords_scaled * math.pi)
        fourier_enc = torch.cat([sin_enc, cos_enc], dim=-1).reshape(coords.size(0), -1)
        
        if self.include_input:
            return torch.cat([coords, fourier_enc], dim=-1)  # [N, 3 + 3*2*L]
        else:
            return fourier_enc  # [N, 3*2*L]


class VS3DPEEncoder(nn.Module):
    """View-Specific 3D Positional Encoding Encoder.
    
    For each 3D point:
    1. Transform to each visible camera's local coordinate system
    2. Apply Fourier positional encoding
    3. Pass through learnable MLP
    4. Fuse across multiple views (mean or attention-weighted)
    
    Args:
        num_frequencies: Fourier encoding frequency bands (default: 10)
        hidden_dim: MLP hidden dimension (default: 128)
        output_dim: Output PE feature dimension (default: 64)
        fusion: Fusion strategy - 'mean' or 'attention' (default: 'mean')
    """
    
    def __init__(self, num_frequencies=10, hidden_dim=128, output_dim=64, fusion='mean'):
        super().__init__()
        self.fusion = fusion
        
        # Fourier encoding: [3] -> [3 + 3*2*L] = [3 + 60] = 63 (when L=10)
        self.fourier = FourierPositionalEncoding(num_frequencies, include_input=True)
        fourier_dim = 3 + 3 * 2 * num_frequencies  # 63
        
        # Learnable MLP: 63 -> 128 -> 128 -> 64 (direct to final dim, no extra projection needed!)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Optional attention for view fusion
        if fusion == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim // 4, 1)
            )
    
    def forward(self, coords_world_batch, pe_metadata_batch):
        """Compute VS3D-PE for a batch of points.
        
        Args:
            coords_world_batch: [N_total, 3] tensor, world coordinates for all points in batch
            pe_metadata_batch: List[N_total] of dicts, each containing:
                - 'visible_view_ids': List of view indices
                - 'camera_extrinsics': [n_views, 4, 4] numpy array
        
        Returns:
            [N_total, output_dim] tensor, fused PE features for each point
        """
        device = coords_world_batch.device
        N_total = coords_world_batch.shape[0]
        output_dim = self.mlp[-1].out_features
        
        # VECTORIZED: Batch all (point, view) pairs for GPU acceleration
        all_coords = []
        all_extrinsics = []
        view_counts = []
        
        for i in range(N_total):
            metadata = pe_metadata_batch[i]
            camera_extrinsics = metadata['camera_extrinsics']  # numpy [n_views, 4, 4]
            n_views = camera_extrinsics.shape[0]
            view_counts.append(n_views)
            
            if n_views == 0:
                continue
            
            # Replicate point coords for each view
            coords_world = coords_world_batch[i]  # [3]
            all_coords.extend([coords_world] * n_views)
            all_extrinsics.append(torch.from_numpy(camera_extrinsics).float())  # [n_views, 4, 4]
        
        if len(all_coords) == 0:
            # No views, return zeros
            return torch.zeros(N_total, output_dim, device=device)
        
        # Stack: [total_views, ...]
        all_coords = torch.stack(all_coords).to(device)  # [total_views, 3]
        all_extrinsics = torch.cat(all_extrinsics, dim=0).to(device)  # [total_views, 4, 4]
        
        # Vectorized transform: p_local = R^T @ (p_world - t)
        R = all_extrinsics[:, :3, :3]  # [total_views, 3, 3]
        t = all_extrinsics[:, :3, 3]   # [total_views, 3]
        p_local = torch.bmm((all_coords - t).unsqueeze(1), R).squeeze(1)  # [total_views, 3]
        
        # Batch Fourier + MLP
        fourier_enc = self.fourier(p_local)  # [total_views, 63]
        pe_all_views = self.mlp(fourier_enc)  # [total_views, output_dim]
        
        # Aggregate by point (mean fusion)
        pe_final = []
        view_idx = 0
        for n_views in view_counts:
            if n_views == 0:
                pe_final.append(torch.zeros(output_dim, device=device))
            else:
                pe_final.append(pe_all_views[view_idx:view_idx+n_views].mean(dim=0))
                view_idx += n_views
        
        return torch.stack(pe_final, dim=0)  # [N_total, output_dim]


# Utility function for testing
def test_vs3d_pe():
    """Test VS3D-PE encoder with dummy data."""
    print("Testing VS3D-PE Encoder...")
    
    # Create dummy data
    batch_size = 4
    coords_world = torch.randn(batch_size, 3).cuda()
    
    pe_metadata = []
    for i in range(batch_size):
        n_views = torch.randint(1, 10, (1,)).item()
        pe_metadata.append({
            'visible_view_ids': list(range(n_views)),
            'camera_extrinsics': torch.randn(n_views, 4, 4).numpy()
        })
    
    # Create encoder
    encoder = VS3DPEEncoder(
        num_frequencies=10,
        hidden_dim=64,
        output_dim=32,
        fusion='mean'
    ).cuda()
    
    # Forward pass
    pe_features = encoder(coords_world, pe_metadata)
    
    print(f"Input coords: {coords_world.shape}")
    print(f"Output PE: {pe_features.shape}")
    print(f"Expected: [{batch_size}, 32]")
    assert pe_features.shape == (batch_size, 32), "Shape mismatch!"
    print("✅ Test passed!")
    
    return encoder, pe_features


if __name__ == "__main__":
    test_vs3d_pe()

