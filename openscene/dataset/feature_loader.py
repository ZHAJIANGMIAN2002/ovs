'''Dataloader for fused point features.'''

import copy
from glob import glob
from os.path import join
import torch
import numpy as np
import SharedArray as SA

from dataset.point_loader import Point3DLoader

class FusedFeatureLoader(Point3DLoader):
    '''Dataloader for fused point features.'''

    def __init__(self,
                 datapath_prefix,
                 datapath_prefix_feat,
                 voxel_size=0.05,
                 split='train', aug=False, memcache_init=False,
                 identifier=7791, loop=1, eval_all=False,
                 input_color = False,
                 ):
        super().__init__(datapath_prefix=datapath_prefix, voxel_size=voxel_size,
                                           split=split, aug=aug, memcache_init=memcache_init,
                                           identifier=identifier, loop=loop,
                                           eval_all=eval_all, input_color=input_color)
        self.aug = aug
        self.input_color = input_color # decide whether we use point color values as input

        # prepare for 3D features
        self.datapath_feat = datapath_prefix_feat

        # Precompute the occurances for each scene
        # for training sets, ScanNet and Matterport has 5 each, nuscene 1
        # for evaluation/test sets, all has just one
        if 'nuscenes' in self.dataset_name: # only one file for each scene
            self.list_occur = None
        else:
            self.list_occur = []
            for data_path in self.data_paths:
                if 'scannet' in self.dataset_name:
                    scene_name = data_path[:-15].split('/')[-1]
                else:
                    scene_name = data_path[:-4].split('/')[-1]
                    scene_name = data_path[:-4].split('/')[-1]
                file_dirs = glob(join(self.datapath_feat, scene_name + '_*.pt'))
                self.list_occur.append(len(file_dirs))
            # some scenes in matterport have no features at all
            ind = np.where(np.array(self.list_occur) != 0)[0]
            if np.any(np.array(self.list_occur)==0):
                data_paths, list_occur = [], []
                for i in ind:
                    data_paths.append(self.data_paths[i])
                    list_occur.append(self.list_occur[i])
                self.data_paths = data_paths
                self.list_occur = list_occur

        if len(self.data_paths) == 0:
            raise Exception('0 file is loaded in the feature loader.')

    def __getitem__(self, index_long):

        index = index_long % len(self.data_paths)
        if self.use_shm:
            locs_in = SA.attach("shm://%s_%s_%06d_locs_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
            feats_in = SA.attach("shm://%s_%s_%06d_feats_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
            labels_in = SA.attach("shm://%s_%s_%06d_labels_%08d" % (
                self.dataset_name, self.split, self.identifier, index)).copy()
        else:
            locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
            labels_in[labels_in == -100] = 255
            labels_in = labels_in.astype(np.uint8)
            if np.isscalar(feats_in) and feats_in == 0:
                # no color in the input point cloud, e.g nuscenes lidar
                feats_in = np.zeros_like(locs_in)
            else:
                feats_in = (feats_in + 1.) * 127.5

        # load 3D features
        if self.dataset_name == 'scannet_3d':
            scene_name = self.data_paths[index][:-15].split('/')[-1]
        else:
            scene_name = self.data_paths[index][:-4].split('/')[-1]

        if 'nuscenes' not in self.dataset_name:
            n_occur = self.list_occur[index]
            if n_occur > 1:
                nn_occur = np.random.randint(n_occur)
            elif n_occur == 1:
                nn_occur = 0
            else:
                raise NotImplementedError

            processed_data = torch.load(join(
                self.datapath_feat, scene_name+'_%d.pt'%(nn_occur)))
        else:
            # no repeated file
            processed_data = torch.load(join(self.datapath_feat, scene_name+'.pt'))

        flag_mask_merge = False
        
        # Check if Fourier PE is available (new optimized format with 4 keys)
        has_fourier_pe = 'fourier_pe_packed' in processed_data and 'view_counts' in processed_data
        
        if has_fourier_pe:  # New optimized format with pre-computed Fourier PE (5 keys: feat + fourier_pe + view_counts + mask_full + mask)
            flag_mask_merge = True
            feat_3d = processed_data['feat']  # [N_sample, 512] (e.g., 20000, includes zero features)
            mask_chunk = processed_data['mask_full']  # [N_full] bool (which points are sampled)
            mask_visible = processed_data.get('mask', None)  # [N_visible] indices (which sampled points have features)
            fourier_pe_3d = processed_data['fourier_pe_packed']  # [total_views, 63] fp16
            view_counts_3d = processed_data['view_counts']       # [N_sample] int32
            if isinstance(mask_chunk, np.ndarray):
                mask_chunk = torch.from_numpy(mask_chunk)
            
            # Create visibility mask: mark which points (among sampled) have non-zero features
            if mask_visible is not None:
                # Convert indices to bool mask for the sampled subset
                mask_vis_bool = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
                mask_vis_bool[mask_visible] = True
            else:
                # Fallback: infer from view_counts (points with vc>0 are visible)
                mask_vis_bool = (view_counts_3d > 0)
            
            if self.split == 'train':
                # For training, use visibility mask directly (will be further filtered by voxelization)
                mask = copy.deepcopy(mask_chunk)  # Start with sampled points mask
                # Store visibility info for later use in voxelization branch
                self._mask_vis_bool = mask_vis_bool
            else:  # val or test set
                mask = copy.deepcopy(mask_chunk)  # mask for sampled points in full point cloud
            
            if self.split != 'train': # val or test set
                # Squeeze trailing singleton dim if present (baseline compatibility)
                if len(feat_3d.shape) > 2 and feat_3d.shape[-1] == 1:
                    feat_3d = feat_3d[..., 0]
                # Expand features to full point cloud (feat_3d is [N_sample, 512])
                feat_3d_new = torch.zeros((locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype)
                feat_3d_new[mask] = feat_3d  # Place all 20000 features (including zeros)
                feat_3d = feat_3d_new
                
                # Expand visibility mask to full point cloud
                mask_vis_full = torch.zeros(locs_in.shape[0], dtype=torch.bool)
                mask_vis_full[mask] = mask_vis_bool  # Only visible sampled points are True
                mask = mask_vis_full  # Now mask indicates visibility in full point cloud
                
                # Expand PE data to match full point cloud size
                view_counts_new = torch.zeros(locs_in.shape[0], dtype=view_counts_3d.dtype)
                view_counts_new[mask_chunk] = view_counts_3d  # Use mask_chunk (all sampled), not mask
                view_counts_3d = view_counts_new
                # fourier_pe_3d stays as is (indexed by view_counts offsets)
                
                mask_chunk = torch.ones_like(mask_chunk)  # All points valid for voxelization
        elif len(processed_data.keys())==2:  # Legacy: feat + mask_full only
            flag_mask_merge = True
            feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
            fourier_pe_3d, view_counts_3d = None, None  # No PE data
            if isinstance(mask_chunk, np.ndarray):
                mask_chunk = torch.from_numpy(mask_chunk)
            mask = copy.deepcopy(mask_chunk)
            if self.split != 'train': # val or test set
                # Squeeze trailing singleton dim if present (baseline compatibility)
                if len(feat_3d.shape) > 2 and feat_3d.shape[-1] == 1:
                    feat_3d = feat_3d[..., 0]
                feat_3d_new = torch.zeros((locs_in.shape[0], feat_3d.shape[1]), dtype=feat_3d.dtype)
                feat_3d_new[mask] = feat_3d
                feat_3d = feat_3d_new
                mask_chunk = torch.ones_like(mask_chunk)
        elif len(processed_data.keys())>2: # legacy, for old processed features
            feat_3d, mask_visible, mask_chunk = processed_data['feat'], processed_data['mask'], processed_data['mask_full']
            fourier_pe_3d, view_counts_3d = None, None  # No PE data
            mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
            mask[mask_visible] = True # mask out points without feature assigned

        if len(feat_3d.shape)>2:
            feat_3d = feat_3d[..., 0]

        locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in

        # calculate the corresponding point features after voxelization
        if self.split == 'train' and flag_mask_merge:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True)
            vox_ind = torch.from_numpy(vox_ind)
            
            # mask_chunk: which points in full cloud are sampled (e.g., 20000 True in 50802 points)
            # For visibility, we need to track which sampled points have features
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            # index1 must cover the entire locs_in range, not just mask_chunk
            index1 = - torch.ones(locs_in.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1!=-1]

            index2 = torch.zeros(locs_in.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization (start with sampled points)
            feat_3d = feat_3d[indices]
            
            # Build visibility mask for ALL voxelized points (not just sampled ones)
            # mask should have length = len(locs) = total voxelized points
            mask_full_vox = torch.zeros(len(locs), dtype=torch.bool)
            
            # Find which voxelized points correspond to sampled points
            # index1[vox_ind] != -1 means the voxelized point came from a sampled point
            sampled_vox_mask = (index1 != -1)  # [len(locs)] bool
            
            # Mark sampled voxelized points as True
            mask_full_vox[sampled_vox_mask] = True
            
            # Determine visibility per voxelized-sampled point
            if hasattr(self, '_mask_vis_bool') and self._mask_vis_bool is not None:
                mask_vis_voxelized = self._mask_vis_bool[indices]
            else:
                # Fallback: infer visibility from view counts if available later; default to all sampled visible
                mask_vis_voxelized = torch.ones_like(sampled_vox_mask[sampled_vox_mask], dtype=torch.bool)
            # Keep only sampled AND visible in full mask
            mask_full_vox[sampled_vox_mask] = mask_vis_voxelized
            # Also filter feat_3d to only visible points so it matches mask.sum()
            feat_3d = feat_3d[mask_vis_voxelized]
            
            mask = mask_full_vox  # [len(locs)] bool
            
            # Apply same indexing to PE data if available
            if view_counts_3d is not None:
                # Subselect fourier rows to keep only views of the kept (voxelized sampled) points
                view_counts_full = view_counts_3d  # per-sampled-point view counts before voxelization
                point_ids_full = torch.repeat_interleave(
                    torch.arange(view_counts_full.shape[0], dtype=torch.long), view_counts_full
                )
                selected_points_mask = torch.zeros(view_counts_full.shape[0], dtype=torch.bool)
                selected_points_mask[indices] = True  # only voxelized sampled points
                keep_view_rows = selected_points_mask[point_ids_full]
                fourier_pe_3d = fourier_pe_3d[keep_view_rows]

                # Expand view_counts to full voxelized size: zeros for non-sampled points
                vc_points = view_counts_full[indices]  # counts for voxelized sampled points
                vc_full_vox = torch.zeros(len(locs), dtype=vc_points.dtype)
                vc_full_vox[sampled_vox_mask] = vc_points
                view_counts_3d = vc_full_vox
        elif self.split == 'train' and not flag_mask_merge: # legacy, for old processed features
            feat_3d = feat_3d[mask] # get features for visible points
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs_in, feats_in, labels_in, return_ind=True)
            mask_chunk[mask_chunk.clone()] = mask
            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind] # voxelized visible mask for entire point clouds
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = - torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1!=-1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)
            # get the indices of corresponding masked point features after voxelization
            indices = index3[chunk_ind] - 1

            # get the corresponding features after voxelization
            feat_3d = feat_3d[indices]
            
            # Apply same indexing to PE data if available
            if view_counts_3d is not None:
                # We must also subselect fourier_pe rows to keep only views of the selected (voxelized+masked) points
                # 1) Keep a copy of full per-point view counts
                vc_full = view_counts_3d
                # 2) Build mapping from each view row -> point id
                pid_full = torch.repeat_interleave(
                    torch.arange(vc_full.shape[0], dtype=torch.long), vc_full
                )
                # 3) Build boolean mask over points we keep after voxelization
                keep_points = torch.zeros(vc_full.shape[0], dtype=torch.bool)
                keep_points[indices] = True
                # 4) Mask view rows that belong to kept points
                keep_views_mask = keep_points[pid_full]
                # 5) Subselect fourier rows and view_counts accordingly
                fourier_pe_3d = fourier_pe_3d[keep_views_mask]
                view_counts_3d = vc_full[indices]
        else:
            locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs[mask_chunk], feats_in[mask_chunk], labels_in[mask_chunk], return_ind=True)
            vox_ind = torch.from_numpy(vox_ind)
            feat_3d = feat_3d[vox_ind]
            mask = mask[vox_ind]
            
            # Apply same voxelization to PE data if available
            if view_counts_3d is not None:
                # Subselect fourier rows and view_counts to voxelized points
                vc_full = view_counts_3d
                pid_full = torch.repeat_interleave(
                    torch.arange(vc_full.shape[0], dtype=torch.long), vc_full
                )
                keep_points = torch.zeros(vc_full.shape[0], dtype=torch.bool)
                keep_points[vox_ind] = True
                keep_views_mask = keep_points[pid_full]
                fourier_pe_3d = fourier_pe_3d[keep_views_mask]
                view_counts_3d = vc_full[vox_ind]

        if self.eval_all: # during evaluation, no voxelization for GT labels
            labels = labels_in
        if self.aug:
            locs, feats, labels = self.input_transforms(locs, feats, labels)
        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        if self.input_color:
            feats = torch.from_numpy(feats).float() / 127.5 - 1.
        else:
            # hack: directly use color=(1, 1, 1) for all points
            feats = torch.ones(coords.shape[0], 3)
        labels = torch.from_numpy(labels).long()

        if self.eval_all:
            # Return PE data as well if available so validation can also use PE
            if view_counts_3d is not None:
                return (coords, feats, labels, feat_3d, mask,
                        torch.from_numpy(inds_reconstruct).long(),
                        fourier_pe_3d, view_counts_3d)
            else:
                return coords, feats, labels, feat_3d, mask, torch.from_numpy(inds_reconstruct).long()
        
        # Return PE data if available (new Fourier format)
        if view_counts_3d is not None:
            return coords, feats, labels, feat_3d, mask, fourier_pe_3d, view_counts_3d
        else:
            return coords, feats, labels, feat_3d, mask

def collation_fn(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (batch,x,y,z)
                feats:  N x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N x 4 x V (B,H,W,mask)
                [optional] fourier_pe: [total_views, 63] fp16 tensor
                [optional] view_counts: N x 1 int32 tensor

    '''
    # Check if batch contains PE data (variable-length tuple)
    has_pe_data = len(batch[0]) == 7
    
    if has_pe_data:
        coords, feats, labels, feat_3d, mask_chunk, fourier_pe, view_counts = list(zip(*batch))
    else:
        coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))

    for i in range(len(coords)):
        coords[i][:, 0] *= i

    if has_pe_data:
        # Concatenate across batch
        coords_cat = torch.cat(coords)
        feats_cat = torch.cat(feats)
        labels_cat = torch.cat(labels)
        feat3d_cat = torch.cat(feat_3d)
        mask_cat = torch.cat(mask_chunk)
        vc_cat = torch.cat(view_counts, dim=0)

        # Align shapes: ensure view_counts length matches coords length
        if vc_cat.numel() != coords_cat.shape[0]:
            # If vc covers only visible points, expand to full length using mask
            if mask_cat.dtype != torch.bool:
                mask_cat = mask_cat.bool()
            if vc_cat.numel() == int(mask_cat.sum().item()):
                vc_full = torch.zeros(coords_cat.shape[0], dtype=vc_cat.dtype)
                vc_full[mask_cat] = vc_cat
                vc_cat = vc_full
            else:
                raise RuntimeError(f"[collation_fn] view_counts size mismatch: vc={vc_cat.numel()} vs coords={coords_cat.shape[0]}, mask_sum={int(mask_cat.sum().item())}")

        # Pre-convert Fourier PE to fp32 once per batch
        fourier_pe_fp32 = tuple(f.float() if f.dtype == torch.float16 else f for f in fourier_pe)
        
        return (coords_cat, feats_cat, labels_cat, feat3d_cat, mask_cat,
                fourier_pe_fp32, vc_cat)
    else:
        return (torch.cat(coords), torch.cat(feats), torch.cat(labels),
                torch.cat(feat_3d), torch.cat(mask_chunk))


def collation_fn_eval_all(batch):
    '''
    :param batch:
    :return:    coords: N x 4 (x,y,z,batch)
                feats:  N x 3
                labels: N
                feat_3d: N x C
                mask:    N
                inds_recons: N
                [optional] fourier_pe: tuple(T_i x 63)
                [optional] view_counts: N

    '''
    # Support optional PE data in eval
    has_pe = len(batch[0]) == 8

    if has_pe:
        coords, feats, labels, feat_3d, mask, inds_recons, fourier_pe, view_counts = list(zip(*batch))
    else:
        coords, feats, labels, feat_3d, mask, inds_recons = list(zip(*batch))

    inds_recons = list(inds_recons)

    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] *= i
        inds_recons[i] = accmulate_points_num + inds_recons[i]
        accmulate_points_num += coords[i].shape[0]

    base = (
        torch.cat(coords), torch.cat(feats), torch.cat(labels),
        torch.cat(feat_3d), torch.cat(mask), torch.cat(inds_recons)
    )

    if has_pe:
        # view_counts can be concatenated; fourier_pe stays per-sample tuple
        view_counts_cat = torch.cat(view_counts, dim=0)
        return base + (fourier_pe, view_counts_cat)
    else:
        return base