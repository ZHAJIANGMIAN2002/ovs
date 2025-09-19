"""
Preprocessing Script for nuScenes Information with DINO features from multiple cameras.
Original nuScenes info preprocessing modified from OpenPCDet.
DINO feature extraction part inspired by discussions and aims for dense LiDAR point features.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com) and GitHub Copilot
Please cite relevant works if the code is helpful to you.
"""

import os
from pathlib import Path
import numpy as np
# from typing import Tuple # Not strictly needed for this version
import argparse
import tqdm
import pickle
from functools import reduce
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud

# DINO related dependencies
import sys
# It's generally better to manage paths via PYTHONPATH or virtual environments
# sys.path.insert(0, "/home/jmzhou/dinov2") # Example path, adjust as needed
import torch
import torch.nn.functional as F # For F.interpolate
from PIL import Image
import torchvision
import traceback


map_name_from_general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

# cls_attr_dist remains the same as in your provided file
# ... (cls_attr_dist definition) ...
cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0, "cycle.without_rider": 0, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 0, "vehicle.parked": 0, "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791, "cycle.without_rider": 8946, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 0, "vehicle.parked": 0, "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0, "cycle.without_rider": 0, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 9092, "vehicle.parked": 3294, "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0, "cycle.without_rider": 0, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 114304, "vehicle.parked": 330133, "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0, "cycle.without_rider": 0, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 882, "vehicle.parked": 11549, "vehicle.stopped": 2102,
    },
    "ignore": { # Assuming 'ignore' class might have some attributes, though typically not
        "cycle.with_rider": 307, "cycle.without_rider": 73, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 165, "vehicle.parked": 400, "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233, "cycle.without_rider": 8326, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 0, "vehicle.parked": 0, "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0, "cycle.without_rider": 0, "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939, "pedestrian.standing": 46530,
        "vehicle.moving": 0, "vehicle.parked": 0, "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0, "cycle.without_rider": 0, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 0, "vehicle.parked": 0, "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0, "cycle.without_rider": 0, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 3421, "vehicle.parked": 19224, "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0, "cycle.without_rider": 0, "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0, "pedestrian.standing": 0,
        "vehicle.moving": 21339, "vehicle.parked": 55626, "vehicle.stopped": 11097,
    },
}


def get_available_scenes(nusc):
    # This function remains the same as in your preprocess_nuscenes_info.py
    available_scenes = []
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        # Optimization: Check only the first frame's lidar path
        lidar_path, _, _ = nusc.get_sample_data(sd_rec["token"])
        if not Path(lidar_path).exists():
            # print(f"Scene {scene_token} first LIDAR_TOP path {lidar_path} does not exist, skipping scene.")
            continue
        available_scenes.append(scene)
    return available_scenes


def get_sample_data_original(nusc, sample_data_token, selected_anntokens=None):
    # Renamed to avoid conflict if you have another get_sample_data for DINO
    # This function remains the same as in your preprocess_nuscenes_info.py
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])
    data_path = nusc.get_sample_data_path(sample_data_token)
    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    else:
        cam_intrinsic = None
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)
        box_list.append(box)
    return data_path, box_list, cam_intrinsic


def quaternion_yaw(q: Quaternion) -> float:
    # This function remains the same
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    yaw = np.arctan2(v[1], v[0])
    return yaw


def obtain_sensor2top(
    nusc, sensor_token, l2e_t_ref, l2e_r_mat_ref, e2g_t_ref, e2g_r_mat_ref, sensor_type="lidar"
):
    # This function remains the same as in your preprocess_nuscenes_info.py
    # Note: parameters renamed slightly for clarity (e.g. l2e_t_ref)
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    # Transform from current sensor to reference LIDAR_TOP
    # Sensor -> Ego (sensor @ T_s_e) -> Global (ego @ T_e_g) -> Ego (ref @ T_g_e_ref) -> LIDAR_TOP (ref @ T_e_l_ref)
    # T_sensor_to_lidar_ref = inv(T_lidar_ref_to_ego_ref) @ inv(T_ego_ref_to_global_ref) @ T_ego_sensor_to_global_sensor @ T_sensor_to_ego_sensor

    # Transformation from sensor to ego
    sensor_to_ego = transform_matrix(l2e_t_s, Quaternion(l2e_r_s), inverse=False)
    # Transformation from ego to global for the sensor's timestamp
    ego_to_global_sensor = transform_matrix(e2g_t_s, Quaternion(e2g_r_s), inverse=False)

    # Transformation from global to ego for the reference LIDAR's timestamp
    global_to_ego_ref = transform_matrix(e2g_t_ref, Quaternion(e2g_r_mat_ref.T), inverse=True) # Inverse of T_ego_ref_to_global
    # Transformation from ego to reference LIDAR for the reference LIDAR's timestamp
    ego_to_lidar_ref = transform_matrix(l2e_t_ref, Quaternion(l2e_r_mat_ref.T), inverse=True) # Inverse of T_lidar_ref_to_ego_ref

    # Combined transformation: Sensor -> Global -> LIDAR_TOP (reference frame)
    # Order of matrix multiplication is important: P_lidar_ref = T_ego_to_lidar_ref @ T_global_to_ego_ref @ T_ego_to_global_sensor @ T_sensor_to_ego @ P_sensor
    full_transform = reduce(np.dot, [ego_to_lidar_ref, global_to_ego_ref, ego_to_global_sensor, sensor_to_ego])

    sweep["sensor2lidar_rotation"] = full_transform[:3, :3]
    sweep["sensor2lidar_translation"] = full_transform[:3, 3]
    # For applying to points: P_lidar = P_sensor @ R.T + T
    # So we store R and T such that P_lidar = R @ P_sensor + T
    # Nuscenes devkit: points = view_points(points[:3, :], transform_matrix, normalize=False)
    # where transform_matrix is sensor2lidar.
    # Here, we want points_lidar = R_s_l @ points_sensor + T_s_l
    # The full_transform is sensor_to_lidar_ref.
    # So, R = full_transform[:3,:3], T = full_transform[:3,3]
    # For points @ R.T + T format, we'd need to adjust.
    # Let's stick to the convention: P_new = R @ P_old + T
    # The matrix 'full_transform' directly maps points from sensor to lidar_ref: P_lidar_ref_homo = full_transform @ P_sensor_homo
    sweep["transform_matrix"] = full_transform # This is the matrix to transform points from sensor to ref lidar
    return sweep


def project_lidar_to_image(points_3d_lidar, cam_intrinsic, lidar_to_cam_rt):
    # This function remains the same as in your DINO-specific script
    points_hom = np.hstack((points_3d_lidar, np.ones((points_3d_lidar.shape[0], 1))))
    points_cam_hom = (lidar_to_cam_rt @ points_hom.T).T
    points_cam = points_cam_hom[:, :3]
    depth_cam = points_cam[:, 2].copy()
    points_2d_image = np.full((points_3d_lidar.shape[0], 2), -1e5, dtype=np.float32)
    valid_depth_mask = depth_cam > 1e-3 # Points in front of camera
    if np.sum(valid_depth_mask) == 0:
        return points_2d_image, depth_cam, valid_depth_mask

    # Project valid points to 2D image plane
    points_img_hom_valid = (cam_intrinsic @ points_cam[valid_depth_mask].T).T
    # Normalize to get 2D coordinates (u, v)
    points_2d_valid = points_img_hom_valid[:, :2] / (points_img_hom_valid[:, 2, np.newaxis] + 1e-8)
    points_2d_image[valid_depth_mask] = points_2d_valid
    return points_2d_image, depth_cam, valid_depth_mask


def fill_trainval_infos(
    data_path, nusc, train_scenes, test=False, max_sweeps=10,
    dino_model=None, device=None, config_dino=None # DINO specific args from main config
):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(
        total=len(nusc.sample), desc="Create Info", dynamic_ncols=True
    )

    # DINO parameters from config_dino (passed from main)
    target_h = getattr(config_dino, "dino_target_h", 518)
    target_w = getattr(config_dino, "dino_target_w", 518)
    # patch_size = getattr(dino_model, "patch_size", 14) # Get from loaded model if possible
    # dino_embed_dim = getattr(dino_model, "embed_dim", 384) # Get from loaded model
    
    # It's safer to get patch_size and embed_dim after model is loaded, or pass via config_dino
    # For now, using defaults if not available in config_dino, but ideally these come from the model or config
    patch_size = getattr(config_dino, "dino_patch_size", 14)
    dino_embed_dim = getattr(config_dino, "dino_embed_dim", 384) # e.g., 384 for ViT-S, 768 for ViT-B
    padding_dino_feature = np.zeros(dino_embed_dim, dtype=np.float16)

    # ImageNet normalization for DINO
    dino_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((target_h, target_w)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    camera_types = [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        ref_lidar_token = sample["data"]["LIDAR_TOP"]
        ref_lidar_sd_rec = nusc.get("sample_data", ref_lidar_token)
        ref_lidar_cs_rec = nusc.get("calibrated_sensor", ref_lidar_sd_rec["calibrated_sensor_token"])
        ref_lidar_pose_rec = nusc.get("ego_pose", ref_lidar_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_lidar_sd_rec["timestamp"]

        ref_lidar_path_str = nusc.get_sample_data_path(ref_lidar_token)
        
        # Load original full LiDAR point cloud
        try:
            pc_full = LidarPointCloud.from_file(ref_lidar_path_str)
            points_lidar_full_original = pc_full.points[:3, :].T  # (N_full, 3)
        except Exception as e_pc:
            print(f"\n[ERROR] Sample {sample['token']}: Failed to load LiDAR point cloud from {ref_lidar_path_str}: {e_pc}. Skipping sample.")
            continue
        
        num_full_lidar_points = points_lidar_full_original.shape[0]
        # Initialize DINO features for all LiDAR points with padding
        all_lidar_points_dino_feat = np.tile(padding_dino_feature, (num_full_lidar_points, 1))
        # Keep track of which points have already been assigned a feature (to implement simple "first-come" strategy)
        lidar_point_has_dino_feature = np.zeros(num_full_lidar_points, dtype=bool)

        # Basic info dict, will be augmented by DINO and original processing
        info = {
            "lidar_path": Path(ref_lidar_path_str).relative_to(data_path).__str__(),
            "lidar_token": ref_lidar_token,
            "token": sample["token"],
            "timestamp": ref_time,
            "sweeps": [], # To be filled by original sweep logic
            # Add other essential fields from original script if needed early
        }
        
        # ========== DINO Feature Extraction for Original LiDAR Points (Multi-Camera) ==========
        if dino_model is not None and device is not None and config_dino is not None:
            try:
                for cam_name in camera_types:
                    cam_token = sample["data"][cam_name]
                    cam_sd_rec = nusc.get("sample_data", cam_token)
                    cam_cs_rec = nusc.get("calibrated_sensor", cam_sd_rec["calibrated_sensor_token"])
                    cam_pose_rec = nusc.get("ego_pose", cam_sd_rec["ego_pose_token"]) # Ego pose at camera timestamp
                    
                    cam_image_path_str = nusc.get_sample_data_path(cam_token)
                    if not os.path.exists(cam_image_path_str):
                        # print(f"\n[Warning] Sample {sample['token']}, Cam {cam_name}: Image path {cam_image_path_str} not found. Skipping this camera.")
                        continue
                    
                    image_pil = Image.open(cam_image_path_str).convert('RGB')
                    original_w, original_h = image_pil.size

                    # 1. Get DINO features from the current camera image (dense map)
                    img_tensor_transformed = dino_transform(image_pil).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        # Assuming dino_model.get_intermediate_layers or similar for patch features
                        # For ViT, forward_features often returns patch tokens before class token
                        # Example: feat_dict = dino_model.forward_features(img_tensor_transformed)
                        # dino_output_key = 'x_norm_patchtokens' # or other key based on your model
                        # patch_tokens = feat_dict[dino_output_key] # (B, NumPatches, C)
                        
                        # Using a generic way to get patch tokens, adjust if your model differs
                        # This might need specific adaptation based on DINOv1 vs DINOv2 and model variant
                        if hasattr(dino_model, 'get_intermediate_layers'): # DINOv1 style
                             # Return last layer before cls token, use reshape_transform=True for ViT
                            patch_tokens = dino_model.get_intermediate_layers(img_tensor_transformed, n=1, reshape=True)[0]
                            # patch_tokens shape (B, H_patch*W_patch, C) or (B, C, H_patch, W_patch) after reshape
                            if patch_tokens.ndim == 4 and patch_tokens.shape[1] == dino_embed_dim: # (B, C, H_patch, W_patch)
                                pass # Already in map format
                            elif patch_tokens.ndim == 3: # (B, NumPatches, C)
                                b, num_patches, c_dim = patch_tokens.shape
                                h_patch = target_h // patch_size
                                w_patch = target_w // patch_size
                                if num_patches != h_patch * w_patch:
                                    print(f"\n[Warning] Sample {sample['token']}, Cam {cam_name}: DINO patch token count mismatch. Expected {h_patch*w_patch}, got {num_patches}.")
                                    continue
                                patch_tokens = patch_tokens.permute(0, 2, 1).reshape(b, c_dim, h_patch, w_patch) # (B, C, H_patch, W_patch)
                            else:
                                print(f"\n[Warning] Sample {sample['token']}, Cam {cam_name}: Unexpected DINO patch_tokens shape {patch_tokens.shape}. Skipping DINO for this camera.")
                                continue

                        elif hasattr(dino_model, 'forward_features'): # DINOv2 style
                            feat_dict = dino_model.forward_features(img_tensor_transformed)
                            dino_output_key = getattr(config_dino, "dino_output_key", 'x_norm_patchtokens')
                            if dino_output_key not in feat_dict:
                                print(f"\n[Warning] Sample {sample['token']}, Cam {cam_name}: DINO output key '{dino_output_key}' not in feat_dict. Keys: {feat_dict.keys()}. Skipping DINO for this camera.")
                                continue
                            patch_tokens_raw = feat_dict[dino_output_key] # (B, NumPatches, C)
                            b, num_patches, c_dim = patch_tokens_raw.shape
                            h_patch = target_h // patch_size
                            w_patch = target_w // patch_size
                            if num_patches != h_patch * w_patch:
                                print(f"\n[Warning] Sample {sample['token']}, Cam {cam_name}: DINO patch token count mismatch. Expected {h_patch*w_patch}, got {num_patches}.")
                                continue
                            patch_tokens = patch_tokens_raw.permute(0, 2, 1).reshape(b, c_dim, h_patch, w_patch) # (B, C, H_patch, W_patch)
                        else:
                            raise NotImplementedError("DINO model feature extraction method not recognized.")

                    # Interpolate patch tokens to dense pixel-level feature map
                    # Output shape (B, C, target_h, target_w)
                    dino_dense_feat_map = F.interpolate(
                        patch_tokens,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # (C, target_h, target_w)

                    # 2. Calculate LiDAR to current Camera transformation
                    # LIDAR_TOP -> ego_vehicle (at LIDAR timestamp)
                    lidar_to_ego_lidar_ts = transform_matrix(
                        translation=ref_lidar_cs_rec["translation"],
                        rotation=Quaternion(ref_lidar_cs_rec["rotation"]),
                        inverse=False
                    )
                    # ego_vehicle (at LIDAR timestamp) -> global
                    ego_lidar_ts_to_global = transform_matrix(
                        translation=ref_lidar_pose_rec["translation"],
                        rotation=Quaternion(ref_lidar_pose_rec["rotation"]),
                        inverse=False
                    )
                    # global -> ego_vehicle (at CAMERA timestamp)
                    global_to_ego_cam_ts = transform_matrix(
                        translation=cam_pose_rec["translation"],
                        rotation=Quaternion(cam_pose_rec["rotation"]),
                        inverse=True # from global to ego
                    )
                    # ego_vehicle (at CAMERA timestamp) -> CURRENT_CAMERA
                    ego_cam_ts_to_cam = transform_matrix(
                        translation=cam_cs_rec["translation"],
                        rotation=Quaternion(cam_cs_rec["rotation"]),
                        inverse=True # from ego to sensor
                    )
                    lidar_to_current_cam_rt = reduce(np.dot, [ego_cam_ts_to_cam, global_to_ego_cam_ts, ego_lidar_ts_to_global, lidar_to_ego_lidar_ts])

                    # 3. Adjust camera intrinsics for DINO model's input image size
                    cam_intrinsic_original = np.array(cam_cs_rec["camera_intrinsic"])
                    scale_w_dino = target_w / original_w
                    scale_h_dino = target_h / original_h
                    adjusted_cam_intrinsic_dino = np.copy(cam_intrinsic_original)
                    adjusted_cam_intrinsic_dino[0, 0] *= scale_w_dino  # fx
                    adjusted_cam_intrinsic_dino[1, 1] *= scale_h_dino  # fy
                    adjusted_cam_intrinsic_dino[0, 2] *= scale_w_dino  # cx
                    adjusted_cam_intrinsic_dino[1, 2] *= scale_h_dino  # cy

                    # 4. Project LiDAR points to this camera's DINO-scaled image plane
                    projected_2d_pts, depths_in_cam, valid_depth_mask = project_lidar_to_image(
                        points_lidar_full_original, adjusted_cam_intrinsic_dino, lidar_to_current_cam_rt
                    )

                    # 5. Create mask for points validly projected into the DINO-scaled image
                    valid_projection_in_image_mask = (
                        (projected_2d_pts[:, 0] >= 0) & (projected_2d_pts[:, 0] < target_w -1) & # -1 for safe indexing
                        (projected_2d_pts[:, 1] >= 0) & (projected_2d_pts[:, 1] < target_h -1) &
                        valid_depth_mask # Ensure points are in front of camera
                    )
                    
                    # Consider only LiDAR points that haven't received a DINO feature yet AND are validly projected in this cam
                    updatable_points_mask = valid_projection_in_image_mask & (~lidar_point_has_dino_feature)
                    
                    if np.any(updatable_points_mask):
                        # Get 2D coordinates of these points for sampling from the dense DINO map
                        coords_to_sample = projected_2d_pts[updatable_points_mask].astype(np.int32)
                        # Clip coordinates to be within feature map bounds (already partially handled by mask)
                        coords_to_sample[:, 0] = np.clip(coords_to_sample[:, 0], 0, target_w - 1)
                        coords_to_sample[:, 1] = np.clip(coords_to_sample[:, 1], 0, target_h - 1)

                        # Sample features: dino_dense_feat_map is (C, H, W)
                        # coords_to_sample are (Num_updatable, 2) with (u,v) i.e. (x,y) -> (W,H)
                        sampled_features_tensor = dino_dense_feat_map[:, coords_to_sample[:, 1], coords_to_sample[:, 0]] # Indexing C, H_idx, W_idx
                        sampled_features_np = sampled_features_tensor.T.cpu().numpy().astype(np.float16) # (Num_updatable, C)

                        # Update the global DINO feature array and the assignment tracker
                        all_lidar_points_dino_feat[updatable_points_mask] = sampled_features_np
                        lidar_point_has_dino_feature[updatable_points_mask] = True
                
                # After processing all cameras for this sample, save the DINO features
                dino_feat_dir = os.path.join(config_dino.output_root, "dino_features_multicam_lidar")
                os.makedirs(dino_feat_dir, exist_ok=True)
                dino_feat_filename = f"{sample['token']}.npz"
                dino_feat_abs_path = os.path.join(dino_feat_dir, dino_feat_filename)
                
                np.savez(
                    dino_feat_abs_path,
                    feat=all_lidar_points_dino_feat
                )
                # Store relative path in info, relative to data_root (which is config_dino.dataset_root)
                info["dino_feat_path"] = Path(dino_feat_abs_path).relative_to(config_dino.dataset_root).__str__()

            except Exception as e_dino_outer:
                print(f"\n[ERROR] Sample {sample['token']}: Failed during DINO multi-cam processing: {e_dino_outer}")
                traceback.print_exc()
                if "dino_feat_path" in info:
                    del info["dino_feat_path"] # Mark as failed for this sample
        # ========== DINO Feature Extraction END ==========


        # ========== Original nuScenes Info Processing (Sweeps, GT Boxes, etc.) ==========
        # This part is largely from your preprocess_nuscenes_info.py
        # Ensure variable names used here (like ref_cs_rec, ref_pose_rec) are correctly defined from LIDAR_TOP
        ref_cs_rec = ref_lidar_cs_rec # Alias for clarity in original code block
        ref_pose_rec = ref_lidar_pose_rec # Alias

        # Get bounding boxes (for GT processing) in the reference LIDAR_TOP frame
        # The get_sample_data_original function transforms boxes to the sensor frame it's called with.
        # Here, ref_lidar_token is for LIDAR_TOP, so boxes are in LIDAR_TOP frame.
        _, ref_boxes, _ = get_sample_data_original(nusc, ref_lidar_token)

        # For camera info in original script (if with_camera is True for original logic)
        # This part collects camera extrinsics/intrinsics relative to LIDAR_TOP
        # It's separate from DINO feature extraction cameras but uses similar nuScenes calls.
        if config_dino.with_camera_info: # Use a specific flag for this original camera info part
            info["cams"] = dict()
            # These are transformations for the *reference LIDAR_TOP* sensor
            l2e_t_ref = ref_cs_rec["translation"]
            l2e_r_mat_ref = Quaternion(ref_cs_rec["rotation"]).rotation_matrix
            e2g_t_ref = ref_pose_rec["translation"]
            e2g_r_mat_ref = Quaternion(ref_pose_rec["rotation"]).rotation_matrix

            for cam_name_orig in camera_types: # Iterate through cameras for original info
                cam_token_orig = sample["data"][cam_name_orig]
                # cam_path_orig, _, cam_intrinsics_orig = get_sample_data_original(nusc, cam_token_orig) # This gives path and boxes in cam frame
                
                # We need cam_sd_rec to get its own pose and calibration
                cam_sd_rec_orig = nusc.get("sample_data", cam_token_orig)
                cam_path_orig = nusc.get_sample_data_path(cam_token_orig)
                cam_cs_rec_orig = nusc.get("calibrated_sensor", cam_sd_rec_orig["calibrated_sensor_token"])
                cam_intrinsics_orig = np.array(cam_cs_rec_orig["camera_intrinsic"])


                cam_info_entry = obtain_sensor2top( # This transforms sensor to LIDAR_TOP
                    nusc, cam_token_orig, l2e_t_ref, l2e_r_mat_ref, e2g_t_ref, e2g_r_mat_ref, cam_name_orig
                )
                cam_info_entry["data_path"] = (
                    Path(cam_info_entry["data_path"]).relative_to(data_path).__str__() # data_path is dataset_root
                )
                cam_info_entry.update(camera_intrinsics=cam_intrinsics_orig) # Add original intrinsics
                info["cams"].update({cam_name_orig: cam_info_entry})

        # Sweeps processing (from original script)
        # ref_from_car and car_from_global are for the main LIDAR_TOP
        ref_from_car_transform = transform_matrix( # LIDAR_TOP to Ego
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )
        car_from_global_transform = transform_matrix( # Ego to Global (inverse) -> Global to Ego
            ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True,
        )
        # Store these if your original script uses them directly in info
        # info["ref_from_car"] = ref_from_car_transform
        # info["car_from_global"] = car_from_global_transform

        current_sd_rec_sweep = ref_lidar_sd_rec
        sweeps_list = []
        while len(sweeps_list) < max_sweeps: # Original script does max_sweeps-1 for prev, here let's do max_sweeps including current
            if current_sd_rec_sweep["prev"] == "" and len(sweeps_list) > 0: # No more previous sweeps
                # Duplicate last sweep if not enough, or handle as per original logic
                # For simplicity, if no prev, and we need more, we might break or duplicate
                # The original script seems to fill up to max_sweeps-1 for *previous* sweeps
                # Let's adjust to match: we need max_sweeps in total, current is one.
                break # Stop if no more previous frames

            if len(sweeps_list) == 0: # Current (reference) frame
                sweep_info = {
                    "data_path": Path(ref_lidar_path_str).relative_to(data_path).__str__(),
                    "sample_data_token": current_sd_rec_sweep["token"],
                    "transform_matrix": np.eye(4), # Identity for ref frame to itself
                    "time_lag": 0.0,
                    # Add sensor2ego and ego2global for this sweep if needed by downstream
                    "sensor2ego_translation": ref_cs_rec["translation"],
                    "sensor2ego_rotation": ref_cs_rec["rotation"],
                    "ego2global_translation": ref_pose_rec["translation"],
                    "ego2global_rotation": ref_pose_rec["rotation"],
                }
                sweeps_list.append(sweep_info)
                if max_sweeps == 1: break # Only need current frame
            
            if not current_sd_rec_sweep["prev"]: # Check again before getting prev
                 if len(sweeps_list) < max_sweeps: # If we still need more sweeps but no more prev
                    # Duplicate the last available sweep to fill up to max_sweeps
                    # This is a common strategy if strict number of sweeps is required
                    while len(sweeps_list) < max_sweeps:
                        sweeps_list.append(sweeps_list[-1].copy()) # Append a copy
                 break

            current_sd_rec_sweep = nusc.get("sample_data", current_sd_rec_sweep["prev"])
            
            # Get transformation from this sweep's LIDAR to reference LIDAR_TOP
            # This uses the same obtain_sensor2top logic, but sensor is a past LIDAR
            sweep_transform_info = obtain_sensor2top(
                nusc, current_sd_rec_sweep["token"],
                ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]).rotation_matrix,
                ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]).rotation_matrix,
                "LIDAR_TOP_SWEEP" # Just a type name
            )

            time_lag = ref_time - 1e-6 * current_sd_rec_sweep["timestamp"]
            sweep_info = {
                "data_path": Path(nusc.get_sample_data_path(current_sd_rec_sweep["token"])).relative_to(data_path).__str__(),
                "sample_data_token": current_sd_rec_sweep["token"],
                "transform_matrix": sweep_transform_info["transform_matrix"], # Matrix from sweep to ref_lidar
                "time_lag": time_lag,
                "sensor2ego_translation": sweep_transform_info["sensor2ego_translation"],
                "sensor2ego_rotation": sweep_transform_info["sensor2ego_rotation"],
                "ego2global_translation": sweep_transform_info["ego2global_translation"],
                "ego2global_rotation": sweep_transform_info["ego2global_rotation"],
            }
            sweeps_list.append(sweep_info)
            if len(sweeps_list) >= max_sweeps:
                break
        
        # Ensure exactly max_sweeps by duplicating if necessary
        while len(sweeps_list) > 0 and len(sweeps_list) < max_sweeps:
            sweeps_list.append(sweeps_list[-1].copy())

        info["sweeps"] = sweeps_list
        # assert len(info["sweeps"]) == max_sweeps, \
        #     f"Sample {sample['token']} ended up with {len(info['sweeps'])} sweeps, expected {max_sweeps}."


        # GT Box processing (from original script)
        if not test:
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]
            # Filter annotations: only keep those with points in LIDAR or RADAR
            num_lidar_pts_anno = np.array([anno["num_lidar_pts"] for anno in annotations])
            num_radar_pts_anno = np.array([anno["num_radar_pts"] for anno in annotations])
            mask_valid_anno = num_lidar_pts_anno + num_radar_pts_anno > 0

            # ref_boxes are already in LIDAR_TOP frame from get_sample_data_original(nusc, ref_lidar_token)
            # We need to ensure `ref_boxes` (which are Box objects) correspond to `annotations`
            # The `sample["anns"]` tokens are annotation tokens. `nusc.get_boxes(ref_lidar_token)` also uses these.
            # So, `ref_boxes` should align with `annotations` if both are derived from `sample["anns"]` or `ref_lidar_token`'s annotations.

            if len(ref_boxes) == len(annotations):
                locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # l,w,h
                velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3) # vx, vy, vz
                rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
                names = np.array([b.name for b in ref_boxes])
                tokens_anno = np.array([b.token for b in ref_boxes]) # These are annotation tokens from Box objects

                gt_boxes_concat = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1) # Use only vx, vy for 2D velocity

                info["gt_boxes"] = gt_boxes_concat[mask_valid_anno, :]
                info["gt_boxes_velocity"] = velocity[mask_valid_anno, :] # Store full 3D velocity
                info["gt_names"] = np.array(
                    [map_name_from_general_to_detection.get(name, "ignore") for name in names]
                )[mask_valid_anno]
                info["gt_boxes_token"] = tokens_anno[mask_valid_anno]
                info["num_lidar_pts"] = num_lidar_pts_anno[mask_valid_anno]
                info["num_radar_pts"] = num_radar_pts_anno[mask_valid_anno]
            else:
                print(f"\n[Warning] Sample {sample['token']}: Mismatch between ref_boxes ({len(ref_boxes)}) and annotations ({len(annotations)}). GT box info might be incorrect.")
                # Fallback or skip GT for this sample
                info["gt_boxes"] = np.empty((0,9)) # Assuming 9 for loc,dim,rot,vel(2)
                info["gt_boxes_velocity"] = np.empty((0,3))
                info["gt_names"] = np.empty((0), dtype=str)
                info["gt_boxes_token"] = np.empty((0), dtype=str)
                info["num_lidar_pts"] = np.empty((0), dtype=int)
                info["num_radar_pts"] = np.empty((0), dtype=int)


            # GT Segmentation processing
            if "lidarseg" in nusc.lidarseg_idx_name_mapping: # Check if lidarseg data is available
                try:
                    lidarseg_anno = nusc.get("lidarseg", ref_lidar_token)
                    segment_path_rel = lidarseg_anno["filename"]
                    info["gt_segment_path"] = segment_path_rel
                except KeyError:
                    # This can happen if a sample doesn't have a lidarseg annotation, even if the split does.
                    # print(f"\n[Info] Sample {sample['token']}: No lidarseg annotation found for this specific sample data token.")
                    info["gt_segment_path"] = None # Or an empty string
            else:
                info["gt_segment_path"] = None


        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
        # ========== Original nuScenes Info Processing END ==========

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess nuScenes dataset info with DINO features.")
    parser.add_argument(
        "--dataset_root", required=True, help="Path to the nuScenes dataset (e.g., /data/sets/nuscenes)."
    )
    parser.add_argument(
        "--output_root", required=True, help="Output path where processed information and DINO features will be located (can be same as dataset_root or a new dir)."
    )
    parser.add_argument(
        "--max_sweeps", default=10, type=int, help="Max number of sweeps (including current). Default: 10."
    )
    parser.add_argument(
        "--with_camera_info", action="store_true", default=False,
        help="Whether to include original camera calibration and pose information in the .pkl file (separate from DINO processing)."
    )
    # DINO specific arguments
    parser.add_argument(
        "--dino_model_name", type=str, default="vit_small", help="DINO model name (e.g., vit_small, vit_base from dinov2 library)."
    )
    parser.add_argument(
        "--dino_checkpoint_path", type=str, required=True, help="Path to the DINO pretrained checkpoint (.pth file)."
    )
    parser.add_argument(
        "--dino_patch_size", type=int, default=14, help="Patch size for the DINO model."
    )
    parser.add_argument(
        "--dino_embed_dim", type=int, default=384, help="Embedding dimension of DINO model (e.g., 384 for ViT-S, 768 for ViT-B)."
    )
    parser.add_argument(
        "--dino_target_h", type=int, default=518, help="Target height for images fed to DINO model."
    )
    parser.add_argument(
        "--dino_target_w", type=int, default=518, help="Target width for images fed to DINO model."
    )
    parser.add_argument(
        "--dino_output_key", type=str, default="x_norm_patchtokens", help="Key for patch tokens in DINO model's forward_features output (for DINOv2)."
    )
    parser.add_argument(
        "--dino_library_path", type=str, default=None, help="Path to the DINO library (e.g., /path/to/dinov2) if not in PYTHONPATH."
    )
    parser.add_argument(
        "--version", type=str, default="v1.0-trainval", help="NuScenes dataset version."
    )


    config_args = parser.parse_args()

    if config_args.dino_library_path:
        sys.path.insert(0, config_args.dino_library_path)
    try:
        if config_args.dino_model_name == "vit_small":
            from dinov2.models.vision_transformer import vit_small as dino_vit_model_func
        elif config_args.dino_model_name == "vit_base":
            from dinov2.models.vision_transformer import vit_base as dino_vit_model_func
        # Add other DINO models as needed
        else:
            raise ImportError(f"Unsupported DINO model name: {config_args.dino_model_name}")
    except ImportError:
        print(f"Failed to import DINO model: {config_args.dino_model_name}. Ensure DINO library is correctly installed and path is set.")
        sys.exit(1)

    print(f"Loading nuScenes tables for version {config_args.version}...")
    nusc = NuScenes(
        version=config_args.version, dataroot=config_args.dataset_root, verbose=False
    )
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    # print(f"Total scenes in {config_args.version}: {len(nusc.scene)}")
    # print(f"Available scenes with data: {len(available_scenes)}")
    # Basic check, can be more robust
    # assert len(available_scenes) > 0, "No available scenes found. Check dataset_root and data integrity."


    # Determine train/val splits based on version
    if "trainval" in config_args.version:
        train_scene_names_split = splits.train
        # val_scene_names_split = splits.val # If you want to create a separate val .pkl
    elif "test" in config_args.version: # For processing test set if needed
        train_scene_names_split = splits.test # All scenes in test set are for "training" info generation
    else: # mini
        train_scene_names_split = splits.mini_train
        # val_scene_names_split = splits.mini_val

    train_scenes_tokens = set()
    for s_name in train_scene_names_split:
        try:
            idx = available_scene_names.index(s_name)
            train_scenes_tokens.add(available_scenes[idx]["token"])
        except ValueError:
            print(f"[Warning] Scene name '{s_name}' from official split not found in available scenes. Skipping.")
    
    # print(f"Processing {len(train_scenes_tokens)} scenes for the primary split.")

    # ========== Load DINO model ==========
    print("Loading DINO model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize DINO model (img_size might be fixed for some DINO models, or passed via dino_target_h/w)
    # block_chunks and init_values are DINOv2 specific, might not apply to all DINO versions/models
    dino_model_instance = dino_vit_model_func(
        patch_size=config_args.dino_patch_size,
        img_size=config_args.dino_target_h, # Assuming square images for DINO
        # init_values=1e-5, # DINOv2 specific
        # block_chunks=0    # DINOv2 specific
    )
    
    if not os.path.exists(config_args.dino_checkpoint_path):
        print(f"[ERROR] DINO checkpoint file not found: {config_args.dino_checkpoint_path}")
        sys.exit(1)
    checkpoint = torch.load(config_args.dino_checkpoint_path, map_location="cpu") # Load to CPU first
    
    # Handle different checkpoint structures (common in DINOv1 vs DINOv2)
    actual_state_dict = None
    if 'teacher' in checkpoint: # DINOv2 often uses 'teacher'
        actual_state_dict = checkpoint['teacher']
    elif 'student' in checkpoint: # DINOv1 often uses 'student'
        actual_state_dict = checkpoint['student']
    elif 'model' in checkpoint:
        actual_state_dict = checkpoint['model']
    else: # Assume checkpoint is the state_dict itself
        actual_state_dict = checkpoint

    # Filter out unnecessary keys (like head for classification if not present in model)
    model_dict = dino_model_instance.state_dict()
    filtered_state_dict = {k: v for k, v in actual_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    missing_keys, unexpected_keys = dino_model_instance.load_state_dict(filtered_state_dict, strict=False)
    if missing_keys:
        print(f"[Warning] DINO model missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[Warning] DINO model unexpected keys in checkpoint: {unexpected_keys}")

    dino_model_instance.eval()
    dino_model_instance.to(device)
    print(f"DINO model {config_args.dino_model_name} loaded to {device}.")
    # ========== Load DINO model END ==========

    print(f"Filling information for {config_args.version} (this may take a while)...")
    # Pass the main config_args as config_dino for DINO parameters
    # data_path for fill_trainval_infos should be dataset_root
    all_nusc_infos, _ = fill_trainval_infos( # Assuming fill_trainval_infos now handles train/val internally or based on train_scenes
        data_path=config_args.dataset_root,
        nusc=nusc,
        train_scenes=train_scenes_tokens, # This defines which scenes are "train" for the output .pkl
        test=False, # We are generating info, not in test mode of a model
        max_sweeps=config_args.max_sweeps,
        dino_model=dino_model_instance,
        device=device,
        config_dino=config_args # Pass the full config for DINO and other params
    )
    
    # For nuScenes, typically one processes "v1.0-trainval" and then splits the resulting .pkl
    # or uses the `train_scenes_tokens` to differentiate.
    # Here, `all_nusc_infos` will contain info for scenes in `train_scenes_tokens`.
    # If you need a separate val.pkl, you'd call fill_trainval_infos again with val_scene_tokens.
    # For simplicity, we save one file based on the primary split (e.g., train from v1.0-trainval).

    output_info_dir = os.path.join(config_args.output_root, "info")
    os.makedirs(output_info_dir, exist_ok=True)
    
    # Determine output filename based on version and if it's a train/val/test split
    split_name_for_file = "train" # Default for v1.0-trainval's training part
    if "mini_train" in train_scene_names_split: split_name_for_file = "mini_train"
    elif "mini_val" in train_scene_names_split: split_name_for_file = "mini_val" # If processing val separately
    elif "test" in config_args.version: split_name_for_file = "test"


    output_pkl_filename = f"nuscenes_infos_{split_name_for_file}_{config_args.max_sweeps}sweeps_dino_multicam.pkl"
    output_pkl_path = os.path.join(output_info_dir, output_pkl_filename)

    print(f"Saving nuScenes information to {output_pkl_path}...")
    print(f"Total samples processed for this split: {len(all_nusc_infos)}")
    with open(output_pkl_path, "wb") as f:
        pickle.dump(all_nusc_infos, f)

    print("Preprocessing complete.")
    print(f"Info file saved to: {output_pkl_path}")
    print(f"DINO features NPZ files saved in: {os.path.join(config_args.output_root, 'dino_features_multicam_lidar')}")
