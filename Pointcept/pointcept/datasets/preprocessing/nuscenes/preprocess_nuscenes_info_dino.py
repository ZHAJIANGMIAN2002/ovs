
"""
Preprocessing Script for nuScenes Informantion
modified from OpenPCDet (https://github.com/open-mmlab/OpenPCDet)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
from pathlib import Path
import numpy as np
from typing import Tuple
import argparse
import tqdm
import pickle
from functools import reduce
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud
#DINO相关依赖
import sys
sys.path.insert(0, "/home/jmzhou/dinov2")
import torch
from PIL import Image
import torchvision
import torch.nn.functional as F
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


cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bicycle": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}


def get_available_scenes(nusc):
    available_scenes = []
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    return available_scenes


def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor"s coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get("sensor", cs_record["sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
    else:
        cam_intrinsic = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def obtain_sensor2top(
    nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: "lidar".

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get("sample_data", sensor_token)
    cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
    data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
    # if os.getcwd() in data_path:  # path from lyftdataset is absolute path
    #     data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
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

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep

def project_lidar_to_image(points_3d: np.ndarray, 
                           cam_intrinsic: np.ndarray, 
                           lidar_to_cam_rt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D LiDAR points to 2D image plane.

    Args:
        points_3d (np.ndarray): LiDAR points in shape (N, 3), in LiDAR frame.
        cam_intrinsic (np.ndarray): Camera intrinsic matrix (3, 3).
        lidar_to_cam_rt (np.ndarray): Transformation matrix from LiDAR to Camera frame (4, 4).
                                      This matrix should transform points from LiDAR to Camera: P_cam = lidar_to_cam_rt @ P_lidar_hom.

    Returns:
        points_2d (np.ndarray): Projected 2D points in image coordinates (pixel coords, origin top-left) (N, 2).
        depth (np.ndarray): Depth of the points in the camera frame (N,).
    """
    if points_3d.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    num_points = points_3d.shape[0]
    
    # Convert points to homogeneous coordinates (N, 4)
    points_3d_hom = np.hstack((points_3d, np.ones((num_points, 1))))
    
    # Transform points from LiDAR frame to Camera frame
    # P_cam_hom = lidar_to_cam_rt @ P_lidar_hom
    # points_3d_hom.T is (4, N), lidar_to_cam_rt is (4,4)
    # result is (4,N), then transpose to (N,4)
    points_cam_hom = (lidar_to_cam_rt @ points_3d_hom.T).T
    
    # Get points in camera frame (N, 3)
    points_cam = points_cam_hom[:, :3]
    
    # Extract depth (Z coordinate in camera frame)
    depth = points_cam[:, 2]
    
    # Project points to image plane using intrinsic matrix
    # P_img_hom = K @ P_cam
    # points_cam.T is (3,N), cam_intrinsic is (3,3)
    # result is (3,N), then transpose to (N,3)
    points_img_hom = (cam_intrinsic @ points_cam.T).T
    
    # Normalize by the Z coordinate (depth) to get pixel coordinates
    # (x, y) = (X/Z, Y/Z)
    # Add a small epsilon to avoid division by zero, though points with depth <= 0 should be filtered later.
    epsilon = 1e-8
    points_2d = points_img_hom[:, :2] / (points_img_hom[:, 2, np.newaxis] + epsilon)
    
    return points_2d, depth
def fill_trainval_infos(
    data_path, nusc, train_scenes, test=False, max_sweeps=10, with_camera=False, max_frames=None,
    dino_model_instance=None,
    torch_device=None,
    output_root_path=None,
    dino_target_h=518,
    dino_target_w=518,
    dino_patch_s=14,
    dino_embed_dimension=64
):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(
        total=len(nusc.sample) if max_frames is None else max_frames, desc="create_info", dynamic_ncols=True
    )

    ref_chan = "LIDAR_TOP"  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.
    dino_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((dino_target_h, dino_target_w), interpolation=Image.BICUBIC),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    camera_types_list_for_dino = [ # 确保 camera_types 在DINO部分被正确引用
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]
    
    for index, sample in enumerate(nusc.sample):
        if max_frames is not None and index >= max_frames:
            break  # 限制处理的帧数
        progress_bar.update()

        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = nusc.get("sample_data", ref_sd_token)
        ref_cs_rec = nusc.get(
            "calibrated_sensor", ref_sd_rec["calibrated_sensor_token"]
        )
        ref_pose_rec = nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample["data"]["CAM_FRONT"]
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )
        info = {
            "lidar_path": Path(ref_lidar_path).relative_to(data_path).__str__(),
            "lidar_token": ref_sd_token,
            "cam_front_path": Path(ref_cam_path).relative_to(data_path).__str__(),
            "cam_intrinsic": ref_cam_intrinsic,
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": ref_from_car,
            "car_from_global": car_from_global,
            "timestamp": ref_time,
        }
        if with_camera:
            info["cams"] = dict()
            l2e_r = ref_cs_rec["rotation"]
            l2e_t = (ref_cs_rec["translation"],)
            e2g_r = ref_pose_rec["rotation"]
            e2g_t = ref_pose_rec["translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"][cam]
                cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
                cam_info = obtain_sensor2top(
                    nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info["data_path"] = (
                    Path(cam_info["data_path"]).relative_to(data_path).__str__()
                )
                cam_info.update(camera_intrinsics=camera_intrinsics)
                info["cams"].update({cam: cam_info})

        sample_data_token = sample["data"][chan]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec["prev"] == "":
                if len(sweeps) == 0:
                    sweep = {
                        "lidar_path": Path(ref_lidar_path)
                        .relative_to(data_path)
                        .__str__(),
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": None,
                        "time_lag": curr_sd_rec["timestamp"] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

                # Get past pose
                current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])
                global_from_car = transform_matrix(
                    current_pose_rec["translation"],
                    Quaternion(current_pose_rec["rotation"]),
                    inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"]
                )
                car_from_current = transform_matrix(
                    current_cs_rec["translation"],
                    Quaternion(current_cs_rec["rotation"]),
                    inverse=False,
                )

                tm = reduce(
                    np.dot,
                    [ref_from_car, car_from_global, global_from_car, car_from_current],
                )

                lidar_path = nusc.get_sample_data_path(curr_sd_rec["token"])

                time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                sweep = {
                    "lidar_path": Path(lidar_path).relative_to(data_path).__str__(),
                    "sample_data_token": curr_sd_rec["token"],
                    "transform_matrix": tm,
                    "global_from_car": global_from_car,
                    "car_from_current": car_from_current,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        assert len(info["sweeps"]) == max_sweeps - 1, (
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, "
            f"you should duplicate to sweep num {max_sweeps - 1}"
        )

         # ========== DINO特征提取 (多相机融合版本) ==========
        try:
            # 1. 加载原始LiDAR点云
            lidar_abs_path = os.path.join(data_path, info["lidar_path"])
            pc_obj = LidarPointCloud.from_file(lidar_abs_path)
            points_lidar_raw_xyz = pc_obj.points[:3, :].T  # (N_lidar, 3)
            num_lidar_points = points_lidar_raw_xyz.shape[0]
            
            if num_lidar_points == 0:
                print(f"[警告] 样本 {sample['token']} 的LiDAR点云为空。跳过DINO特征提取。")
                # 仍然需要为 all_lidar_points_dino_feat 创建一个空数组或符合期望的占位符
                all_lidar_points_dino_feat = np.zeros((0, dino_embed_dimension), dtype=np.float32)
                # 保存空的DINO特征
                dino_feat_dir_rel = "dino_features_multi_cam_full_lidar"
                dino_feat_filename = f"{sample['token']}.npz"
                info_dino_feat_path = os.path.join(dino_feat_dir_rel, dino_feat_filename)
                dino_feat_abs_path = os.path.join(output_root_path, info_dino_feat_path) # 使用config_args.output_root
                os.makedirs(os.path.dirname(dino_feat_abs_path), exist_ok=True)
                np.savez(dino_feat_abs_path, feat=all_lidar_points_dino_feat)
                info["dino_feat_path"] = info_dino_feat_path
                # 后续的GT处理等仍会进行
            else:
                # 2. 初始化DINO特征数组和辅助数组 (用于特征选择)
                all_lidar_points_dino_feat = np.zeros((num_lidar_points, dino_embed_dimension), dtype=np.float32)
                # 用于记录每个LiDAR点已分配特征的来源相机的深度，以便后续比较选择更优的
                # 初始化为无穷大，表示尚未分配或深度非常远
                assigned_depth_for_lidar_points = np.full(num_lidar_points, np.inf, dtype=np.float32)
                # 标记是否从CAM_FRONT获取了特征
                assigned_from_cam_front = np.zeros(num_lidar_points, dtype=bool)


                # 3. 遍历所有相关相机视角
                for cam_type in camera_types_list_for_dino:
                    cam_sd_token = sample["data"][cam_type]
                    cam_sd_rec = nusc.get("sample_data", cam_sd_token)
                    cam_cs_rec = nusc.get("calibrated_sensor", cam_sd_rec["calibrated_sensor_token"])
                    cam_pose_rec = nusc.get("ego_pose", cam_sd_rec["ego_pose_token"]) # 相机时刻的自车位姿

                    # 加载图像
                    cam_img_path_abs = nusc.get_sample_data_path(cam_sd_token)
                    if not os.path.exists(cam_img_path_abs):
                        # print(f"[警告] 样本 {sample['token']}, 相机 {cam_type}: 图像文件 {cam_img_path_abs} 未找到。")
                        continue
                    
                    try:
                        image_pil = Image.open(cam_img_path_abs).convert('RGB')
                    except Exception as e_img:
                        print(f"[错误] 样本 {sample['token']}, 相机 {cam_type}: 加载图像 {cam_img_path_abs} 失败: {e_img}")
                        continue
                    
                    original_w, original_h = image_pil.size
                    
                    # 获取原始相机内参
                    cam_intrinsic_orig = np.array(cam_cs_rec["camera_intrinsic"])

                    # 计算从LiDAR坐标系到当前相机坐标系的变换矩阵 (lidar_to_camera_rt)
                    # LIDAR_TOP -> ego_vehicle (at LIDAR timestamp)
                    lidar_to_ego_lidar_ts = transform_matrix(
                        translation=ref_cs_rec["translation"],
                        rotation=Quaternion(ref_cs_rec["rotation"]),
                        inverse=False
                    )
                    # ego_vehicle (at LIDAR timestamp) -> global
                    ego_lidar_ts_to_global = transform_matrix(
                        translation=ref_pose_rec["translation"],
                        rotation=Quaternion(ref_pose_rec["rotation"]),
                        inverse=False
                    )
                    # global -> ego_vehicle (at CAMERA timestamp)
                    global_to_ego_cam_ts = transform_matrix(
                        translation=cam_pose_rec["translation"],
                        rotation=Quaternion(cam_pose_rec["rotation"]),
                        inverse=True
                    )
                    # ego_vehicle (at CAMERA timestamp) -> CURRENT_CAMERA
                    ego_cam_ts_to_current_cam = transform_matrix(
                        translation=cam_cs_rec["translation"],
                        rotation=Quaternion(cam_cs_rec["rotation"]),
                        inverse=True
                    )
                    lidar_to_current_camera_rt = reduce(
                        np.dot,
                        [
                            ego_cam_ts_to_current_cam,
                            global_to_ego_cam_ts,
                            ego_lidar_ts_to_global,
                            lidar_to_ego_lidar_ts,
                        ],
                    )

                    # 提取稠密的DINO特征图
                    img_tensor = dino_transform(image_pil).unsqueeze(0).to(torch_device)
                    with torch.inference_mode():
                        # 假设DINO模型输出 'x_norm_patchtokens'
                        # 如果你的模型输出不同，需要调整这里的键名
                        dino_output = dino_model.forward_features(img_tensor)
                        patch_tokens = dino_output.get('x_norm_patchtokens', dino_output.get('x_prenorm')) 
                        if patch_tokens is None and isinstance(dino_output, torch.Tensor): # 有些模型可能直接输出tensor
                            patch_tokens = dino_output
                        if patch_tokens is None:
                             raise KeyError(f"DINO model output does not contain 'x_norm_patchtokens' or 'x_prenorm'. Available keys: {dino_output.keys() if isinstance(dino_output, dict) else 'N/A (Tensor output)'}")


                    # 将patch token重塑为 (B, C, H_patch, W_patch) 以便插值
                    # DINO ViT输出通常是 (B, NumPatches, C)
                    # NumPatches = (target_h // patch_size) * (target_w // patch_size)
                    num_patches_h = dino_target_h // dino_patch_s
                    num_patches_w = dino_target_w // dino_patch_s
                    patch_tokens_reshaped = patch_tokens.permute(0, 2, 1).reshape(1, dino_embed_dimension, num_patches_h, num_patches_w)
                    
                    # 使用双线性插值上采样到目标图像尺寸 (target_h, target_w)
                    # 输出形状 (B, C, target_h, target_w)
                    dense_dino_feat_map = F.interpolate(
                        patch_tokens_reshaped,
                        size=(dino_target_h, dino_target_w),
                        mode='bilinear',
                        align_corners=False # 通常对于特征图设为False
                    ).squeeze(0)  # (C, target_h, target_w)

                    # 调整相机内参以匹配DINO模型的输入图像尺寸 (target_h, target_w)
                    scale_w_dino = dino_target_w / original_w
                    scale_h_dino = dino_target_h / original_h
                    adjusted_cam_intrinsic_dino = np.copy(cam_intrinsic_orig)
                    adjusted_cam_intrinsic_dino[0, 0] *= scale_w_dino
                    adjusted_cam_intrinsic_dino[1, 1] *= scale_h_dino
                    adjusted_cam_intrinsic_dino[0, 2] *= scale_w_dino
                    adjusted_cam_intrinsic_dino[1, 2] *= scale_h_dino

                    # 投影LiDAR点到当前相机处理后的图像平面
                    projected_uv, depth_in_cam_coord = project_lidar_to_image(
                        points_lidar_raw_xyz,
                        adjusted_cam_intrinsic_dino,
                        lidar_to_current_camera_rt
                    )

                    # 筛选有效投影点并采样DINO特征
                    # valid_projection_mask: 点在相机前方 (depth > 0) 且在图像边界内
                    valid_mask_current_cam = (
                        (depth_in_cam_coord > 1e-3) &
                        (projected_uv[:, 0] >= 0) & (projected_uv[:, 0] < dino_target_w -1) & # -1 for grid_sample safety
                        (projected_uv[:, 1] >= 0) & (projected_uv[:, 1] < dino_target_w -1)  # -1 for grid_sample safety
                    )
                    
                    if not np.any(valid_mask_current_cam):
                        continue # 当前相机没有有效的LiDAR点投影

                    # 获取有效投影点的2D坐标 (归一化到 [-1, 1] for grid_sample)
                    # grid_sample期望的坐标是 (N, H_out, W_out, 2) 或 (N, 2, H_out, W_out)
                    # 这里我们为每个点采样，所以 H_out=1, W_out=1
                    # (u,v) 映射到 (x,y) in [-1,1]: x = 2*u/W - 1, y = 2*v/H - 1
                    uv_valid_pixels = projected_uv[valid_mask_current_cam]
                    normalized_uv = np.zeros_like(uv_valid_pixels)
                    normalized_uv[:, 0] = 2.0 * uv_valid_pixels[:, 0] / (dino_target_w -1) - 1.0 # x
                    normalized_uv[:, 1] = 2.0 * uv_valid_pixels[:, 1] / (dino_target_h -1) - 1.0 # y
                    
                    # 将归一化坐标转换为Tensor (1, N_valid, 1, 2) for grid_sample
                    # grid_sample的输入是 (B, C, H_in, W_in), grid是 (B, H_out, W_out, 2)
                    # 我们要对每个点采样，所以 H_out=N_valid, W_out=1
                    grid = torch.from_numpy(normalized_uv).float().to(torch_device).unsqueeze(0).unsqueeze(2) # (1, N_valid, 1, 2)

                    # dense_dino_feat_map is (C, H, W), unsqueeze to (1, C, H, W)
                    sampled_features_tensor = F.grid_sample(
                        dense_dino_feat_map.unsqueeze(0), # (1, C, target_h, target_w)
                        grid,                               # (1, N_valid, 1, 2)
                        mode='bilinear',
                        padding_mode='border', # 'zeros', 'border', 'reflection'
                        align_corners=False
                    ).squeeze().T # (N_valid, C) after squeeze and transpose
                    
                    sampled_features_np = sampled_features_tensor.cpu().numpy().astype(np.float32)

                    # 更新 all_lidar_points_dino_feat
                    # 策略：
                    # 1. 如果点之前未被CAM_FRONT赋值，且当前是CAM_FRONT，则赋值。
                    # 2. 如果点之前未被任何相机赋值 (depth is inf)，则赋值。
                    # 3. 如果点已被赋值，但当前相机提供的深度更小，则更新。
                    # 4. CAM_FRONT有优先权覆盖非CAM_FRONT的赋值，前提是CAM_FRONT的深度也有效。
                    
                    indices_of_valid_points_in_original_lidar = np.where(valid_mask_current_cam)[0]
                    depths_for_these_valid_points = depth_in_cam_coord[valid_mask_current_cam]

                    for i, original_idx in enumerate(indices_of_valid_points_in_original_lidar):
                        current_depth = depths_for_these_valid_points[i]
                        
                        # 优先CAM_FRONT的逻辑
                        if cam_type == "CAM_FRONT":
                            # 如果之前未被CAM_FRONT赋值，或者当前深度更小
                            if not assigned_from_cam_front[original_idx] or \
                               current_depth < assigned_depth_for_lidar_points[original_idx]:
                                all_lidar_points_dino_feat[original_idx] = sampled_features_np[i]
                                assigned_depth_for_lidar_points[original_idx] = current_depth
                                assigned_from_cam_front[original_idx] = True
                        else: # 非 CAM_FRONT
                            # 只有在点未被CAM_FRONT赋值，并且当前深度更优时才更新
                            if not assigned_from_cam_front[original_idx] and \
                               current_depth < assigned_depth_for_lidar_points[original_idx]:
                                all_lidar_points_dino_feat[original_idx] = sampled_features_np[i]
                                assigned_depth_for_lidar_points[original_idx] = current_depth
                
                dino_feat_dir_rel = "dino_features_multi_cam_full_lidar" # 新的目录名
                dino_feat_filename = f"{sample['token']}.npz"
                info_dino_feat_path = os.path.join(dino_feat_dir_rel, dino_feat_filename)
                
                # 保存的绝对路径使用 config_args.output_root
                dino_feat_abs_path = os.path.join(output_root_path, info_dino_feat_path)
                os.makedirs(os.path.dirname(dino_feat_abs_path), exist_ok=True)
                
                np.savez(
                    dino_feat_abs_path,
                    feat=all_lidar_points_dino_feat # (N_lidar, dino_embed_dim)
                )
                info["dino_feat_path"] = info_dino_feat_path

        except Exception as e:
            print(f"[DINO提取错误] 处理样本 {sample['token']} 时发生错误: {e}")
            traceback.print_exc()
            # 如果出错，确保info中没有错误的dino_feat_path，或者保存一个空的/占位符npz
            # 这里选择不设置dino_feat_path，让后续加载时处理缺失
            if "dino_feat_path" in info:
                del info["dino_feat_path"]
            # 也可以选择保存一个全零的特征文件，以避免后续加载失败
            # num_lidar_points_at_error = LidarPointCloud.from_file(os.path.join(data_path, info["lidar_path"])).points.shape[1]
            # error_feat = np.zeros((num_lidar_points_at_error, dino_embed_dim), dtype=np.float32)
            # error_dino_feat_dir_rel = "dino_features_multi_cam_full_lidar"
            # error_dino_feat_filename = f"{sample['token']}_error.npz" # 标记为错误文件
            # error_info_dino_feat_path = os.path.join(error_dino_feat_dir_rel, error_dino_feat_filename)
            # error_dino_feat_abs_path = os.path.join(config_args.output_root, error_info_dino_feat_path)
            # os.makedirs(os.path.dirname(error_dino_feat_abs_path), exist_ok=True)
            # np.savez(error_dino_feat_abs_path, feat=error_feat)
            # info["dino_feat_path"] = error_info_dino_feat_path # 指向错误文件

        # ========== DINO特征提取 END ==========

        if not test:
            # processing gt bbox
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array([anno["num_lidar_pts"] for anno in annotations])
            num_radar_pts = np.array([anno["num_radar_pts"] for anno in annotations])
            mask = num_lidar_pts + num_radar_pts > 0

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[
                :, [1, 0, 2]
            ]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(
                -1, 1
            )
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)

            info["gt_boxes"] = gt_boxes[mask, :]
            info["gt_boxes_velocity"] = velocity[mask, :]
            info["gt_names"] = np.array(
                [map_name_from_general_to_detection[name] for name in names]
            )[mask]
            info["gt_boxes_token"] = tokens[mask]
            info["num_lidar_pts"] = num_lidar_pts[mask]
            info["num_radar_pts"] = num_radar_pts[mask]

            # processing gt segment
            segment_path = nusc.get("lidarseg", ref_sd_token)["filename"]
            info["gt_segment_path"] = segment_path

        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, help="Path to the nuScenes dataset."
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where processed information located.",
    )
    parser.add_argument(
        "--max_sweeps", default=10, type=int, help="Max number of sweeps. Default: 10."
    )
    parser.add_argument(
        "--max_frames", default=None, type=int, help="Maximum number of frames to process (for debugging)"
    )
    parser.add_argument(
        "--with_camera",
        action="store_true",
        default=False,
        help="Whether use camera or not.",
    )
    config = parser.parse_args()

    print(f"Loading nuScenes tables for version v1.0-trainval...")
    nusc_trainval = NuScenes(
        version="v1.0-trainval", dataroot=config.dataset_root, verbose=False
    )
    available_scenes_trainval = get_available_scenes(nusc_trainval)
    available_scene_names_trainval = [s["name"] for s in available_scenes_trainval]
    print("total scene num:", len(nusc_trainval.scene))
    print("exist scene num:", len(available_scenes_trainval))
    assert len(available_scenes_trainval) == len(nusc_trainval.scene) == 850

    # print(f"Loading nuScenes tables for version v1.0-test...")
    # nusc_test = NuScenes(
    #     version="v1.0-test", dataroot=config.dataset_root, verbose=False
    # )
    # available_scenes_test = get_available_scenes(nusc_test)
    # available_scene_names_test = [s["name"] for s in available_scenes_test]
    # print("total scene num:", len(nusc_test.scene))
    # print("exist scene num:", len(available_scenes_test))
    # assert len(available_scenes_test) == len(nusc_test.scene) == 150

    train_scenes = splits.train
    train_scenes = set(
        [
            available_scenes_trainval[available_scene_names_trainval.index(s)]["token"]
            for s in train_scenes
        ]
    )
    # test_scenes = splits.test
    # test_scenes = set(
    #     [
    #         available_scenes_test[available_scene_names_test.index(s)]["token"]
    #         for s in test_scenes
    #     ]
    # )
    # ========== 加载DINO模型 ==========
    print("Loading DINO model...")
    from dinov2.models.vision_transformer import vit_small
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 这些是DINO ViT-S/14 的标准参数
    dino_model_patch_size = 14
    dino_model_img_size = 518 # DINO模型训练时的图像大小
    dino_model_embed_dim = 64 # ViT-S 的嵌入维度
    dino_model = vit_small(patch_size=dino_model_patch_size, img_size=dino_model_img_size,init_values=1e-5,block_chunks=0)
    checkpoint = torch.load("/home/jmzhou/dinov2/dinov2_vits14_pretrain.pth", map_location=device)
    dino_model.load_state_dict(checkpoint)
    dino_model.eval()
    dino_model.to(device)
    # ========== 加载DINO模型 END ==========
    print(f"Filling trainval information...")
    train_nusc_infos, val_nusc_infos = fill_trainval_infos(
        config.dataset_root,
        nusc_trainval,
        train_scenes,
        test=False,
        max_sweeps=config.max_sweeps,
        with_camera=config.with_camera,
        max_frames=config.max_frames,
        dino_model_instance=dino_model,
        torch_device=device,
        output_root_path=config.output_root,
        dino_target_h=dino_model_img_size,
        dino_target_w=dino_model_img_size,
        dino_patch_s=dino_model_patch_size,
        dino_embed_dimension=dino_model_embed_dim
    )
    # print(f"Filling test information...")
    # test_nusc_infos, _ = fill_trainval_infos(
    #     config.dataset_root,
    #     nusc_test,
    #     test_scenes,
    #     test=True,
    #     max_sweeps=config.max_sweeps,
    #     with_camera=config.with_camera,
    # )

    print(f"Saving nuScenes information...")
    os.makedirs(os.path.join(config.output_root, "info"), exist_ok=True)
    print(
        f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
    )
    with open(
        os.path.join(
            config.output_root,
            "info",
            f"nuscenes_infos_{config.max_sweeps}sweeps_train.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(train_nusc_infos, f)
    with open(
        os.path.join(
            config.output_root,
            "info",
            f"nuscenes_infos_{config.max_sweeps}sweeps_val.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(val_nusc_infos, f)
    # with open(
    #     os.path.join(
    #         config.output_root,
    #         "info",
    #         f"nuscenes_infos_{config.max_sweeps}sweeps_test.pkl",
    #     ),
    #     "wb",
    # ) as f:
    #     pickle.dump(test_nusc_infos, f)
