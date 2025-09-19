"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from collections.abc import Sequence
import pickle

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class NuScenesDataset(DefaultDataset):
    def __init__(self, sweeps=10, ignore_index=-1, **kwargs):
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        strength = points[:, 3].reshape([-1, 1]) / 255  # scale strength to [0, 1]

        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        
        # 加载 DINO 特征（如果存在）
        if "dino_feat_path" in data.keys() and data["dino_feat_path"] is not None:
            dino_feat_path = os.path.join(self.data_root, data["dino_feat_path"])
            if os.path.exists(dino_feat_path):
                dino_data = np.load(dino_feat_path)
                dino_feat = dino_data["feat"]   # (N, C)
            else:
                dino_feat = np.zeros((0, 64), dtype=np.float32)  # 假设降维到 64
        else:
            dino_feat = np.zeros((0, 64), dtype=np.float32)
        # # debug
        print(f"Coord shape: {coord.shape}")  # 点云坐标
        print(f"Strength shape: {strength.shape}")  # 点云强度
        print(f"DINO Feat shape: {dino_feat.shape}")  # DINO 特征
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            dino_feat=dino_feat,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["lidar_token"]

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map
