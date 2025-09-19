# 极端过拟合测试 - 验证蒸馏损失的理论下限
import os
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data

from MinkowskiEngine import SparseTensor
from util import config
from dataset.feature_loader import FusedFeatureLoader, collation_fn
from models.disnet import DisNet as Model
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='Extreme overfitting test for loss bounds.')
    parser.add_argument('--config', type=str,
                        default='config/scannet/train_ptv3.yaml',
                        help='config file')
    args_in = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    return cfg

def main():
    args = get_parser()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    print("=== 极端过拟合测试：验证蒸馏损失下限 ===")
    
    # 创建模型
    model = Model(cfg=args).cuda()
    model.train()

    # 使用更激进的优化策略
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 更高学习率
    
    # 获取单个batch
    train_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                    datapath_prefix_feat=args.data_root_2d_fused_feature,
                                    voxel_size=args.voxel_size,
                                    split='train', aug=False,
                                    memcache_init=False,
                                    loop=1,
                                    input_color=args.input_color)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=collation_fn)
    
    single_batch = next(iter(train_loader))
    print("获取batch完成，开始极端过拟合...")

    best_loss = float('inf')
    patience_counter = 0
    
    # 极端过拟合：5000次迭代
    for i in tqdm(range(5000)):
        (coords, feat, _, feat_3d, mask) = single_batch
        
        coords_gpu = coords.cuda(non_blocking=True)
        feat_gpu = feat.cuda(non_blocking=True)
        feat_3d_gpu = feat_3d.cuda(non_blocking=True)
        mask_gpu = mask.cuda(non_blocking=True)

        sinput = SparseTensor(feat_gpu, coords_gpu)
        output_3d = model(sinput)
        output_3d_masked = output_3d[mask_gpu]
        
        loss = (1 - torch.nn.CosineSimilarity()(output_3d_masked, feat_3d_gpu)).mean()

        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录最佳损失
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        # 动态学习率调整
        if i == 1000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.005
        elif i == 2000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
        elif i == 3000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0005

        if (i + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'迭代 [{i+1}/5000], 当前损失: {loss.item():.6f}, 最佳损失: {best_loss:.6f}, LR: {current_lr:.6f}')

        # 早停机制：如果1000次迭代没有改善，提前停止
        if patience_counter > 1000:
            print(f"在迭代 {i+1} 处早停，已连续 {patience_counter} 次迭代无改善")
            break

    print(f"\n=== 极端过拟合测试完成 ===")
    print(f"最终损失: {loss.item():.6f}")
    print(f"最佳损失: {best_loss:.6f}")
    print(f"对应余弦相似度: {1 - best_loss:.6f}")
    
    if best_loss < 0.15:
        print("✅ 优秀：模型能够达到很低的蒸馏损失")
    elif best_loss < 0.20:
        print("✅ 良好：您之前的0.19-0.20结果已经非常接近理论最优")
    elif best_loss < 0.25:
        print("⚠️  一般：可能存在优化空间")
    else:
        print("❌ 较差：可能存在模型或数据问题")

if __name__ == '__main__':
    main() 