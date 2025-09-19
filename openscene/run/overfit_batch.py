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
from torch.utils.tensorboard import SummaryWriter

def get_parser():
    '''Parse the config file.'''
    parser = argparse.ArgumentParser(description='Overfit a single batch for debugging.')
    parser.add_argument('--config', type=str,
                        default='config/scannet/train_ptv3.yaml',
                        help='config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    return cfg

def get_logger():
    '''Define logger.'''
    logger_name = "main-logger"
    logger_in = logging.getLogger(logger_name)
    logger_in.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger_in.addHandler(handler)
    return logger_in

def main():
    '''Main function for overfitting.'''
    args = get_parser()
    
    # Simple setup, single GPU only for overfitting test
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.train_gpu[0])  # Only use first GPU
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    logger = get_logger()
    logger.info(args)

    # 新增：初始化TensorBoard
    writer = SummaryWriter(args.save_path if hasattr(args, 'save_path') else './out/overfit_debug')

    # Get model
    logger.info("=> creating model ...")
    model = Model(cfg=args).cuda()
    model.train()

    # Get optimizer
    if hasattr(args, 'optimizer') and args.optimizer.type == 'AdamW':
        if hasattr(args, 'param_dicts'):
            param_groups = []
            for p_dict in args.param_dicts:
                keyword = p_dict['keyword']
                params = [p for n, p in model.named_parameters() if keyword in n and p.requires_grad]
                group_config = p_dict.copy()
                del group_config['keyword']
                group_config['params'] = params
                param_groups.append(group_config)
            
            assigned_params = [p for group in param_groups for p in group['params']]
            assigned_params_set = set(assigned_params)
            default_params = [p for p in model.parameters() if p.requires_grad and p not in assigned_params_set]
            
            if default_params:
                param_groups.append({'params': default_params})
            param_dicts = param_groups
        else:
            param_dicts = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(param_dicts, lr=args.optimizer.lr, weight_decay=0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    # Add scheduler to match the main training script
    scheduler = None
    total_steps = 1000  # Match the overfitting loop iterations
    if hasattr(args, 'scheduler') and args.scheduler.type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.scheduler.max_lr,
            total_steps=total_steps,  # Total number of iterations in the overfitting loop
            pct_start=args.scheduler.pct_start,
            anneal_strategy=args.scheduler.anneal_strategy,
            div_factor=args.scheduler.div_factor,
            final_div_factor=args.scheduler.final_div_factor
        )

    # Get data loader
    train_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                    datapath_prefix_feat=args.data_root_2d_fused_feature,
                                    voxel_size=args.voxel_size,
                                    split='train', aug=False,  # No augmentation for overfitting
                                    memcache_init=False,
                                    loop=1,
                                    input_color=args.input_color)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,
                                               shuffle=True,  # Shuffle to get a random batch
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=collation_fn)
    
    # Fetch a single batch
    logger.info("Fetching a single batch to overfit...")
    single_batch = next(iter(train_loader))
    logger.info("Batch fetched. Starting overfitting loop...")

    # Overfitting loop
    for i in tqdm(range(1000)):
        (coords, feat, _, feat_3d, mask) = single_batch
        
        # Move to GPU
        coords_gpu = coords.cuda(non_blocking=True)
        feat_gpu = feat.cuda(non_blocking=True)
        feat_3d_gpu = feat_3d.cuda(non_blocking=True)
        mask_gpu = mask.cuda(non_blocking=True)

        # --- Correct Data Flow Fix ---
        # Input all points to the model, but only compute loss on points with 2D features
        
        # Use all points as input to the 3D model
        sinput = SparseTensor(feat_gpu, coords_gpu)
        
        # Get 3D features for all points
        output_3d = model(sinput)
        
        # Only extract features for points that have corresponding 2D features
        output_3d_masked = output_3d[mask_gpu]
        features_for_input = feat_3d_gpu

        if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
            loss = (1 - torch.nn.CosineSimilarity()(output_3d_masked, features_for_input)).mean()
        elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
            loss = torch.nn.L1Loss()(output_3d_masked, features_for_input)
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        # 新增：写入loss到TensorBoard
        writer.add_scalar('Loss/train', loss.item(), i+1)

        if (i + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Iteration [{i+1}/1000], Loss: {loss.item():.8f}, LR: {current_lr:.6f}')

    logger.info("Overfitting finished. Final Loss: {:.8f}".format(loss.item()))
    if loss.item() < 0.1:
        logger.info("SUCCESS: Loss has converged to a small value. The training pipeline is likely correct.")
    else:
        logger.warning("WARNING: Loss did not converge to a small value. There might be an issue in the pipeline.")

    # 新增：关闭writer
    writer.close()

if __name__ == '__main__':
    main() 