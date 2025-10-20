import os
import time
import random
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from MinkowskiEngine import SparseTensor
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, \
    poly_learning_rate, save_checkpoint, \
    export_pointcloud, get_palette, convert_labels_with_palette, extract_clip_feature, extract_text_feature
from dataset.label_constants import *
from dataset.feature_loader import FusedFeatureLoader, collation_fn
from dataset.point_loader import Point3DLoader, collation_fn_eval_all
from models.disnet import DisNet as Model
from models.petr_pe import TeacherPETRPEHead
from tqdm import tqdm


best_iou = 0.0
top_k_checkpoints = []  # List of (epoch, mIoU) tuples for top-k models


def worker_init_fn(worker_id):
    '''Worker initialization.'''
    random.seed(time.time() + worker_id)


def get_parser():
    '''Parse the config file.'''

    parser = argparse.ArgumentParser(description='OpenScene 3D distillation.')
    parser.add_argument('--config', type=str,
                        default='config/scannet/distill_openseg.yaml',
                        help='config file')
    parser.add_argument('opts',
                        default=None,
                        help='see config/scannet/distill_openseg.yaml for all options',
                        nargs=argparse.REMAINDER)
    args_in = parser.parse_args()
    assert args_in.config is not None
    cfg = config.load_cfg_from_cfg_file(args_in.config)
    if args_in.opts:
        cfg = config.merge_cfg_from_list(cfg, args_in.opts)
    os.makedirs(cfg.save_path, exist_ok=True)
    model_dir = os.path.join(cfg.save_path, 'model')
    result_dir = os.path.join(cfg.save_path, 'result')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_dir + '/last', exist_ok=True)
    os.makedirs(result_dir + '/best', exist_ok=True)
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


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    '''Main function.'''

    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    
    # By default we use shared memory for training
    if not hasattr(args, 'use_shm'):
        args.use_shm = True

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    
    # Handle sync_bn: read from config, but override for single GPU
    if not hasattr(args, 'sync_bn'):
        args.sync_bn = False  # Default value if not specified in config
    
    if len(args.train_gpu) == 1:
        args.sync_bn = False  # Force disable for single GPU
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False
    else:
        # Multi-GPU: keep sync_bn from config, but ensure distributed is enabled
        if args.sync_bn and not args.distributed:
            print("Warning: sync_bn=True requires distributed training. Enabling distributed mode.")
            args.distributed = True

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node,
                 args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    global top_k_checkpoints
    args = argss

    # Set CUDA device EARLY to ensure all modules are created on the correct GPU
    try:
        if args.distributed:
            torch.cuda.set_device(gpu)
        else:
            dev0 = args.train_gpu[0] if isinstance(args.train_gpu, (list, tuple)) else int(args.train_gpu)
            torch.cuda.set_device(dev0)
    except Exception:
        pass

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)

    model = get_model(args)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")

    # Add sync BatchNorm conversion for multi-GPU training (MUST be before DDP wrapping)
    if args.sync_bn and args.distributed:
        import MinkowskiEngine as ME
        if main_process():
            print(f"Converting model to synchronized BatchNorm (sync_bn={args.sync_bn}, distributed={args.distributed})")
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        if main_process():
            print("=> Successfully converted model to use synchronized BatchNorm")

    # ####################### Optimizer ####################### #
    # Build TeacherPETRPEHead (online & ema) early so its params are part of optimizer & scheduler
    petr_cfg_early = getattr(args, 'TEACHER_PETRPE', None)
    if petr_cfg_early is not None:
        pe_enable = getattr(petr_cfg_early, 'enable', False)
        pe_use_fpe = getattr(petr_cfg_early, 'use_fpe', True)
        pe_lambda = getattr(petr_cfg_early, 'lambda_pe', 1.0)
        pe_ema_m = getattr(petr_cfg_early, 'ema_m', 0.996)
        pe_head_w = getattr(petr_cfg_early, 'head_loss_weight', 0.1)
        pe_pc_range = getattr(petr_cfg_early, 'pc_range', None)
        pe_debug = getattr(petr_cfg_early, 'debug', False)
        pe_embed_dim = getattr(petr_cfg_early, 'embed_dim', 512)
    else:
        # flattened fallback
        pe_enable = getattr(args, 'enable', False)
        pe_use_fpe = getattr(args, 'use_fpe', True)
        pe_lambda = getattr(args, 'lambda_pe', 1.0)
        pe_ema_m = getattr(args, 'ema_m', 0.996)
        pe_head_w = getattr(args, 'head_loss_weight', 0.1)
        pe_pc_range = getattr(args, 'pc_range', None)
        pe_debug = getattr(args, 'debug', False)
        pe_embed_dim = getattr(args, 'embed_dim', 512)
    # Cache EMA hyperparams on args
    args._petrpe_ema_m = pe_ema_m
    args._petrpe_head_w = pe_head_w
    if pe_enable and (not hasattr(args, '_teacher_petrpe_head_online') or not hasattr(args, '_teacher_petrpe_head_ema')):
        try:
            args._teacher_petrpe_head_online = TeacherPETRPEHead(embed_dim=pe_embed_dim, use_fpe=pe_use_fpe,
                                                                 lambda_pe=pe_lambda, pc_range=pe_pc_range,
                                                                 debug=pe_debug).cuda()
            args._teacher_petrpe_head_ema = TeacherPETRPEHead(embed_dim=pe_embed_dim, use_fpe=pe_use_fpe,
                                                              lambda_pe=pe_lambda, pc_range=pe_pc_range,
                                                              debug=False).cuda()
            # EMA params不参与梯度
            for p in args._teacher_petrpe_head_ema.parameters():
                p.requires_grad = False
            # 初始化 ema = online
            args._teacher_petrpe_head_ema.load_state_dict(args._teacher_petrpe_head_online.state_dict())
        except Exception:
            args._teacher_petrpe_head_online = None
            args._teacher_petrpe_head_ema = None
    if hasattr(args, 'optimizer') and args.optimizer.type == 'AdamW':
        # Build param groups without duplication: default group = all params minus special groups
        special_groups = []
        assigned_params: set = set()
        if hasattr(args, 'param_dicts'):
            for p_dict in args.param_dicts:
                keyword = p_dict['keyword']
                special = [p for n, p in model.named_parameters() if keyword in n and p.requires_grad]
                if len(special) == 0:
                    continue
                assigned_params.update(special)
                group_config = {k: v for k, v in p_dict.items() if k != 'keyword'}
                group_config['params'] = special
                special_groups.append(group_config)

        default_params = [p for p in model.parameters() if p.requires_grad and p not in assigned_params]
        # Append ONLY online TeacherPETRPEHead params if present/enabled
        if pe_enable and hasattr(args, '_teacher_petrpe_head_online') and args._teacher_petrpe_head_online is not None:
            default_params.extend([p for p in args._teacher_petrpe_head_online.parameters() if p.requires_grad])
        param_groups = [{'params': default_params}] + special_groups
        optimizer = torch.optim.AdamW(param_groups, lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    scheduler = None
    resume_scheduler_state = None
    args.index_split = 0

    if args.distributed:
        # torch.cuda.set_device(gpu) already called above
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        # Move model to GPU first, then wrap with DDP
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            # Cache scheduler state if available; we'll restore it after creating the scheduler
            resume_scheduler_state = checkpoint.get('scheduler', None)
            # Load top-k checkpoints list if available
            if 'top_k_checkpoints' in checkpoint:
                top_k_checkpoints = checkpoint['top_k_checkpoints']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info(
                    "=> no checkpoint found at '{}'".format(args.resume))

    # ####################### Data Loader ####################### #
    if not hasattr(args, 'input_color'):
        # by default we do not use the point color as input
        args.input_color = False
    train_data = FusedFeatureLoader(datapath_prefix=args.data_root,
                                    datapath_prefix_feat=args.data_root_2d_fused_feature,
                                    voxel_size=args.voxel_size,
                                    split='train', aug=args.aug,
                                    memcache_init=args.use_shm, loop=args.loop,
                                    input_color=args.input_color
                                    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                            shuffle=(train_sampler is None),
                                            num_workers=args.workers, pin_memory=True,
                                            sampler=train_sampler,
                                            drop_last=True, collate_fn=collation_fn,
                                            worker_init_fn=worker_init_fn)

    # Create scheduler after train_loader is created
    if hasattr(args, 'scheduler') and args.scheduler.type == 'OneCycleLR':
        # Infer last_epoch when resuming without an explicit scheduler state in the checkpoint
        inferred_last_epoch = -1
        if args.resume and resume_scheduler_state is None and args.start_epoch > 0:
            inferred_last_epoch = args.start_epoch * len(train_loader) - 1

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.scheduler.max_lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.scheduler.pct_start,
            anneal_strategy=args.scheduler.anneal_strategy,
            div_factor=args.scheduler.div_factor,
            final_div_factor=args.scheduler.final_div_factor,
            last_epoch=inferred_last_epoch
        )

        # If we have an explicit scheduler state, load it to continue from exact step
        if resume_scheduler_state is not None:
            try:
                scheduler.load_state_dict(resume_scheduler_state)
            except Exception as e:
                if main_process():
                    print(f"[resume] Failed to load scheduler state ({e}); fall back to inferred last_epoch={inferred_last_epoch}.")
                scheduler.last_epoch = inferred_last_epoch
        # Print param groups and corresponding max_lr mapping for verification
        if main_process():
            try:
                from pprint import pformat
                print("OneCycle param_groups:")
                for idx, g in enumerate(optimizer.param_groups):
                    group_lr = g.get('lr', args.optimizer.lr)
                    print(f"  group[{idx}] params={len(g['params'])} initial_lr={group_lr}")
                print(f"OneCycle max_lr setting: {args.scheduler.max_lr}")
            except Exception:
                pass

    if args.evaluate:
        val_data = Point3DLoader(datapath_prefix=args.data_root,
                                 voxel_size=args.voxel_size,
                                 split='val', aug=False,
                                 memcache_init=args.use_shm,
                                 eval_all=True,
                                 input_color=args.input_color)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data) if args.distributed else None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                shuffle=False,
                                                num_workers=args.workers, pin_memory=True,
                                                drop_last=False, collate_fn=collation_fn_eval_all,
                                                sampler=val_sampler)

        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu) # for evaluation

    # ####################### Distill ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            if args.evaluate:
                val_sampler.set_epoch(epoch)
        loss_train = distill(train_loader, model, optimizer, scheduler, epoch)
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)

        is_best = False
        is_top_k = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                val_loader, model, criterion)
            # raise NotImplementedError

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)
                
                # Update top-k checkpoints list (keep top-3)
                K = 3
                top_k_checkpoints.append((epoch_log, mIoU_val))
                # Sort by mIoU descending, then by epoch descending (prefer later epochs if tied)
                top_k_checkpoints.sort(key=lambda x: (-x[1], -x[0]))
                # Keep only top-K
                top_k_checkpoints[:] = top_k_checkpoints[:K]
                
                # Check if current epoch is in top-k
                is_top_k = any(ep == epoch_log for ep, _ in top_k_checkpoints)
                
                logger.info('Top-{} checkpoints: {}'.format(
                    K, [(ep, f'{iou:.4f}') for ep, iou in top_k_checkpoints]))

        if (epoch_log % args.save_freq == 0) and main_process():
            # Always save model_last.pth.tar
            checkpoint_dict = {
                'epoch': epoch_log,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_iou': best_iou,
                'top_k_checkpoints': top_k_checkpoints
            }
            if scheduler is not None:
                try:
                    checkpoint_dict['scheduler'] = scheduler.state_dict()
                except Exception:
                    pass
            save_checkpoint(
                checkpoint_dict, is_best, os.path.join(args.save_path, 'model')
            )
            
            # Synchronize Top-K checkpoint files: always rewrite rank1..K based on latest list
            if args.evaluate and top_k_checkpoints:
                try:
                    K = min(3, len(top_k_checkpoints))
                    model_dir = os.path.join(args.save_path, 'model')
                    desired_epochs = [ep for ep, _ in top_k_checkpoints[:K]]

                    # Build mapping from epoch -> checkpoint content
                    epoch_to_ckpt = {}
                    # Prefer current epoch's in-memory checkpoint when applicable
                    if epoch_log in desired_epochs:
                        epoch_to_ckpt[epoch_log] = checkpoint_dict

                    # Load existing rank files to recover other epochs' checkpoints
                    for existing_rank in range(1, K + 1):
                        existing_path = os.path.join(model_dir, f'model_best_rank{existing_rank}.pth.tar')
                        if not os.path.isfile(existing_path):
                            continue
                        try:
                            ckpt = torch.load(existing_path, map_location='cpu')
                            ep_saved = ckpt.get('epoch', None)
                            if ep_saved in desired_epochs and ep_saved not in epoch_to_ckpt:
                                # Update embedded top-k list to the latest for consistency
                                ckpt['top_k_checkpoints'] = top_k_checkpoints
                                epoch_to_ckpt[ep_saved] = ckpt
                        except Exception:
                            pass

                    # Rewrite rank files according to latest ordering
                    for rank, (ep, iou) in enumerate(top_k_checkpoints[:K], 1):
                        topk_path = os.path.join(model_dir, f'model_best_rank{rank}.pth.tar')
                        ckpt_to_save = epoch_to_ckpt.get(ep, None)
                        if ckpt_to_save is None:
                            logger.info(f'[TopK] Missing checkpoint for epoch {ep}; skip writing rank {rank}.')
                            continue
                        # Ensure metadata reflects latest ordering
                        ckpt_to_save['top_k_checkpoints'] = top_k_checkpoints
                        torch.save(ckpt_to_save, topk_path)
                    logger.info(f'Synchronized Top-{K} checkpoint files: {[(ep, f"{iou:.4f}") for ep, iou in top_k_checkpoints[:K]]}')
                except Exception as e:
                    logger.info(f'[TopK] Synchronization skipped due to error: {e}')
    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))
        if top_k_checkpoints:
            logger.info('Top-3 checkpoints saved:')
            for rank, (ep, iou) in enumerate(top_k_checkpoints, 1):
                logger.info('  Rank {}: Epoch {} - mIoU {:.4f}'.format(rank, ep, iou))


def get_model(cfg):
    '''Get the 3D model.'''

    model = Model(cfg=cfg)

    # Optionally load a pretrained 3D checkpoint (e.g., supervised PTV3)
    if hasattr(cfg, 'pretrained_3d') and cfg.pretrained_3d:
        ckpt_path = cfg.pretrained_3d
        print(f"[pretrained_3d] Try loading: {ckpt_path}")
        if os.path.isfile(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state = checkpoint['state_dict']
                else:
                    state = checkpoint if isinstance(checkpoint, dict) else {}
            except Exception as ex:
                print(f"[pretrained_3d] Failed to load checkpoint: {ex}")
                state = {}

            msd = model.state_dict()
            loaded, skipped, mismatched = 0, 0, 0

            def candidate_keys(k: str):
                # Generate a list of candidate names in current model for a ckpt key
                names = []
                n = k.replace('module.', '')
                n = n.replace('model.', '').replace('segmentor.', '').replace('DefaultSegmentorV2.', '')
                names.append(n)
                # if contains backbone., map to net3d.backbone.*
                if 'backbone.' in n:
                    suf = n.split('backbone.', 1)[1]
                    names.append('net3d.backbone.' + suf)
                # direct net3d.*
                names.append('net3d.' + n)
                # direct (already prefixed) passthrough
                return names

            sample_print = 0
            for k, v in state.items():
                matched = False
                for cand in candidate_keys(k):
                    if cand in msd and msd[cand].shape == v.shape:
                        msd[cand].copy_(v)
                        loaded += 1
                        if sample_print < 20:
                            print(f"[pretrained_3d] + {k} -> {cand} {tuple(v.shape)}")
                            sample_print += 1
                        matched = True
                        break
                if not matched:
                    # shape-mismatch diagnostics
                    for cand in candidate_keys(k):
                        if cand in msd and msd[cand].shape != getattr(v, 'shape', None):
                            # Special-case: first conv 6->3 channel slice (spconv weight [out,kx,ky,kz,in])
                            if (
                                isinstance(v, torch.Tensor)
                                and isinstance(msd[cand], torch.Tensor)
                                and v.ndim == msd[cand].ndim == 5
                                and v.shape[:-1] == msd[cand].shape[:-1]
                                and v.shape[-1] > msd[cand].shape[-1]
                                and ('stem.conv.weight' in k or 'stem.conv.weight' in cand)
                            ):
                                msd[cand].copy_(v[..., : msd[cand].shape[-1]])
                                loaded += 1
                                if sample_print < 20:
                                    print(f"[pretrained_3d] ~ sliced load {k} -> {cand} {tuple(v.shape)} -> {tuple(msd[cand].shape)}")
                                    sample_print += 1
                                matched = True
                                break
                            mismatched += 1
                            if sample_print < 20:
                                print(f"[pretrained_3d] ! shape mismatch {k}->{cand}: ckpt {getattr(v,'shape',None)} vs model {msd[cand].shape}")
                                sample_print += 1
                            matched = True
                            break
                    if not matched:
                        skipped += 1

            model.load_state_dict(msd)
            total = len(state)
            print(f"[pretrained_3d] Loaded from {ckpt_path}: matched={loaded}, mismatched={mismatched}, skipped={skipped}, total_keys={total}")
        else:
            print(f"[pretrained_3d] File not found: {ckpt_path}")
    return model

def obtain_text_features_and_palette():
    '''obtain the CLIP text feature and palette.'''

    if 'scannet' in args.data_root:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other'
        palette = get_palette()
        dataset_name = 'scannet'
    elif 'matterport' in args.data_root:
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
        dataset_name = 'matterport'
    elif 'nuscenes' in args.data_root:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap='nuscenes16')
        dataset_name = 'nuscenes'

    # Use the same text feature pipeline as inference (supports prompt_eng)
    text_features = extract_text_feature(labelset, args)
    # Normalize text prototypes once to use cosine similarity at inference
    try:
        text_features = torch.nn.functional.normalize(text_features.float(), dim=1)
    except Exception:
        pass
    return text_features, palette


def distill(train_loader, model, optimizer, scheduler, epoch):
    '''Distillation pipeline.'''

    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    text_features, palette = obtain_text_features_and_palette()

    # --- Log PETR-PE/FPE status once ---
    if main_process():
        try:
            petr_cfg = getattr(args, 'TEACHER_PETRPE', None)
            if petr_cfg is not None:
                enable = getattr(petr_cfg, 'enable', False)
                use_fpe = getattr(petr_cfg, 'use_fpe', True)
                lambda_pe = getattr(petr_cfg, 'lambda_pe', 1.0)
                has_pc = hasattr(petr_cfg, 'pc_range')
                pc_desc = 'fixed pc_range' if has_pc else 'per-scene min-max'
                debug = getattr(petr_cfg, 'debug', False)
                logger.info(f"[PETR-PE] enable={enable} use_fpe={use_fpe} lambda_pe={lambda_pe} norm={pc_desc} debug={debug}")
            else:
                # Fallback: some configs get flattened; try top-level keys
                has_flat = any(hasattr(args, k) for k in ('enable', 'use_fpe', 'lambda_pe', 'debug'))
                if has_flat:
                    enable = getattr(args, 'enable', False)
                    use_fpe = getattr(args, 'use_fpe', True)
                    lambda_pe = getattr(args, 'lambda_pe', 1.0)
                    has_pc = hasattr(args, 'pc_range')
                    pc_desc = 'fixed pc_range' if has_pc else 'per-scene min-max'
                    debug = getattr(args, 'debug', False)
                    logger.info(f"[PETR-PE] (fallback) enable={enable} use_fpe={use_fpe} lambda_pe={lambda_pe} norm={pc_desc} debug={debug}")
                else:
                    logger.info('[PETR-PE] disabled (no TEACHER_PETRPE in cfg)')
        except Exception:
            pass

    # start the distillation process
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        (coords, feat, label_3d, feat_3d, mask) = batch_data

        # Light random translation while keeping integer coordinates
        # coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)

        # Move to GPU before building SparseTensor and sanitize fused 3D features
        feat_3d = torch.nan_to_num(feat_3d, nan=0.0, posinf=1e4, neginf=-1e4)
        sinput = SparseTensor(
            feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
        feat_3d, mask = feat_3d.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # --- Teacher PETR-PE/FPE injection (config-gated) ---
        # Support both nested (args.TEACHER_PETRPE) and flattened configs
        petr_cfg = getattr(args, 'TEACHER_PETRPE', None)
        if petr_cfg is not None:
            enable = getattr(petr_cfg, 'enable', False)
            lambda_pe = getattr(petr_cfg, 'lambda_pe', 1.0)
            use_fpe = getattr(petr_cfg, 'use_fpe', True)
            pc_range = getattr(petr_cfg, 'pc_range', None)
            debug = getattr(petr_cfg, 'debug', False)
        else:
            enable = getattr(args, 'enable', False)
            lambda_pe = getattr(args, 'lambda_pe', 1.0)
            use_fpe = getattr(args, 'use_fpe', True)
            pc_range = getattr(args, 'pc_range', None)
            debug = getattr(args, 'debug', False)

        if enable:
            embed_dim = feat_3d.shape[1]
            # create modules once
            if not hasattr(args, '_teacher_petrpe_head_online') or not hasattr(args, '_teacher_petrpe_head_ema'):
                args._teacher_petrpe_head_online = TeacherPETRPEHead(embed_dim=embed_dim, use_fpe=use_fpe, lambda_pe=lambda_pe,
                                                                     pc_range=pc_range, debug=debug).cuda()
                args._teacher_petrpe_head_ema = TeacherPETRPEHead(embed_dim=embed_dim, use_fpe=use_fpe, lambda_pe=lambda_pe,
                                                                  pc_range=pc_range, debug=False).cuda()
                for p in args._teacher_petrpe_head_ema.parameters():
                    p.requires_grad = False
                args._teacher_petrpe_head_ema.load_state_dict(args._teacher_petrpe_head_online.state_dict())
                # add only online params to optimizer if missing
                try:
                    optimizer.add_param_group({'params': [p for p in args._teacher_petrpe_head_online.parameters() if p.requires_grad]})
                except Exception:
                    pass
            # forward (align coords with fused features length using mask when needed)
            coords_pe = coords.cuda(non_blocking=True)
            if coords_pe.size(0) != feat_3d.size(0):
                try:
                    coords_pe = coords_pe[mask]
                except Exception:
                    pass
            # online / ema outputs
            t_online = args._teacher_petrpe_head_online(coords_pe, args.voxel_size, feat_3d)
            with torch.no_grad():
                t_ema = args._teacher_petrpe_head_ema(coords_pe, args.voxel_size, feat_3d)
            # debug
            if debug and main_process() and getattr(args._teacher_petrpe_head_online, '_last_debug', None) is not None:
                dbg = args._teacher_petrpe_head_online._last_debug
                try:
                    logger.info('[PETR-PE-debug] pos_min={} pos_max={} ratio_mean={:.4f} p10={:.4f} p90={:.4f} gate_mean={} gate_p10={} gate_p90={}'
                                .format(tuple(dbg['pos_min'].tolist()), tuple(dbg['pos_max'].tolist()),
                                        dbg['ratio_mean'], dbg['ratio_p10'], dbg['ratio_p90'],
                                        (None if dbg['gate_mean'] is None else f"{dbg['gate_mean']:.4f}"),
                                        (None if dbg['gate_p10'] is None else f"{dbg['gate_p10']:.4f}"),
                                        (None if dbg['gate_p90'] is None else f"{dbg['gate_p90']:.4f}")))
                except Exception:
                    pass
        else:
            t_online = feat_3d
            t_ema = feat_3d

        output_3d = model(sinput)
        # Sanitize network outputs to prevent propagation of non-finite values
        output_3d = torch.nan_to_num(output_3d, nan=0.0, posinf=1e4, neginf=-1e4)
        output_3d = output_3d[mask]

        if hasattr(args, 'loss_type') and args.loss_type == 'cosine':
            # Filter out invalid rows (zero targets or non-finite) to avoid 0/0 in cosine
            with torch.no_grad():
                valid_rows = (
                    torch.isfinite(output_3d).all(dim=1)
                    & torch.isfinite(t_ema).all(dim=1)
                    & (t_ema.abs().sum(dim=1) > 0)
                )
            if valid_rows.any():
                cosv = nn.functional.cosine_similarity(
                    output_3d[valid_rows], t_ema[valid_rows].detach(), dim=1, eps=1e-6
                )
                loss = (1 - cosv).mean()
            else:
                loss = torch.zeros((), device=output_3d.device, dtype=output_3d.dtype)
        elif hasattr(args, 'loss_type') and args.loss_type == 'l1':
            loss = torch.nn.L1Loss()(output_3d, t_ema.detach())
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Optional: online teacher head small loss to keep semantics (no student signal)
        if enable and args._petrpe_head_w > 0:
            with torch.no_grad():
                f_det = feat_3d.detach()
            # cosine: 1 - cos(t_online, f_det)
            sem_cos = nn.functional.cosine_similarity(t_online, f_det, dim=1, eps=1e-6)
            loss_sem = (1 - sem_cos).mean()
            # geometric residual alignment (use full PE, not orthogonalized)
            # get last pe projection from online head (already with grad)
            p = getattr(args._teacher_petrpe_head_online, '_last_pe', None)
            if p is None:
                p = torch.zeros_like(t_online)
            # residual towards geometry (encourage residual to align with full p)
            res = t_online - f_det
            geo_cos = nn.functional.cosine_similarity(res, p, dim=1, eps=1e-6)
            loss_geo = (1 - geo_cos).mean()
            # combine
            loss_head = float(args._petrpe_head_w) * (loss_sem + 0.5 * loss_geo)
            loss_head.backward()
        optimizer.step()

        # EMA update after optimizer.step()
        if enable and hasattr(args, '_teacher_petrpe_head_ema') and args._teacher_petrpe_head_ema is not None:
            with torch.no_grad():
                m = float(args._petrpe_ema_m)
                for p_ema, p_o in zip(args._teacher_petrpe_head_ema.parameters(), args._teacher_petrpe_head_online.parameters()):
                    p_ema.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        
        current_iter = epoch * len(train_loader) + i + 1

        # adjust learning rate
        if scheduler is not None:
            scheduler.step()
        else: # Fallback to original poly learning rate
            current_lr = poly_learning_rate(
                args.base_lr, current_iter, max_iter, power=args.power)

            for index in range(0, args.index_split):
                optimizer.param_groups[index]['lr'] = current_lr
            for index in range(args.index_split, len(optimizer.param_groups)):
                optimizer.param_groups[index]['lr'] = current_lr * 10

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(
            int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                            batch_time=batch_time, data_time=data_time,
                                                            remain_time=remain_time,
                                                            loss_meter=loss_meter))
        if main_process():
            # writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], current_iter)

        end = time.time()

    # Export: fix device mismatch by indexing with tensors on the same device as mask
    coords_device = coords.to(mask.device, non_blocking=True) if isinstance(mask, torch.Tensor) else coords
    mask_first = (coords_device[mask][:, 0] == 0)
    output_3d = output_3d[mask_first]
    # use t_ema (teacher with 3D PE/FPE) for visualization, detach to CPU
    t_ema_vis = t_ema[mask_first] if isinstance(t_ema, torch.Tensor) else feat_3d[mask_first]
    logits_pred = output_3d.half() @ text_features.t()
    logits_img = t_ema_vis.half() @ text_features.t()
    logits_pred = torch.max(logits_pred, 1)[1].cpu().numpy()
    logits_img = torch.max(logits_img, 1)[1].cpu().numpy()
    mask = mask.cpu().numpy()
    logits_gt = label_3d.numpy()[mask][mask_first.cpu().numpy()]
    logits_gt[logits_gt == 255] = args.classes

    pcl = coords_device[:, 1:].detach().cpu().numpy()

    seg_label_color = convert_labels_with_palette(
        logits_img, palette)
    pred_label_color = convert_labels_with_palette(
        logits_pred, palette)
    gt_label_color = convert_labels_with_palette(
        logits_gt, palette)
    pcl_part = pcl[mask][mask_first.cpu().numpy()]

    export_pointcloud(os.path.join(args.save_path, 'result', 'last', '{}_{}.ply'.format(
        args.feature_2d_extractor, epoch)), pcl_part, colors=seg_label_color)
    export_pointcloud(os.path.join(args.save_path, 'result', 'last',
                        'pred_{}.ply'.format(epoch)), pcl_part, colors=pred_label_color)
    export_pointcloud(os.path.join(args.save_path, 'result', 'last',
                        'gt_{}.ply'.format(epoch)), pcl_part, colors=gt_label_color)

    return loss_meter.avg


def validate(val_loader, model, criterion):
    '''Validation.'''

    torch.backends.cudnn.enabled = False
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    # obtain the CLIP feature
    text_features, _ = obtain_text_features_and_palette()

    # Ensure evaluation mode (affects BatchNorm/Dropout)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch_data in tqdm(val_loader):
            (coords, feat, label, inds_reverse) = batch_data
            sinput = SparseTensor(
                feat.cuda(non_blocking=True), coords.cuda(non_blocking=True))
            label = label.cuda(non_blocking=True)
            output = model(sinput)
            output = output[inds_reverse, :]
            text = text_features
            # Align with inference: use half precision on both sides, no normalization or scaling
            logits = (output.half() @ text.half().t())
            loss = criterion(logits.float(), label)
            output = torch.max(logits, 1)[1]

            intersection, union, target = intersectionAndUnionGPU(output, label.detach(),
                                                                  args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(
                    union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu(
            ).numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(
                union), target_meter.update(target)

            loss_meter.update(loss.item(), args.batch_size)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    # Restore training mode
    if was_training:
        model.train()
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()