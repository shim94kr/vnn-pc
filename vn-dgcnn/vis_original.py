import os
import sys
import copy
import argparse
import importlib

import trimesh
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from pdb import set_trace
from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.utils import trans_rmat_to_pmat, trans_quat_to_pmat, \
    quaternion_to_rmat, save_pc, Rotation3D


@torch.no_grad()
def visualize(cfg):
    # Initialize dataloaders
    val_loader, _ = build_dataloader(cfg)
    val_dst = val_loader.dataset

    # save some predictions for visualization
    vis_lst, loss_lst = [], []
    for batch in tqdm(val_loader):
        batch = {k: v.float().cuda() for k, v in batch.items()}
        # convert all the rotations to quaternion for simplicity
        out_dict = {
            'data_id': batch['data_id'].long(),
            'gt_trans': batch['part_trans'],
            'gt_quat': batch['part_quat'].to_quat(),
            'part_valids': batch['part_valids'].long(),
        }
        
    # apply the predicted transforms to the original meshes and save them
    save_dir = os.path.join(
        os.path.dirname(cfg.exp.weight_file), 'vis', args.category)
    top_idx = range(len(val_loader))
    for rank, idx in enumerate(top_idx):
        out_dict = vis_lst[idx]
        data_id = out_dict['data_id']
        mesh_dir = os.path.join(val_dst.data_dir, val_dst.data_list[data_id])
        mesh_files = os.listdir(mesh_dir)
        mesh_files.sort()
        assert len(mesh_files) == out_dict['part_valids'].sum()
        subfolder_name = f"rank{rank}-{len(mesh_files)}pcs-"\
                         f"{mesh_dir.split('/')[-1]}"
        cur_save_dir = os.path.join(save_dir,
                                    mesh_dir.split('/')[-2], subfolder_name)
        os.makedirs(cur_save_dir, exist_ok=True)
        for i, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(os.path.join(mesh_dir, mesh_file))
            mesh.export(os.path.join(cur_save_dir, mesh_file))
            # R^T (mesh - T) --> init_mesh
            gt_trans, gt_quat = \
                out_dict['gt_trans'][i], out_dict['gt_quat'][i]
            gt_rmat = quaternion_to_rmat(gt_quat)
            init_trans = -(gt_rmat.T @ gt_trans)
            init_rmat = gt_rmat.T
            init_pmat = trans_rmat_to_pmat(init_trans, init_rmat)
            init_mesh = mesh.apply_transform(init_pmat)
            init_mesh.export(os.path.join(cur_save_dir, f'input_{mesh_file}'))
            init_pc = trimesh.sample.sample_surface(init_mesh,
                                                    val_dst.num_points)[0]
            save_pc(init_pc,
                    os.path.join(cur_save_dir, f'input_{mesh_file[:-4]}.ply'))
            # predicted pose
            
    print(f'Saving {len(top_idx)} predictions for visualization...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument('--cfg_file', required=True, type=str, help='.py')
    parser.add_argument('--category', type=str, default='', help='data subset')
    parser.add_argument('--min_num_part', type=int, default=-1)
    parser.add_argument('--gpus', nargs='+', default=[3], type=int)
    parser.add_argument('--max_num_part', type=int, default=-1)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--vis', type=int, default=-1, help='visualization')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.cfg_file))
    cfg = importlib.import_module(os.path.basename(args.cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()
    parallel_strategy = 'ddp'  # 'dp'
    cfg.exp.gpus = args.gpus
    # manually increase batch_size according to the number of GPUs in DP
    # not necessary in DDP because it's already per-GPU batch size
    if len(cfg.exp.gpus) > 1 and parallel_strategy == 'dp':
        cfg.exp.batch_size *= len(cfg.exp.gpus)
        cfg.exp.num_workers *= len(cfg.exp.gpus)
    if args.category:
        cfg.data.category = args.category
    if args.min_num_part > 0:
        cfg.data.min_num_part = args.min_num_part
    if args.max_num_part > 0:
        cfg.data.max_num_part = args.max_num_part
    if args.weight:
        cfg.exp.weight_file = args.weight

    cfg_backup = copy.deepcopy(cfg)
    cfg.freeze()
    print(cfg)

    if not args.category:
        args.category = 'all'
    visualize(cfg)