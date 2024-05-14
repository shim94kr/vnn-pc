#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Congyue Deng
@Contact: congyue@stanford.edu
@File: main.py
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, GeometryPartDataset
from model import DGCNN_cls
from model_equi import EQCNN_cls
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations


def _init_():
    if not os.path.exists('results/cls'):
        os.makedirs('results/cls')
    if not os.path.exists('results/cls/'+args.exp_name):
        os.makedirs('results/cls/'+args.exp_name)
    if not os.path.exists('results/cls/'+args.exp_name+'/'+'models'):
        os.makedirs('results/cls/'+args.exp_name+'/'+'models')
    os.system('cp main_cls.py results/cls'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model_equi.py results/cls' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py results/cls' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py results/cls' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    data_dict = dict(
        data_dir=args.data_dir,
        data_fn=args.data_fn.format('train'),
        data_keys=('part_ids', ),
        category=args.category,
        num_points=args.num_pc_points,
        min_num_part=args.min_num_part,
        max_num_part=args.max_num_part,
        shuffle_parts=False,
        rot_range=args.rot_range,
        overfit=-1,
    )
    train_set = GeometryPartDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    print('Len of Train Loader: ', len(train_loader))
    data_dict['data_fn'] = args.data_fn.format('val')
    data_dict['shuffle_parts'] = False
    test_set = GeometryPartDataset(**data_dict)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    #train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
    #                          batch_size=args.batch_size, shuffle=True, drop_last=True)
    #test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
    #                         batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'eqcnn':
        model = EQCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss
    
    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(data.shape[0])*360, axis="Z", degrees=True, device=device)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(data.shape[0]), device=device)
            
            data, label = data.to(dtype=torch.float32, device=device), label.to(device).squeeze()
            if trot is not None:
                data = trot.transform_points(data)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            
            trot = None
            if args.rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(data.shape[0])*360, axis="Z", degrees=True, device=device)
            elif args.rot == 'so3':
                trot = Rotate(R=random_rotations(data.shape[0]), device=device)
            
            data, label = data.to(dtype=torch.float32, device=device), label.to(device).squeeze()
            if trot is not None:
                data = trot.transform_points(data)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'results/cls/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'eqcnn':
        model = EQCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    nparameters = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %d' % nparameters)
    model = nn.DataParallel(model)
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('results/cls', args.exp_name, 'models/model.t7')
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    
    #trot = RotateAxisAngle(angle=120, axis="Y", degrees=True, device=device)
    for data, label in test_loader:
        
        trot = None
        if args.rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(data.shape[0])*360, axis="Z", degrees=True, device=device)
        elif args.rot == 'so3':
            trot = Rotate(R=random_rotations(data.shape[0]), device=device)

        data, label = data.to(device), label.to(device).squeeze()
        if trot is not None:
            data = trot.transform_points(data)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    # python main_cls.py --exp_name=dgcnn_vnn --model=eqcnn --rot=ROTATION

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='dgcnn_vnn', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='eqcnn', metavar='N',
                        choices=['dgcnn', 'eqcnn'],
                        help='Model to use, [dgcnn, eqcnn]')
    parser.add_argument('--dataset', type=str, default='geometry', metavar='N',
                        choices=['modelnet40','geometry'])
    parser.add_argument('--data_dir', type=str, default='./BBdataset', metavar='N',
                        help='Path to the data directory')
    parser.add_argument('--data_fn', type=str, default='data_split/everyday.{}.txt', metavar='N',
                        help='Function for data split')
    parser.add_argument('--category', type=str, default='', metavar='N',
                        help='empty means all categories')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--min_num_part', type=int, default=2, help='Minimum number of parts')
    parser.add_argument('--max_num_part', type=int, default=8, help='Maximum number of parts')
    parser.add_argument('--num_pc_points', type=int, default=512, help='Number of point clouds')
    parser.add_argument('--rot_range', type=float, default=-1., help='rotation range for curriculum learning')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default=None, metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--rot', type=str, default='aligned', metavar='N',
                        choices=['aligned', 'z', 'so3'],
                        help='Rotation augmentation to input data')
    parser.add_argument('--pooling', type=str, default='mean', metavar='N',
                        choices=['mean', 'max'],
                        help='VNN only: pooling method.')
    args = parser.parse_args()

    _init_()

    io = IOStream('results/cls/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
