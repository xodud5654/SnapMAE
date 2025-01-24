# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
# from util.mixup_multi_label import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from util.multi_label_loss import SoftTargetBinaryCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, build_dataset_chest_xray
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torch.nn.functional as F
import models_vit

from engine_finetune import train_one_epoch, evaluate_chestxray, test_metric
from util.sampler import RASampler
# from apex.optimizers import FusedAdam
# from libauc import losses
from torchvision import models
import timm.optim.optim_factory as optim_factory
from collections import OrderedDict
# from BY_trainmodel import TrainModel  # trainmodel 모듈에서 TrainModel 클래스 import


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,#
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=75, type=int)#
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')


    parser.add_argument('--model', default='vit_small_patch16', type=str, metavar='MODEL',#
                        help='Name of model to train')    
    
    parser.add_argument('--output_dir', default='',#
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--log_dir', default='',#
                        help='path where to tensorboard log')
    
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',#
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')



    parser.add_argument('--blr', type=float, default=2.5e-3, metavar='LR',#
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.55,#
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',#
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m6-mstd0.5-inc1', metavar='NAME',#
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',#
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    

    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,#
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1, type=int,#
                        help='number of the classification types')


    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=12, type=int)#
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument("--train_list", default="", type=str, help="file for train list")#
    parser.add_argument("--val_list", default="", type=str, help="file for val list")#
    parser.add_argument("--test_list", default="", type=str, help="file for test list")#
    parser.add_argument('--eval_interval', default=10, type=int)
    parser.add_argument('--vit_dropout_rate', type=float, default=0,
                        help='Dropout rate for ViT blocks (default: 0.0)')
    parser.add_argument("--build_timm_transform", action='store_true', default=True)#
    parser.add_argument("--aug_strategy", default='default', type=str, help="strategy for data augmentation")
    parser.add_argument("--dataset", default='chestxray', type=str)

    parser.add_argument('--repeated-aug', action='store_true', default=False)

    parser.add_argument("--optimizer", default='adamw', type=str)

    parser.add_argument('--ThreeAugment', action='store_true')  # 3augment

    parser.add_argument('--src', action='store_true')  # simple random crop

    parser.add_argument('--loss_func', default=None, type=str)

    parser.add_argument("--norm_stats", default=None, type=str)

    parser.add_argument("--checkpoint_type", default=None, type=str)

    parser.add_argument("--best_model", default="accuracy", type=str)

    parser.add_argument("--data_type", default="multi-label", type=str)

    parser.add_argument("--fixed_lr", default=False)
    parser.add_argument("--average", default="macro",type=str)
    parser.add_argument("--weight",default=False)
    parser.add_argument("--name",default="", type = str)
    parser.add_argument("--model_type",default=False)
    return parser


def main(args):
    print(args.seed)
    misc.init_distributed_mode(args)        ###시간 정보 보여주는 아이

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print(device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset_chest_xray(split='train', args=args)
    dataset_val = build_dataset_chest_xray(split='val', args=args)
    dataset_test = build_dataset_chest_xray(split='test', args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    log_writer = None
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    drop_last = False
    if args.mixup>0. or args.cutmix>0.:
        drop_last = True
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
    )
    print(len(data_loader_train))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if 'vit' in args.model:
        model = models_vit.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_rate=args.vit_dropout_rate,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )

    elif 'densenet' in args.model or 'resnet' in args.model:
        model = models.__dict__[args.model](pretrained=False, num_classes=args.nb_classes)          ###??

    elif args.model == "mae-b":
        model = timm.create_model("vit_base_patch16_224.mae",pretrained=True,num_classes=args.nb_classes)
    elif args.model == "mae-s":
        model = timm.create_model("vit_small_patch16_224.mae",pretrained=True,num_classes=args.nb_classes)
    elif args.model == "VIT-b":
        model = timm.models.vision_transformer.vit_base_patch16_224(pretrained=True,num_classes=args.nb_classes)
    elif args.model == "VIT-s":
        model = timm.models.vision_transformer.vit_small_patch16_224(pretrained=True,num_classes=args.nb_classes)
    elif 'moco' in args.model:
        model = models.__dict__["resnet50"](num_classes=args.nb_classes)
    elif 'BYOL' in args.model:
        model = None
    
    elif "efficient_NS" in args.model:
        model = timm.create_model('tf_efficientnet_b4_ns',pretrained=True, num_classes=args.nb_classes)
    elif "efficient_IN" in args.model:
        model = timm.create_model('efficientnet_b4',pretrained=True, num_classes=args.nb_classes)
    elif "efficient_FS" in args.model:
        model = timm.create_model('efficientnet_b4',pretrained=False, num_classes=args.nb_classes)

    
    else:
        raise NotImplementedError


    if args.finetune and not args.eval:
        if 'vit' in args.model:
            if isinstance(args.finetune, list):
                for j, model_path in enumerate(args.finetune):
                    NUM_MODELS = len(args.finetune)
                    print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

                    assert os.path.exists(model_path)
                    # state_dict1 = torch.load(model_path,map_location=torch.device('cpu'))["model_state_dict"]
                    state_dict1 = torch.load(model_path,map_location=torch.device('cpu'))["model"]
                    # if hasattr(state_dict, 'online_model.'):
                    #     delattr(state_dict, 'online_model.')
                    # if hasattr(state_dict, 'encoder.'):
                    #     delattr(state_dict, 'encoder.')    
                    # if hasattr(state_dict, 'encoder'):
                    #     delattr(state_dict, 'encoder')    
                    state_dict = {}
                    for i in state_dict1:
                            if "encoder" in i:
                                state_dict[i[8:]] = state_dict1[i]
                            else:
                                state_dict[i] = state_dict1[i]
                    if j == 0:
                        uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
                    else:
                        uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

                checkpoint_model = uniform_soup
                print(checkpoint_model)
            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')
                # print(checkpoint["model"])
                print("Load pre-trained checkpoint from: %s" % args.finetune)
                if args.model_type:
                    try:
                        print("a")
                        # # chest X-ray
                        state_dict = checkpoint["model"]
                        new_state_dict = {}
                        # load된 model의 key에 model. 이 붙어있을 경우 제거
                        for k, v in state_dict.items():
                            if "online_model." in k:
                                name = k[13:]
                                new_state_dict[name] = v
                        checkpoint_model = new_state_dict
                    except:
                        print("b")
                    # ecg
                        checkpoint_model = checkpoint['model_state_dict'] # medical_mae_main code
                        new_state_dict = {}
                        for i in checkpoint_model:
                            if "encoder" in i:
                                new_state_dict[i[8:]] = checkpoint_model[i]
                            elif "online_model." in i:
                                new_state_dict[i[len('online_model.'):]] = checkpoint_model[i]
                            else:
                                new_state_dict[i] = checkpoint_model[i]
                        checkpoint_model = new_state_dict
                else:
                    checkpoint_model = checkpoint['model'] # medical_mae_main code
            state_dict = model.state_dict()

            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            if args.global_pool:
                for k in ['fc_norm.weight', 'fc_norm.bias']:
                    try:
                        del checkpoint_model[k]
                    except:
                        pass
            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)


            # if args.global_pool:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            # else:
            #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

            # manually initialize fc layer
            trunc_normal_(model.head.weight, std=2e-5)
        elif 'densenet' in args.model or 'resnet' in args.model:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            if 'state_dict' in checkpoint.keys():
                checkpoint_model = checkpoint['state_dict']
            elif 'model' in checkpoint.keys():
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            if args.checkpoint_type == 'smp_encoder':
                state_dict = checkpoint_model

                new_state_dict = OrderedDict()

                for key, value in state_dict.items():
                    if 'model.encoder.' in key:
                        new_key = key.replace('model.encoder.', '')
                        new_state_dict[new_key] = value
                checkpoint_model = new_state_dict
            model_state_dict = model.state_dict()
            for key in checkpoint_model:
                if key in model_state_dict and model_state_dict[key].shape == checkpoint_model[key].shape:

                    model_state_dict[key] = checkpoint_model[key]
            msg = model.load_state_dict(model_state_dict, strict=False)
            print(msg)
        elif "moco" in args.model:
            if isinstance(args.finetune, list):
                model = models.__dict__["resnet50"](num_classes=args.nb_classes)
                for j, model_path in enumerate(args.finetune):
                    NUM_MODELS = len(args.finetune)
                    print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')
                    print(model_path)
                    assert os.path.exists(model_path)
                    state_dict1 = torch.load(model_path,map_location=torch.device('cpu'))["state_dict"]
  
                    state_dict = {}
                    for k, v in state_dict1.items():
                        if "module.encoder_q." in k:
                            name = k[17:]
                            state_dict[name] = v
                    
                    if j == 0:
                        uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
                    else:
                        uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

                new_state_dict = uniform_soup
            else:
                model = models.__dict__["resnet50"](num_classes=args.nb_classes)
                checkpoint = torch.load(args.finetune, map_location='cpu')
                state_dict = checkpoint["state_dict"]
    
                new_state_dict = {}
                
                # load된 model의 key에 model. 이 붙어있을 경우 제거
                for k, v in state_dict.items():
                    if "module.encoder_q." in k:
                        name = k[17:]
                        new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)

        elif 'BYOL' in args.model:
            if isinstance(args.finetune, list):
                for j, model_path in enumerate(args.finetune):
                    NUM_MODELS = len(args.finetune)
                    print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')
                    print(model_path)
                    assert os.path.exists(model_path)
                    model = torch.load(model_path,map_location=torch.device('cpu'))["online_network_model"]
                    # state_dict = torch.nn.Sequential(*list(state_dict.children())[:-1]) 
                    encoder = torch.nn.Sequential(*list(model.children())[:-1]) 
                    state_dict = encoder.state_dict()
                    # print(state_dict)
                    
                    if j == 0:
                        uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
                    else:
                        uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

                encoder.load_state_dict(uniform_soup)
                print(encoder)

            else:
                checkpoint = torch.load(args.finetune, map_location='cpu')
                encoder = checkpoint['online_network_model']
                encoder = torch.nn.Sequential(*list(encoder.children())[:-1]) 
            model = TrainModel(encoder,output_dim = args.nb_classes)
            
            

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # print("actual lr: %.2e" % args.lr)

    # print("accumulate grad iterations: %d" % args.accum_iter)
    # print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    if 'vit' in args.model:
        param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
        )
    else:
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.adam(param_groups, lr=args.lr)
    else:
        raise NotImplementedError

    loss_scaler = NativeScaler()

    # last_activation = torch.nn.Softmax()
    last_activation = None
    if mixup_fn is not None:
        criterion = SoftTargetBinaryCrossEntropy()
    elif args.data_type == "multi-label":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.data_type == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.data_type == "multi-class":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
        
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate_chestxray(data_loader_test, model, device, args)
        print(f"Average AUC of the network on the test set images: {test_stats['auc_avg']:.4f}")
        if args.dataset == 'covidx':
            print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    max_accuracy = 0.0
    max_auc = 0.0
    min_loss = 99999999999

    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer, weight=args.weight,       
            args=args, last_activation=last_activation
        )
        
            

        test_stats = evaluate_chestxray(data_loader_val, model, device, args)
        print(test_stats)
        # print(f"Average AUC on the test set images: {test_stats['auc_avg']:.4f}")
        max_auc = max(max_auc, test_stats['auc_avg'])
        print(f'Max Average AUC: {max_auc:.4f}', {max_auc})



        if log_writer is not None:#[auc_avg, acc_avg, f1_avg, pre_avg, rec_avg]
            log_writer.add_scalar('perf/auc_avg', test_stats['auc_avg'], epoch)
            log_writer.add_scalar('perf/validation_loss', test_stats['loss'], epoch)
            log_writer.add_scalar('perf/validation_acc', test_stats['acc_avg'], epoch)
            log_writer.add_scalar('perf/tvalidation_precision', test_stats['pre_avg'], epoch)
            log_writer.add_scalar('perf/validation_recall', test_stats['rec_avg'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'validation_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        ##########################################################################################
        # save best model of args.best_model(loss / accuracy / auc )
        ##########################################################################################

        import copy
        if args.best_model == "accuracy":
            if max_accuracy < test_stats["acc_avg"]:
                max_accuracy = test_stats["acc_avg"]
                best_model = copy.deepcopy(model)
                misc.save_model(
                    args=args, model=best_model, model_without_ddp=best_model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
        
        elif args.best_model == "auc":
            if max_auc < test_stats["auc_avg"]:
                max_auc = test_stats["auc_avg"]
                best_model = copy.deepcopy(model)
                misc.save_model(
                    args=args, model=best_model, model_without_ddp=best_model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                
        elif args.best_model == "loss":
            if min_loss > test_stats["loss"]:
                min_loss = test_stats["loss"]
                best_model = copy.deepcopy(model)
                misc.save_model(
                    args=args, model=best_model, model_without_ddp=best_model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        else:
            raise NotImplementedError

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    test_stats = evaluate_chestxray(data_loader_test, best_model, device, args)
    print(test_stats)

    ######################################################################################revised performance
    if args.fixed_lr: save_lr = args.lr
    else: save_lr = args.blr

    test_result = [args.name,save_lr,args.seed]
    test_result_col = ["model","lr","trial"]
    if args.nb_classes == 2 and (args.mixup > 0 or args.cutmix > 0.): nb_classes = 1
    else:    nb_classes = args.nb_classes

    for i in test_stats.keys():
        if i == "auc_each_class" and nb_classes!=1:
            test_result_col.extend([f"auc : {j}" for j in range(nb_classes)])
            test_result.extend([auc for auc in test_stats[i]])
        else:
            if "acc" in i:
                test_result.extend([test_stats[i]])
            else:
                test_result.extend([test_stats[i]])
            test_result_col.extend([i])
    test_result.extend([total_time_str])       
    test_result_col.extend(["train_time"])
    ######################################################################################revised performance
    import pandas as pd
    print(test_result,test_result_col)
    test_stat = pd.DataFrame([test_result],columns=test_result_col)
    test_stat.to_csv(f"{args.output_dir}/test_result.csv",index=None)
    test_metric(args.output_dir)
    log_writer.close()

    


"""
MAE snapshot (pediCXR)
"""
path = "/mnt/hdd2/result/SSL_CXR/snapshot_ensemble/finetune/PediCXR"
train_path = "/mnt/hdd1/data/Chest_X_ray/pediCXR"
PATH = "/mnt/hdd2/result/SSL_CXR/snapshot_ensemble/pretrain"
if __name__ == '__main__':  
    for lr in [0.001,0.0025,0.0001]:
        for epoch in [200,400,600,800]:
            for m,f,n in [
                ["vit_small_patch16", f"{PATH}/CXR_CheXpert_800epoch_256batch_0.001lr0.65ratio4cycle/checkpoint-{epoch}.pth", f"snapshot-MAE(0.65)"],
                ["vit_small_patch16", f"{PATH}/CXR_CheXpert_800epoch_256batch_0.001lr0.75ratio4cycle/checkpoint-{epoch}.pth", f"snapshot-MAE(0.75)"],
                ["vit_small_patch16", f"{PATH}/CXR_CheXpert_800epoch_256batch_0.001lr0.85ratio4cycle/checkpoint-{epoch}.pth", f"snapshot-MAE(0.85)"],
                 ]:
                for seed in [42,1004,2023]:

                    args                = get_args_parser()
                    args                = args.parse_args()

                    args.average        = "weighted"
                    args.data_type      = "multi-label"
                    args.best_model     = "loss"
                    args.optimizer      = "adamw"
                    args.fixed_lr       = False
                    # args.lr           = lr
                    args.blr            = lr
                    args.nb_classes     = 6
                    args.batch_size     = 128
                    args.warmup_epochs  = 5
                    args.epochs         = 75
                   
                    args.weight         = True
                    args.model          = m
                    args.finetune       = f
                    args.name           = n
                    args.seed           = seed
                    args.model_type     = True
                    args.train_list     = f"{train_path}/train_1.txt"
                    args.val_list       = f"{train_path}/valid.txt"
                    args.test_list      = f"{train_path}/test.txt"
                    args.output_dir     = f"{path}/{n}/{lr}/{epoch}E/{args.seed}"
                    args.log_dir        = args.output_dir
                    if args.output_dir:
                        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                    main(args)
