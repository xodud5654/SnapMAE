# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.learning_rate_cosine_with_cycle(optimizer, data_iter_step, epoch, len(data_loader), args)
        if isinstance(samples, list):
            imgs = samples[0].to(device, non_blocking=True)
            heatmaps = samples[1].to(device, non_blocking=True)
        else:
            imgs = samples.to(device, non_blocking=True)
            heatmaps = None

        with torch.cuda.amp.autocast():
            if heatmaps is not None:
                loss, _, _ = model(imgs, mask_ratio=args.mask_ratio, heatmaps=heatmaps)
            else:
                loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import cv2   
from torch.utils.data import Dataset
class CustomDatasets(Dataset):
    def __init__(self, data_dir, label=False, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label = label
        # self.lst_data = sorted(glob2.glob(f"{data_dir}/*png"))

    
    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self, index):
        data_BGR = cv2.imread(self.data_dir[index])
        data = cv2.cvtColor(data_BGR,cv2.COLOR_BGR2RGB)
        if self.transform:
            data = self.transform(data)
        if self.label:
            data_label = self.data_dir[index].split("/")[-2]
            return data, data_label
        return data