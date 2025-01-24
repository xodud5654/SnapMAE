# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if hasattr(args, 'fixed_lr') and args.fixed_lr:
            lr = args.lr
        else:
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
        print(lr)
    return lr


def learning_rate_cosine_with_cycle(optimizer, iter, epoch, nBatches, args):
    n_epochs_sub = math.floor(args.epochs / args.cycle)
    # print(n_epochs_sub)
    n_epochs_last = args.epochs - (args.cycle - 1) * n_epochs_sub
    # print(n_epochs_last)
    n_epochs_cur = n_epochs_last if epoch > (args.cycle - 1) * n_epochs_sub else n_epochs_sub
    # print(n_epochs_cur)
    t_total = n_epochs_cur * nBatches # data / batch
    # print(t_total)
    t_cur = ((epoch - 1) % n_epochs_cur) * nBatches + iter
    # print(t_cur)
    cur_lr = 0.5 * args.lr * (1 + math.cos(math.pi * t_cur / t_total))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = cur_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = cur_lr

    return cur_lr
