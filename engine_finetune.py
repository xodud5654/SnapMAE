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
from typing import Iterable, Optional

import numpy as np
import torch

from timm.data import Mixup
from timm.utils import accuracy
import torch.distributed as dist
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics._ranking import roc_auc_score
import torch.nn.functional as F
from libauc import losses
from sklearn.metrics import confusion_matrix
from util.multi_label_loss import SoftTargetBinaryCrossEntropy
import copy
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, last_activation=None, weight=False, class_weight = None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    loss_a = []
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:# and args.fixed_lr == False:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if targets.shape[1] == 1 and args.data_type !="binary": 
            targets = targets.to(torch.long).squeeze()      #float타입으로 들어와서 long타입으로 바꾸고 [[0],[1]]의 형식에서 [0,1]로 변환
            # print(targets)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if last_activation is not None:
            # print("last_activation")
            if last_activation == 'sigmoid':
                last_activation = torch.nn.Sigmoid()
        with torch.cuda.amp.autocast():
            outputs = model(samples)
  

            if last_activation is not None:
                outputs = last_activation(outputs)

            if weight == True:

                # class_weight = torch.Tensor([1.0/(torch.sum(targets[:,i])/128 + 1e-6) for i in range(len(targets[0]))]).to(device)  #used_calss_weight
                
                # w/o epsilon : class_weight "inf"
                # class_weight = []                                                                                                   #pos_weight
                # for i in range(0,args.nb_classes):
                #     class_weight.append((targets[:,i]==0.).sum()/targets[:,i].sum())
                # class_weight = torch.Tensor(class_weight).to(device)
                # print(class_weight)

                # w epsilon
                if class_weight == None:
                    
                    class_weight = []                                                                                                   #pos_weight
                    for i in range(0,args.nb_classes):
                        class_weight.append((targets[:,i]==0.).sum()/(targets[:,i].sum()+1e-6))
                class_weight = torch.Tensor(class_weight).to(device)

                if mixup_fn is not None:
                    criterion = SoftTargetBinaryCrossEntropy()
                elif args.data_type == "multi-label":
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight)
                elif args.data_type == "binary":
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight)
                elif args.data_type == "multi-class":
                    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

            loss = criterion(outputs, targets)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"target : {targets}")
            print(f"output : {outputs}")
            print("Loss is {}, stopping training".format(loss_value))
            print(args.output_dir)
            sys.exit(1)
            

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


from sklearn.metrics import *


def computeMETRIC(targets, outputs, classCount, args):
    
    if args.data_type == "multi-label":
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(torch.Tensor(outputs)).numpy()
        each_auc = roc_auc_score(targets,outputs,multi_class="ovr",average=None)
        auc = roc_auc_score(targets,outputs,multi_class="ovr",average="macro")
        acc = 0
        for i in range(classCount):
            outputs[:,i] = np.where(outputs[:, i]>=0.5,1,0)
            acc += accuracy_score(targets[:,i], outputs[:,i])
        
        acc /= classCount
        recall = recall_score(targets,outputs,average=args.average)
        precision = precision_score(targets,outputs,average=args.average)
        f1 = f1_score(targets,outputs,average=args.average)
        LIST = each_auc.tolist()
        LIST.extend([auc,acc,recall,precision,f1])
        return LIST

    elif args.data_type == "binary":
        sigmoid = torch.nn.Sigmoid()
        outputs = sigmoid(torch.Tensor(outputs)).numpy()
        each_auc = roc_auc_score(targets,outputs)
        auc = roc_auc_score(targets,outputs)
##############################################################################################################################
        # outputs_pred = outputs.argmax(axis=1)
        # if len(targets.shape) != 1:                                                                                                                     
        #     targets  = targets[:,1]
        for i in range(classCount):
            outputs[:,i] = np.where(outputs[:, i]>=0.5,1,0)
            # acc += accuracy_score(targets[:,i], outputs[:,i])
        outputs_pred = outputs
        # outputs = np.argmax(outputs)
        acc = accuracy_score(targets,outputs_pred)
        recall = recall_score(targets,outputs_pred,average=args.average)
        precision = precision_score(targets,outputs_pred,average=args.average)
        f1 = f1_score(targets,outputs_pred,average=args.average)
        LIST = [each_auc, auc,acc,recall,precision,f1]
        return LIST
        
    elif args.data_type == "multi-class":
        Softmax = torch.nn.Softmax(dim=1)
        outputs = Softmax(torch.Tensor(outputs)).numpy()

        each_auc = roc_auc_score(targets,outputs,multi_class="ovr",average=None)
        auc = roc_auc_score(targets,outputs,multi_class="ovr",average="macro")
        try:
            targets = targets.argmax(axis=1)
        except:
            targets = targets
        preds = outputs.argmax(axis=1)
        acc = accuracy_score(targets, preds)
        recall = recall_score(targets, preds,average=args.average)
        precision = precision_score(targets, preds,average=args.average)
        f1 = f1_score(targets, preds,average=args.average)
        LIST = each_auc.tolist()
        LIST.extend([auc,acc,recall,precision,f1])
        return LIST

    


@torch.no_grad()
def evaluate_chestxray(data_loader, model, device, args):

    # if args.mixup > 0.:
    #     criterion = SoftTargetBinaryCrossEntropy()
    if args.data_type == "multi-label":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.data_type == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.data_type == "multi-class":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError


    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs = []
    targets = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if target.shape[1] == 1 and args.data_type !="binary": 
            target = target.to(torch.long).squeeze()     

        with torch.cuda.amp.autocast():
            output = model(images)
##############################################################################################################################
            if args.mixup>0. or args.cutmix>0.:                                                                                     
                target = torch.full((target.size()[0], args.nb_classes), 0., device=device).scatter_(1, target.long(), 1.)


            loss = criterion(output, target)



        outputs.append(output)
        targets.append(target)


        metric_logger.update(loss=loss.item())
##############################################################################################################################
    if args.nb_classes == 2 and (args.mixup > 0 or args.cutmix > 0.):                                                               
        num_classes = 1
    else:
        num_classes = args.nb_classes
    targets = torch.cat(targets, dim=0).cpu().numpy()
    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    print(target.size())
    print(output.size())

    np.save(args.log_dir + '/' + 'y_gt.npy', targets)
    np.save(args.log_dir + '/' + 'y_pred.npy', outputs)

    LIST = computeMETRIC(targets, outputs, num_classes,args)
    auc_each_class = LIST[:num_classes]
    auc_avg = LIST[num_classes:num_classes+1][0]
    acc_avg = LIST[num_classes+1:num_classes+2][0]
    rec_avg = LIST[num_classes+2:num_classes+3][0]
    pre_avg = LIST[num_classes+3:num_classes+4][0]
    f1_avg = LIST[num_classes+4:num_classes+5][0]

    metric_logger.synchronize_between_processes()

    return {**{k: meter.global_avg for k, meter in metric_logger.meters.items()},
            **{'auc_each_class': auc_each_class, 'auc_avg': auc_avg, \
                'acc_avg': acc_avg, 'rec_avg': rec_avg, 'pre_avg': pre_avg, 'f1_avg': f1_avg}}



@torch.no_grad()
def evaluate_chestxray2(data_loader, model, device, args):

    # if args.mixup > 0.:
    #     criterion = SoftTargetBinaryCrossEntropy()
    if args.data_type == "multi-label":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.data_type == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.data_type == "multi-class":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError


    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    outputs = []
    targets = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if target.shape[1] == 1 and args.data_type !="binary": 
            target = target.to(torch.long).squeeze()   


        with torch.cuda.amp.autocast():
            output = model(images)
##############################################################################################################################
            if args.mixup>0. or args.cutmix>0.:                                                                                     
                target = torch.full((target.size()[0], args.nb_classes), 0., device=device).scatter_(1, target.long(), 1.)


            loss = criterion(output, target)



        outputs.append(output)
        targets.append(target)


        metric_logger.update(loss=loss.item())
##############################################################################################################################
    if args.nb_classes == 2 and (args.mixup > 0 or args.cutmix > 0.):                                                               
        num_classes = 1
    else:
        num_classes = args.nb_classes
    targets = torch.cat(targets, dim=0).cpu().numpy()
    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    print(target.size())
    print(output.size())

    np.save(args.log_dir + '/' + 'y_gt.npy', targets)
    np.save(args.log_dir + '/' + 'y_pred.npy', outputs)


import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
def test_metric(log_path):
    # 텍스트 파일 읽기
    txt_filename = f"{log_path}/log.txt"
    csv_filename = f"{log_path}/log.csv"
    # CSV 파일 생성 및 데이터 작성
    with open(csv_filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        # CSV 헤더 작성
        csvwriter.writerow(["epoch","train_lr", "train_loss", "validation_loss", "validation_auc_avg", "validation_acc_avg", "validation_f1_avg", "validation_pre_avg", "validation_rec_avg"])

        # 텍스트 파일을 줄 단위로 읽어서 JSON 파싱 후 CSV 데이터로 작성
        with open(txt_filename, "r") as txtfile:
            for line in txtfile:
                data = json.loads(line)
                csvwriter.writerow([data["epoch"], data["train_lr"], data["train_loss"], data["validation_loss"], data["validation_auc_avg"], data["validation_acc_avg"], data["validation_f1_avg"], data["validation_pre_avg"], data["validation_rec_avg"]])

    print(f"CSV 파일 '{csv_filename}'로 변환되었습니다.")
