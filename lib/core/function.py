               # ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import torch
import sys
import os
from core.evaluate import accuracy


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        output = model(input)
        target = target.cuda(non_blocking=True)

        loss = criterion(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        prec1, prec5 = accuracy(output, target, (1, 5))

        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Accuracy@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, top1=top1, top5=top5)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_top1', top1.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1


def validate(filesave, valdir, config, val_loader, model, writer_dict=None):   

    # output_dir = os.path.join(folder_save, valdir.split('/')[6])
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # switch to evaluate mode
    model.eval()
    count = 0
    # print('len(val_loader) ', len(val_loader), valdir)
    with torch.no_grad():
        features = np.array([])
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output, feat = model(input)
            feat = feat.data.cpu().numpy().copy()
            print('feat ', feat.shape)

            features = feat if features.shape[0] == 0 else np.vstack((features, feat))

            count += 1
            print(count, 'features: ', features.shape)
            if count == 200:
                print('break')
                break

        np.save(filesave, features)
        print('saved to >> ', filesave)
        # count += 1

    return feat

def validate2(filesave, valdir, config, val_loader, model, writer_dict=None):   

    # output_dir = os.path.join(folder_save, valdir.split('/')[6])
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # switch to evaluate mode
    model.eval()
    count = 0
    # print('len(val_loader) ', len(val_loader), valdir)
    with torch.no_grad():
        features = np.array([])
        for i, (input, target) in enumerate(val_loader):
            # compute output
            output, feat = model(input)
            feat = feat.data.cpu().numpy().copy()

            features = feat if features.shape[0] == 0 else np.vstack((features, feat))

            count += 1
            print(count, 'features: ', features.shape)
            if count == 100:
                print('break')
                break

        np.save(filesave, features)
        print('saved to >> ', filesave)
        # count += 1

    return feat    




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
