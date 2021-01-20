# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.function import validate, validate2
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

def get_feat(args, folder_save, line):

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)

    if config.TEST.MODEL_FILE:
        
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        
        model.load_state_dict(torch.load(model_state_file))

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    valdir = line.split(' ')[0]
    num_frames = int(line.split(' ')[1])
    
    filename = valdir.split('/')
    namefile = filename[len(filename)-1]
    filesave = os.path.join(folder_save, namefile)
    valdir = os.path.join(valdir, 'rgb')
    # print(' valdir >>', valdir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    validate2(filesave, valdir, config, valid_loader, model, None)


def main():
    args = parse_args()

    folder_save = '/media/ee401_2/Didik/disertation/results/final/feat/hrnet/64C/'
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    f = open('/media/ee401_2/Didik/disertation/anomaly-det/models/final/hrnet/tools/ucf_cek.list', "r")
    lines = sorted(list(f))

    count = 0
    for i in range(len(lines)):
        if i == 8:
            line = lines[i].strip('\n')
            print(count, line)
            get_feat(args, folder_save, line)
        count += 1

   


if __name__ == '__main__':
    main()

# corrupted ffiles
# [1, 90. 248, 1127, 1293, 1330, 1352, 1378, 1553]
