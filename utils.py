import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import Dataset
import csv
import PIL.Image as Image
import os
import torchvision.transforms as T
import pickle
#import cv2
import math
from gluoncv.torch.engine.config import get_cfg_defaults
 
# config info
# refer to https://cv.gluon.ai/model_zoo/action_recognition.html
CONFIG_ROOT = './config/' # config paths
CONFIG_PATHS = {
    'i3d_resnet50': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet50_v1_kinetics400.yaml'),
    'i3d_resnet101': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet101_v1_kinetics400.yaml'),
    'slowfast_resnet50': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet50_kinetics400.yaml'),
    'slowfast_resnet101': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet101_kinetics400.yaml'),
    'tpn_resnet50': os.path.join(CONFIG_ROOT, 'tpn_resnet50_f32s2_kinetics400.yaml'),
    'tpn_resnet101': os.path.join(CONFIG_ROOT, 'tpn_resnet101_f32s2_kinetics400.yaml')
}

# ucf model infos
UCF_MODEL_ROOT = './checkpoints/' # ckpt file path of UCF101
MODEL_TO_CKPTS = {
    'i3d_resnet50': os.path.join(UCF_MODEL_ROOT, 'i3d_resnet50.pth'),
    'i3d_resnet101': os.path.join(UCF_MODEL_ROOT, 'i3d_resnet101.pth'),
    'slowfast_resnet50': os.path.join(UCF_MODEL_ROOT, 'slowfast_resnet50.pth'),
    'slowfast_resnet101': os.path.join(UCF_MODEL_ROOT, 'slowfast_resnet101.pth'),
    'tpn_resnet50': os.path.join(UCF_MODEL_ROOT, 'tpn_resnet50.pth'),
    'tpn_resnet101': os.path.join(UCF_MODEL_ROOT, 'tpn_resnet101.pth')
}
# ucf dataset
UCF_DATA_ROOT = './UCF101-Examples/' # ucf101 dataset path
Kinetic_DATA_ROOT = './Kinetics-Examples/' # kinetics dataset path

def change_cfg(cfg, batch_size):
    # modify video paths and pretrain setting.
    cfg.CONFIG.DATA.VAL_DATA_PATH = Kinetic_DATA_ROOT
    cfg.CONFIG.DATA.VAL_ANNO_PATH = './kinetics400_attack_samples.csv'
    cfg.CONFIG.MODEL.PRETRAINED = True
    cfg.CONFIG.VAL.BATCH_SIZE = batch_size
    return cfg

def get_cfg_custom(cfg_path, batch_size=16):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg = change_cfg(cfg, batch_size)
    return cfg

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


class Normalize(nn.Module):
    def __init__(self,dataset_name):
        super(Normalize, self).__init__()
        assert dataset_name in ['imagenet', 'cifar100', 'inc', 'tensorflow'], 'check dataset_name'
        self.mode = dataset_name
        if dataset_name == 'imagenet':
            self.normalize = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        elif dataset_name == 'cifar100':
            self.normalize = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        else:
            self.normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

    def forward(self, input):
        # import ipdb; ipdb.set_trace()
        x = input.clone()
        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        else:
            for i in range(x.shape[1]):
                x[:,i] = (x[:,i] - self.normalize[0][i]) / self.normalize[1][i]
        return x

