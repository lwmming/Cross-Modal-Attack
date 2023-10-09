from math import gamma
import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import argparse
import os
import utils
import transforms_ucf101
import math

import torchattacks
# import pytorch_msssim
from torchvision.transforms import ToPILImage, ToTensor
# from gluoncv.torch.engine.config import get_cfg_defaults
# from gluoncv.torch.model_zoo import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size ')
parser.add_argument('--vis', type=lambda x: (str(x).lower() == 'true'), default=False, help='vis')
parser.add_argument('--gpu', type=str, default='9', help='gpu-id')
parser.add_argument('--adv_path', type=str, default='/mnt/data/rk/ucf-ens-ours-GIE', help='ImageNet-val directory.')

args = parser.parse_args()
if not os.path.exists(args.adv_path):
    os.makedirs(args.adv_path)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

UCF_MODEL_ROOT = './checkpoints/' # ckpt file path of UCF101
MODEL_TO_CKPTS = {
    'i3d_resnet50': os.path.join(UCF_MODEL_ROOT, 'i3d_resnet50.pth'),
    'i3d_resnet101': os.path.join(UCF_MODEL_ROOT, 'i3d_resnet101.pth'),
    'slowfast_resnet50': os.path.join(UCF_MODEL_ROOT, 'slowfast_resnet50.pth'),
    'slowfast_resnet101': os.path.join(UCF_MODEL_ROOT, 'slowfast_resnet101.pth'),
    'tpn_resnet50': os.path.join(UCF_MODEL_ROOT, 'tpn_resnet50.pth'),
    'tpn_resnet101': os.path.join(UCF_MODEL_ROOT, 'tpn_resnet101.pth')
}
CONFIG_ROOT = './config/'
CONFIG_PATHS = {
    'i3d_resnet50': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet50_v1_kinetics400.yaml'),
    'i3d_resnet101': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet101_v1_kinetics400.yaml'),
    'slowfast_resnet50': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet50_kinetics400.yaml'),
    'slowfast_resnet101': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet101_kinetics400.yaml'),
    'tpn_resnet50': os.path.join(CONFIG_ROOT, 'tpn_resnet50_f32s2_kinetics400.yaml'),
    'tpn_resnet101': os.path.join(CONFIG_ROOT, 'tpn_resnet101_f32s2_kinetics400.yaml')
}

def test_transform():
    input_size = 224
    scale_ratios = '1.0, 0.8'
    scale_ratios = [float(i) for i in scale_ratios.split(',')]
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    norm_method = transforms_ucf101.Normalize(default_mean, default_std)
    spatial_transform = transforms_ucf101.spatial_Compose([
       transforms_ucf101.Scale(int(input_size / 1.0)),
        transforms_ucf101.CornerCrop(input_size, 'c'),
        transforms_ucf101.ToTensor(), norm_method
        ])
    temporal_transform = transforms_ucf101.LoopPadding(32)#transforms_ucf101.LoopPadding(32)
    return spatial_transform, temporal_transform


def transform_video(video, mode='forward'):
    r'''
    Transform the video into [0, 1]
    '''
    dtype = video.dtype
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=dtype).cuda()
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=dtype).cuda()
    if mode == 'forward':
        video.sub_(mean[:, None, None]).div_(std[:, None, None])
    elif mode == 'back':
        video.mul_(std[:, None, None]).add_(mean[:, None, None])
    return video

if __name__ == '__main__':

    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # model_incv3 = nn.Sequential(utils.Normalize('imagenet'),torchvision.models.inception_v3(pretrained=True, transform_input=True)).cuda().eval()
    # model_vgg19 = nn.Sequential(utils.Normalize('imagenet'),torchvision.models.vgg19(pretrained=True)).cuda().eval()
    model_res101 = nn.Sequential(utils.Normalize('imagenet'),torchvision.models.resnet101(pretrained=True)).cuda().eval()
    # model_dense121 = nn.Sequential(utils.Normalize('imagenet'),torchvision.models.densenet121(pretrained=True)).cuda().eval()
    model_vgg16 = nn.Sequential(utils.Normalize('imagenet'),torchvision.models.vgg16(pretrained=True)).cuda().eval()
    model_alex = nn.Sequential(utils.Normalize('imagenet'),torchvision.models.alexnet(pretrained=True)).cuda().eval()
    model_sqn = nn.Sequential(utils.Normalize('imagenet'),torchvision.models.squeezenet1_1(pretrained=True)).cuda().eval()

    test_spa_trans, test_temp_trans = test_transform()
    test_dataset = transforms_ucf101.attack_ucf101(spatial_transform=test_spa_trans, temporal_transform=test_temp_trans)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=9, pin_memory=True)

    # attacks methods
    # ADAMTAP10 = torchattacks.I2V(model_vgg16, eps=16.0/255, alpha=0.005, steps=60, momentum=1.0) # baseline
    ADAMTAP10 = torchattacks.GIE(model_res101, eps=16.0/255, alpha=0.005, steps=60, momentum=1.0) # ours
    # ADAMTAP10 = torchattacks.I2V_ENS([model_res101,model_vgg16,model_alex,model_sqn], eps=16.0/255, alpha=0.005, steps=60, momentum=1.0) #ens-baseline
    # ADAMTAP10 = torchattacks.GIE_ENS([model_res101,model_vgg16,model_alex,model_sqn], eps=16.0/255, alpha=0.005, steps=60, momentum=1.0) #ens-ours

    attacks = [ADAMTAP10]

    # testing
    for attack_id, attack in enumerate(attacks):
        cnt_steps = 0
        if args.vis and not os.path.exists(attack.attack):
            os.mkdir(attack.attack)
        for idd, data in enumerate(val_loader):
            cnt_steps += 1
            print('process %.4f%%...' % (100.0 * (idd / len(val_loader))))
            # Data.
            X = data[0].cuda()
            y = data[1].cuda()
            b,c,f,h,w = X.shape #1,3,32,224,224
            X = X.permute([0,2,1,3,4])
            X = X.reshape(b*f, c, h, w)
            X = transform_video(X.clone().detach(), mode='back') # [0, 1]
            adv_X = attack(X, y.repeat(b*f))

            adv_X = transform_video(adv_X.clone().detach(), mode='forward')
            adv_X = adv_X.reshape(b,f,c,h,w)
            adv_X = adv_X.permute([0,2,1,3,4])

            for ind,label in enumerate(y):
                adv = adv_X[ind].detach().cpu().numpy()
                np.save(os.path.join(args.adv_path, '{}-adv'.format(label.item())), adv)

            if args.vis:
                for kk in range(X.shape[0]):
                    ToPILImage()(adv_X[kk].detach().squeeze()).save(os.path.join(attack.attack, '%03d.png' % (idd*args.batch_size+kk)), quality=200)
                    # ToPILImage()(X[kk].detach().squeeze()).save(os.path.join('ori', '%03d.png' % (idd*args.batch_size+kk)), quality=200)

        print(attack)
