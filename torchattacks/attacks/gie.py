import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

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


mid_outputs = []
class GIE:
    '''
    our proposed GIE and LCD.
    '''
    def __init__(self, model, eps=8/255, alpha=1/255, steps=8, momentum=0.9, targeted=False):
        self.model = model
        self.eps = eps
        self.steps = steps
        self.momentum = momentum
        self.tar = targeted
        if self.tar:
            self.alpha = -alpha
        else:
            self.alpha = alpha

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        return images

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        loss = nn.CrossEntropyLoss()
        modif = torch.Tensor(images.shape).fill_(0.01/255).cuda()
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.alpha)
        # feature_layers = ['layer3']  # res101
        # feature_layers = ['Mixed_6c'] # incv3
        # feature_layers = ['9']     # incv4
        # feature_layers = ['mixed_6a']      # incResv2
        # feature_layers = ['20']     # vgg
        feature_layers = ['layer2']      # Res152
        # feature_layers = ['6']     # sqn
        # feature_layers = ['7']     # alex
        # feature_layers = ['denseblock2']      # dense121

        global mid_outputs

        def get_mid_output(m, i, o):
            global mid_outputs
            mid_outputs.append(o)

        hs = []
        for layer_name in feature_layers:
            #print(layer_name)
            # import ipdb; ipdb.set_trace()
            if self.model[1]._modules.get(layer_name) is not None:
                hs.append(self.model[1]._modules.get(layer_name).register_forward_hook(get_mid_output))
            # hs.append(self.model[1]._modules.get('features')._modules.get(layer_name).expand3x3_activation.register_forward_hook(get_mid_output)) # incv4
            # hs.append(self.model[1]._modules.get('features')._modules.get(layer_name).register_forward_hook(get_mid_output)) # incv4

        libs = torch.zeros(32,32,512,28,28) #384,13,13, incv3: 768,12,12, dense121:512,28,28,sqn:128,27,27
        for ii in range(32):
            for jj in range(32):
                # hhs = self.model[1]._modules.get('features')._modules.get(layer_name).register_forward_hook(get_mid_output) # alex
                _ = self.model((0.5 * images[ii] + 0.5 * images[jj])[None])  # 32,3,224,224
                # import ipdb;ipdb.set_trace()
                libs[ii,jj] =  mid_outputs[0].cpu().detach()
                mid_outputs = []
                # hhs.remove()
        mid_outputs = []

        for i in range(self.steps):
            adv_images = torch.clamp(images + torch.clamp(modifier, min=-self.eps, max=self.eps), min=0, max=1)
            ind = [x for x in range(adv_images.size(0))]
            ori_fea = libs[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.model(0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid = F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            loss_adj = 0.1 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.model(0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += 1.0 * F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.model(0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += 1.0 * F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            mid_outputs = []

            # ind = [x for x in range(adv_images.size(0))]
            # random.shuffle(ind)
            # ori_fea = libs[[x for x in range(32)], ind].cuda().reshape(32, -1)
            # outputs = self.model(0.5 * adv_images + 0.5 * adv_images[ind])
            # loss_mid += 1.0 * F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            # mid_outputs = []

            # ind = [x for x in range(adv_images.size(0))]
            # random.shuffle(ind)
            # ori_fea = libs[[x for x in range(32)], ind].cuda().reshape(32, -1)
            # outputs = self.model(0.5 * adv_images + 0.5 * adv_images[ind])
            # loss_mid += 1.0 * F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            # mid_outputs = []

            loss_mid /= 3.0
            loss_mid += loss_adj
            cost = loss_mid.cuda()
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            mid_outputs = []

        for h in hs:
            h.remove()
        return torch.clamp(images + torch.clamp(modifier, min=-self.eps, max=self.eps), min=0, max=1).detach()