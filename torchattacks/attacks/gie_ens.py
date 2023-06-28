import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import numpy as np

mid_outputs = []
class GIE_ENS:
    r"""
    our proposed GIE and LCD (ens). 
    """
    def __init__(self, models, eps=8/255, alpha=1/255, steps=8, momentum=0.9, targeted=False):
        self.models = models
        self.eps = eps
        #self.alpha = alpha
        self.steps = steps
        self.momentum = momentum
        self.tar = targeted
        if self.tar:
            self.alpha = -alpha
        else:
            self.alpha = alpha

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        for mm in self.models:
            mm.eval()
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        modif = torch.Tensor(images.shape).fill_(0.01/255).cuda()
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.alpha)
        global mid_outputs

        def get_mid_output(m, i, o):
            global mid_outputs
            mid_outputs.append(o)

        hs = []
        hs.append(self.models[0][1]._modules.get('layer2').register_forward_hook(get_mid_output))
        hs.append(self.models[1][1]._modules.get('features')._modules.get('20').register_forward_hook(get_mid_output))
        hs.append(self.models[2][1]._modules.get('features')._modules.get('7').register_forward_hook(get_mid_output))
        hs.append(self.models[3][1]._modules.get('features')._modules.get('6').expand3x3_activation.register_forward_hook(get_mid_output))


        libs_res = torch.zeros(32,32,512,28,28) #384,13,13, incv3: 768,12,12, dense121:512,28,28, sqn:128,27,27
        for ii in range(32):
            for jj in range(32):
                _ = self.models[0]((0.5 * images[ii] + 0.5 * images[jj])[None])  # 32,3,224,224
                libs_res[ii,jj] =  mid_outputs[0].cpu().detach()
                mid_outputs = []
        mid_outputs = []

        libs_vgg = torch.zeros(32,32,512,28,28) #384,13,13, incv3: 768,12,12, dense121:512,28,28, sqn:128,27,27
        for ii in range(32):
            for jj in range(32):
                _ = self.models[1]((0.5 * images[ii] + 0.5 * images[jj])[None])  # 32,3,224,224
                libs_vgg[ii,jj] =  mid_outputs[0].cpu().detach()
                mid_outputs = []
        mid_outputs = []

        libs_alex = torch.zeros(32,32,384,13,13) #384,13,13, incv3: 768,12,12, dense121:512,28,28, sqn:128,27,27
        for ii in range(32):
            for jj in range(32):
                # hhs = self.model[1]._modules.get('features')._modules.get(layer_name).expand3x3_activation.register_forward_hook(get_mid_output) # incv4
                _ = self.models[2]((0.5 * images[ii] + 0.5 * images[jj])[None])  # 32,3,224,224
                libs_alex[ii,jj] =  mid_outputs[0].cpu().detach()
                mid_outputs = []
        mid_outputs = []

        libs_sqn = torch.zeros(32,32,128,27,27) #384,13,13, incv3: 768,12,12, dense121:512,28,28, sqn:128,27,27
        for ii in range(32):
            for jj in range(32):
                _ = self.models[3]((0.5 * images[ii] + 0.5 * images[jj])[None])  # 32,3,224,224
                libs_sqn[ii,jj] =  mid_outputs[0].cpu().detach()
                mid_outputs = []
        mid_outputs = []

        for i in range(self.steps):
            #tmpp = input_diversity2(adv_images, 0.9, 299, 330)
            adv_images = torch.clamp(images + torch.clamp(modifier, min=-self.eps, max=self.eps), min=0, max=1)
            ind = [x for x in range(adv_images.size(0))]
            ori_fea = libs_res[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[0](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid = F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            loss_adj = 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            ori_fea = libs_vgg[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[1](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            loss_adj += 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            ori_fea = libs_alex[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[2](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            loss_adj += 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            ori_fea = libs_sqn[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[3](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            loss_adj += 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs_res[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[0](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs_vgg[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[1](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj = 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs_alex[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[2](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj = 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs_sqn[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[3](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj = 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs_res[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[0](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs_vgg[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[1](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj = 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs_alex[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[2](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj = 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            ind = [x for x in range(adv_images.size(0))]
            random.shuffle(ind)
            ori_fea = libs_sqn[[x for x in range(32)], ind].cuda().reshape(32, -1)
            outputs = self.models[3](0.5 * adv_images + 0.5 * adv_images[ind])
            loss_mid += F.cosine_similarity(ori_fea, mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj = 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            print('mid:', loss_mid.item()/12.0)
            cost = loss_mid/12.0 + loss_adj/4.0

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            mid_outputs = []

        for h in hs:
            h.remove()

        return torch.clamp(images + torch.clamp(modifier, min=-self.eps, max=self.eps), min=0, max=1).detach()