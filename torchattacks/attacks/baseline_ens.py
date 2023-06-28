import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

mid_outputs = []
class I2V_ENS:
    r"""
    our baseline (ens).
    """
    def __init__(self, models, eps=8/255, alpha=1/255, steps=8, momentum=0.9, targeted=False):
        self.models = models
        self.eps = eps
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


        out = self.models[0](images)
        mid_originals_m1 = []
        for mid_output in mid_outputs:
            mid_original = torch.zeros(mid_output.size()).cuda()
            mid_originals_m1.append(mid_original.copy_(mid_output))
        mid_outputs = []

        out = self.models[1](images)
        mid_originals_m2 = []
        for mid_output in mid_outputs:
            mid_original = torch.zeros(mid_output.size()).cuda()
            mid_originals_m2.append(mid_original.copy_(mid_output))
        mid_outputs = []

        out = self.models[2](images)
        mid_originals_m3 = []
        for mid_output in mid_outputs:
            mid_original = torch.zeros(mid_output.size()).cuda()
            mid_originals_m3.append(mid_original.copy_(mid_output))
        mid_outputs = []

        out = self.models[3](images)
        mid_originals_m4 = []
        for mid_output in mid_outputs:
            mid_original = torch.zeros(mid_output.size()).cuda()
            mid_originals_m4.append(mid_original.copy_(mid_output))
        mid_outputs = []
        # import ipdb;ipdb.set_trace()
        for i in range(self.steps):
            #tmpp = input_diversity2(adv_images, 0.9, 299, 330)
            adv_images = torch.clamp(images + torch.clamp(modifier, min=-self.eps, max=self.eps), min=0, max=1)
            outputs = self.models[0](adv_images)
            mid_originals_ = []
            for mid_original in mid_originals_m1:
                mid_originals_.append(mid_original.detach())
            loss_mid = F.cosine_similarity(mid_originals_[0].reshape(32, -1), mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj = 0.1 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            outputs = self.models[1](adv_images)
            mid_originals_ = []
            for mid_original in mid_originals_m2:
                mid_originals_.append(mid_original.detach())
            loss_mid += F.cosine_similarity(mid_originals_[0].reshape(32, -1), mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj += 0.1 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            outputs = self.models[2](adv_images)
            mid_originals_ = []
            for mid_original in mid_originals_m3:
                mid_originals_.append(mid_original.detach())
            loss_mid += F.cosine_similarity(mid_originals_[0].reshape(32, -1), mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj += 0.1 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            outputs = self.models[3](adv_images)
            mid_originals_ = []
            for mid_original in mid_originals_m4:
                mid_originals_.append(mid_original.detach())
            loss_mid += F.cosine_similarity(mid_originals_[0].reshape(32, -1), mid_outputs[0].reshape(32, -1)).mean()
            # loss_adj += 0.1 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            mid_outputs = []

            print('mid:', loss_mid.item()/4.0)
            cost = loss_mid/4.0 #+ loss_adj/4.0

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            mid_outputs = []

        for h in hs:
            h.remove()

        return torch.clamp(images + torch.clamp(modifier, min=-self.eps, max=self.eps), min=0, max=1).detach()