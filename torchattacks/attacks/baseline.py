import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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
class I2V:
    r"""
    our baseline.
    """
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
        # videomodel.eval()
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()
        loss = nn.CrossEntropyLoss()
        modif = torch.Tensor(images.shape).fill_(0.01/255).cuda()
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=self.alpha)

        # feature_layers = list(self.model[1]._modules.keys())
        # import ipdb; ipdb.set_trace()
        #feature_layers = list(self.model[1]._modules.keys())[7:8]  # res101
        # feature_layers = ['layer2']  # res101
        # feature_layers = ['Mixed_6c'] # incv3
        # feature_layers = ['9']     # incv4
        # feature_layers = ['mixed_6a']      # incResv2
        feature_layers = ['20']     # vgg: 20, cifar100:25
        # feature_layers = ['layer2']      # Res152
        # feature_layers = ['6']     # sqn
        # feature_layers = ['7']     # alex
        # feature_layers = ['stage_3']  # resnext

        #mid_outputs = []
        global mid_outputs

        def get_mid_output(m, i, o):
            global mid_outputs
            mid_outputs.append(o)

        hs = []
        for layer_name in feature_layers:
            #print(layer_name)
            # import ipdb; ipdb.set_trace()
            # if self.model[1]._modules.get(layer_name) is not None:
            #     hs.append(self.model[1]._modules.get(layer_name).register_forward_hook(get_mid_output))
            # hs.append(self.model[1]._modules.get('features')._modules.get(layer_name).expand3x3_activation.register_forward_hook(get_mid_output)) # incv4
            hs.append(self.model[1]._modules.get('features')._modules.get(layer_name).register_forward_hook(get_mid_output)) # incv4

        out = self.model(images)
        # import ipdb;ipdb.set_trace()
        mid_originals = []
        for mid_output in mid_outputs:
            mid_original = torch.zeros(mid_output.size()).cuda()
            mid_originals.append(mid_original.copy_(mid_output))
        mid_outputs = []
        # import ipdb; ipdb.set_trace()
        # losses = []
        # grad_norm = []
        for i in range(self.steps):
            #tmpp = input_diversity2(adv_images, 0.9, 299, 330)
            adv_images = torch.clamp(images + torch.clamp(modifier, min=-self.eps, max=self.eps), min=0, max=1)
            outputs = self.model(adv_images) #F.interpolate(adv_images, (112, 112), mode='bilinear')
            #print(outputs.mean())
            # import ipdb;ipdb.set_trace()
            mid_originals_ = []
            for mid_original in mid_originals:
                mid_originals_.append(mid_original.detach())

            loss_mid = F.cosine_similarity(mid_originals_[0].reshape(32, -1), mid_outputs[0].reshape(32, -1)).mean()
            # loss_mid += 0.01 * F.cosine_similarity(mid_outputs[0].reshape(32, -1)[:-1], mid_outputs[0].reshape(32, -1)[1:]).mean()
            print('mid:', loss_mid.item())
            # print(F.cross_entropy(outputs, labels))
            # if i == self.steps-1:
            #     import ipdb;ipdb.set_trace()
            # losses.append(F.cross_entropy(outputs, labels).item())
            # input_video = transform_video(adv_images.clone().detach(), mode='forward')
            # input_video = input_video.reshape(1,32,3,224,224)
            # input_video = input_video.permute([0,2,1,3,4])
            #import ipdb;ipdb.set_trace()
            # losses.append(F.cross_entropy(videomodel(input_video), labels[0].unsqueeze(dim=0)).item())

            cost = loss_mid.cuda()
            optimizer.zero_grad()
            cost.backward()
            # print(modifier.grad.norm(p=1))
            # grad_norm.append(modifier.grad.norm(dim=(1,2,3),p=1).mean().item())
            optimizer.step()
            mid_outputs = []

        for h in hs:
            h.remove()
        return torch.clamp(images + torch.clamp(modifier, min=-self.eps, max=self.eps), min=0, max=1).detach()#, np.array(losses)