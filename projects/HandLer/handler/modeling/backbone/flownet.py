import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_

__all__ = [
    'flownets', 'flownets_bn'
]

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def offset_loss(pred_offset, cur_locs, pre_locs):
    MSELoss = nn.MSELoss(reduction='sum')
    loss = 0
    bs = cur_locs[0].shape[0]

    idx = 0
    for cur_loc, pre_loc in zip(cur_locs, pre_locs):
        for offset in pred_offset:
            offset_norm = torch.min(torch.sum(cur_loc), torch.sum(pre_loc))
            offset = offset[idx].unsqueeze(0).permute(0, 2, 3, 1)
            size = offset.shape[-3:-1]
            cur_loc_resize, pre_loc_resize = F.interpolate(cur_loc, size=size, mode="bilinear", align_corners=True), \
                                             F.interpolate(pre_loc, size=size, mode="bilinear", align_corners=True)
            pred_pre = F.grid_sample(cur_loc_resize, offset, mode="bilinear", padding_mode="zeros", align_corners=True)
            if offset_norm == 0:
                loss += MSELoss(pred_pre, pre_loc_resize)
            else:
                loss += MSELoss(pred_pre, pre_loc_resize) / (offset_norm + 1e-5)
        idx += 1
    return {'offset_loss': loss*0.1}

def pose_offset_loss(offsets, cur_hm, pre_hm):
    MSELoss = nn.MSELoss(reduction='mean')
    loss=MSELoss(cur_hm, pre_hm)
    return {'pose_offset_loss': loss}


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetS,self).__init__()

        self.batchNorm = batchNorm

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv2, out_conv3, out_conv4, out_conv5, out_conv6 = x

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        # if self.training:
        return flow2,flow3,flow4,flow5,flow6
        # else:
        #     return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownets(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)
    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def flownets_bn(data=None, device=0):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)
    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=True)
    if data is not None:
        data = torch.load(data, map_location=torch.device(device))
        model.load_state_dict(data['state_dict'], strict=False)
    return model