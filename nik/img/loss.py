import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nik.utils.util_img_io import get_img_offset

EPSILON = 1e-6

class NikLosses(object):
    def __init__(self, img_height, img_width, device):
        self.kernel_cross_17 = torch.tensor(
                                [[[[0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                     [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0]]],
                                 [[[0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1., 0, 0, 0, 0]]]]).to(device)

        self._img_height = img_height
        self._img_width = img_width
        self._img_offset = torch.tensor(get_img_offset(self._img_height, self._img_width).transpose(2, 0, 1)).to(device)

        self._logger = logging.getLogger()
        self._mask_border_1 = torch.tensor(self.get_border_mask(img_height, img_width, 3)).to(device)
        # self._mask_border_2 = torch.tensor(self.get_border_mask(img_height//8, img_width//8, 4)).to(device)
        # self._mask_border_4 = torch.tensor(self.get_border_mask(img_height//16, img_width//16, 3)).to(device)
        # self._mask_border_8 = torch.tensor(self.get_border_mask(img_height//32, img_width//32, 2)).to(device)

    def multi_loss(self, o, o0, o1, o2, o3, fp, mask):
        # print(o.shape,o0.shape,o1.shape,o2.shape,o3.shape,fp.shape,mask.shape)
        l3 = self.simple_loss(o3, F.interpolate(fp, scale_factor=1 / 8))
        l2 = self.simple_loss(o2, F.interpolate(fp, scale_factor=1 / 4))
        l1 = self.simple_loss(o1, F.interpolate(fp, scale_factor=1 / 2))
        l0 = self.simple_loss(o0, fp)
        l = self.total_loss(o, fp, mask, self._mask_border_1)
        return l, l0, l1, l2, l3

    def get_border_mask(self, height, width, d):
        mask = np.zeros([height, width], dtype=np.float32)
        mask[0:d, :] = 1.0
        mask[:, 0:d] = 1.0
        mask[height - d : height, :] = 1.0
        mask[:, width - d : width] = 1.0
        return mask

    def total_loss(self, input, target, mask, mask_border):

        '''one'''
        # bs = input.shape[0]
        loss_l1_all = F.smooth_l1_loss(input, target, reduction='mean')
        loss_l1_mask = torch.sum(F.smooth_l1_loss(input, target, reduction='none') * mask) / (torch.sum(mask) + EPSILON)
        # loss_l1_border = torch.sum(F.smooth_l1_loss(input, target, reduction='none') * mask_border) / (torch.sum(mask_border * bs) + EPSILON)

        '''two'''
        i_t = target - input
        # i_t_avg = F.conv2d(F.pad(i_t, (4, 4, 4, 4), mode='replicate'), self.kernel_cross_17, padding=0, groups=2) / 17
        # # loss_local = F.smooth_l1_loss(i_t, i_t_avg,  reduction='sum') / torch.sum(mask)
        # loss_local = F.smooth_l1_loss(i_t, i_t_avg, reduction='mean')
        local = torch.pow(F.conv2d(F.pad(i_t, (4, 4, 4, 4), mode='replicate'), self.kernel_cross_17, padding=0, groups=2) - i_t*17, 2)
        weights = F.conv2d(F.pad(torch.ones_like(i_t), (4, 4, 4, 4), mode='constant', value=0), self.kernel_cross_17, padding=0, groups=2)
        weights = 1/weights*17
        loss_local = torch.mean(local*weights)

        return  0.5 * loss_l1_all + loss_l1_mask + 0.5 * loss_local

    def simple_loss(self, input, target):

        loss_l1_all = F.smooth_l1_loss(input, target, reduction='mean')

        '''two'''
        i_t = target - input
        local = torch.pow(F.conv2d(F.pad(i_t, (4, 4, 4, 4), mode='replicate'), self.kernel_cross_17, padding=0, groups=2) - i_t*17, 2)
        weights = F.conv2d(F.pad(torch.ones_like(i_t), (4, 4, 4, 4), mode='constant', value=0), self.kernel_cross_17, padding=0, groups=2)
        weights = 1/weights*17
        loss_local = torch.mean(local*weights)

        return  loss_l1_all + 0.5 * loss_local

    def loss_fn4_v5_r_4(self, fp_m, fp_t, mask, img, ori):
        # self._logger.info("loss_fn4_v5_r_4: input:{} target:{} mask:{}".format(input.shape, target.shape, mask.shape))

        fp_m = fp_m * mask
        fp_t = fp_t * mask
        i_t = fp_m - fp_t
        # self._logger.info("loss_fn4_v5_r_4: i_t:{}".format(i_t.shape))

        '''one'''
        loss_l1 = F.smooth_l1_loss(fp_m*mask, fp_t, reduction='sum') / torch.sum(mask)

        '''two'''
        mask_num = F.conv2d(F.pad(mask, (4, 4, 4, 4), mode='constant'), self.kernel_cross_17, padding=0)
        i_t_2 = F.conv2d(F.pad(i_t, (4, 4, 4, 4), mode='constant'), self.kernel_cross_17, padding=0, groups=2) * mask - mask_num * i_t
        loss_local = torch.sum(torch.pow(i_t_2, 2)) / torch.sum(mask_num)

        # '''gen'''
        fp_m = fp_m + self._img_offset
        fp_m = (fp_m - 512) / 512
        fp_m = fp_m.transpose(1, 2).transpose(2, 3)
        ori_warp = F.grid_sample(img, fp_m, align_corners=True)
        loss_gen = nn.functional.l1_loss(ori_warp * mask, ori, reduction="sum") / torch.sum(mask)

        return loss_l1, loss_local, loss_gen



