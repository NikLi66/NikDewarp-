# sub-parts of the U-Net model
import logging

import torch
import torch.nn as nn
from nik.img.util.mscan_util import msca_attention

def layer_input(in_channels, out_channels, bn_fn, act_fn, groups=1):
    return nn.Sequential(
        bn_fn(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=4, padding=1, groups=groups),
        # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=groups),
        bn_fn(out_channels),
        act_fn(inplace=True)
    )

def layer_down(in_channels, out_channels, bn_fn, act_fn, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=groups),
        bn_fn(out_channels),
        act_fn(inplace=True)
    )

def layer_out(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1)

def layer_block_mscan(channels, bn_fn, act_fn, groups=1):
    return nn.Sequential(
        msca_attention(channels, groups=groups),
        bn_fn(channels),
        act_fn(inplace=True)
    )

def layer_block_conv3x3(channels, bn_fn, act_fn, groups=1 ):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=groups),
        bn_fn(channels),
        act_fn(inplace=True)
    )

def layer_block_dilation(channels, dilation, bn_fn, act_fn, groups=1):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, groups=groups),
        bn_fn(channels),
        act_fn(inplace=True)
    )

class down_layer(nn.Module):

    def __init__(self, in_channels, out_channels, depth, layer_type, block_type, bn_fn, act_fn, groups):
        super(down_layer,self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._layer_type = layer_type
        self._block_type = block_type
        self._act_fn = act_fn
        self._bn_fn = bn_fn

        self._logger = logging.getLogger()
        self._logger.info("down_layer: in_channels:{} out_channels:{} depth:{} group:{} layer_type:{} block_type:{}".format(in_channels, out_channels, depth, groups, layer_type, block_type))

        if layer_type == "input":
            self._conv1 = layer_input(self._in_channels, self._out_channels, bn_fn=self._bn_fn, act_fn=self._act_fn, groups=groups)
        elif layer_type == "down":
            self._conv1 = layer_down(self._in_channels, self._out_channels, bn_fn=self._bn_fn, act_fn=self._act_fn, groups=groups)
        elif layer_type == "bridge":
            self._conv1 = layer_down(self._in_channels, self._out_channels, bn_fn=self._bn_fn, act_fn=self._act_fn, groups=groups)
        else:
            raise Exception("no such layer type")

        self._block_list = nn.ModuleList()
        for i in range(depth):
            if block_type == "mscan":
                block = layer_block_mscan(self._out_channels, bn_fn=self._bn_fn, act_fn=self._act_fn, groups=groups)
            elif block_type == "dilation":
                block = layer_block_dilation(self._out_channels, dilation=i+1, bn_fn=self._bn_fn, act_fn=self._act_fn, groups=groups)
            elif block_type == "conv3x3":
                block = layer_block_conv3x3(self._out_channels, bn_fn=self._bn_fn, act_fn=self._act_fn, groups=groups)
            else:
                raise Exception("no such block type")
            self._block_list.append(block)

        self._drop_out = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self._conv1(x)
        if self._layer_type == "bridge":
            b = []
            b.append(x)
            for blk in self._block_list:
                x = blk(x)
                b.append(x)
            x = torch.mean(torch.stack(b), dim=0)
        else:
            for blk in self._block_list:
                x = blk(x)
        return x

class up_layer(nn.Module):

    def __init__(self, in_channels, out_channels, depth, block_type, bn_fn, act_fn):
        super(up_layer,self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._block_type = block_type
        self._act_fn = act_fn
        self._bn_fn = bn_fn

        self._logger = logging.getLogger()
        self._logger.info("up_layer: in_channels:{} out_channels:{} depth:{} block_type:{}".format(in_channels, out_channels, depth, block_type))

        self._conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=1, output_padding=1),
            self._bn_fn(out_channels),
            self._act_fn(inplace=True),
        )
        self._conv2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, groups=1),
            self._bn_fn(out_channels),
            self._act_fn(inplace=True),
        )

        self._block_list = nn.ModuleList()
        for i in range(depth):
            if block_type == "mscan":
                block = layer_block_mscan(self._out_channels, bn_fn=self._bn_fn, act_fn=self._act_fn)
            elif block_type == "dilation":
                block = layer_block_dilation(self._out_channels, dilation=i+1, bn_fn=self._bn_fn, act_fn=self._act_fn)
            elif block_type == "conv3x3":
                block = layer_block_conv3x3(self._out_channels, bn_fn=self._bn_fn, act_fn=self._act_fn)
            else:
                raise Exception("no such layer type")
            self._block_list.append(block)

        self._drop_out = nn.Dropout2d(0.5)

    def forward(self, x1, x2):
        x1 = self._conv1(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self._conv2(x)
        for blk in self._block_list:
            x = blk(x)
        return x


