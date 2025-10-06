import logging

import torch
import torch.nn as nn
from nik.img.util.unet_util import up_layer, down_layer, layer_out
import torch.nn.functional as F


class ImgUNet(nn.Module):
    def __init__(self, in_channels, out_channels, use_ori=1, use_canny_input=0,  model_name="unet"):
        super(ImgUNet, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._use_ori = use_ori
        self._use_canny_input = use_canny_input
        self._act_fn = nn.ReLU
        self._bn_fn = nn.InstanceNorm2d
        self._logger = logging.getLogger()

        if model_name == "unet1":
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "down", "down", "bridge"]  # input down bridge
            self._down_block_depths = [1, 1, 1, 6]
            self._down_block_groups = [1, 1, 1, 1]
            self._down_block_types = ["conv3x3", "conv3x3", "conv3x3", "dilation"]  # mscan conv3x3 dilation
            self._up_block_depths = [1, 1, 1]
            self._up_block_types = ["conv3x3", "conv3x3", "conv3x3"]
        elif model_name == "unet2":
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "down", "down", "bridge"]  # input down bridge
            self._down_block_depths = [2, 2, 2, 5]
            self._down_block_groups = [2, 2, 2, 1]
            self._down_block_types = ["conv3x3", "conv3x3", "conv3x3", "dilation"]  # mscan conv3x3 dilation
            self._up_block_depths = [2, 2, 2]
            self._up_block_types = ["conv3x3", "conv3x3", "conv3x3"]
        elif model_name == "unet3" :
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "down", "down", "bridge"]  # input down bridge
            self._down_block_depths = [2, 2, 2, 5]
            self._down_block_groups = [2, 2, 2, 1]
            self._down_block_types = ["conv3x3", "conv3x3", "conv3x3", "dilation"]  # mscan conv3x3 dilation
            self._up_block_depths = [1, 1, 1]
            self._up_block_types = ["conv3x3", "conv3x3", "conv3x3"]
        elif model_name == "unet4" :
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "down", "down", "bridge"]  # input down bridge
            self._down_block_depths = [2, 2, 2, 5]
            self._down_block_groups = [2, 2, 2, 1]
            self._down_block_types = ["conv3x3", "conv3x3", "conv3x3", "dilation"]  # mscan conv3x3 dilation
            self._up_block_depths = [4, 4, 4]
            self._up_block_types = ["conv3x3", "conv3x3", "conv3x3"]
        elif model_name == "unet5":
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "down", "down", "bridge"]  # input down bridge
            self._down_block_depths = [4, 4, 4, 5]
            self._down_block_groups = [2, 2, 2, 1]
            self._down_block_types = ["conv3x3", "conv3x3", "conv3x3", "dilation"]  # mscan conv3x3 dilation
            self._up_block_depths = [1, 1, 1]
            self._up_block_types = ["conv3x3", "conv3x3", "conv3x3"]
        elif model_name == "unet6":
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "down", "down", "bridge"]  # input down bridge
            self._down_block_depths = [2, 2, 2, 10]
            self._down_block_groups = [2, 2, 2, 1]
            self._down_block_types = ["conv3x3", "conv3x3", "conv3x3", "dilation"]  # mscan conv3x3 dilation
            self._up_block_depths = [1, 1, 1]
            self._up_block_types = ["conv3x3", "conv3x3", "conv3x3"]
        elif model_name == "unet7":
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "down", "down", "bridge"]  # input down bridge
            self._down_block_depths = [2, 2, 2, 5]
            self._down_block_groups = [2, 2, 2, 1]
            self._down_block_types = ["dilation", "dilation", "dilation", "dilation"]  # mscan conv3x3 dilation
            self._up_block_depths = [1, 1, 1]
            self._up_block_types = ["conv3x3", "conv3x3", "conv3x3"]
        elif model_name == "mscan1":
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "down", "down", "down"]  # input down bridge
            self._down_block_depths = [2, 2, 4, 4]
            self._down_block_groups = [1, 1, 1, 1]
            self._down_block_types = ["mscan", "mscan", "mscan", "mscan"]  # mscan conv3x3 dilation
            self._up_block_depths = [1, 1, 1]
            self._up_block_types = ["conv3x3", "conv3x3", "conv3x3"]
        elif model_name == "mscan2":
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "bridge", "bridge", "bridge"]  # input down bridge
            self._down_block_depths = [3, 3, 5, 2]
            self._down_block_groups = [1, 1, 1, 1]
            self._down_block_types = ["mscan", "mscan", "mscan", "mscan"]  # mscan conv3x3 dilation
            self._up_block_depths = [1, 1, 1]
            self._up_block_types = ["conv3x3", "conv3x3", "conv3x3"]
        elif model_name == "mscan3":
            self._layer_channels = [64, 128, 256, 512]
            self._layer_types = ["input", "bridge", "bridge", "bridge"]  # input down bridge
            self._down_block_depths = [3, 3, 5, 2]
            self._down_block_groups = [2, 2, 2, 1]
            self._down_block_types = ["mscan", "mscan", "mscan", "mscan"]  # mscan conv3x3 dilation
            self._up_block_depths = [2, 2, 2]
            self._up_block_types = ["mscan", "mscan", "mscan"]
        else:
            raise Exception("no such model name")

        #
        if self._use_ori == 0:
            if self._use_canny_input == 0:
                self._down_block_groups = [1, 1, 1, 1]
            elif self._use_canny_input == 1:
                self._down_block_groups = [1, 1, 1, 1]
            else:
                self._down_block_groups = [2, 2, 2, 1]
        else:
            if self._use_canny_input == 0:
                self._down_block_groups = [2, 2, 2, 1]
            elif self._use_canny_input == 1:
                self._down_block_groups = [2, 2, 2, 1]
            else:
                self._down_block_groups = [4, 4, 4, 1]
                # self._down_block_groups = [2, 2, 2, 1]

        self._input = down_layer(in_channels=self._in_channels, out_channels=self._layer_channels[0],
                                     depth=self._down_block_depths[0], layer_type=self._layer_types[0], block_type=self._down_block_types[0],
                                     bn_fn=self._bn_fn, act_fn=self._act_fn, groups=self._down_block_groups[0])
        self._down_1 = down_layer(in_channels=self._layer_channels[0], out_channels=self._layer_channels[1],
                                      depth=self._down_block_depths[1], layer_type=self._layer_types[1], block_type=self._down_block_types[1],
                                      bn_fn=self._bn_fn, act_fn=self._act_fn, groups=self._down_block_groups[1])
        self._down_2 = down_layer(in_channels=self._layer_channels[1], out_channels=self._layer_channels[2],
                                      depth=self._down_block_depths[2], layer_type=self._layer_types[2], block_type=self._down_block_types[2],
                                      bn_fn=self._bn_fn, act_fn=self._act_fn, groups=self._down_block_groups[2])
        self._down_3 = down_layer(in_channels=self._layer_channels[2], out_channels=self._layer_channels[3],
                                      depth=self._down_block_depths[3], layer_type=self._layer_types[3], block_type=self._down_block_types[3],
                                      bn_fn=self._bn_fn, act_fn=self._act_fn, groups=self._down_block_groups[3])

        self._up_1 = up_layer(in_channels=self._layer_channels[1], out_channels=self._layer_channels[0],
                                  depth=self._up_block_depths[0], block_type=self._up_block_types[0], bn_fn=self._bn_fn, act_fn=self._act_fn)
        self._up_2 = up_layer(in_channels=self._layer_channels[2], out_channels=self._layer_channels[1],
                                  depth=self._up_block_depths[1], block_type=self._up_block_types[1], bn_fn=self._bn_fn, act_fn=self._act_fn)
        self._up_3 = up_layer(in_channels=self._layer_channels[3], out_channels=self._layer_channels[2],
                                  depth=self._up_block_depths[2], block_type=self._up_block_types[2], bn_fn=self._bn_fn, act_fn=self._act_fn)

        self._out_0 = layer_out(self._layer_channels[0], self._out_channels)
        self._out_1 = layer_out(self._layer_channels[1], self._out_channels)
        self._out_2 = layer_out(self._layer_channels[2], self._out_channels)
        self._out_3 = layer_out(self._layer_channels[3], self._out_channels)

        self._output = layer_out(sum(self._layer_channels), self._out_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                nn.init.xavier_normal_(m.weight, gain=0.2)

    def forward(self, x):

        d = []
        x = self._input(x)
        d.append(x)
        x = self._down_1(x)
        d.append(x)
        x = self._down_2(x)
        d.append(x)
        x = self._down_3(x)

        mix_out = []
        mix_out.append(F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=False))
        o3 = self._out_3(x)

        x = self._up_3(x, d[2])
        mix_out.append(F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False))
        o2 = self._out_2(x)

        x = self._up_2(x, d[1])
        mix_out.append(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        o1 = self._out_1(x)

        x = self._up_1(x, d[0])
        mix_out.append(x)
        o0 = self._out_0(x)

        o = self._output(torch.cat(mix_out, dim=1))
        # self._logger.info("output shape {}".format(x.shape))
        return o, o0, o1, o2, o3

    def print_param(self):
        total_params = 0

        num_params = sum(p.numel() for p in self._input.parameters())
        total_params += num_params
        self._logger.info("_input parameters: %d " % num_params)

        num_params = sum(p.numel() for p in self._down_1.parameters())
        total_params += num_params
        self._logger.info("_down_1 parameters: %d " % num_params)

        num_params = sum(p.numel() for p in self._down_2.parameters())
        total_params += num_params
        self._logger.info("_down_2 parameters: %d " % num_params)

        num_params = sum(p.numel() for p in self._down_3.parameters())
        total_params += num_params
        self._logger.info("_down_3 parameters: %d " % num_params)

        num_params = sum(p.numel() for p in self._up_1.parameters())
        total_params += num_params
        self._logger.info("_up_1 parameters: %d " % num_params)

        num_params = sum(p.numel() for p in self._up_2.parameters())
        total_params += num_params
        self._logger.info("_up_2 parameters: %d " % num_params)

        num_params = sum(p.numel() for p in self._up_3.parameters())
        total_params += num_params
        self._logger.info("_up_3 parameters: %d " % num_params)

        num_params = sum(p.numel() for p in self._output.parameters())
        total_params += num_params
        self._logger.info("_output parameters: %d " % num_params)

        self._logger.info("total parameters: %d " % total_params)
