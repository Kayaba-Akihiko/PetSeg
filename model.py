#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch.nn as nn
import torch
import torchvision.models as models
from utils.TorchHelper import TorchHelper


class UNet(nn.Module):

    def __init__(self, n_class):
        super(UNet, self).__init__()

        self.encoder = models.mobilenet_v3_large(pretrained=True).features
        self.skip_layer_ids = [1, 3, 6, 12, 16]
        skip_layer_nc = [16, 24, 40, 112, 960]
        upsampler_nc = [n_class, 64, 128, 256, 512, 0]

        self.decoder = []
        for i in reversed(range(len(skip_layer_nc))):
            layer = self._build_upsampler(skip_layer_nc[i] + upsampler_nc[i + 1], upsampler_nc[i])
            self.decoder.append(layer)
        self.decoder = nn.ModuleList(self.decoder)
        self.decoder = TorchHelper.init_weights(self.decoder)

        # self.upsamplers = nn.ModuleList([self._build_upsampler(960, 512),
        #                                  self._build_upsampler(512 + 112, 256),
        #                                  self._build_upsampler(256 + 40, 128),
        #                                  self._build_upsampler(128 + 24, 64),
        #                                  self._build_upsampler(64 + 16, n_class)])

    @staticmethod
    def _build_upsampler(input_nc, output_nc):
        return nn.Sequential(nn.ConvTranspose2d(in_channels=input_nc,
                                                out_channels=output_nc,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1,
                                                bias=False),
                             nn.BatchNorm2d(output_nc),
                             nn.ReLU(True))

    def forward(self, x):
        y_list = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_layer_ids:
                y_list.append(x)

        x_lsit = y_list[:: -1]
        y = self.decoder[0](x_lsit[0])

        for i in range(1, len(self.decoder)):
            y = self.decoder[i](torch.cat((y, x_lsit[i]), dim=1))
        return y
