#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch
import torchvision.models as models
from model import UNet


def main():
    device = torch.device('cpu')
    model = models.mobilenet_v3_large(pretrained=True)
    print(model.parameters())

    x = torch.randn(2, 3, 224, 224, device=device)
    print(x.shape)

    layers = [1, 3, 6, 12, 16]
    for i, layer in enumerate(model.features):
        # print(i)
        x = layer(x)
        print(i, x.shape)



    #
    # x0 = model.features[0](x)
    # x1 = model.features[1](x0)
    # print(x0.shape)
    pass

def main2():
    device = torch.device('cpu')
    model = UNet(3).to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    y = model(x)
    print(y.shape)

    pass





if __name__ == '__main__':
    main2()