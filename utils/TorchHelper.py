#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
#
import os
from collections import OrderedDict
import torch
from torch.nn import init
from typing import Optional
from abc import ABC


class TorchHelper(ABC):

    @staticmethod
    def load_network_by_path(net: torch.nn.Module or torch.nn.DataParallel,
                             path: str,
                             device: torch.device,
                             strict=True):
        load_net = net.module if isinstance(net, torch.nn.DataParallel) else net
        missing_keys = list(name for name, _ in load_net.named_parameters())
        if not os.path.exists(path):
            msg = f"Weights not found at {path}."
            if strict:
                raise RuntimeError(msg)
            else:
                print(msg + " skipped.")
            return missing_keys
        pretrained_dict = torch.load(path, map_location=str(device))

        if strict:
            load_net.load_state_dict(pretrained_dict, strict=strict)
        else:
            try:
                missing_keys, unexpected_keys = load_net.load_state_dict(pretrained_dict, strict=strict)
            except RuntimeError:
                loaded_keys = []
                model_dict = net.state_dict()
                for key, value in pretrained_dict.items():
                    if key in model_dict:
                        if model_dict[key].shape == value.shape:
                            model_dict[key] = value
                            loaded_keys.append(key)
                loaded_keys = set(loaded_keys)
                missing_keys = list(set(model_dict.keys()) - loaded_keys)
                unexpected_keys = list(set(pretrained_dict.keys()) - loaded_keys)
                load_net.load_state_dict(OrderedDict(model_dict))
                print(f"loaded keys in {path}")
                print(loaded_keys)
            if missing_keys is not None and len(missing_keys) > 0:
                print(f"missing keys in loading {path}")
                print(missing_keys)
                print(f"unexpected keys in loading {path}")
                print(unexpected_keys)
        print(path, "loaded.")
        return missing_keys

    @staticmethod
    def save_network(net: torch.nn.Module, path: str):
        if isinstance(net, torch.nn.Module):
            torch.save(net.state_dict(), path)
        else:
            torch.save(net.module.state_dict(), path)
        print(path, "wrote.")

    @staticmethod
    def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
        """Initialize network weights.

        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if debug:
                    print(classname)
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        net.apply(init_func)  # apply the initialization function <init_func