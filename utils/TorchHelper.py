#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
#
import os
from collections import OrderedDict
import torch
from torch.nn import init
from torch.optim import lr_scheduler
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from typing import Optional
import copy
from .TypeHelper import TypeHelper



class TorchHelper:


    @staticmethod
    def get_scheduler(optimizer, opt: Namespace, epochs=None, steps_per_epoch=None):

        """Return a learning rate scheduler

        Parameters:
            optimizer          -- the optimizer of the network
            opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                                  opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

        For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
        and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
        For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
        See https://pytorch.org/docs/stable/optim.html for more details.
        """

        if opt.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
        elif opt.lr_policy == "cosine_warm":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=opt.T_0,
                                                                 T_mult=opt.T_mult)
        elif opt.lr_policy == "multi_step":
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=opt.milestones,
                                                 gamma=opt.milestones_gamma)
        elif opt.lr_policy == "one_cycle":
            assert epochs is not None and steps_per_epoch is not None
            scheduler = lr_scheduler.OneCycleLR(optimizer,
                                                max_lr=opt.max_lr,
                                                epochs=epochs,
                                                steps_per_epoch=steps_per_epoch)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler

    @staticmethod
    def modify_lr_policy_parser(parser: ArgumentParser, args: Sequence[str]) -> ArgumentParser:
        parser.add_argument("--lr_policy", type=str, default="linear", choices=["linear",
                                                                                "cosine",
                                                                                "cosine_warm",
                                                                                "multi_step",
                                                                                "one_cycle"])
        opt, _ = parser.parse_known_args(args=args)
        if opt.lr_policy == "linear":
            parser.add_argument("--n_epochs_decay", type=int, required=True)
        elif opt.lr_policy == "cosine_warm":
            parser.add_argument("--T_0", type=int, default=10)
            parser.add_argument("--T_mult", type=int, default=2)
        elif opt.lr_policy == "multi_step":
            parser.add_argument("--milestones", type=TypeHelper.str2intlist, required=True)
            parser.add_argument("--milestones_gamma", type=float, default=0.1)
        elif opt.lr_policy == "one_cycle":
            parser.add_argument("--max_lr", type=float, required=True)
        return parser

    @staticmethod
    def parallelize_net(net: torch.nn.Module or torch.nn.DataParallel, gpu_ids: Sequence[int]):
        if isinstance(net, torch.nn.DataParallel):
            return torch.nn.DataParallel(net.module, gpu_ids)
        assert isinstance(net, torch.nn.Module)
        return torch.nn.DataParallel(net, gpu_ids)

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

        new_pretrained_dict = OrderedDict()
        for name, val in pretrained_dict.items():
            if name.startswith('_model'):
                name = name[1:]
            new_pretrained_dict[name] = val
        pretrained_dict = new_pretrained_dict

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
    def data_to_device(data: dict,
                       device: torch.device,
                       torch_dtype=torch.float32,
                       in_place=True) -> dict:
        # if in_place:
        #     re_data = data
        # else:
        #     re_data = {}
        #     for key, img_dao in data.items():
        #         re_data[key] = ImageDAO(**(img_dao.to_dict()), copy_data=True)
        #
        # for key, img_dao in re_data.items():
        #     img_dao: ImageDAO
        #     if img_dao.img_data is not None:
        #         img_dao.img_data = img_dao.img_data.to(device)
        #         img_dao.img_data.requires_grad = True
        #     for info_key, val in img_dao.case_info.items():
        #         img_dao.case_info[info_key] = val.to(device)

        if in_place:
            re_data = data
        else:
            re_data = copy.deepcopy(data)

        for key, val in re_data.items():
            if isinstance(val, torch.Tensor):
                re_data[key] = val.to(torch_dtype).to(device)

        return re_data


    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def trigger_model(model: torch.nn.Module or torch.nn.DataParallel, train: bool):
        if isinstance(model, torch.nn.Module):
            trig_model = model
        else:
            trig_model = model.module
        if train:
            trig_model.train()
        else:
            trig_model.eval()

    @staticmethod
    def trigger_models(nets, train: bool):
        for net in nets:
            TorchHelper.trigger_model(model=net, train=train)

    @staticmethod
    def init_net(net: torch.nn.Module,
                 init_type='normal',
                 init_gain=0.02,
                 device: Optional[torch.device] = torch.device("cpu"),
                 initialize_weights=True) -> torch.nn.Module:
        """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
        :param device:
        :param init_gain: scaling factor for normal, xavier and orthogonal.
        :param init_type: the name of an initialization method: normal | xavier | kaiming | orthogonal
        :param net: the network to be initialized
        :param initialize_weights:
        """
        net = net.to(device)
        if initialize_weights:
            TorchHelper.init_weights(net, init_type, init_gain=init_gain, debug=False)
        return net

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