#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
#

import argparse
import json


class TypeHelper:
    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def json(v: str):
        # if isinstance(v, list):
        #     return v
        return json.loads(v)

    @staticmethod
    def str2intlist(v: str):
        str_ids = TypeHelper.json(v)
        gpu_ids = []
        for str_id in str_ids:
            id_ = int(str_id)
            if id_ >= 0:
                gpu_ids.append(id_)
        return gpu_ids
