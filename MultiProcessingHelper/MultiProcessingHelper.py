#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
#

from typing import Dict

from typing import List, Tuple, Union
from tqdm import tqdm
import multiprocessing.pool as mpp


class MultiProcessingHelper:

    def __init__(self, pool_class: str = None):
        if pool_class is None:
            pool_class = ""
        if pool_class in ["torch", "pytorch"]:
            from torch.multiprocessing.pool import Pool
        else:
            from multiprocessing.pool import Pool

        Pool.istarmap = istarmap
        self.__pool_class = Pool

    @staticmethod
    def _function_proxy(fun, kwargs: Dict):
        return fun(**kwargs)

    def run(self,
            args: Union[List, Tuple],
            n_workers: int,
            func=None,
            desc: str = None,
            mininterval: float = None,
            maxinterval: float = None,
            process_bar=False):
        assert len(args) > 0
        tqdm_args = {"total": len(args), "desc": desc}
        if mininterval is not None:
            tqdm_args["mininterval"] = mininterval
        if maxinterval is not None:
            tqdm_args["maxinterval"] = maxinterval

        if n_workers > 0:
            with self.__pool_class(n_workers) as pool:
                if process_bar:
                    return [ret for ret in tqdm(pool.istarmap(MultiProcessingHelper._function_proxy
                                                              if func is None else func,
                                                              iterable=args),
                                                **tqdm_args)]
                else:
                    return [ret for ret in pool.istarmap(MultiProcessingHelper._function_proxy
                                                         if func is None else func,
                                                         iterable=args)]
        else:
            """
                Version 1
            """
            # re = []
            # if func is None:
            #     for fun, kwargs in tqdm(args, **tqdm_args):
            #         re.append(self._function_proxy(fun=fun, kwargs=kwargs))
            # else:
            #     for arg in tqdm(args, **tqdm_args):
            #         re.append(func(*arg))
            # return re

            exec_fun = MultiProcessingHelper._function_proxy if func is None else func
            """
                Version 2
            """
            # re = []
            # for item in tqdm(args, **tqdm_args):
            #     re.append(exec_fun(*item))
            # return re

            """
                Version 3
            """
            if process_bar:
                return [exec_fun(*item) for item in tqdm(args, **tqdm_args)]
            else:
                return [exec_fun(*item) for item in args]


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)
