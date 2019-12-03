import torch
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module
class LoadPersonID(object):
    def __call__(self, results):
        results["gt_pids"] = DC(torch.from_numpy(results["ann_info"]["person_ids"]))
        return results

    def __repr__(self):
        return self.__class__.__name__
