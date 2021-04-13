from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.ctseg import CTSegDataset

from src.lib.datasets.dataset.coco import COCO
from src.lib.datasets.dataset.pascal import PascalVOC
from src.lib.datasets.dataset.kitti import KITTI
from src.lib.datasets.dataset.coco_hp import COCOHP
from src.lib.datasets.dataset.coco_seg import COCOSEG


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'coco_seg': COCOSEG
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset,
  'ctseg': CTSegDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
