from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset
from .sample.ctdetSpotNet2 import CTDetDatasetSpotNet2
from .sample.ctdetSpotNetVid import CTDetDatasetSpotNetVid
from .sample.ctdetVid import CTDetDatasetVid

from .dataset.uadetrac1on10_b import UADETRAC1ON10_b
from .dataset.uav import UAV


dataset_factory = {
  'uadetrac1on10_b': UADETRAC1ON10_b,
  'uav': UAV,
}

_sample_factory = {
  'ctdet': CTDetDataset,
  'ctdetSpotNet2': CTDetDatasetSpotNet2,
  'ctdetSpotNetVid': CTDetDatasetSpotNetVid,
  'ctdetVid': CTDetDatasetVid,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
