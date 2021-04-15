from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ctdetSpotNet2 import CtdetTrainerSpotNet2
from .ctdetMultiSpot import CtdetTrainerMultiSpot
from .ctdetVid import CtdetTrainerVid
from .ctdetSpotNetVid import CtdetTrainerSpotNetVid
from .ctdetWipeNet import CtdetWipeNetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer


train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'ctdetSpotNet2': CtdetTrainerSpotNet2,
  'ctdetMultiSpot': CtdetTrainerMultiSpot,
  'ctdetSpotNetVid': CtdetTrainerSpotNetVid,
  'ctdetVid': CtdetTrainerVid,
  'multi_pose': MultiPoseTrainer, 
  'ctdetWipeNet': CtdetWipeNetTrainer,
  
}
