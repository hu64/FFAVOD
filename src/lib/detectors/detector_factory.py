from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetDetector
from .ctdetVid import CtdetDetectorVid
from .ctdetSpotNetVid import CtdetDetectorSpotNetVid

detector_factory = {
  'ctdet': CtdetDetector,
  'ctdetSpotNet2': CtdetDetector,
  'ctdetVid': CtdetDetectorVid,
  'ctdetSpotNetVid': CtdetDetectorSpotNetVid,
}
