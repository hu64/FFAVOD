from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.large_hourglass import get_large_hourglass_net
from .networks.large_hourglass_SpotNet2 import get_large_hourglass_net_spotnet2
from .networks.large_hourglass_SpotNetVid import get_large_hourglass_net_SpotNetVid
from .networks.large_hourglass_vid import get_large_hourglass_net_vid


_model_factory = {
  'hourglass': get_large_hourglass_net,
  'hourglassSpotnet2': get_large_hourglass_net_spotnet2,
  'hourglassVid': get_large_hourglass_net_vid,
  'hourglassSpotNetVid': get_large_hourglass_net_SpotNetVid,
}

def create_model(arch, heads, head_conv, nbr_frames=None):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  if arch == 'hourglassVid' or 'hourglassSpotNetVid':
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv, nbr_frames=nbr_frames)
  else:
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None, nbr_frames=5):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  if 'epoch' in checkpoint:
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}

  FUSION = False
  if FUSION:
    # fusion_checkpoint = torch.load('/usagers2/huper/dev/SpotNet2/exp/uadetrac1on10_b/ctdetVid/fromSpotNetNoBias/model_best.pth', map_location=lambda storage, loc: storage)
    # fusion_checkpoint = torch.load('/usagers2/huper/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/AblationTestfromDetrac3/model_best.pth',map_location=lambda storage, loc: storage)
    # fusion_checkpoint = torch.load('/usagers2/huper/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromSpotNet2/FusionFromDetrac.pth',map_location=lambda storage, loc: storage)
    # fusion_checkpoint = torch.load('/store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_best.pth',map_location=lambda storage, loc: storage)
    # fusion_checkpoint = torch.load('/store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNetVid/fromDetrac/model_1.pth',map_location=lambda storage, loc: storage)
    fusion_checkpoint = torch.load('../exp/uav/ctdetSpotNetVid/fromCOCOB/model_last.pth',map_location=lambda storage, loc: storage)
    fusion_state_dict = fusion_checkpoint['state_dict']

  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  if FUSION:
    for k in fusion_state_dict:
      if 'Fusion' in k:
        print('fusion: ', k)
        # print(fusion_state_dict[k])
        model_state_dict[k] = fusion_state_dict[k]
        state_dict[k] = fusion_state_dict[k]

  EXT_SEG = False
  if EXT_SEG:
    # SEG_W = torch.load('../exp/uav/ctdetSpotNet2/fromSNVID/model_best.pth',map_location=lambda storage, loc: storage)
    SEG_W = torch.load('/store/datasets/UA-DetracResults/exp/ctdet/SpotNet2_ADD_DB_PYFLOW_LR2e6/model_best.pth',map_location=lambda storage, loc: storage)
    # SEG_W = torch.load('/usagers2/huper/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNet2/ct_attResElliptical/model_best.pth', map_location=lambda storage, loc: storage)
    # SEG_W = torch.load('/usagers2/huper/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNet2/2attResElliptical/model_best.pth', map_location=lambda storage, loc: storage)
    seg_state_dict = SEG_W['state_dict']
    for k in seg_state_dict:
      if 'seg' in k:
        print('seg: ', k)
        model_state_dict[k] = seg_state_dict[k]
        state_dict[k] = seg_state_dict[k]
              
  EXT_CT_ATT = False
  if EXT_CT_ATT:
    CT_ATT_W = torch.load('/store/dev/SpotNet2/exp/uadetrac1on10_b/ctdetSpotNet2/2attResElliptical/model_best.pth', map_location=lambda storage, loc: storage)
    ct_state_dict = CT_ATT_W['state_dict']
    for k in ct_state_dict:
      if 'ct_att' in k:
        model_state_dict[k] = ct_state_dict[k]
        state_dict[k] = ct_state_dict[k]

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'



  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))

        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
    else:
      do_nothing = True
      # hughes freeze model
      # model_state_dict[k].requires_grad = False

  # """
  import numpy as np
  # fusion
  if False and FUSION:
    for k in fusion_state_dict:
      if 'FusionModule' in k:
        print('Loaded Fusion: ', k)
        if nbr_frames == 3:
          state_dict[k] = fusion_state_dict[k][:, 1:-1, :, :]
        elif nbr_frames == 5:
          state_dict[k] = fusion_state_dict[k]
        elif nbr_frames == 7:
          state_dict[k] = np.concatenate([np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          fusion_state_dict[k][:, :, :, :],
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1)], 1)
          state_dict[k] = torch.Tensor(state_dict[k])
        elif nbr_frames == 9:
          state_dict[k] = np.concatenate([np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          fusion_state_dict[k][:, :, :, :],
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1)], 1)
        elif nbr_frames == 11:
          state_dict[k] = np.concatenate([np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          fusion_state_dict[k][:, :, :, :],
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1)], 1)
        elif nbr_frames == 13:
          state_dict[k] = np.concatenate([np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, 0, :, :], 1),
                                          fusion_state_dict[k][:, :, :, :],
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1),
                                          np.expand_dims(fusion_state_dict[k][:, -1, :, :], 1)], 1)
        state_dict[k] = torch.Tensor(state_dict[k])
        print('k.shape: ', state_dict[k].shape)
  # """
  loaded_state_dict = state_dict.copy()
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')

  FREEZE_LAYERS = False
  if FREEZE_LAYERS:
    for name, param in model.named_parameters():
      if name in loaded_state_dict and not 'Fusion' in name:
      # if name in loaded_state_dict and not 'Fusion' in name:
      # if name in loaded_state_dict and not 'reg' in name and not 'seg' in name and ('kps' in name or 'pre' in name):
        # print('Freeze: ', name)
        param.requires_grad = False
        param.freeze = True
      else:
        print('Not freezing: ', name)
        param.freeze = False

  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

