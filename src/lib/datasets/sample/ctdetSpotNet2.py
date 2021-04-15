from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_ellipse_gaussian
from utils.image import draw_dense_reg
import math

class CTDetDatasetSpotNet2(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)

    # """
    HM_ATT = False
    PYFLOW = True
    ONE_CLASS_ONLY = True

    if not HM_ATT:
      if PYFLOW:
        if 'uav' in self.opt.dataset:
          seg_path = os.path.join('/store/datasets/UAV/bgsubs',
                                  os.path.dirname(file_name).split('/')[-1],
                                  os.path.basename(file_name).replace('jpg', 'png'))
        else:
          seg_path = os.path.join('/store/datasets/OlderUA-Detrac/pyflow-bgsubs',
                                  os.path.dirname(file_name).split('/')[-1],
                                  os.path.basename(file_name).replace('jpg', 'png'))
    # """

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)

    channel_counter = len(self.coco.getCatIds())
    if not HM_ATT:
      bboxes = {}
      for ann in anns:
        if str(ann['category_id']) in bboxes:
          bboxes[str(ann['category_id'])].append([int(ann['bbox'][0]),
                        int(ann['bbox'][1]),
                       int(ann['bbox'][0] + ann['bbox'][2]),
                        int(ann['bbox'][1] + ann['bbox'][3])])
        else:
          bboxes[str(ann['category_id'])] = [[int(ann['bbox'][0]),
                                       int(ann['bbox'][1]),
                                       int(ann['bbox'][0] + ann['bbox'][2]),
                                       int(ann['bbox'][1] + ann['bbox'][3])]]
    # for ann in anns:
     #    bboxes.append([int(ann['bbox'][0]),
      #                  int(ann['bbox'][1]),
       #                 int(ann['bbox'][0] + ann['bbox'][2]),
        #                int(ann['bbox'][1] + ann['bbox'][3])])
    num_objs = min(len(anns), self.max_objs)
    # print(img_path)
    img = cv2.imread(img_path)
    if not HM_ATT:
      if PYFLOW:
        seg_img = cv2.imread(seg_path, 0)  # hughes


      if not PYFLOW:
        if 'coco' in img_path:
          if 'val' in img_path:
            seg_dir = '/store/datasets/coco/annotations/stuff_val2017_pixelmaps'
          else:
            seg_dir = '/store/datasets/coco/annotations/stuff_train2017_pixelmaps'
          stuff_img = cv2.imread(os.path.join(seg_dir, file_name.replace('.jpg', '.png')))
          seg_img = np.zeros([img.shape[0], img.shape[1]])
          seg_img[stuff_img[:, :, 0] == 0] += 1
          seg_img[stuff_img[:, :, 1] == 214] += 1
          seg_img[stuff_img[:, :, 2] == 255] += 1
          seg_img[seg_img == 3] = 255
          seg_img[seg_img < 255] = 0
        else:
          if not ONE_CLASS_ONLY:
            seg_img = np.zeros([channel_counter, img.shape[0], img.shape[1]])
            for label in range(1, channel_counter+1):
              if str(label) in bboxes:
                for bbox in bboxes[str(label)]:
                  seg_img[label-1, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
          else:
            seg_img = np.zeros([img.shape[0], img.shape[1]])
            for label in range(1, channel_counter + 1):
              if str(label) in bboxes:
                for bbox in bboxes[str(label)]:
                  seg_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    # seg_img = np.zeros([img.shape[0], img.shape[1]])
    # for bbox in bboxes:
    #   seg_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", os.path.basename(file_name.replace('.jpg', '_rgb.jpg'))), seg_img_rgb)
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", os.path.basename(file_name)), seg_img)
    # exit()
    # print("IMG_SHAPE: ", img.shape, " MEAN: ", np.mean(img))
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", os.path.basename(file_name)), img)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        if not HM_ATT:
          if ONE_CLASS_ONLY:
            seg_img = seg_img[:, ::-1]
          else:
            seg_img = seg_img[:, ::-1, :]
        # print('img.shape: ', img.shape)
        # print('seg_img.shape: ', seg_img.shape)
        # exit()
        # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)), img)
        # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)).replace('.jpg', '_seg.jpg'), seg_img)
        c[0] =  width - c[0] - 1


    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    # print('TRANS INPUT SHAPE: ', trans_input.shape)
    inp = cv2.warpAffine(img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)

    if not HM_ATT:
      if ONE_CLASS_ONLY:
        seg_inp = cv2.warpAffine(seg_img, trans_input,
                                (input_w, input_h),
                                flags=cv2.INTER_LINEAR)
      else:
        seg_inp = np.zeros((seg_img.shape[0], input_w, input_h))
        for channel in range(seg_img.shape[0]):
          seg_inp[channel, :, :] = cv2.warpAffine(seg_img[channel, :, :], trans_input,
                                   (input_w, input_h),
                                   flags=cv2.INTER_LINEAR)

    inp = (inp.astype(np.float32) / 255.)
    if not HM_ATT:
      seg_inp = (seg_inp.astype(np.float32) / 255.) # hughes
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)


    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)), inp)
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)).replace('.jpg', '_seg.jpg'), seg_inp)

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    # print('MEAN: ', np.average(seg_inp))

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    if self.opt.elliptical_gt:
      draw_gaussian = draw_ellipse_gaussian
    else:
      draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        if self.opt.elliptical_gt:
          radius_x = radius if h > w else int(radius * (w / h))
          radius_y = radius if w >= h else int(radius * (h / w))
          # radius_x = radius if w > h else int(radius / (w/h))
          # radius_y = radius if h >= w else int(radius / (h/w))
          draw_gaussian(hm[cls_id], ct_int, radius_x, radius_y)
        else:
          draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    if not HM_ATT:
      if ONE_CLASS_ONLY:
        # scale_percent = 25  # percent of original size
        # width = int(seg_inp.shape[1] * scale_percent / 100)
        # height = int(seg_inp.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # seg_inp = cv2.resize(seg_inp, dim, interpolation=cv2.INTER_AREA)
        seg_inp = np.expand_dims(seg_inp, 0)
    # print(seg_inp.shape)
    # print(hm.shape)
    # print(inp.shape)
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'seg': seg_inp, 'ct_att': hm}  # 'seg': seg_inp}  # 'seg': np.expand_dims(seg_inp, 0)}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta

    # ret['seg'] = ret['hm']
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)), (inp.transpose(1, 2, 0)* 255).astype(np.uint8))
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/COCO-format/img_tests/", "inp_" + os.path.basename(file_name)).replace('.jpg', '_seg.jpg'), (seg_inp.squeeze(0) * 255).astype(np.uint8))
    # cv2.imwrite(os.path.join("/store/datasets/UA-Detrac/test_images/hm/", "hm_" + os.path.basename(file_name)), (hm.squeeze(0) * 255).astype(np.uint8))

    return ret
