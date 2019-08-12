#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : demo.py
#   Author      : chr
#   Created date: 2019-07-23 13:42:37
#   Description :
#
#================================================================

from mmdet.apis import inference_detector, show_result
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

config_file = 'configs/retinanet_r101_fpn_1x.py'
checkpoint_file = 'checkpoint/RetinaNet/retinanet_r101_fpn_1x_20181129-f016f384.pth'

cfg = mmcv.Config.fromfile(config_file)
cfg.model.pretrained = None

model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, checkpoint_file)

img = './demo/000001.jpg'
#img = mmcv.imread(img_path)
result = inference_detector(model, img, cfg)
show_result(img, result)
