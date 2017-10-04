#!/usr/bin/python3 
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

from config import cfg
from VOCDataset import VOCDataset
from ssd import SSDFeatures, DetectorHead
import mxnet as mx
from utils import random_flip, imagenetNormalize, img_resize, random_square_crop
from anchor_generator import ssd_generate_anchors, map_anchors

def train_transformation(data, label):
    data, label = random_flip(data, label)
    data, label = random_square_crop(data, label)
    data = imagenetNormalize(data)
    return data, label

train_dataset = VOCDataset(cfg.annotation_dir, cfg.img_dir, cfg.dataset_index, transform=train_transformation, resize_func=img_resize)
train_datait = mx.gluon.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
ctx = mx.gpu(0)
ssd_features = SSDFeatures()
ssd_features.init_by_vgg(ctx)
ssd_heads = [DetectorHead(21, 5), DetectorHead(21, 5), DetectorHead(21, 5), DetectorHead(21, 5)]
for h in ssd_heads:
    h.init_params(ctx)

for it, (data, label) in enumerate(train_datait):
    data = data.as_in_context(ctx)
    _n, _c, h, w = data.shape
    label = label.as_in_context(ctx)
    with mx.autograd.record():
        features = ssd_features(data)
        for fi, (f, dh) in enumerate(zip(features, ssd_heads)):
            cls_head, reg_head = dh(f)
            ref_anchors = ssd_generate_anchors(cfg.anchor_scales[fi] * h, ratios=cfg.anchor_ratios)
            anchors = map_anchors(ref_anchors, reg_head.shape, h, ctx)
            from IPython import embed; embed()
            break
    #TODO
    if it >= 5:
        break
