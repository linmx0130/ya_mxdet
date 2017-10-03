#!/usr/bin/python3 
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

from config import cfg
from VOCDataset import VOCDataset
from ssd import SSDFeatures
import mxnet as mx
from utils import random_flip, imagenetNormalize, img_resize

def train_transformation(data, label):
    data, label = random_flip(data, label)
    data = imagenetNormalize(data)
    return data, label

train_dataset = VOCDataset(cfg.annotation_dir, cfg.img_dir, cfg.dataset_index, transform=train_transformation, resize_func=img_resize)
train_datait = mx.gluon.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
ctx = mx.gpu(0)
ssd_features = SSDFeatures()
ssd_features.init_by_vgg(ctx)

for it, (data, label) in enumerate(train_datait):
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)
    features = ssd_features(data)
    #TODO
    from IPython import embed; embed()
    if it >= 5:
        break
