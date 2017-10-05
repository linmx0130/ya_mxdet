#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>
from mxnet import nd

def generate_ssd_scales(feature_map_count):
    return [0.2 + (0.9 - 0.2) / (feature_map_count - 1) * i for i in range(feature_map_count)]

class _Config:
    # Dataset config
    annotation_dir='VOC2007Train/Annotations/'
    dataset_index='VOC2007Train/ImageSets/Main/trainval.txt'
    img_dir='VOC2007Train/JPEGImages/'
    resize_short_size = 300

    # Anchors
    feature_map_count = 4
    anchor_scales = generate_ssd_scales(feature_map_count)
    anchor_ratios = nd.array([1, 2, 3, 1.0/2, 1.0/3])

    # Ground truth assignment
    iou_thresh = 0.7

cfg = _Config()