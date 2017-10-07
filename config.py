#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>
from mxnet import nd

def generate_ssd_scales(feature_map_count):
    return [0.2 + (0.9 - 0.2) / (feature_map_count - 1) * i for i in range(feature_map_count)]

class _Config:
    # Dataset config
    annotation_dir='VOC2007Train/Annotations/'
    #dataset_index='VOC2007Train/ImageSets/Main/person_train.txt'
    dataset_index='person.txt'
    img_dir='VOC2007Train/JPEGImages/'
    resize_short_size = 600

    # Model saved
    model_path_pattern='./model_dump/epoch-{}.gluonmodel'

    # Anchors
    anchor_ratios = nd.array([0.5, 1, 2])
    anchor_scales = 2**nd.arange(3, 6)

    # Ground truth assignment
    iou_positive_thresh = 0.5
    iou_negative_thresh = 0.3

    # RPN Test
    show_top_bbox_count = 10

cfg = _Config()
