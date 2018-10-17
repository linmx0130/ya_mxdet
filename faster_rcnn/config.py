#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>
from mxnet import nd

def generate_ssd_scales(feature_map_count):
    return [0.2 + (0.9 - 0.2) / (feature_map_count - 1) * i for i in range(feature_map_count)]

class _Config:
    # Dataset config
    annotation_dir='VOCdevkit/VOC2007/Annotations/'
    dataset_index='VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    #dataset_index='person.txt'
    img_dir='VOCdevkit/VOC2007/JPEGImages/'
    resize_short_size = 600
    num_classes = 21 # added background

    # Model saved
    model_path_pattern='./model_dump/epoch-{}.gluonmodel'

    # Anchors
    anchor_ratios = nd.array([0.5, 1, 2])
    anchor_scales = 2**nd.arange(3, 6)

    # Ground truth assignment
    iou_positive_thresh = 0.7
    iou_negative_thresh = 0.3
    rpn_fg_fraction = 0.5
    rpn_batchsize = 256
    rcnn_fg_thresh = 0.5
    rcnn_bg_lo_thresh = 0.1
    rcnn_train_sample_size = 256
    rcnn_test_sample_size = 256
    rcnn_train_fg_fraction = 0.5
    rcnn_nms_thresh = 0.3
    rcnn_score_thresh = 0.1

    # RPN Test
    bbox_count_before_nms = 2000
    rpn_nms_thresh = 0.7
    test_annotation_dir='VOCdevkit/VOC2007/Annotations/'
    test_dataset_index='VOCdevkit/VOC2007/ImageSets/Main/test.txt'
    test_img_dir='VOCdevkit/VOC2007/JPEGImages/'

class _ConfigVOC2012:
    # Dataset config
    annotation_dir='VOCdevkit/VOC2012/Annotations'
    dataset_index='VOCdevkit/VOC2012/ImageSets/Main/train.txt'
    #dataset_index='person.txt'
    img_dir='VOCdevkit/VOC2012/JPEGImages/'
    resize_short_size = 600
    num_classes = 21 # added background

    # Model saved
    model_path_pattern='./model_dump/epoch-{}.gluonmodel'

    # Anchors
    anchor_ratios = nd.array([0.5, 1, 2])
    anchor_scales = 2**nd.arange(3, 6)

    # Ground truth assignment
    iou_positive_thresh = 0.7
    iou_negative_thresh = 0.3
    rpn_fg_fraction = 0.5
    rpn_batchsize = 256
    rcnn_fg_thresh = 0.5
    rcnn_bg_lo_thresh = 0.1
    rcnn_train_sample_size = 256
    rcnn_test_sample_size = 256
    rcnn_train_fg_fraction = 0.5
    rcnn_nms_thresh = 0.3
    rcnn_score_thresh = 0.5

    # RPN Test
    bbox_count_before_nms = 2000
    rpn_nms_thresh = 0.7
    test_annotation_dir='VOCdevkit/VOC2012/Annotations/'
    test_dataset_index='VOCdevkit/VOC2012/ImageSets/Main/val.txt'
    test_img_dir='VOCdevkit/VOC2012/JPEGImages/'


# Choose a config
# config for voc 2012
# cfg = _ConfigVOC2012()
# config for voc 2007
cfg = _Config()
