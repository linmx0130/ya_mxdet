#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>
from mxnet import nd

def generate_ssd_scales(feature_map_count):
    return [0.2 + (0.9 - 0.2) / (feature_map_count - 1) * i for i in range(feature_map_count)]

class _Config:
    # Dataset config
    annotation_dir='VOC2007Train/Annotations/'
    dataset_index='VOC2007Train/ImageSets/Main/trainval.txt'
    #dataset_index='person.txt'
    img_dir='VOC2007Train/JPEGImages/'
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
    test_annotation_dir='VOC2007Train/Annotations/'
    test_dataset_index='VOC2007Train/ImageSets/Main/test.txt'
    test_img_dir='VOC2007Train/JPEGImages/'

class _ConfigVOC2012:
    # Dataset config
    annotation_dir='VOC2012/Annotations'
    dataset_index='VOC2012/ImageSets/Main/train.txt'
    #dataset_index='person.txt'
    img_dir='VOC2012/JPEGImages/'
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
    test_annotation_dir='VOC2012/Annotations/'
    test_dataset_index='VOC2012/ImageSets/Main/test.txt'
    test_img_dir='VOC2012/JPEGImages/'


class _ConfigTUPUFace:
    # Dataset config
    input_json_files = [
            "/home/zhanglinghan/face-detect-yolov2-mxnet/dataset/data_train.json",
            # "/world/data-c9/dl-data/5673875659a518eb6c2af055/5af12891bf0e3f59cd610488/15257540018230.7161486853115395_list.json"
            ]
    resize_short_size = 600
    num_classes = 2  # added background

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
    rcnn_score_thresh = 0.8

    # RPN Test
    bbox_count_before_nms = 2000
    rpn_nms_thresh = 0.7
    test_dataset_json_lst = ["/world/data-gpu-57/AR/zmp_working_dir/face_detect/wh9_16_shuffle_test.json"]


# Choose a config
# config for voc 2012
# cfg = _ConfigVOC2012()
# config for voc 2007
# cfg = _Config()
cfg = _ConfigTUPUFace()
