#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

class _Config:
    # Dataset config
    annotation_dir='VOC2007Train/Annotations/'
    dataset_index='VOC2007Train/ImageSets/Main/trainval.txt'
    img_dir='VOC2007Train/JPEGImages/'
    resize_short_size = 300


cfg = _Config()