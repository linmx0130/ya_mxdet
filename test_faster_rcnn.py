#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

import argparse
from config import cfg
from VOCDataset import VOCDataset
from faster_rcnn import FasterRCNN
import mxnet as mx
from utils import imagenetNormalize, img_resize, bbox_inverse_transform
from vis_tool import show_detection_result
from rpn_proposal import proposal_test

def parse_args():
    parser = argparse.ArgumentParser(description="Test Faster RCNN")
    parser.add_argument('model_file', metavar='model_file', type=str)
    return parser.parse_args()


def test_transformation(data, label):
    data = imagenetNormalize(data)
    return data, label

test_dataset = VOCDataset(annotation_dir=cfg.test_annotation_dir,
                           img_dir=cfg.test_img_dir,
                           dataset_index=cfg.test_dataset_index,
                           transform=test_transformation,
                           resize_func=img_resize)

test_datait = mx.gluon.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
ctx = mx.gpu(0)
net = FasterRCNN(len(cfg.anchor_ratios) * len(cfg.anchor_scales), cfg.num_classes)
net.init_params(ctx)
args = parse_args()
print("Load model: {}".format(args.model_file))
net.collect_params().load(args.model_file, ctx)

for it, (data, label) in enumerate(test_datait):
    data = data.as_in_context(ctx)
    _n, _c, h, w = data.shape
    label = label.as_in_context(ctx)
    rpn_cls, rpn_reg, f = net.rpn(data)
    f_height = f.shape[2]
    f_width = f.shape[3]
    rpn_bbox_pred = proposal_test(rpn_cls, rpn_reg, f.shape, data.shape, ctx)

    # RCNN part 
    # add batch dimension
    rpn_bbox_pred_attach_batchid = mx.nd.concatenate([mx.nd.zeros((rpn_bbox_pred.shape[0], 1), ctx), rpn_bbox_pred], axis=1)
    f = mx.nd.ROIPooling(f, rpn_bbox_pred_attach_batchid, (7, 7), 1.0/16) # VGG16 based spatial stride=16
    rcnn_cls, rcnn_reg = net.rcnn(f)
    rcnn_bbox_pred = mx.nd.zeros(rcnn_reg.shape)
    for i in range(len(test_dataset.voc_class_name)):
        rcnn_bbox_pred[:, i*4:(i+1)*4] = bbox_inverse_transform(rpn_bbox_pred, rcnn_reg[:, i*4:(i+1)*4])
    rcnn_cls = mx.nd.softmax(rcnn_cls)
    show_detection_result(data, label, rcnn_bbox_pred, rcnn_cls, test_dataset.voc_class_name)
