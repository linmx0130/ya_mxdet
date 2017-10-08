#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

import mxnet as mx
from anchor_generator import generate_anchors, map_anchors
from utils import bbox_overlaps, bbox_transform
from config import cfg

def rpn_gt_opr(reg_shape, label, ctx, img_h, img_w, return_anchors=False):
    _fn, _fc, feature_height, feature_width = reg_shape
    label_count = label.shape[1]
    anchor_counts = _fc // 4
    # only batch size=1 is supported
    ref_anchors = generate_anchors(base_size=16, ratios=cfg.anchor_ratios, scales=cfg.anchor_scales)
    anchors = map_anchors(ref_anchors, reg_shape, img_h, img_w, ctx)
    anchors = anchors.reshape((1, -1, 4, feature_height, feature_width))
    # anchors is [1, H, W, A, 4]
    anchors = mx.nd.transpose(anchors, (0, 3, 4, 1, 2))
    # anchors is [(1, H, W, A), 4]
    anchors = anchors.reshape((-1, 4))

    # So until now, anchors are N * 4, the order is [(H, W, A), 4]
    overlaps = bbox_overlaps(anchors, label[:, :, :4].reshape((-1, 4)))
    # Any gt has its own bbox, gt_assignment in [(1, H, W, A), #{gt}]
    gt_assignment = mx.nd.argmax(overlaps, axis=0)
    max_overlaps = mx.nd.max(overlaps, axis=1)
    bbox_cls_gt = (max_overlaps >= cfg.iou_positive_thresh)
    bbox_cls_gt = bbox_cls_gt - ((max_overlaps > cfg.iou_negative_thresh) * (max_overlaps < cfg.iou_positive_thresh))
    bbox_assignment = mx.nd.argmax(overlaps, axis=1)
    for i in gt_assignment.asnumpy():
        bbox_cls_gt[int(i)] = 1
    bbox_cls_gt = bbox_cls_gt.reshape((1, feature_height, feature_width, anchor_counts))
    bbox_cls_gt = mx.nd.transpose(bbox_cls_gt, (0, 3, 1, 2))
    bbox_assignment = bbox_assignment.reshape((1, feature_height, feature_width, anchor_counts))
    bbox_assignment = mx.nd.transpose(bbox_assignment, (0, 3, 1, 2))
    
    # reshape acnhors to [1, A, H, W, 4]
    anchors = anchors.reshape((1, feature_height, feature_width, anchor_counts, 4))
    anchors = mx.nd.transpose(anchors, (0, 3, 1, 2, 4))

    reg_label_extend = label[:,:,:4].reshape(
                (1, 1, 1, 1, label_count, 4)).broadcast_to(
                (1, anchor_counts, feature_height, feature_width, label_count, 4))
    # Due to the mother-fucked MXNet slice operator, the operation to get regression target 
    # is picked as following.
    bbox_reg_gt = mx.nd.concatenate(
        [mx.nd.pick(reg_label_extend[0][:,:,:,:,0], bbox_assignment[0]).reshape((1, anchor_counts, feature_height, feature_width)),
         mx.nd.pick(reg_label_extend[0][:,:,:,:,1], bbox_assignment[0]).reshape((1, anchor_counts, feature_height, feature_width)),
         mx.nd.pick(reg_label_extend[0][:,:,:,:,2], bbox_assignment[0]).reshape((1, anchor_counts, feature_height, feature_width)),
         mx.nd.pick(reg_label_extend[0][:,:,:,:,3], bbox_assignment[0]).reshape((1, anchor_counts, feature_height, feature_width))], axis=0)
    bbox_reg_gt = mx.nd.transpose(bbox_reg_gt, (1, 2, 3, 0)).reshape((1, anchor_counts, feature_height, feature_width, 4))
    bbox_reg_gt = bbox_transform(anchors.reshape((-1, 4)), bbox_reg_gt.reshape((-1, 4))).reshape((1, anchor_counts, feature_height, feature_width, 4))

    if not return_anchors:
        return bbox_cls_gt, bbox_reg_gt
    else:
        return bbox_cls_gt, bbox_reg_gt, anchors
