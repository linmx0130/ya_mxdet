#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

import mxnet as mx
import numpy as np
from anchor_generator import generate_anchors, map_anchors
from utils import bbox_inverse_transform, bbox_overlaps, bbox_transform
from nms import nms
from config import cfg


def proposal_train(rpn_cls, rpn_reg, gt, feature_shape, image_shape, ctx):

    # Stop gradient to stop gradient recording
    rpn_cls = mx.nd.stop_gradient(rpn_cls)
    rpn_reg = mx.nd.stop_gradient(rpn_reg)

    # Get basic information of the feature and the image
    _n, _c, f_height, f_width = feature_shape
    _in, _ic, img_height, img_width = image_shape
    rpn_cls = rpn_cls.reshape((1, -1, 2, f_height, f_width))
    anchors_count = rpn_cls.shape[1]
    
    # Recover RPN prediction with anchors
    ref_anchors = generate_anchors(base_size=16, ratios=cfg.anchor_ratios, scales=cfg.anchor_scales)
    anchors = map_anchors(ref_anchors, rpn_reg.shape, img_height, img_width, ctx)
    anchors = anchors.reshape((1, -1, 4, f_height, f_width))
    anchors = mx.nd.transpose(anchors, (0, 3, 4, 1, 2))
    rpn_anchor_scores = mx.nd.softmax(mx.nd.transpose(rpn_cls, (0, 3, 4, 1, 2)), axis=4)[:,:,:,:,1]
    rpn_reg = mx.nd.transpose(rpn_reg.reshape((1, -1, 4, f_height, f_width)), (0, 3, 4, 1, 2))
    rpn_bbox_pred = bbox_inverse_transform(anchors.reshape((-1, 4)), rpn_reg.reshape((-1, 4))).reshape((1, f_height, f_width, anchors_count, 4))

    # Use NMS to filter out too many boxes
    rpn_bbox_pred = rpn_bbox_pred.asnumpy().reshape((-1, 4))
    rpn_anchor_scores = rpn_anchor_scores.asnumpy().reshape((-1, ))
    rpn_anchor_scores, rpn_bbox_pred = nms(rpn_anchor_scores, rpn_bbox_pred, cfg.rpn_nms_thresh, use_top_n=cfg.bbox_count_before_nms)
    rpn_bbox_pred = mx.nd.array(rpn_bbox_pred, ctx)
    del rpn_anchor_scores

    # append ground truth
    rpn_bbox_pred = mx.nd.concatenate([rpn_bbox_pred, gt[0][:,:4]])

    # assign label for rpn_bbox_pred
    overlaps = bbox_overlaps(rpn_bbox_pred, gt[0][:, :4].reshape((-1, 4)))
    gt_assignment = mx.nd.argmax(overlaps, axis=1).asnumpy()
    max_overlaps = mx.nd.max(overlaps, axis=1).asnumpy()
    gt_labels = gt[0][:, 4].reshape((-1,)).asnumpy()
    gt_bboxes = gt[0][:, :4].reshape((-1, 4)).asnumpy()
    cls_labels = gt_labels[gt_assignment]
    rpn_bbox_pred_np = rpn_bbox_pred.asnumpy()
    reg_target = gt_bboxes[gt_assignment, :]
    cls_labels = cls_labels * (max_overlaps >= cfg.rcnn_fg_thresh)

    # sample positive and negative ROIs
    fg_inds = np.where(max_overlaps >= cfg.rcnn_fg_thresh)[0]
    bg_inds = np.where((max_overlaps >= cfg.rcnn_bg_lo_thresh) * (max_overlaps < cfg.rcnn_fg_thresh))[0]
    fg_nums = int(cfg.rcnn_train_sample_size * cfg.rcnn_train_fg_fraction)
    bg_nums = cfg.rcnn_train_sample_size - fg_nums
    if (len(fg_inds) > fg_nums):
        fg_inds = np.random.choice(fg_inds, size=fg_nums, replace=False)
    if (len(bg_inds) > bg_nums):
        bg_inds = np.random.choice(bg_inds, size=bg_nums, replace=False)
    cls_labels = np.concatenate([cls_labels[fg_inds], cls_labels[bg_inds]])
    reg_target = np.concatenate([reg_target[fg_inds], reg_target[bg_inds]])
    rpn_bbox_pred_np = np.concatenate([rpn_bbox_pred_np[fg_inds], rpn_bbox_pred_np[bg_inds]])
    cls_labels = mx.nd.array(cls_labels, ctx)
    reg_target = mx.nd.array(reg_target, ctx)
    rpn_bbox_pred = mx.nd.array(rpn_bbox_pred_np, ctx)
    reg_target = bbox_transform(rpn_bbox_pred, reg_target)
    
    return rpn_bbox_pred, reg_target, cls_labels