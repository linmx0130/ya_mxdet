#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>
# Original code is from Faster R-CNN Python implementation.
#
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
#

import mxnet.ndarray as nd

def generate_anchors(base_size=16, ratios=nd.array([0.5, 1, 2]), scales=2**nd.arange(3,6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    This implementation matches the original Faster-RCNN RPN generate_anchors().
    But all calculations are on mxnet.ndarray.NDArray.

    Refer to 
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/generate_anchors.py
    """

    base_anchor = nd.array([1, 1, base_size, base_size]) -1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = nd.concatenate([_scale_enum(ratio_anchors[i, :], scales)
                                 for i in range(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor:nd.NDArray):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws.reshape((-1, 1))
    hs = hs.reshape((-1, 1))
    anchors = nd.concatenate(
                        [x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)], axis=1)
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = nd.round(nd.sqrt(size_ratios))
    hs = nd.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
