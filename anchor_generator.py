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


def ssd_generate_anchors(scale, ratios=nd.array([0.5, 1, 2]), append_scale=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, scale, scale) window.
    
    append_scale is used to generate an extra anchor whose scale is 
    \sqrt{scale*append_scale}. Set append_scale=None to disenable this 
    extra anchor.
    """
    base_anchor = nd.array([1, 1, scale, scale]) - 1
    anchors = _ratio_enum(base_anchor, ratios)
    if append_scale is not None:
        ns = int(scale * append_scale)
        append_anchor = nd.round(nd.sqrt(nd.array([[1, 1, ns, ns]])))-1
        anchors = nd.concatenate([anchors, append_anchor], axis=0)
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

def map_anchors(ref_anchors, target_shape, scale, ctx):
    ref_anchors = ref_anchors.as_in_context(ctx)
    ref_anchors = ref_anchors.reshape((1, -1, 1, 1))
    ref_anchors = ref_anchors.broadcast_to(target_shape)
    _n, _c, h, w = ref_anchors.shape
    ref_x = nd.arange(h).as_in_context(ctx).reshape((h, 1)) / h 
    ref_x = ref_x * scale
    ref_x = ref_x.broadcast_to((h, w))
    ref_y = nd.arange(w).as_in_context(ctx).reshape((1, w)) / w
    ref_y = ref_y * scale
    ref_y = ref_y.broadcast_to((h, w))
    for anchor_i in range(_c//4):
        ref_anchors[0, anchor_i * 4] += ref_x
        ref_anchors[0, anchor_i * 4 + 1] += ref_y
        ref_anchors[0, anchor_i * 4 + 2] += ref_x
        ref_anchors[0, anchor_i * 4 + 3] += ref_y
    return ref_anchors
