#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

import mxnet as mx
from mxnet import nd
from .config import cfg
import numpy as np 
import cv2

# Model utils
def bbox_transform(anchor, bbox):
    w = anchor[:, 2] - anchor[:, 0]
    h = anchor[:, 3] - anchor[:, 1]
    cx = (anchor[:, 0] + anchor[:, 2]) / 2.0
    cy = (anchor[:, 1] + anchor[:, 3]) / 2.0

    
    g_w = bbox[:, 2] - bbox[:, 0]
    g_h = bbox[:, 3] - bbox[:, 1]
    g_cx = (bbox[:, 0] + bbox[:, 2]) / 2.0
    g_cy = (bbox[:, 1] + bbox[:, 3]) / 2.0
    
    g_w = mx.ndarray.log(g_w / w)
    g_h = mx.ndarray.log(g_h / h)
    g_cx = (g_cx - cx) / w 
    g_cy = (g_cy - cy) / h
    return mx.ndarray.concatenate([
                g_w.reshape((-1, 1)), 
                g_h.reshape((-1, 1)), 
                g_cx.reshape((-1, 1)), 
                g_cy.reshape((-1, 1))], axis=1)


def bbox_inverse_transform(anchor, bbox):
    w = anchor[:, 2] - anchor[:, 0]
    h = anchor[:, 3] - anchor[:, 1]
    cx = (anchor[:, 0] + anchor[:, 2]) / 2.0
    cy = (anchor[:, 1] + anchor[:, 3]) / 2.0

    g_w = mx.ndarray.exp(bbox[:, 0]) * w
    g_h = mx.ndarray.exp(bbox[:, 1]) * h
    g_cx = bbox[:, 2] * w + cx
    g_cy = bbox[:, 3] * h + cy

    g_x1 = g_cx - g_w / 2
    g_y1 = g_cy - g_h / 2
    g_x2 = g_cx + g_w / 2
    g_y2 = g_cy + g_h / 2
    return mx.ndarray.concatenate([
                        g_x1.reshape((-1, 1)),
                        g_y1.reshape((-1, 1)),
                        g_x2.reshape((-1, 1)),
                        g_y2.reshape((-1, 1))], axis=1)


def _get_area(bbox:mx.nd.NDArray):
    zeros = mx.nd.zeros_like(bbox[:, 0])
    width = mx.nd.max(nd.stack(bbox[:, 2] - bbox[:, 0], zeros), axis=0)
    height = mx.nd.max(nd.stack(bbox[:, 3] - bbox[:, 1], zeros), axis=0)
    return width * height


def bbox_overlaps(anchors:mx.nd.NDArray, gt:mx.nd.NDArray):
    """
    Get IoU of the anchors and ground truth bounding boxes.
    The shape of anchors and gt should be (N, 4) and (M, 4)
    So the shape of return value is (N, M)
    """
    ret = []
    for i in range(gt.shape[0]):
        cgt = gt[i].reshape((1, 4)).broadcast_to(anchors.shape)
        # inter
        x0 = nd.max(nd.stack(anchors[:,0], cgt[:,0]), axis=0)
        y0 = nd.max(nd.stack(anchors[:,1], cgt[:,1]), axis=0)
        x1 = nd.min(nd.stack(anchors[:,2], cgt[:,2]), axis=0)
        y1 = nd.min(nd.stack(anchors[:,3], cgt[:,3]), axis=0)
        
        inter = _get_area(nd.concatenate([x0.reshape((-1, 1)), 
                                         y0.reshape((-1, 1)), 
                                         x1.reshape((-1, 1)), 
                                         y1.reshape((-1, 1))], axis=1))
        outer = _get_area(anchors) + _get_area(cgt) - inter
        iou = inter / outer
        ret.append(iou.reshape((-1, 1)))
    ret=nd.concatenate(ret, axis=1)
    return ret


#
# Data argumentation and normalization
#
def random_flip(data, label):
    if np.random.uniform() > 0.0:
        c, h, w = data.shape
        data = np.flip(data, axis=2)
        x0 = label[:, 0].copy()
        x1 = label[:, 2].copy()
        label[:, 0] = w - x1
        label[:, 2] = w - x0
    return data, label


def imagenetNormalize(img):
    mean = mx.nd.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = mx.nd.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = mx.nd.array(img / 255)
    img = mx.image.color_normalize(img, mean, std)
    return img


def img_resize(img):
    h, w, c = img.shape
    if h > w:
        # align width to cfg.short_size
        scale = cfg.resize_short_size / w
        nw = int(w * scale)
        nh = int(h * scale)
        img = cv2.resize(img, (nw, nh))
    else:
        # align height to cfg.short_size
        scale = cfg.resize_short_size / h
        nw = int(w * scale)
        nh = int(h * scale)
        img = cv2.resize(img, (nw, nh))
    return img, scale


def random_square_crop(img, label):
    c, h, w = img.shape
    if h>w:
        x = np.random.randint(0, h-w)
        img = img[:, x: x+w, :]
        label[:, 1] -= x
        label[:, 3] -= x
    else:
        x = np.random.randint(0, w - h)
        img = img[:, :, x: x+h]
        label[:, 0] -= x
        label[:, 2] -= x
    return img, label


def select_class_generator(class_id):
    def select_class(img, label):
        ret_label = []
        for item in label:
            if item[4] == class_id:
                ret_label.append(item)
        return img, np.stack(ret_label)
    return select_class


def softmax_celoss_with_ignore(F, label, ignore_label):
    output = mx.nd.log_softmax(F)
    label_matrix = mx.nd.zeros(output.shape, ctx=output.context)
    for i in range(label_matrix.shape[1]):
        label_matrix[:, i] = (label==i)
    ignore_unit = (label == ignore_label)
    loss = -mx.nd.sum(output * label_matrix, axis=1)
    return mx.nd.sum(loss) / (output.shape[0] - mx.nd.sum(ignore_unit))

