#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>
import numpy as np

def _get_area(bbox):
    width = np.maximum(bbox[:, 2] - bbox[:, 0], 0)
    height = np.maximum(bbox[:, 3] - bbox[:, 1], 0)
    return width * height


def _bbox_overlaps(bbox, target):
    # inter
    x0 = np.maximum(bbox[:,0], target[0])
    y0 = np.maximum(bbox[:,1], target[1])
    x1 = np.minimum(bbox[:,2], target[2])
    y1 = np.minimum(bbox[:,3], target[3])
    inter = _get_area(np.concatenate([x0.reshape((-1, 1)), 
                                     y0.reshape((-1, 1)), 
                                     x1.reshape((-1, 1)), 
                                     y1.reshape((-1, 1))], axis=1))
    outer = _get_area(bbox) + _get_area(target.reshape(1, 4)) - inter
    iou = inter / outer
    return iou


def nms(bbox_scores, bbox_pred, iou_thresh):
    bbox_inds = np.argsort(-bbox_scores)
    bbox_pred = bbox_pred[bbox_inds]
    bbox_scores = bbox_scores[bbox_inds]
    keep_mask = np.ones(bbox_inds.shape)
    for idx in range(keep_mask.shape[0]):
        if keep_mask[idx] == 1:
            iou = _bbox_overlaps(bbox_pred, bbox_pred[idx])
            current_mask = (iou < iou_thresh)
            current_mask[:idx+1] = 1
            keep_mask *= current_mask
    keep_inds = np.where(keep_mask)
    return bbox_scores[keep_inds], bbox_pred[keep_inds]

def test_nms():
    bbox = np.array([
        [10, 10, 50, 50],
        [20, 20, 60, 60],
        [30, 30, 70, 70]
    ], dtype=np.float)
    scores = np.array([0.3, 0.4, 0.5])
    print(nms(scores, bbox, 0.3))

if __name__=='__main__':
    test_nms()
