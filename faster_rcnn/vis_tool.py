#!/usr/bin/python3 
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>
import mxnet as mx 
import numpy as np
import cv2
from .nms import nms
from .config import cfg

def show_anchors(data, label, anchors, anchors_chosen, count=None):
    """
    show image, ground truth and anchors in the same window
    """
    data = data[0].as_in_context(mx.cpu(0))
    data[0] = data[0] * 0.229 + 0.485
    data[1] = data[1] * 0.224 + 0.456
    data[2] = data[2] * 0.225 + 0.406
    label = label[0].asnumpy()
    img = data.asnumpy()
    img = np.array(np.round(img * 255), dtype=np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for item in label:
        cv2.rectangle(img, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), color=(255, 0, 0), thickness=2)
        #cv2.putText(img, ds.voc_class_name[int(item[4])], (int(item[0]), int(item[3])),0, 0.5,(0, 255, 0))
    anchors = anchors[0].asnumpy()
    anchors_chosen = anchors_chosen[0].asnumpy()
    anchors = anchors.reshape((-1, 4))
    anchors_chosen = anchors_chosen.reshape((-1,))
    for anchor_id, c in enumerate(anchors_chosen):
        if c==1:
            anc = anchors[anchor_id]
            cv2.rectangle(img, (int(anc[0]), int(anc[1])), (int(anc[2]), int(anc[3])), color=(0,0, 255), thickness=1)
            print((int(anc[0]), int(anc[1])), (int(anc[2]), int(anc[3])))
        if count is not None:
            count = count - 1
            if count == 0:
                break
    cv2.imshow("Img", img)
    cv2.waitKey(0)


def show_detection_result(data, label, bboxes, cls_scores, class_name_list):
    data = data[0].as_in_context(mx.cpu(0))
    data[0] = data[0] * 0.229 + 0.485
    data[1] = data[1] * 0.224 + 0.456
    data[2] = data[2] * 0.225 + 0.406
    label = label[0].asnumpy()
    img = data.asnumpy()
    img = np.array(np.round(img * 255), dtype=np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bboxes = bboxes.asnumpy()
    cls_scores = cls_scores.asnumpy()

    # Show ground truth
    for item in label:
        cv2.rectangle(img, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), color=(255, 0, 0), thickness=2)
        cv2.putText(img, class_name_list[int(item[4])], (int(item[0]), int(item[3])),0, 0.5,(0, 255, 0))

    # NMS by class
    for cls_id in range(1, len(class_name_list)):
        cur_scores = cls_scores[:, cls_id]
        bboxes_pick = bboxes[:, cls_id * 4: (cls_id+1)*4]
        cur_scores, bboxes_pick = nms(cur_scores, bboxes_pick, cfg.rcnn_nms_thresh)
        for i in range(len(cur_scores)):
            if cur_scores[i] >= cfg.rcnn_score_thresh:
                bbox = bboxes_pick[i]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=1)
                cv2.putText(img, "{}: {:.4}".format(class_name_list[cls_id], cur_scores[i]), (int(bbox[0]), int(bbox[3])),0, 0.5,(255, 255, 0))
    cv2.imshow("Img", img)
    cv2.waitKey(0)
