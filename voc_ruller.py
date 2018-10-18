#!/usr/bin/python3
# Copyright 2018, Mengxiao Lin <linmx0130@gmail.com>

import argparse
from VOCDataset import VOCDataset
from faster_rcnn.config import cfg
import json
import numpy as np

def get_area(x0, y0, x1, y1):
    width = max(x1 - x0, 0)
    height = max(y1 - y0, 0)
    return width * height

def get_box_iou(bbox, target):
    # inter
    x0 = max(bbox[0], target[0])
    y0 = max(bbox[1], target[1])
    x1 = min(bbox[2], target[2])
    y1 = min(bbox[3], target[3])
    inter = get_area(x0, y0, x1, y1)
    outer =  get_area(*bbox) + get_area(*target) - inter
    iou = inter / (outer + 1e-5)
    return iou


def voc_ap(det_result, gt, iou_thresh, 
           average_recall_points=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5 ,0.6, 0.7, 0.8, 0.9, 1.0]):
    total_bboxes_in_gt = sum([len(t) for t in gt.values()])
    true_positive = 0
    visited_pred = 0
    visited_gt = 0
    ap_values = []
    current_recall_point = 0
    for pred_det in det_result:
        if not pred_det['image_id'] in gt:
            visited_pred += 1
            print("missing image", pred_det['image_id'])
            continue
        gt_for_query = gt[pred_det['image_id']]
        gt_for_query = [t for t in gt_for_query if t['category_id'] == pred_det['category_id'] and (not t['visited'])]
        pred_box = pred_det['bbox']
        pred_box = pred_box[0], pred_box[1], pred_box[2] + pred_box[0], pred_box[1]+pred_box[3]
        iou = [get_box_iou(pred_box, t['bbox']) for t in gt_for_query]
        if (len(iou) > 0):
            gt_choice = np.argmax(iou)
            if (iou[gt_choice] >= iou_thresh):
                true_positive += 1
                visited_gt += 1
                gt_for_query[gt_choice]['visited'] = True
        visited_pred += 1
        if visited_gt / total_bboxes_in_gt >= average_recall_points[current_recall_point]:
            ap_values.append(true_positive / visited_pred)
            current_recall_point += 1 
    while current_recall_point < len(average_recall_points):
        current_recall_point += 1
        ap_values.append(0.0)
    return ap_values

def parse_args():
    parser = argparse.ArgumentParser(description="Measure performance of detection with VOC dataset with VOC07 metrics.")
    parser.add_argument("result_file", type=str)
    parser.add_argument("--hit_iou", type=float, default=0.5)
    return parser.parse_args()

args = parse_args()
test_dataset = VOCDataset(annotation_dir=cfg.test_annotation_dir,
                           img_dir=cfg.test_img_dir,
                           dataset_index=cfg.test_dataset_index)
with open(args.result_file) as f:
    det_result = json.load(f)

det_result = sorted(det_result, key=lambda x:x['score'], reverse=True)

# collect all bboxes in test_dataset
gt_labels = {}
for it, data_id in enumerate(range(len(test_dataset))):
    data, label = test_dataset[data_id]
    data_id = test_dataset.dataset_index[data_id]
    del data
    bboxes = []
    for item in label:
        bboxes.append({'bbox':item[:4], 'category_id':item[4], 'visited': False})
    gt_labels[str(data_id)] = bboxes

ap_values = voc_ap(det_result, gt_labels, args.hit_iou)
mAP = np.mean(ap_values)
print("AP = ", ap_values)
print("mAP = ", mAP)
