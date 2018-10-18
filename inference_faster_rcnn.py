#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

import argparse
from faster_rcnn.config import cfg
from VOCDataset import VOCDataset
from faster_rcnn.faster_rcnn import FasterRCNN
import mxnet as mx
from faster_rcnn.utils import imagenetNormalize, img_resize, bbox_inverse_transform, bbox_clip
from faster_rcnn.vis_tool import show_detection_result
from faster_rcnn.rpn_proposal import proposal_test
from faster_rcnn.nms import nms
from tqdm import tqdm 
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Faster RCNN")
    parser.add_argument('model_file', metavar='model_file', type=str)
    parser.add_argument('--feature_name', default='vgg0_conv12_fwd_output', type=str, help='feature to be extracted')
    parser.add_argument('--output_file', metavar='output_file', type=str, default="inference_output.json")
    return parser.parse_args()


def test_transformation(data, label):
    data = imagenetNormalize(data)
    return data, label

test_dataset = VOCDataset(annotation_dir=cfg.test_annotation_dir,
                           img_dir=cfg.test_img_dir,
                           dataset_index=cfg.test_dataset_index,
                           transform=test_transformation,
                           resize_func=None)

# test_datait = mx.gluon.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
ctx = mx.gpu(0)
args = parse_args()
print("Load model: {}".format(args.model_file))

net = FasterRCNN(len(cfg.anchor_ratios) * len(cfg.anchor_scales), cfg.num_classes, feature_name=args.feature_name)
net.init_params(ctx)
net.collect_params().load(args.model_file, ctx)
det_result = []
prograss_bar = tqdm(total=len(test_dataset))

for it, data_id in enumerate(range(len(test_dataset))):
    data, label = test_dataset[data_id]
    data = data.asnumpy()
    data = data.transpose(1, 2, 0)
    data, scale = img_resize(data)
    data = data.transpose(2, 0, 1)
    data_id = test_dataset.dataset_index[data_id]
    data = mx.nd.array(data, ctx=ctx)
    _c, h, w = data.shape
    data = data.reshape(1, _c, h, w)
    # label = label.as_in_context(ctx)
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
        rcnn_bbox_pred[:, i*4:(i+1)*4] = bbox_clip(bbox_inverse_transform(rpn_bbox_pred, rcnn_reg[:, i*4:(i+1)*4]), h, w)
    rcnn_cls = mx.nd.softmax(rcnn_cls)

    # NMS by class 
    keep_boxes = []
    for cls_id in range(1, len(test_dataset.voc_class_name)):
        cur_scores = rcnn_cls[:, cls_id].asnumpy()
        bboxes_pick = rcnn_bbox_pred[:, cls_id * 4: (cls_id+1)*4].asnumpy()
        cur_scores, bboxes_pick = nms(cur_scores, bboxes_pick, cfg.rcnn_nms_thresh)
        for i in range(len(cur_scores)):
            if cur_scores[i] >= cfg.rcnn_score_thresh:
                bbox = bboxes_pick[i] / scale
                bbox_x, bbox_y , bbox_w, bbox_h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
                keep_boxes.append({'image_id':  data_id,
                                   'category_id': cls_id, 
                                   'bbox': [bbox_x, bbox_y, bbox_w, bbox_h],
                                   'score': float(cur_scores[i])})
    det_result.extend(keep_boxes)
    prograss_bar.update()
    # show_detection_ddresult(data, label, rcnn_bbox_pred, rcnn_cls, test_dataset.voc_class_na
prograss_bar.close()
with open(args.output_file, 'w') as f:
    json.dump(det_result, f)
