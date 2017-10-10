#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

from config import cfg
from VOCDataset import VOCDataset
from faster_rcnn import FasterRCNN
import mxnet as mx
from utils import random_flip, imagenetNormalize, img_resize, random_square_crop, select_class_generator, bbox_inverse_transform, softmax_celoss_with_ignore
from rpn_gt_opr import rpn_gt_opr
from debug_tool import show_anchors
from rpn_proposal import proposal_train

def train_transformation(data, label):
    data, label = random_flip(data, label)
    data = imagenetNormalize(data)
    return data, label

train_dataset = VOCDataset(annotation_dir=cfg.annotation_dir,
                           img_dir=cfg.img_dir,
                           dataset_index=cfg.dataset_index,
                           transform=train_transformation,
                           resize_func=img_resize)

train_datait = mx.gluon.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
ctx = mx.gpu(0)
net = FasterRCNN(len(cfg.anchor_ratios) * len(cfg.anchor_scales), cfg.num_classes)
net.init_params(ctx)
trainer = mx.gluon.trainer.Trainer(net.collect_params(), 
                                    'sgd', 
                                    {'learning_rate': 0.001,
                                     'wd': 0.0005,
                                     'momentum': 0.9})

anchors_count = len(cfg.anchor_ratios) * len(cfg.anchor_scales)

for epoch in range(20):
    for it, (data, label) in enumerate(train_datait):
        data = data.as_in_context(ctx)
        _n, _c, h, w = data.shape
        label = label.as_in_context(ctx).reshape((1, -1, 5))
        with mx.autograd.record():
            rpn_cls, rpn_reg, f = net.rpn(data)
            f_height = f.shape[2]
            f_width = f.shape[3]
            rpn_cls_gt, rpn_reg_gt = rpn_gt_opr(rpn_reg.shape, label, ctx, h, w)
            rpn_bbox_sampled, rcnn_reg_target, rcnn_cls_target = proposal_train(rpn_cls, rpn_reg, label, f.shape, data.shape, ctx)

            # RPN Loss part
            # Reshape and transpose to the shape of gt
            rpn_cls = rpn_cls.reshape((1, -1, 2, f_height, f_width))
            rpn_cls = mx.nd.transpose(rpn_cls, (0, 1, 3, 4, 2))
            rpn_reg = mx.nd.transpose(rpn_reg.reshape((1, -1, 4, f_height, f_width)), (0, 1, 3, 4, 2))
            mask = (rpn_cls_gt==1).reshape((1, anchors_count, f_height, f_width, 1)).broadcast_to((1, anchors_count, f_height, f_width, 4))
            rpn_loss_reg = mx.nd.sum(mx.nd.smooth_l1((rpn_reg - rpn_reg_gt) * mask, 3.0)) / mx.nd.sum(mask)
            rpn_loss_cls = softmax_celoss_with_ignore(rpn_cls.reshape((-1, 2)), rpn_cls_gt.reshape((-1, )), -1)

            # RCNN part 
            # add batch dimension
            rpn_bbox_sampled = mx.nd.concatenate([mx.nd.zeros((rpn_bbox_sampled.shape[0], 1), ctx), rpn_bbox_sampled], axis=1)
            f = mx.nd.ROIPooling(f, rpn_bbox_sampled, (7, 7), 1.0/16) # VGG16 based spatial stride=16
            rcnn_cls, rcnn_reg = net.rcnn(f)
            mask = (rcnn_cls_target > 0).reshape((rcnn_cls_target.shape[0], 1)).broadcast_to((rcnn_cls_target.shape[0], 4*cfg.num_classes))
            rcnn_loss_reg = mx.nd.sum(mx.nd.smooth_l1((rcnn_reg - rcnn_reg_target) * mask, 1.0)) / mx.nd.sum(mask)
            rcnn_loss_cls = mx.nd.softmax_cross_entropy(rcnn_cls, rcnn_cls_target) / rcnn_cls.shape[0]
            
            loss = rpn_loss_cls + rpn_loss_reg + rcnn_loss_cls + rcnn_loss_reg
            from IPython import embed; embed()

        loss.backward()
        print("Epoch {} Iter {}: loss={:.4}, rpn_loss_cls={:.4}, rpn_loss_reg={:.4}, rcnn_loss_cls={:.4}, rcnn_loss_reg={:.4}".format(
                epoch,it, loss.asscalar(), 
                rpn_loss_cls.asscalar(), rpn_loss_reg.asscalar(),
                rcnn_loss_cls.asscalar(), rcnn_loss_reg.asscalar()))
        trainer.step(data.shape[0])
    net.collect_params().save(cfg.model_path_pattern.format(epoch)) 
