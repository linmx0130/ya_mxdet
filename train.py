#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

from config import cfg
from VOCDataset import VOCDataset
from rpn import RPNFeatureExtractor, DetectorHead
import mxnet as mx
from utils import random_flip, imagenetNormalize, img_resize, random_square_crop
from rpn_gt_opr import rpn_gt_opr

def train_transformation(data, label):
    data, label = random_flip(data, label)
    #data, label = random_square_crop(data, label)
    data = imagenetNormalize(data)
    return data, label

train_dataset = VOCDataset(annotation_dir=cfg.annotation_dir,
                           img_dir=cfg.img_dir,
                           dataset_index=cfg.dataset_index,
                           transform=train_transformation,
                           resize_func=img_resize)
train_datait = mx.gluon.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
ctx = mx.gpu(0)
feature_extractor = RPNFeatureExtractor()
feature_extractor.init_by_vgg(ctx)
rpn_head = DetectorHead(len(cfg.anchor_ratios) * len(cfg.anchor_scales))
rpn_head.init_params(ctx)
cls_loss_func = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=2)
trainer_head = mx.gluon.trainer.Trainer(rpn_head.collect_params(), 
                                    'sgd', 
                                    {'learning_rate': 0.01,
                                     'wd': 0.0001,
                                     'momentum': 0.9})

trainer_feature = mx.gluon.trainer.Trainer(feature_extractor.collect_params(), 
                                    'sgd', 
                                    {'learning_rate': 0.01,
                                     'wd': 0.0001,
                                     'momentum': 0.9})

for it, (data, label) in enumerate(train_datait):
    data = data.as_in_context(ctx)
    _n, _c, h, w = data.shape
    label = label.as_in_context(ctx).reshape((-1, 5))
    background_bndbox = mx.nd.array([[0, 0, 1, 1, 0]], ctx=ctx)
    label = mx.nd.concatenate([background_bndbox, label], axis=0).reshape((1, -1, 5))
    with mx.autograd.record():
        f = feature_extractor(data)
        rpn_cls, rpn_reg = rpn_head(f)
        rpn_cls_gt, rpn_reg_gt = rpn_gt_opr(rpn_reg.shape, label, ctx, h, w)
        f_height = f.shape[2]
        f_width = f.shape[3]
        # Reshape and transpose to the shape of gt
        rpn_cls = rpn_cls.reshape((1, -1, 2, f_height, f_width))
        rpn_reg = mx.nd.transpose(rpn_reg.reshape((1, -1, 4, f_height, f_width)), (0, 1, 3, 4, 2))
        loss_cls = cls_loss_func(rpn_cls, rpn_cls_gt)
        anchors_count = rpn_cls.shape[1]
        mask = rpn_cls_gt.reshape((1, anchors_count, f_height, f_width, 1)).broadcast_to((1, anchors_count, f_height, f_width, 4))
        loss_reg = mx.nd.sum(mx.nd.smooth_l1((rpn_reg - rpn_reg_gt) * mask, 3.0)) / (mx.nd.sum(rpn_cls_gt) + 1) # avoid all zeros
        loss = loss_cls + loss_reg
    loss.backward()
    print("Iteration {}: loss={:.4}, loss_cls={:.4}, loss_reg={:.4}".format(it, loss.asscalar(), loss_cls.asscalar(), loss_reg.asscalar()))
    trainer_head.step(data.shape[0])
    trainer_feature.step(data.shape[0])
