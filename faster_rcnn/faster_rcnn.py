#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

import mxnet as mx
from .config import cfg
from .rpn import RPNBlock

def _set_dense_weights(lv: mx.gluon.nn.Dense, rv: mx.gluon.nn.Dense):
    lv.weight.set_data(rv.weight.data())
    lv.bias.set_data(rv.bias.data())


class RCNNBlock(mx.gluon.Block):
    def __init__(self, num_classes, **kwargs):
        super(RCNNBlock, self).__init__(**kwargs)
        self.fc6 = mx.gluon.nn.Dense(units=4096, activation='relu')
        self.fc7 = mx.gluon.nn.Dense(in_units=4096, units=4096, activation='relu')
        self.cls_fc = mx.gluon.nn.Dense(in_units=4096, units=num_classes, activation=None)
        self.reg_fc = mx.gluon.nn.Dense(in_units=4096, units=num_classes*4, activation=None)
    
    def forward(self, f, **kwargs):
        f = self.fc6(f)
        f = self.fc7(f)
        cls_output = self.cls_fc(f)
        reg_output = self.reg_fc(f)
        return cls_output, reg_output
    
    def init_by_vgg(self, ctx):
        self.collect_params().initialize(mx.init.Normal(), ctx=ctx)
        vgg16 = mx.gluon.model_zoo.vision.vgg16(pretrained=True)
        # _set_dense_weights(self.fc6, vgg16.features[31])
        # _set_dense_weights(self.fc7, vgg16.features[33])


class FasterRCNN(mx.gluon.Block):
    def __init__(self, num_anchors, num_classes, **kwargs):
        super(FasterRCNN, self).__init__()
        self.rpn = RPNBlock(num_anchors, feature_name=kwargs["feature_name"])
        self.rcnn = RCNNBlock(num_classes)
        
    def forward(self, x, **kwargs):
        raise NotImplementedError
    
    def init_params(self, ctx):
        self.rpn.init_params(ctx)
        self.rcnn.init_by_vgg(ctx)
