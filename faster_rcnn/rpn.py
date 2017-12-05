#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

"""
RPN: Region Proposal Network
"""

import mxnet as mx

def setConvWeights(lv: mx.gluon.nn.Conv2D, rv: mx.gluon.nn.Conv2D):
    lv.weight.set_data(rv.weight.data())
    lv.bias.set_data(rv.bias.data())

class DetectorHead(mx.gluon.Block):
    def __init__(self, num_anchors, **kwargs):
        super(DetectorHead, self).__init__(**kwargs)
        self.conv1 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), padding=(1,1), activation='relu', weight_initializer=mx.init.Normal(0.01))
        self.conv_cls = mx.gluon.nn.Conv2D(channels=2*num_anchors, kernel_size=(1, 1),padding=(0, 0), weight_initializer=mx.init.Normal(0.01))
        self.conv_reg = mx.gluon.nn.Conv2D(channels=4*num_anchors, kernel_size=(1, 1), padding=(0, 0), weight_initializer=mx.init.Normal(0.01))
    
    def forward(self, feature, *args):
        f = self.conv1(feature)
        f_cls = self.conv_cls(f)
        f_reg = self.conv_reg(f)
        return f_cls, f_reg

    def init_params(self, ctx):
        self.collect_params().initialize(ctx=ctx)


class RPNBlock(mx.gluon.Block):
    def __init__(self, num_anchors, pretrained_model=mx.gluon.model_zoo.vision.vgg16, feature_name='vgg0_conv12_fwd_output', **kwargs):
        super(RPNBlock, self).__init__(**kwargs)
        self.feature_extractor = None
        self.feature_model = pretrained_model
        self.feature_name = feature_name
        self.head = DetectorHead(num_anchors)
    
    def forward(self, data, *args):
        f = self.feature_exactor(data)
        f = f[0]
        f_cls, f_reg = self.head(f)
        return f_cls, f_reg, f
    
    def init_params(self, ctx):
        # get feature exactor
        feature_model = self.feature_model(pretrained=True, ctx=ctx)
        input_var = mx.sym.var('data')
        out_var = feature_model(input_var)
        internals = out_var.get_internals()
        feature_list = internals.list_outputs()
        # make sure the feature user want exists
        assert self.feature_name in feature_list
        feature_requested = internals[self.feature_name]
        self.feature_exactor = mx.gluon.SymbolBlock(feature_requested, input_var, params=feature_model.collect_params())
        self.head.init_params(ctx)
