#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

"""
RPN: Region Proposal Network
"""

import mxnet as mx

def setConvWeights(lv: mx.gluon.nn.Conv2D, rv: mx.gluon.nn.Conv2D):
    lv.weight.set_data(rv.weight.data())
    lv.bias.set_data(rv.bias.data())

class RPNFeatureExtractor(mx.gluon.Block):
    def __init__(self, **kwargs):
        super(RPNFeatureExtractor, self).__init__(**kwargs)
        self.conv1_1 = mx.gluon.nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=3, activation='relu')
        self.conv1_2 = mx.gluon.nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=64, activation='relu')
        self.pool1 = mx.gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2), padding=0, ceil_mode=False)

        self.conv2_1 = mx.gluon.nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=64, activation='relu')
        self.conv2_2 = mx.gluon.nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=128, activation='relu')
        self.pool2 = mx.gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2), padding=0, ceil_mode=False)

        self.conv3_1 = mx.gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=128, activation='relu')
        self.conv3_2 = mx.gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=256, activation='relu')
        self.conv3_3 = mx.gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=256, activation='relu')
        self.pool3 = mx.gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2), padding=0, ceil_mode=False)

        self.conv4_1 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=256, activation='relu')
        self.conv4_2 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=512, activation='relu')
        self.conv4_3 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=512, activation='relu')
        self.pool4 = mx.gluon.nn.MaxPool2D(pool_size=(2,2), strides=(2,2), padding=0, ceil_mode=False)
        
        self.conv5_1 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=512, activation='relu')
        self.conv5_2 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=512, activation='relu')
        self.conv5_3 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), in_channels=512, activation='relu')

    def init_by_vgg(self, ctx):
        self.collect_params().initialize(mx.init.Normal(), ctx=ctx)
        vgg16 = mx.gluon.model_zoo.vision.vgg16(pretrained=True)
        setConvWeights(self.conv1_1, vgg16.features[0])
        setConvWeights(self.conv1_2, vgg16.features[2])
        setConvWeights(self.conv2_1, vgg16.features[5])
        setConvWeights(self.conv2_2, vgg16.features[7])
        setConvWeights(self.conv3_1, vgg16.features[10])
        setConvWeights(self.conv3_2, vgg16.features[12])
        setConvWeights(self.conv3_3, vgg16.features[14])
        setConvWeights(self.conv4_1, vgg16.features[17])
        setConvWeights(self.conv4_2, vgg16.features[19])
        setConvWeights(self.conv4_3, vgg16.features[21])
        setConvWeights(self.conv5_1, vgg16.features[24])
        setConvWeights(self.conv5_2, vgg16.features[26])
        setConvWeights(self.conv5_3, vgg16.features[28])

    def forward(self, x, *args):
        features = []
        f = self.conv1_1(x)
        f = self.conv1_2(f)
        f = self.pool1(f)
        
        f = self.conv2_1(f)
        f = self.conv2_2(f)
        f = self.pool2(f)

        f = self.conv3_1(f)
        f = self.conv3_2(f)
        f = self.conv3_3(f)
        f = self.pool3(f)

        f = self.conv4_1(f)
        f = self.conv4_2(f)
        f = self.conv4_3(f)
        f = self.pool4(f)
        f = self.conv5_1(f)
        f = self.conv5_2(f)
        f = self.conv5_3(f)
        return f


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
    def __init__(self, num_anchors, **kwargs):
        super(RPNBlock, self).__init__(**kwargs)
        self.feature_exactor = RPNFeatureExtractor()
        self.head = DetectorHead(num_anchors)
    
    def forward(self, data, *args):
        f = self.feature_exactor(data)
        f_cls, f_reg = self.head(f)
        return f_cls, f_reg, f
    
    def init_params(self, ctx):
        self.feature_exactor.init_by_vgg(ctx)
        self.head.init_params(ctx)
