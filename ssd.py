#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

"""
SSD: Single Shot Detector
"""

import mxnet as mx

def setConvWeights(lv: mx.gluon.nn.Conv2D, rv: mx.gluon.nn.Conv2D):
    lv.weight.set_data(rv.weight.data())
    lv.bias.set_data(rv.bias.data())

class SSD(mx.gluon.Block):
    """
    SSD: Single Shot Detector block
    """
    def __init__(self, **kwargs):
        super(SSD, self).__init__(**kwargs)
        self.conv1_1 = mx.gluon.nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv1_2 = mx.gluon.nn.Conv2D(channels=64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.pool1 = mx.gluon.nn.MaxPool2D(pool_size=(2,2), stride=(2,2), padding=0, ceil_mode=False)

        self.conv2_1 = mx.gluon.nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv2_2 = mx.gluon.nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.pool2 = mx.gluon.nn.MaxPool2D(pool_size=(2,2), stride=(2,2), padding=0, ceil_mode=False)

        self.conv3_1 = mx.gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv3_2 = mx.gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv3_3 = mx.gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.pool3 = mx.gluon.nn.MaxPool2D(pool_size=(2,2), stride=(2,2), padding=0, ceil_mode=False)

        self.conv4_1 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv4_2 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv4_3 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.pool4 = mx.gluon.nn.MaxPool2D(pool_size=(2,2), stride=(2,2), padding=0, ceil_mode=False)
        
        self.conv5_1 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv5_2 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv5_3 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')

        self.conv6 = mx.gluon.nn.Conv2D(channels=1024, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation='relu')
        self.conv7 = mx.gluon.nn.Conv2D(channels=1024, kernel_size=(1, 1), activation='relu')
        self.conv8_1 = mx.gluon.nn.Conv2D(channels=256, kernel_size=(1, 1), activation='relu')
        self.conv8_2 = mx.gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(2, 2), activation='relu')
        self.conv9_1 = mx.gluon.nn.Conv2D(channels=128, kernel_size=(1, 1), activation='relu')
        self.conv9_2 = mx.gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(2, 2), activation='relu')
        self.conv10_1 = mx.gluon.nn.Conv2D(channels=128, kernel_size=(1, 1), activation='relu')
        self.conv10_2 = mx.gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')

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
        features.append(f) # features of conv4_3
        f = self.pool4(f)
        f = self.conv5_1(f)
        f = self.conv5_2(f)
        f = self.conv5_3(f)
        f = self.conv6(f)
        f = self.conv7(f)
        features.append(f)
        f = self.conv8_1(f)
        f = self.conv8_2(f)
        features.append(f)
        f = self.conv9_1(f)
        f = self.conv9_2(f)
        features.append(f)
        f = self.conv10_1(f)
        f = self.conv10_2(f)
        features.append(f)

        return features
        # do something
