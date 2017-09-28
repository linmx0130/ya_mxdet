#!/usr/bin/python3

import mxnet as mx
import json
import cv2 
import os
import numpy as np

class HICODataset(mx.gluon.data.Dataset):
    """
    Wrapper of HICO Dataset in json file.
    """
    def __init__(self, gtfile, imgpath, transform=None, **kwargs):
        super(HICODataset, self).__init__(**kwargs)
        with open(gtfile) as f:
            self.gt = json.load(f)
        self.imgpath = imgpath
        self.transform = transform
    
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.imgpath, self.gt[idx]['filename']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        hoi = self.gt[idx]['hoi']
        if self.transform is None:
            return img, hoi
        else:
            return self.transform(img, hoi)

    def __len__(self):
        return len(self.gt)

def extractHumanBox(label):
    bboxhuman = []
    for item in label:
        bboxhuman.append(item['bboxhuman'])
    bboxhuman = np.asarray(bboxhuman)

    # convert (x0, y0, w, h) to (x0, y0, x1, y1)
    bboxhuman[:, 2] += bboxhuman[:, 0]
    bboxhuman[:, 3] += bboxhuman[:, 1] 
    return bboxhuman

def imagenetNormalize(img):
    mean = mx.nd.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = mx.nd.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = mx.nd.array(img / 255)
    img = mx.image.color_normalize(img, mean, std)
    return img

def humanDetectorTransform(data, label):
    from IPython import embed; embed()
    return imagenetNormalize(data), extractHumanBox(label)
