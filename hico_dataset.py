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
    def __init__(self, gtfile, imgpath, **kwargs):
        super(HICODataset, self).__init__(**kwargs)
        with open(gtfile) as f:
            self.gt = json.load(f)
        self.imgpath = imgpath
    
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.imgpath, self.gt[idx]['filename']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        hoi = self.gt[idx]['hoi']
        return img, hoi

    def __len__(self):
        return len(self.gt)
