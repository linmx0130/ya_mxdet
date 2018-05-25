#!/usr/bin/python3
# Copyright 2017, Mengxiao Lin <linmx0130@gmail.com>

import os
import cv2
import numpy as np
import mxnet as mx
from .xmlParser import parseFile


class VOCDataset(mx.gluon.data.Dataset):
    """
    Wrapper of HICO Dataset in json file.
    """
    voc_class_name = ['__background__', 'person', 'bird', 'cat', 'cow', 'dog', 
                      'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 
                      'bus', 'car', 'motorbike', 'train', 'bottle', 
                      'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    def __init__(self, annotation_dir: str, img_dir: str, dataset_index:str, transform=None, resize_func=None, **kwargs):
        """
        Args:
            annotation_dir: a string describing the path of annotation XML files.
            img_dir: a string describing the path of JPEG images.
            dataset_index: filename of a file containing the IDs of all images used to constructing this dataset
        """
        super(VOCDataset, self).__init__(**kwargs)
        with open(dataset_index) as f:
            self.dataset_index = [t.strip() for t in f.readlines()]
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.class_to_id = {}
        self.resize_func = resize_func
        for i, class_name in enumerate(self.voc_class_name):
            self.class_to_id[class_name] = i
    
    def __getitem__(self, idx):
        idx = self.dataset_index[idx]
        img_path = os.path.join(self.img_dir, idx+'.jpg')
        img = cv2.imread(img_path)
        if self.resize_func is not None:
            img, scale = self.resize_func(img)
        else:
            scale = 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        anno_path = os.path.join(self.annotation_dir, idx+'.xml')
        gt = self.convert_gt_into_array(parseFile(anno_path))
        gt[:, 0:4] *= scale

        if self.transform is None:
            return img, gt
        else:
            return self.transform(img, gt)

    def __len__(self):
        return len(self.dataset_index)

    def convert_gt_into_array(self, gt, filter_difficult=True):
        """
            Args:
                gt: the ground truth return by parseFile
                filter_difficult: filter out difficult cases or not
            Returns:
                A n * 5 array in the format of [x1, y1, x2, y2, c]
        """
        ret = []
        for obj in gt['objects']:
            if filter_difficult and (obj['difficult']==1):
                continue
            new_array = list(obj['bndbox'])
            new_array.append(self.class_to_id[obj['name']])
            ret.append(new_array)
        return np.asarray(ret, dtype=np.float32)


def show_images(data, label, ds:VOCDataset):
    img = np.transpose(data, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for item in label:
        cv2.rectangle(img, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), color=(255, 0, 0), thickness=2)
        cv2.putText(img, ds.voc_class_name[int(item[4])], (int(item[0]), int(item[3])),0, 0.5,(0, 255, 0))
    cv2.imshow("Img", img)
    cv2.waitKey(0)
