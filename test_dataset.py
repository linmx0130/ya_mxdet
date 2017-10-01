from VOCDataset import *

ds = VOCDataset(
        annotation_dir='VOC2007Train/Annotations/', 
        dataset_index='VOC2007Train/ImageSets/Main/trainval.txt', 
        img_dir='VOC2007Train/JPEGImages/')

show_images(*ds[5], ds)
