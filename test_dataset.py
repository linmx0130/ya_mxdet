from VOCDataset import *
from utils import img_resize

ds = VOCDataset(
        annotation_dir='VOC2007Train/Annotations/', 
        dataset_index='VOC2007Train/ImageSets/Main/trainval.txt', 
        img_dir='VOC2007Train/JPEGImages/', transform=None, resize_func=img_resize)

show_images(*ds[5], ds)
