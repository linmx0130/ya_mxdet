from VOCDataset import *
from utils import img_resize, random_square_crop, random_flip

def transformation(data, label):
    data, label = random_flip(data, label)
    data, label = random_square_crop(data, label)
    return data, label

ds = VOCDataset(
        annotation_dir='VOC2007Train/Annotations/', 
        dataset_index='VOC2007Train/ImageSets/Main/trainval.txt', 
        img_dir='VOC2007Train/JPEGImages/', transform=transformation, resize_func=img_resize)

for i in [5, 10, 15, 20, 25]:
        show_images(*ds[i], ds)
