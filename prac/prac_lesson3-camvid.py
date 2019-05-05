import numpy as np
import cv2
import matplotlib.pyplot as plt

from fastai.vision import *
# from fastai.callbacks.hooks import import *
# from fastai.utils.mem import *

"""
Download & Explore data
"""
path = untar_data(url=URLs.CAMVID)
path_lbl = path/'labels/'
path_img = path/'images/'
get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'

# img_names = get_image_files(c=path_img)
# lbl_names = get_image_files(c=path_lbl)
#
# img = open_image(img_names[0])
# img.show(figsize=(5, 5))
#
# mask = open_mask(get_y_fn(img_names[0]))
# mask.show(figsize=(5, 5), alpha=1)

codes = np.loadtxt(fname=path/'codes.txt', dtype=str)
name2id = {v:k for k, v in enumerate(codes)}
void_code = name2id['Void']

src = (SegmentationItemList.from_folder(path=path_img)
       .split_by_fname_file(fname='../valid.txt')
       .label_from_func(get_y_fn, classes=codes))

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

"""
Train small images
"""
#==========create learner==========
bs = 4
size = (360, 480)
metrics = acc_camvid
wd = 1e-2

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

learn = unet_learner(data=data, arch=models.resnet34, pretrained=True, metrics=metrics, wd=wd)

#==========train==========
lr_find(learn=learn)
learn.recorder.plot()
lr = 3e-3
learn.fit_one_cycle(cyc_len=10, max_lr=slice(lr), pct_start=0.9)
learn.save('small-1')

learn.load('small-1')
lrs = np.array([lr/100, lr/10])
learn.fit_one_cycle(cyc_len=10, max_lr=lrs, pct_start=0.8)
learn.save('small-2')

"""
Train big images
"""
bs = 1
size = (720, 960)

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

learn = unet_learner(data=data, arch=models.resnet34, metrics=acc_camvid, wd=wd)

#==========train==========
lr_find(learn=learn)
learn.recorder.plot()
lr = 1e-3
learn.fit_one_cycle(cyc_len=10, max_lr=lr, pct_start=0.8)

learn.unfreeze()
lrs = slice(start=1e-6, stop=lr/10)
learn.fit_one_cycle(cyc_len=10, max_lr=lrs, pct_start=0.3)
