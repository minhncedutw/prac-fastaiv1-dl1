from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *

import cv2
import matplotlib.pyplot as plt

'''
Download and Explore data
'''
#########Download Data##########
path = untar_data(URLs.CAMVID)
path_lbl = path/'labels'
path_img = path/'images'
get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'

fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)

img = open_image(fnames[0])
img.show(figsize=(5,5))

mask = open_mask(get_y_fn(fnames[0]))
mask.show(figsize=(5,5), alpha=1)

'''
Declare parameters
'''
# free = gpu_mem_get_free_no_cache()
bs = 4
size = (360, 480)
codes = np.loadtxt(path/'codes.txt', dtype=str);
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

'''
Define Data & Learner
'''
def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')
       .label_from_func(get_y_fn, classes=codes))
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))
# data.show_batch(2, figsize=(10,7))

metrics=acc_camvid
wd=1e-2
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

'''
Train model
'''
#==========Gross training==========
lr_find(learn)
learn.recorder.plot()
lr=3e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
learn.save('stage-11')

# learn.load('stage-1')
# learn.show_results(rows=3, figsize=(8,9))

learn.unfreeze()
lrs = slice(lr/400,lr/4)
learn.fit_one_cycle(10, lrs, pct_start=0.8)
learn.save('stage-12')

#==========More Fine-tuning==========
learn.load('stage-12')
lr_find(learn)
learn.recorder.plot()
lr=1e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)
learn.save('stage-21')

learn.load('stage-21')
learn.unfreeze()
lrs = slice(1e-6,lr/10)
learn.fit_one_cycle(10, lrs)
learn.save('stage-22')

learn.load('stage-2-big')
learn.show_results(rows=3, figsize=(10,10))