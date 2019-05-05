#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Description
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "CONG-MINH NGUYEN"
__copyright__ = "Copyright (C) 2019, pointnet_pytorch_itri"
__credits__ = ["CONG-MINH NGUYEN"]
__license__ = "GPL"
__version__ = "1.0.1"
__date__ = "2019-05-05"
__maintainer__ = "CONG-MINH NGUYEN"
__email__ = "minhnc.edu.tw@gmail.com"
__status__ = "Development"  # ["Prototype", "Development", or "Production"]
# Project Style: https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6
# Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting

#==============================================================================
# Imported Modules
#==============================================================================
import argparse
import os.path
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # The GPU id to use, usually either "0" or "1"

from fastai.vision import *
from fastai.utils.mem import *

#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================

#==============================================================================
# Main function
#==============================================================================
def _main_(args):
    print('Hello World! This is {:s}'.format(args.desc))

    # config_path = args.conf
    # with open(config_path) as config_buffer:    
    #     config = json.loads(config_buffer.read())
    '''**************************************************************
    I. Preparation
    '''
    path = untar_data(URLs.CAMVID)
    path_img = path/'images/'
    path_lbl = path / 'labels'
    get_y_fn = lambda x: path_lbl / f'{x.stem}_P{x.suffix}'
    codes = np.loadtxt(path/'codes.txt', dtype=str)

    name2id = {v: k for k, v in enumerate(codes)}
    void_code = name2id['Void']

    def acc_camvid(input, target):
        target = target.squeeze(1)
        mask = target != void_code
        return (input.argmax(dim=1)[mask] == target[mask]).float().mean()

    '''**************************************************************
    II. Train small data
    '''
    # 1: Make data
    free = gpu_mem_get_free_no_cache()
    size = (360, 480)
    bs = 4

    src = (SegmentationItemList.from_folder(path=path_img)
           .split_by_fname_file(fname='../valid.txt')
           .label_from_func(get_y_fn, classes=codes))
    data = (src.transform(get_transforms(), size=size, tfm_y=True)
            .databunch(bs=bs)
            .normalize(imagenet_stats))

    # 2: Make model
    metrics = acc_camvid
    wd = 1e-2
    learn = unet_learner(data=data, arch=models.resnet34, pretrained=True, metrics=metrics, wd=wd)
    learn.summary()

    # 3: Slight training
    # 3.a: find fair learning rate
    lr_find(learn=learn)
    learn.recorder.plot()

    # 3.b: train big learning rate then save weights
    lr = 3e-3
    learn.fit_one_cycle(cyc_len=10, max_lr=slice(lr), pct_start=0.9)
    learn.save(name='small-1')

    # 4: Finetune training
    learn.load(name='small-1')
    # 4.a: unfreeze learner
    learn.unfreeze()

    # 4.b: train small learning rate then save weights
    lrs = slice(lr/400, lr/4)
    learn.fit_one_cycle(cyc_len=12, max_lr=lrs, pct_start=0.8)
    learn.save(name='small-2')

    '''**************************************************************
    III. Train big data
    '''
    # 1: Make data
    free = gpu_mem_get_free_no_cache()

    # 2: Make model

    # 3: Slight training

    # 4: Finetune training


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='pointnet_pytorch_itri')
    # argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
