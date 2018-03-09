#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg, cfg_from_file, cfg_from_list
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib
matplotlib.use('Agg') ##can't visualize inside terminal
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import json
import scipy.io.wavfile as wf

#use symlink to get to om2
import gammatonegram as gtg

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

CLASSES = ('__background__',
           'noise','tone')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def vis_detections(im, class_name, dets, fn, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    figIM, axIM = plt.subplots(figsize=(12, 12))
    axIM.imshow(im, aspect='equal')
    bbox = axIM.get_window_extent().transformed(figIM.dpi_scale_trans.inverted())
    width, height = bbox.width*figIM.dpi, bbox.height*figIM.dpi    

    filename='Track17X'
    sr, y = wf.read('/om2/user/mcusi/bayesianASA/sounds/' + filename + '.wav')
    renderParams={"twin":0.025,"thop":0.01,"nFilts":64,"filtWidth":0.5,"sr":sr}
    sxx, cfs = gtg.gammatonegram(y,sr=renderParams['sr'],twin=renderParams['twin'],thop=renderParams['thop'],
        N=renderParams["nFilts"],fmin=50,fmax=int(renderParams['sr']/2.),width=renderParams["filtWidth"])
    sxx[sxx == 0] = 1e-80
    sxx = 20.*np.log10(sxx)
    sxx[sxx < -60.] = -60.

    fig, ax = plt.subplots(1,1,figsize=(13, 5))
    plt.pcolormesh(sxx,vmin=-60, vmax=0,cmap='Purples') #Blues for samples
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    figsize = fig.get_size_inches()*fig.dpi # size in pixels
    print(figsize)
    t = np.arange(0,figsize[0])*(1./np.shape(sxx)[1])
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        #convert im boudning boxes to sxx boudning boxes
        bbox0=bbox[0]
        bbox1=figsize[1]*bbox[1]/height
        bbox2=bbox[2]
        bbox3=figsize[1]*bbox[3]/height

        bboxcolor = 'red'
        ax.add_patch(
            plt.Rectangle((bbox0, bbox1),
                          bbox2 - bbox0,
                          bbox3 - bbox1, fill=False,
                          edgecolor=bboxcolor, linewidth=3.5)
            )
        ax.text(bbox0+1, bbox3 + 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='black', alpha=0.5),
                fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    #plt.axis('off')
    #plt.tight_layout()
    plt.savefig(fn,bbox_inches='tight',pad_inches=0)
    plt.close('all')

def basademo(net, image_name, dataname, exptname):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/bASA/JPEGImages', image_name)
    im_file = '/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/'+dataname+'/demos/' + image_name + '.jpg'
    print(im_file)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH=[1,0.4,0.6]
    NMS_THRESH=[1,0.5,0.3]
    elements = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH[cls_ind])
        dets = dets[keep.numpy(), :]
        fn = '/om2/user/mcusi/bayesianASA/cogsci2018/fig/' + dataname + image_name + cls + exptname + '.png'
        vis_detections(im, cls, dets, fn, thresh=CONF_THRESH[cls_ind]) 


if __name__ == '__main__':

    folder = ''
    dataname = 'bASAGP1'#os.environ.get('dataname','bASA')
    exptname = 'vgg16_ar4_anch6_mult4'#os.environ.get('exptname','vgg16_ar4')
    iteration = 80000

    cfg_from_file('experiments/cfgs/vgg16.yml') # Add usual config options for network type 
    cfg_from_file('output/vgg16/'+dataname+'_train/default/' + folder + exptname + '.yml') # add config options for that particular trained network
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    # model path
    saved_model = 'output/vgg16/'+dataname+'_train/default/' + folder + exptname + '_iter_' + str(iteration) + '.pth'

    # load network
    net = vgg16()
    net.create_architecture(3,tag='default', anchor_scales=cfg.ANCHOR_SCALES, anchor_ratios=cfg.ANCHOR_RATIOS)
    net.load_state_dict(torch.load(saved_model))
    net.eval()
    net.cuda()
    print('Loaded network {:s}'.format(saved_model))

    #im_names = ['%06d' % i for i in range(50002,50022)]
    #im_names = ['4ii','continuity','Track01fast','Track01slow','Track15higherBetweens','Track15lowerBetweens','Track16capture','Track16nocapture','Track17X','Track32fast','Track32slow','1i','1ii','1iii','1iv','1v','2i','2ii','2iii','2iv','2v'];
    #im_names = ['3ai','3aii','3aiii','3aiv','3av','3bi','3bii','3biii','3biv','3bv','4i','4ii','4iii','4iv','4v'];
    #im_names=['2vi']
    im_names=['Track17X']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/bASA/JPEGImages/{}'.format(im_name))
        basademo(net, im_name, dataname, exptname)

    plt.show()
