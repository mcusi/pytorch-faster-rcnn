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
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        bboxcolor = ['red','green','blue'][i % 3]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=bboxcolor, linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fn)

def convert_detections(class_name, dets, t, f, thresh=0.5):
    
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return []

    elements = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        """Convert bounding boxes to times and frequencies"""
        onset = t[int(bbox[0])]
        duration = t[int(bbox[2])] - t[int(bbox[0])]
        centreBin = f[int((bbox[3] - bbox[1])/2. + bbox[1])]
        logf0 = np.log(centreBin) if class_name == 'tone' else -1

        """Add element to list"""
        elements.append({"onset":np.float64(onset), "duration":np.float64(duration),"voice":class_name,"logf0":np.float64(logf0),"score":np.float64(score)})

    return elements

def compute_conversions(params, timepx):
    y = np.zeros(params['sr'])
    sxx, cfs = gtg.gammatonegram(y, sr=params['sr'],twin=params['twin'],thop=params['thop'],
        N=params["nFilts"],fmin=50,fmax=int(params['sr']/2.),width=params["filtWidth"])
    # samples in image * 1 second per np.shape(sxx)[1] samples
    t = np.arange(0,timepx)*(1./np.shape(sxx)[1])
    return t, cfs


def basademo(nets, image_name, dataname, exptnames):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/bASA/JPEGImages', image_name)
    im_file = '/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/'+dataname+'/demos/' + image_name + '.jpg'
    print(im_file)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores={};boxes={};
    for exptname in exptnames:
        scores[exptname], boxes[exptname] = im_detect(nets[exptname], im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes[exptname].shape[0]))

    # Find frequency and time conversions
    with open('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + dataname + '/' +dataname+'dream.json') as infile:
        params = json.load(infile)
    t, f = compute_conversions(params, np.shape(im)[1])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.5
    elements = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes={};cls_scores={};_dets={};
        for exptname in exptnames:
            cls_boxes[exptname] = boxes[exptname][:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores[exptname] = scores[exptname][:, cls_ind]
            _dets[exptname] = np.hstack((cls_boxes[exptname],
                              cls_scores[exptname][:, np.newaxis])).astype(np.float32)
        dets=np.vstack((_dets['vgg16_ar4'],_dets['vgg16_ar8'],_dets['vgg16_ar16']))
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        fn = '/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/'+dataname+'/demos/' + image_name + '_' + cls + '_combined.jpg'
        vis_detections(im, cls, dets, fn, thresh=CONF_THRESH) 
        els = convert_detections(cls, dets, t, f, thresh=CONF_THRESH)
        for e in els:
            elements.append(e)

    jfn = '/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/'+dataname+'/demos/' + image_name + '_combined.json'
    with open(jfn, 'w') as outfile:
        outfile.write(json.dumps(elements))

if __name__ == '__main__':

    dataname = os.environ.get('dataname','bASAGP')
    exptnames = ['vgg16_ar4','vgg16_ar8','vgg16_ar16']
    nets={}
    for exptname in exptnames:
        folder = ''
        iteration = '50000'

        cfg_from_file('experiments/cfgs/vgg16.yml') # Add usual config options for network type 
        cfg_from_file('output/vgg16/'+dataname+'_train/default/' + folder + exptname + '.yml') # add config options for that particular trained network
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals

        # model path
        saved_model = 'output/vgg16/'+dataname+'_train/default/' + folder + exptname + '_iter_' + iteration + '.pth'

        # load network
        nets[exptname] = vgg16()
        nets[exptname].create_architecture(3,tag='default', anchor_scales=cfg.ANCHOR_SCALES, anchor_ratios=cfg.ANCHOR_RATIOS)
        nets[exptname].load_state_dict(torch.load(saved_model))
        nets[exptname].eval()
        nets[exptname].cuda()
        print('Loaded network {:s}'.format(saved_model))

    #im_names = ['%06d' % i for i in range(50002,50022)]
    im_names = ['continuity','Track01fast','Track01slow','Track15higherBetweens','Track15lowerBetweens','Track16capture','Track16nocapture','Track17X','Track32fast','Track32slow','1i','1ii','1iii','1iv','1v','2i','2ii','2iii','2iv','2v','3ai','3aii','3aiii','3aiv','3av','3bi','3bii','3biii','3biv','3bv','4i','4ii','4iii','4iv','4v'];
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/bASA/JPEGImages/{}'.format(im_name))
        basademo(nets, im_name, dataname, exptnames)

    plt.show()
