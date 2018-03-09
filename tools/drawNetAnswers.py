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

        bboxcolor = ['red','green','blue','yellow','cyan','magenta'][i % 6]
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
    plt.close('all')

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

    # Find frequency and time conversions
    with open('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + dataname + '/' +dataname+'dream.json') as infile:
        params = json.load(infile)
    t, f = compute_conversions(params, np.shape(im)[1])

    # Visualize detections for each class
    CONF_THRESHS = [0.7,0.75]#[0.1, 0.4, 0.6, 0.8]
    NMS_THRESHS = [0.3,0.35,0.4]#[0.3, 0.5, 0.7]
    for CONF_THRESH in CONF_THRESHS:
        for NMS_THRESH in NMS_THRESHS:
            elements = []
            for cls_ind, cls in enumerate(CLASSES[1:]):
                cls_ind += 1 # because we skipped background
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(torch.from_numpy(dets), NMS_THRESH)
                dets = dets[keep.numpy(), :]
                fn = '/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/'+dataname+'/demos/' + image_name + '_' + cls + '_' + exptname + '_c' + str(CONF_THRESH) + '_n' + str(NMS_THRESH) + 'FINALTONES.jpg'
                vis_detections(im, cls, dets, fn, thresh=CONF_THRESH) 
                els = convert_detections(cls, dets, t, f, thresh=CONF_THRESH)
                for e in els:
                    elements.append(e)

            jfn = '/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/'+dataname+'/demos/' + image_name + '_' + exptname + '_c' + str(CONF_THRESH) + '_n' + str(NMS_THRESH) + '.json'
            with open(jfn, 'w') as outfile:
                outfile.write(json.dumps(elements))

if __name__ == '__main__':

    folder = ''
    dataname = os.environ.get('dataname','bASAGP1')
    exptname = os.environ.get('exptname','vgg16_ar4_anch6_mult4')
    iteration = 200000

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

    #im_names=['1iii', '1ii', '1iv', '1i', '1v', '2iii', '2ii', '2iv', '2i', '2vi', '2v', 'continuitynew', 'df10_dt100', 'df10_dt110', 'df10_dt120', 'df10_dt125', 'df10_dt130', 'df10_dt140', 'df10_dt150', 'df10_dt80', 'df10_dt90', 'df12_dt100', 'df12_dt110', 'df12_dt120', 'df12_dt125', 'df12_dt130', 'df12_dt140', 'df12_dt150', 'df12_dt80', 'df12_dt90', 'df15_dt100', 'df15_dt110', 'df15_dt120', 'df15_dt125', 'df15_dt130', 'df15_dt140', 'df15_dt150', 'df15_dt80', 'df15_dt90', 'df2_dt100', 'df2_dt110', 'df2_dt120', 'df2_dt125', 'df2_dt130', 'df2_dt140', 'df2_dt150', 'df2_dt80', 'df2_dt90', 'df5_dt100', 'df5_dt110', 'df5_dt120', 'df5_dt125', 'df5_dt130', 'df5_dt140', 'df5_dt150', 'df5_dt80', 'df5_dt90', 'df7_dt100', 'df7_dt110', 'df7_dt120', 'df7_dt125', 'df7_dt130', 'df7_dt140', 'df7_dt150', 'df7_dt80', 'df7_dt90', '4ii', 'Track15higherBetweens', 'Track15lowerBetweens', 'Track17X', 'Track32fast', 'Track32slow'];
    #im_names=['Track15higherBetweens', 'Track15lowerBetweens', 'Track17X', 'Track32fast', 'Track32slow'];
    #im_names=['1iii', '1ii', '1iv', '1i', '1v', '2iii', '2ii', '2iv', '2i', '2vi', '2v','4ii','Track32fast', 'Track32slow']
    #im_names=['1iii', '1ii', '1iv', '1i', '1v', '2iii', '2ii', '2iv', '2i', '2vi', '2v', 'continuitynew', 'df10_dt100', 'df10_dt110', 'df10_dt120', 'df10_dt125', 'df10_dt130', 'df10_dt140', 'df10_dt150', 'df10_dt80', 'df10_dt90', 'df12_dt100', 'df12_dt110', 'df12_dt120', 'df12_dt125', 'df12_dt130', 'df12_dt140', 'df12_dt150', 'df12_dt80', 'df12_dt90', 'df15_dt100', 'df15_dt110', 'df15_dt120', 'df15_dt125', 'df15_dt130', 'df15_dt140', 'df15_dt150', 'df15_dt80', 'df15_dt90', 'df2_dt100', 'df2_dt110', 'df2_dt120', 'df2_dt125', 'df2_dt130', 'df2_dt140', 'df2_dt150', 'df2_dt80', 'df2_dt90', 'df5_dt100', 'df5_dt110', 'df5_dt120', 'df5_dt125', 'df5_dt130', 'df5_dt140', 'df5_dt150', 'df5_dt80', 'df5_dt90', 'df7_dt100', 'df7_dt110', 'df7_dt120', 'df7_dt125', 'df7_dt130', 'df7_dt140', 'df7_dt150', 'df7_dt80', 'df7_dt90', '4ii', 'Track32fast','Track32slow', 'Track15higherBetweens', 'Track15lowerBetweens', 'Track17X'];
    im_names=['Track15higherBetweens', 'Track15lowerBetweens', 'Track17X', 'continuitynew']
    for df in [2, 8, 14, 20]:
        for dt in [125,150,175,200]:
            im_names.append('df{}_dt{}-shorttone'.format(df,dt))
    for im_name in im_names:
        #for extra in ['','-50']:
        im_name = im_name + '-50'
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/bASA/JPEGImages/{}'.format(im_name))
        basademo(net, im_name, dataname, exptname)

    plt.show()
