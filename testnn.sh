#!/bin/bash
#SBATCH --gres=gpu:titan-x:1
#SBATCH --time=1:00:00 

GPU_ID=0
DATASET=basa
NET=vgg16
ANCHORREF=4
SNAPSHOT="${NET}_ar${ANCHORREF}"
ANCHORS="[1,2,4,8,16,32]"
RATIOS="[0.25,0.5,1,2,4]"

source activate /om/user/mcusi/nnInit/nnonda
./experiments/scripts/test_faster_rcnn.sh 0 basa vgg16 ${SNAPSHOT} ${ANCHORS} ${RATIOS} ${ANCHORREF}; 
echo 'done training'
