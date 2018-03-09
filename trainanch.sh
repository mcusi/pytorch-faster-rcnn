#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --qos=mcdermott

echo Node: $(/bin/hostname)
gpu=`env | grep GPU_DEVICE_ORDINAL | cut -d"=" -f2`
if [ -z "$gpu" ]; then
	echo GPU: None
else
	echo GPU: $gpu
	nvidia-smi
fi

export dataname=bASAGP1

source activate /om/user/mcusi/nnInit/nnonda
cd /om/user/mcusi/nnInit/pytorch-faster-rcnn/
# ./experiments/scripts/train_faster_rcnn.sh $gpu basa vgg16 vgg16_ar4_anch6_mult4 [1,2,4,8,16,32] [0.25,0.5,1,2,4] 4 bASAGP1;
# echo "done training" 
sbatch --qos=mcdermott -o /om/user/mcusi/nnInit/pytorch-faster-rcnn/data/bASAGP1/demos-vgg16_ar4_anch6_mult4.out /om/user/mcusi/nnInit/pytorch-faster-rcnn/smoothtrain/step3.sh bASAGP1 vgg16_ar4_anch6_mult4