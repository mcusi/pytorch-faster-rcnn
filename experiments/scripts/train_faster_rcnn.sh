#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
SNAPSHOT_TRAIN=$4
ANCHORS_TRAIN=$5
RATIOS_TRAIN=$6
ANCHORREF_TRAIN=$7
dataname=$8

export dataname=$dataname

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:8:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ANCHORREF=16
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ANCHORREF=16
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ANCHORREF=16
    ;;
  basa)
    TRAIN_IMDB="basa_train"
    TEST_IMDB="basa_test"
    STEPSIZE="[65000]"
    ITERS=200000
    ANCHORS=${ANCHORS_TRAIN}
    RATIOS=${RATIOS_TRAIN}
    ANCHORREF=${ANCHORREF_TRAIN}
    SNAPSHOT="${SNAPSHOT_TRAIN}"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${dataname}_train/${EXTRA_ARGS_SLUG}/${SNAPSHOT}_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${dataname}_train/default/${SNAPSHOT}_iter_${ITERS}.pth
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.pth \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --tag ${EXTRA_ARGS_SLUG} \
      --net ${NET} \
      --dataname ${dataname} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ANCHOR_REFERENCE ${ANCHORREF} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.pth \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --net ${NET} \
      --dataname ${dataname} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ANCHOR_REFERENCE ${ANCHORREF} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT} ${EXTRA_ARGS}
  fi
fi

./experiments/scripts/test_faster_rcnn.sh $@
