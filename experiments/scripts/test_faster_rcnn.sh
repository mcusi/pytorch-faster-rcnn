#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

echo $@

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
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  basa)
    TRAIN_IMDB="basa_train"
    TEST_IMDB="basa_test"
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

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${dataname}_train/${EXTRA_ARGS_SLUG}/${SNAPSHOT}_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${dataname}_train/default/${SNAPSHOT}_iter_${ITERS}.pth
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ANCHOR_REFERENCE ${ANCHORREF} \
          TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT} ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ANCHOR_REFERENCE ${ANCHORREF}\
          TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT} ${EXTRA_ARGS}
fi

