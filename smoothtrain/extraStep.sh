#!/bin/bash
#SBATCH --time=00:45:00 
#SBATCH --gres=gpu:titan-x:1

## CHECK NET PERFORMANCE ON DEMOS

#make spectrograms of current demos in bayesianASA/sounds
cd /om2/user/mcusi/bayesianASA/src/py/
source activate webponda
python genNNDataFromWav.py

cd /om/user/mcusi/nnInit/pytorch-faster-rcnn/
#test and save results of network on demos
source activate /om/user/mcusi/nnInit/nnonda
python ./tools/drawNetAnswersExtraScript.py
echo 'done testing!! :)'
