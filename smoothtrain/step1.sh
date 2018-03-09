#!/bin/bash
#SBATCH --time=00:20:00 

##GENERATE DATASET FROM GENERATIVE MODEL AND THEN PASS IT OVER TO NEURAL NETOWRKS FOR TRAINING

export dataname=debugprocess
export spectrumdt=0.1
export spectrumdf=2
export nTrain=5
export nTest=1

python defineDream.py

dream='dream'
opt=${dataname}${dream}
cd /om2/user/mcusi/bayesianASA/
source activate webponda
webppl --require . --require webppl-call-async --require webppl-json ./listen.wppl -- --opt=$opt
echo "complete webppl generation" 
cd /om/user/mcusi/nnInit/pytorch-faster-rcnn/smoothtrain/

source deactivate
outputplace=/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/${dataname}/startnetworks.out
sbatch -o $outputplace ./step2.sh $dataname
