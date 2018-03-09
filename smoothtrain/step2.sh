#!/bin/bash
#SBATCH --time=00:20:00 

## MAKE SH FILES THAT RUN A NEURAL NETWORK AND THEN TEST IT ON DEMOS

source activate /om/user/mcusi/nnInit/nnonda
cd /om/user/mcusi/nnInit/pytorch-faster-rcnn/smoothtrain/

export dataname=bASAGP1
export net=vgg16
export anchorref=4
export anchors="[1,3,6,9,21,27]" 
export ratios="[0.25,0.5,1,2,4]"
export snapshot=${net}_ar${anchorref}_anch6_mult3
python makeSH.py
shpath="/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/"
shfile="/train${dataname}${snapshot}.sh"
shcomplete=${shpath}${dataname}${shfile}
echo 'ar4,anch6,mult3'
echo $shcomplete
shlog="/train${dataname}${snapshot}.out"
shlogfile=${shpath}${dataname}${shlog}
echo $shlogfile
sbatch -o ${shlogfile} $shcomplete 



