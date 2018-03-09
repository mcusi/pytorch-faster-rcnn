import os

net=os.environ['net']
anchorref=os.environ['anchorref']
snapshot = os.environ['snapshot']
dataname = os.environ['dataname']
anchors=os.environ['anchors']
ratios=os.environ['ratios']
with open("/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/" + dataname + '/train' + dataname + snapshot + '.sh', "w") as t:
    t.write('#!/bin/bash\n')
    t.write('#SBATCH --time=02:00:00\n')
    t.write('#SBATCH --gres=gpu:titan-x:1\n')
    t.write('\n')
    t.write('echo Node: $(/bin/hostname)\n')
    t.write('gpu=`env | grep GPU_DEVICE_ORDINAL | cut -d"=" -f2`\n')
    t.write('if [ -z "$gpu" ]; then\n')
    t.write('\techo GPU: None\n')
    t.write('else\n')
    t.write('\techo GPU: $gpu\n')
    t.write('\tnvidia-smi\n')
    t.write('fi\n')
    t.write('\n')
    t.write('export dataname={}\n'.format(dataname))
    t.write('\n')
    t.write('source activate /om/user/mcusi/nnInit/nnonda\n')
    t.write('cd /om/user/mcusi/nnInit/pytorch-faster-rcnn/\n')
    t.write('./experiments/scripts/train_faster_rcnn.sh $gpu basa {0} {1} {2} {3} {4} {5};\n'.format(net, snapshot, anchors, ratios, anchorref, dataname))
    t.write('echo "done training" \n')
    outputplace = '/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/{0}/demos-{1}.out'.format(dataname,snapshot)
    t.write('sbatch -o {0} /om/user/mcusi/nnInit/pytorch-faster-rcnn/smoothtrain/step3.sh {1} {2}\n'.format(outputplace, dataname, snapshot))