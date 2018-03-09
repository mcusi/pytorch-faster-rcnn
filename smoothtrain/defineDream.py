import json
import os
import shutil

params={}
params["forward"]=True
params["dataset"]=os.environ['dataname']
params["nTrain"]=int(os.environ.get('nTrain',50000))
params["nTest"]=int(os.environ.get('nTest',10000))
params["sr"]=20000
params["twin"]=float(os.environ.get('twin',0.025))
params["thop"]=float(os.environ.get('thop',0.010))
params["nFilts"]=int(os.environ.get('nFilts',128))
params["filtWidth"]=float(os.environ.get('filtWidth',0.5))
params["spectrum"]={}
params["spectrum"]["dt"]=os.environ['spectrumdt']
params["spectrum"]["df"]=os.environ['spectrumdf']
params["spectrum"]["lowf"]=int(os.environ.get('lowf',50))
params["spectrum"]["highf"]=int(os.environ.get('highf',9000))
params["spectrum"]["tstep"]=float(os.environ.get('tstep',0.1))
params["spectrum"]["fstep"]=float(os.environ.get('fstep',3))
params["f0"]={}
params["f0"]["d"]=float(os.environ.get('f0d',0.5))
params["f0"]["sigma"]=float(os.environ.get('f0sigma',1))
params["cpts"]=int(os.environ.get('cpts',20))
with open('/om2/user/mcusi/bayesianASA/opts/' + os.environ['dataname'] + 'dream.json', 'w') as fp:
    json.dump(params, fp)

os.mkdir('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + os.environ['dataname'] + '/')
os.mkdir('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + os.environ['dataname'] + '/JPEGImages/')
os.mkdir('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + os.environ['dataname'] + '/Annotations/')
os.mkdir('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + os.environ['dataname'] + '/ImageSets/')
os.mkdir('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + os.environ['dataname'] + '/ImageSets/Main/')
os.mkdir('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + os.environ['dataname'] + '/results/')
os.mkdir('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + os.environ['dataname'] + '/results/Main/')
os.mkdir('/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + os.environ['dataname'] + '/demos/')

shutil.copyfile('/om2/user/mcusi/bayesianASA/opts/' + os.environ['dataname'] + 'dream.json', '/om/user/mcusi/nnInit/pytorch-faster-rcnn/data/' + os.environ['dataname'] + '/' + os.environ['dataname'] + 'dream.json')

print('Prepared json params and data directories')