import os

basedir = os.getcwd()
log_path = os.path.join(basedir,'log')
epochs = 300
model_path = os.path.join(basedir,'mdl','mdl')
vocimdir = r"img/"
voclabeldir= r"annotation/"
trainbatch=10
labels=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
                        'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']