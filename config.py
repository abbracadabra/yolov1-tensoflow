import os

basedir = os.getcwd()
log_path = os.path.join(basedir,'log')
epochs = 300
model_path = os.path.join(basedir,'mdl','mdl')
vocimdir = r"D:\Users\yl_gong\Desktop\dl\voc\zzz"
voclabeldir= r"D:\Users\yl_gong\Desktop\dl\voc\combo\label"
trainbatch=1
labels=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
                        'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']