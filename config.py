import os

basedir = os.getcwd()
log_path = os.path.join(basedir,'log')
epochs = 10
model_path = os.path.join(basedir,'mdl','mdl')
vocimdir = r"D:\Users\yl_gong\Desktop\dl\voc\combo\image"
voclabeldir= r"D:\Users\yl_gong\Desktop\dl\voc\combo\label"
trainbatch=10
labels=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
                        'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']