import os

basedir = os.getcwd()
log_path = os.path.join(basedir,'log')
epochs = 300
model_path = os.path.join(basedir,'mdl','mdl')
vocimdir = r"D:\Users\yl_gong\Desktop\dl\voc\sss"
voclabeldir= r"D:\Users\yl_gong\Desktop\dl\voc\VOC2012\Annotations"
trainbatch=10
labels=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
                        'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']