import tensorflow as tf
from config import *
import numpy as np

detector_inp = tf.placeholder(dtype=tf.float32,shape=[None,None,None,512],name='input')#[None,7,7,512]
xy_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1,2])
wh_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1,2])
mask_box = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1,1])
cls_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,20])
ts = tf.shape(detector_inp)[:3]#[None,7,7]
th = tf.cast(tf.shape(detector_inp)[1],tf.float32)
mask_box_s = tf.squeeze(mask_box,axis=-1)#[None,7,7,1]
whsqrt_t = tf.sqrt(tf.maximum(wh_t,1e-5))
box_num = tf.maximum(tf.reduce_sum(mask_box),1.)
coordxy_t = (xy_t+tf.expand_dims(tf.stack(tf.meshgrid(tf.range(th),tf.range(th)),axis=-1),axis=2))/th#[None,7,7,2,2]
lf_t = coordxy_t-wh_t/2#[None,7,7,2,2]
rt_t = coordxy_t+wh_t/2#[None,7,7,2,2]

tmp = tf.layers.conv2d(detector_inp,256,(3,3),activation=tf.nn.leaky_relu,padding='SAME')
tmp = tf.layers.conv2d(tmp,512,(3,3),activation=tf.nn.leaky_relu,padding='SAME')
detector_out = tf.layers.conv2d(tmp,30,(1,1))#[None,7,7,30]
xy = tf.sigmoid(tf.reshape(detector_out[...,0:4],tf.concat([ts,[2,2]],axis=0)))#[None,7,7,2,2]
wh = tf.sigmoid(tf.reshape(detector_out[...,4:8],tf.concat([ts,[2,2]],axis=0)))#[None,7,7,2,2]
whsqrt = tf.sqrt(tf.maximum(wh,1e-5))#[None,7,7,2,2]
iou_p = tf.sigmoid(tf.reshape(detector_out[...,8:10],tf.concat([ts,[2]],axis=0)))#[None,7,7,2]
cls = tf.nn.softmax(detector_out[...,10:30])#[None,7,7,20]

coordxy = (xy+tf.expand_dims(tf.stack(tf.meshgrid(tf.range(th),tf.range(th)),axis=-1),axis=2))/th#[None,7,7,2,2]
lf = coordxy-wh/2#[None,7,7,2,2]
rt = coordxy+wh/2#[None,7,7,2,2]
sectwh = tf.minimum(rt_t,rt)-tf.maximum(lf_t,lf)#[None,7,7,2,2]
sect = tf.multiply(*tf.unstack(sectwh,axis=-1))\
       *tf.cast(tf.greater_equal(sectwh[...,0],0),tf.float32)#[None,7,7,2]
union = tf.maximum(tf.multiply(*tf.unstack(rt-lf,axis=-1))+tf.multiply(*tf.unstack(rt_t-lf_t,axis=-1))-sect,1e-5)#[None,7,7,2]
iou_t = sect/union#[None,7,7,2]

resp_mask = tf.cast(tf.equal(iou_t,tf.reduce_max(iou_t,axis=-1,keepdims=True)),tf.float32)*mask_box_s#[None,7,7,2]
resp_maskb = tf.tile(tf.expand_dims(resp_mask,axis=-1),[1,1,1,1,2])#[None,7,7,2,2]
iouerr = tf.reduce_mean((iou_p-iou_t)**2 * tf.where(tf.equal(resp_mask,1.),resp_mask,tf.ones_like(resp_mask)*0.5))
xyerr = tf.reduce_sum((xy-xy_t)**2 * resp_maskb)/box_num
wherr = tf.reduce_sum((whsqrt_t-whsqrt)**2 * resp_maskb)/box_num
clserr = tf.reduce_sum((cls-cls_t)**2 * cls_t)/box_num
allerr = xyerr+wherr+clserr+iouerr

tf.summary.scalar('xyerr',xyerr)
tf.summary.scalar('wherr',wherr)
tf.summary.scalar('iouerr',iouerr)
tf.summary.scalar('clserr',clserr)
tf.summary.scalar('allerr',allerr)
tf.summary.histogram('xy',xy)
tf.summary.histogram('whsqrt',whsqrt)
tf.summary.histogram('iou_p',iou_p)
tf.summary.histogram('detector_out',detector_out)
log_all = tf.summary.merge_all()







