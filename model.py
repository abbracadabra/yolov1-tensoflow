import tensorflow as tf
from config import *
import numpy as np

detector_inp = tf.placeholder(dtype=tf.float32,shape=[None,None,None,512],name='input')#[None,7,7,512]
ts = tf.shape(detector_inp)[:3]
th = tf.cast(tf.shape(detector_inp)[1],tf.float32)
xy_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1,2])
wh_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1,2])
mask_box = tf.placeholder(dtype=tf.float32,shape=[None,None,None,1,1])
cls_t = tf.placeholder(dtype=tf.float32,shape=[None,None,None,20])
mask_box_s = tf.squeeze(mask_box,axis=-1)#[None,7,7,1]
whsqrt_t = tf.sqrt(tf.maximum(wh_t,1e-2))
box_num = tf.reduce_sum(mask_box)

coordxy_t = (xy_t+tf.expand_dims(tf.stack(tf.meshgrid(tf.range(th),tf.range(th)),axis=-1),axis=2))/th#[None,7,7,2,2]
lf_t = tf.clip_by_value(coordxy_t-wh_t/2,0.,1.)#[None,7,7,2,2]
rt_t = tf.clip_by_value(coordxy_t+wh_t/2,0.,1.)#[None,7,7,2,2]

detector_out = tf.layers.conv2d(detector_inp,30,(1,1))#[None,7,7,30]
xy = tf.nn.sigmoid(tf.reshape(detector_out[...,0:4],tf.concat([ts,[2,2]],axis=0)))#[None,7,7,2,2]
wh = tf.sigmoid(tf.reshape(detector_out[...,4:8],tf.concat([ts,[2,2]],axis=0)))
whsqrt = tf.sqrt(tf.maximum(wh,1e-2))#[None,7,7,2,2]
iou_p = tf.nn.sigmoid(tf.reshape(detector_out[...,8:10],tf.concat([ts,[2]],axis=0)))#[None,7,7,2]
cls = tf.nn.softmax(detector_out[...,10:30])#[None,7,7,20]

coordxy = (xy+tf.expand_dims(tf.stack(tf.meshgrid(tf.range(th),tf.range(th)),axis=-1),axis=2))/th#[None,7,7,2,2]
lf = tf.clip_by_value(coordxy-wh/2,0.,1.)#[None,7,7,2,2]
rt = tf.clip_by_value(coordxy+wh/2,0.,1.)#[None,7,7,2,2]
sect = tf.nn.relu(tf.multiply(*tf.unstack(tf.minimum(rt_t,rt)-tf.maximum(lf_t,lf),axis=-1)))#[None,7,7,2]
union = tf.maximum(tf.multiply(*tf.unstack(rt-lf,axis=-1))+tf.multiply(*tf.unstack(rt_t-lf_t,axis=-1))-sect,1e-2)#[None,7,7,2]
iou_t = sect/union#[None,7,7,2]

resp_mask = tf.cast(tf.equal(iou_t,tf.maximum(iou_t[...,0:1],iou_t[...,1:2])),tf.float32)*mask_box_s#[None,7,7,2]
resp_maskb = tf.tile(tf.expand_dims(resp_mask,axis=-1),[1,1,1,1,2])
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
log_all = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())







