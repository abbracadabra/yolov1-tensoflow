import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
from util import *

z = tf.constant([2.])
aa = tf.cast(tf.equal(z,[2.00]),tf.float32)

sess = tf.Session()
print(sess.run(aa))


