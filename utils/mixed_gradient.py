import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

x = tf.Variable(0.0,trainable=True,dtype=tf.float16)
with tf.GradientTape() as y_tape:
    y = 2.0*tf.cast(x,tf.float16)+3.0*tf.cast(x,tf.float16)
dy_dx = y_tape.gradient(y,x)
print(dy_dx)
    