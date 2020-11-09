import tensorflow as tf
import time
import os 
import sys
import model 
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../../'))
import datasets.BratsPipeLine as train_dataset


physical_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

######################################################################################################
train_path = "G:/Datasets/BraTS/ToCrop/MICCAI_BraTS2020_TrainingData"
test_path = "G:/Datasets/BraTS/ToCrop/MICCAI_BraTS2020_ValidationData"
tmp_path = "D:/Work/Codes_tmp/2DCycleGAN-mixed-wgp-sn-randomCrop128-v96"
out_path = "D:/Work/Codes_tmp/2DCycleGAN-mixed-wgp-sn-randomCrop128-v96/out" #64v(default) 72v 96v 128v 32v 24v
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)
def map_func(imgX,imgY,maskX,maskY,index_buf):
    #240->128
    imgX = imgX[index_buf[0,0]:index_buf[0,1],index_buf[1,0]:index_buf[1,1]]
    imgY = imgY[index_buf[0,0]:index_buf[0,1],index_buf[1,0]:index_buf[1,1]]
    maskX = maskX[index_buf[0,0]:index_buf[0,1],index_buf[1,0]:index_buf[1,1]]
    maskY = maskY[index_buf[0,0]:index_buf[0,1],index_buf[1,0]:index_buf[1,1]]
    X = tf.reshape(imgX,[128,128,1],name="imgX")
    Y = tf.reshape(imgY,[128,128,1],name="imgY")
    mX = tf.reshape(maskX,[128,128,1],name="maskX")
    mY = tf.reshape(maskY,[128,128,1],name="maskY")
    return X,Y,mX,mY
EPOCHES = 200
BATCH_SIZE = 1


num_threads = 4
dataset = train_dataset.DataPipeLine(train_path,target_size=[240,240],patch_size=[128,128])
dataset = tf.data.Dataset.from_generator(dataset.generator,output_types=(tf.float32,tf.float32,tf.float32,tf.float32,tf.int32),output_shapes=([240,240],[240,240],[240,240],[240,240],[2,2]))\
            .map(map_func,num_parallel_calls=num_threads)\
            .batch(BATCH_SIZE)\
            .shuffle(200)\
            .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
            
# import matplotlib.pyplot as plt
# for i,(X,Y,mX,mY) in enumerate(dataset):
#     # print(i+1,X.shape,Y.dtype,mX.dtype,mY.dtype)
#     plt.figure(figsize=(5,5))#图片大一点才可以承载像素
#     plt.subplot(2,2,1)
#     plt.imshow(X[0,:,:,0],cmap='gray')
#     plt.axis('off')
#     plt.subplot(2,2,2)
#     plt.imshow(Y[0,:,:,0],cmap='gray')
#     plt.axis('off')
#     plt.subplot(2,2,3)
#     plt.imshow(mX[0,:,:,0],cmap='gray')
#     plt.axis('off')
#     plt.subplot(2,2,4)
#     plt.imshow(mY[0,:,:,0],cmap='gray')
#     plt.axis('off')
#     plt.show()
test_set = train_dataset.DataPipeLine(test_path,target_size=[240,240],patch_size=[128,128])
test_set = tf.data.Dataset.from_generator(test_set.generator,output_types=(tf.float32,tf.float32,tf.float32,tf.float32,tf.int32),output_shapes=([240,240],[240,240],[240,240],[240,240],[2,2]))\
            .map(map_func,num_parallel_calls=num_threads)\
            .batch(BATCH_SIZE)\
            .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

# for i,(X,Y) in enumerate(dataset):
#     print(i+1,X.shape,Y.dtype)
model = model.CycleGAN(train_set=dataset,
                       test_set=test_set,
                       loss_name="WGAN-GP-SN",
                       mixed_precision=True,
                       learning_rate=1e-4,
                       tmp_path=tmp_path,
                       out_path=out_path)
model.build(X_shape=[None,128,128,1],Y_shape=[None,128,128,1])
model.train(epoches=EPOCHES)