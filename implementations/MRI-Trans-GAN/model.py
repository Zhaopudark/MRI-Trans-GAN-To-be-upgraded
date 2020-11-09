import networks 
import tensorflow as tf
import time
import os 
import sys
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base,'../../'))
from basemodels.GanLosses import GanLoss
from basemodels.GanOptimizers import Adam

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
from PIL import Image
import datetime
###############global paraments###################
"""
记录那些无法被模型定义传递的参数
尤其是@tf.function() 中需要的参数
学习率与损失函数系数则应当在模型训练过程中予以控制
"""
global_input_X_shape = [1,128,128,3,1]
global_input_Y_shape = [1,128,128,3,1]
global_mask_X_shape = [1,128,128,3,1]
global_mask_Y_shape = [1,128,128,3,1]
################################################
class vgg16():
    def __init__(self):
        super(vgg16,self).__init__()
        vgg16_data = np.load('E:\\Datasets\\VGG\\vgg16.npy', encoding='latin1',allow_pickle=True)
        # print(type(vgg16_data))
        self.data_dict = vgg16_data.item()
        # print(type(data_dict))
        # print(data_dict.keys())
        # print(len(data_dict))
        self.vgg_list = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv4_1', 'conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7','fc8']
        self.vgg_list_n = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3']
        # for i in lis:
        #     tmp = data_dict[i]
        #     w, b = tmp
        #     print(w.shape,b.shape)
        self.layer_list = []
        self.wb=[]
        for i,layer_name in enumerate(self.vgg_list_n):
            tmp = self.data_dict[layer_name]
            self.wb.append(tmp)
    def __call__(self,x):
        x = x*255.0
        x = tf.reshape(x,shape=[1,128,128,3])
        # x=tf.broadcast_to(x,shape=[1,128,128,3],name=None)
        out_buf = []
        for i,(w,b) in enumerate(self.wb):
            x = tf.nn.conv2d(x,filters=w,strides=[1,1,1,1],padding="SAME",data_format='NHWC')
            x = x+b
            # plt.imshow(x[0,:,:,0],cmap="gray")
            # plt.show()
            out_buf.append(x)
        for item in out_buf:
            item = item/255.0
        return out_buf

class CycleGAN(tf.keras.Model):
    """
    模型只负责给定训练集和测试(验证)集后的操作
    """
    def __init__(self,
                train_set,
                test_set,
                loss_name="WGAN-GP",
                mixed_precision=False,
                learning_rate=2e-4,
                tmp_path=None,
                out_path=None):
        super(CycleGAN,self).__init__()
        #接收数据集和相关参数
        self.train_set = train_set
        self.test_set = test_set
        self.tmp_path = tmp_path
        self.out_path = out_path
        #定义模型
        self.G = networks.Generator(name="G_X2Y")
        self.F = networks.Generator(name="G_Y2X")
        if loss_name in ["WGAN-SN","WGAN-GP-SN"]:
            self.Dy = networks.Discriminator(name="If_is_real_Y",use_sigmoid=False,sn=True)
            self.Dx = networks.Discriminator(name="If_is_real_X",use_sigmoid=False,sn=True)
            self.loss_name = loss_name[:-3]
        elif loss_name in ["WGAN","WGAN-GP"]:
            self.Dy = networks.Discriminator(name="If_is_real_Y",use_sigmoid=False,sn=False)
            self.Dx = networks.Discriminator(name="If_is_real_X",use_sigmoid=False,sn=False)
            self.loss_name = loss_name
        elif loss_name in ["Vanilla-SN","LSGAN-SN"]:
            self.Dy = networks.Discriminator(name="If_is_real_Y",use_sigmoid=True,sn=True)
            self.Dx = networks.Discriminator(name="If_is_real_X",use_sigmoid=True,sn=True)
            self.loss_name = loss_name[:-3]
        elif loss_name in ["Vanilla","LSGAN"]:
            self.Dy = networks.Discriminator(name="If_is_real_Y",use_sigmoid=True,sn=False)
            self.Dx = networks.Discriminator(name="If_is_real_X",use_sigmoid=True,sn=False)
            self.loss_name = loss_name
        else: 
            raise ValueError("Do not support the loss "+loss_name)
        self.vgg = vgg16()
        self.model_list=[self.G,self.F,self.Dy,self.Dx]
        #定义损失函数 优化器 记录等
        self.gan_loss = GanLoss(self.loss_name)
        self.optimizers_list = self.optimizers_config(mixed_precision=mixed_precision,learning_rate=learning_rate)
        self.mixed_precision = mixed_precision
        self.matrics_list = self.matrics_config()
        self.checkpoint_config()
        self.get_seed()
    def build(self,X_shape,Y_shape):
        """
        input_shape必须切片 因为在底层会被当做各层的输出shape而被改动
        """
        self.G.build(input_shape=X_shape[:])#G X->Y
        self.Dy.build(input_shape=Y_shape[:])#Dy Y or != Y
        self.F.build(input_shape=Y_shape[:])#F Y->X
        self.Dx.build(input_shape=X_shape[:])#Dx X or != X
        self.built = True

    def optimizers_config(self,mixed_precision=False,learning_rate=2e-4):
        self.G_optimizer = Adam(learning_rate=1e-4,beta_1=0.0,beta_2=0.9)
        self.Dy_optimizer = Adam(learning_rate=4e-4,beta_1=0.0,beta_2=0.9)
        self.F_optimizer = Adam(learning_rate=1e-4,beta_1=0.0,beta_2=0.9)
        self.Dx_optimizer = Adam(learning_rate=4e-4,beta_1=0.0,beta_2=0.9)
        # self.G_optimizer = Adam(learning_rate=2e-4)
        # self.Dy_optimizer = Adam(learning_rate=2e-4)
        # self.F_optimizer = Adam(learning_rate=2e-4)
        # self.Dx_optimizer = Adam(learning_rate=2e-4)
        if mixed_precision:
            self.G_optimizer=self.G_optimizer.get_mixed_precision()
            self.Dy_optimizer=self.Dy_optimizer.get_mixed_precision()
            self.F_optimizer=self.F_optimizer.get_mixed_precision()
            self.Dx_optimizer=self.Dx_optimizer.get_mixed_precision()
        return [self.G_optimizer,self.Dy_optimizer,self.F_optimizer,self.Dx_optimizer]
    def matrics_config(self):
        current_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_logdir = self.tmp_path+"/logs/" + current_time
        self.train_summary_writer = tf.summary.create_file_writer(train_logdir)
        self.m_psnr_X2Y = tf.keras.metrics.Mean('psnr_y', dtype=tf.float32)
        self.m_psnr_Y2X = tf.keras.metrics.Mean('psnr_x', dtype=tf.float32)
        self.m_ssim_X2Y = tf.keras.metrics.Mean('ssim_y', dtype=tf.float32)
        self.m_ssim_Y2X = tf.keras.metrics.Mean('ssim_x', dtype=tf.float32)
        return [self.m_psnr_X2Y,self.m_psnr_Y2X,self.m_ssim_X2Y,self.m_ssim_Y2X]
        # return None

    def checkpoint_config(self):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),optimizer=self.optimizers_list,model=self.model_list,dataset=self.train_set)
        self.manager = tf.train.CheckpointManager(self.ckpt,self.tmp_path+'/tf_ckpts', max_to_keep=3)
    def pix_gradient(self,x):
        x = tf.reshape(x,shape=[1,64,64,1])#在各batch和通道上进行像素梯度 对2D单通道而言其实没必要reshape
        dx,dy = tf.image.image_gradients(x)
        return dx,dy

    @tf.function(input_signature=[tf.TensorSpec(shape=global_input_X_shape,dtype=tf.float32),
                                  tf.TensorSpec(shape=global_input_Y_shape,dtype=tf.float32),
                                  tf.TensorSpec(shape=global_mask_X_shape,dtype=tf.float32),
                                  tf.TensorSpec(shape=global_mask_Y_shape,dtype=tf.float32)])
    def train_step_D(self,trainX,trainY,maskX,maskY):
        with tf.GradientTape(persistent=True) as D_tape:
            GeneratedY = self.G(trainX)
            GeneratedY = tf.multiply(GeneratedY,maskY)
            Dy_real_out = self.Dy(trainY)
            Dy_fake_out = self.Dy(GeneratedY)

            GeneratedX = self.F(trainY)
            GeneratedX = tf.multiply(GeneratedX,maskX)
            Dx_real_out = self.Dx(trainX)
            Dx_fake_out = self.Dx(GeneratedX)

            e = tf.random.uniform(shape=self.wgp_shape,minval=0.0,maxval=1.0)
            mid_Y = e*trainY+(1-e)*GeneratedY
            with tf.GradientTape() as gradient_penaltyY:
                gradient_penaltyY.watch(mid_Y)
                inner_loss = self.Dy(mid_Y)
            penalty = gradient_penaltyY.gradient(inner_loss,mid_Y)
            penalty_normY = 10.0*tf.math.square(tf.norm(tf.reshape(penalty,shape=[self.wgp_shape[0],-1]),ord=2,axis=-1)-1)

            e = tf.random.uniform(shape=self.wgp_shape,minval=0.0,maxval=1.0)
            mid_X = e*trainX+(1-e)*GeneratedX
            with tf.GradientTape() as gradient_penaltyX:
                gradient_penaltyX.watch(mid_X)
                inner_loss = self.Dx(mid_X)
            penalty = gradient_penaltyX.gradient(inner_loss,mid_X)
            penalty_normX = 10.0*tf.math.square(tf.norm(tf.reshape(penalty,shape=[self.wgp_shape[0],-1]),ord=2,axis=-1)-1)

            Dy_loss = self.gan_loss.DiscriminatorLoss(Dy_real_out,Dy_fake_out)+tf.reduce_mean(penalty_normY)
            Dx_loss = self.gan_loss.DiscriminatorLoss(Dx_real_out,Dx_fake_out)+tf.reduce_mean(penalty_normX)

            if self.mixed_precision:
                scaled_Dy_loss = self.Dy_optimizer.get_scaled_loss(Dy_loss)
                scaled_Dx_loss = self.Dx_optimizer.get_scaled_loss(Dx_loss)

        if self.mixed_precision:
            scaled_gradients_of_Dy=D_tape.gradient(scaled_Dy_loss,self.Dy.trainable_variables)
            scaled_gradients_of_Dx=D_tape.gradient(scaled_Dx_loss,self.Dx.trainable_variables)
            gradients_of_Dy = self.Dy_optimizer.get_unscaled_gradients(scaled_gradients_of_Dy)
            gradients_of_Dx = self.Dx_optimizer.get_unscaled_gradients(scaled_gradients_of_Dx)
        else:
            gradients_of_Dy = D_tape.gradient(Dy_loss,self.Dy.trainable_variables)
            gradients_of_Dx = D_tape.gradient(Dx_loss,self.Dx.trainable_variables)

        self.Dy_optimizer.apply_gradients(zip(gradients_of_Dy,self.Dy.trainable_variables))
        self.Dx_optimizer.apply_gradients(zip(gradients_of_Dx,self.Dx.trainable_variables))
        return Dy_loss,Dx_loss

    @tf.function(input_signature=[tf.TensorSpec(shape=global_input_X_shape,dtype=tf.float32),
                                  tf.TensorSpec(shape=global_input_Y_shape,dtype=tf.float32),
                                  tf.TensorSpec(shape=global_mask_X_shape,dtype=tf.float32),
                                  tf.TensorSpec(shape=global_mask_Y_shape,dtype=tf.float32)])
    def train_step_G(self,trainX,trainY,maskX,maskY):
        with tf.GradientTape(persistent=True) as G_tape:
            GeneratedY = self.G(trainX)
            GeneratedY = tf.multiply(GeneratedY,maskY)
            # Dy_real_out = self.Dy(trainY)
            Dy_fake_out = self.Dy(GeneratedY)

            GeneratedX = self.F(trainY)
            GeneratedX = tf.multiply(GeneratedX,maskX)
            # Dx_real_out = self.Dx(trainX)
            Dx_fake_out = self.Dx(GeneratedX)

            cycle_consistent_loss_X2Y = tf.reduce_mean(tf.abs(self.F(GeneratedY)-trainX))
            cycle_consistent_loss_Y2X = tf.reduce_mean(tf.abs(self.G(GeneratedX)-trainY))
            cycle_consistent = cycle_consistent_loss_X2Y+cycle_consistent_loss_Y2X


            fake_Y_perceptual = self.vgg(GeneratedY)
            real_Y_perceptual = self.vgg(trainY)
            fake_X_perceptual = self.vgg(GeneratedX)
            real_X_perceptual = self.vgg(trainX)
            reconstruction_loss_X2Y = 0
            reconstruction_loss_Y2X = 0
            for i in range(7):
                reconstruction_loss_X2Y += 0.14*tf.reduce_mean(tf.abs(fake_Y_perceptual[i]-real_Y_perceptual[i]))
                reconstruction_loss_Y2X += 0.14*tf.reduce_mean(tf.abs(fake_X_perceptual[i]-real_X_perceptual[i]))

            # reconstruction_loss_X2Y = tf.reduce_mean(tf.abs(GeneratedY-trainY))
            # reconstruction_loss_Y2X = tf.reduce_mean(tf.abs(GeneratedX-trainX))

            G_loss = self.gan_loss.GeneratorLoss(Dy_fake_out)+10.0*cycle_consistent+reconstruction_loss_X2Y
            F_loss = self.gan_loss.GeneratorLoss(Dx_fake_out)+10.0*cycle_consistent+reconstruction_loss_Y2X

            if self.mixed_precision:
                scaled_G_loss = self.G_optimizer.get_scaled_loss(G_loss)
                scaled_F_loss = self.F_optimizer.get_scaled_loss(F_loss)
        if self.mixed_precision:
            scaled_gradients_of_G=G_tape.gradient(scaled_G_loss,self.G.trainable_variables)
            scaled_gradients_of_F=G_tape.gradient(scaled_F_loss,self.F.trainable_variables)
            gradients_of_G = self.G_optimizer.get_unscaled_gradients(scaled_gradients_of_G)
            gradients_of_F = self.F_optimizer.get_unscaled_gradients(scaled_gradients_of_F)

        else:
            gradients_of_G = G_tape.gradient(G_loss,self.G.trainable_variables)
            gradients_of_F = G_tape.gradient(F_loss,self.F.trainable_variables)

        self.G_optimizer.apply_gradients(zip(gradients_of_G,self.G.trainable_variables))
        self.F_optimizer.apply_gradients(zip(gradients_of_F,self.F.trainable_variables))
        return G_loss,F_loss

    @tf.function(input_signature=[tf.TensorSpec(shape=global_input_X_shape,dtype=tf.float32),
                                  tf.TensorSpec(shape=global_input_Y_shape,dtype=tf.float32),
                                  tf.TensorSpec(shape=global_mask_X_shape,dtype=tf.float32),
                                  tf.TensorSpec(shape=global_mask_Y_shape,dtype=tf.float32)])
    def train_step(self,trainX,trainY,maskX,maskY):
        with tf.GradientTape(persistent=True) as cycle_type:
            GeneratedY = self.G(trainX)
            GeneratedY = tf.multiply(GeneratedY,maskY)
            Dy_real_out = self.Dy(trainY)
            Dy_fake_out = self.Dy(GeneratedY)

            GeneratedX = self.F(trainY)
            GeneratedX = tf.multiply(GeneratedX,maskX)
            Dx_real_out = self.Dx(trainX)
            Dx_fake_out = self.Dx(GeneratedX)

            cycle_consistent_loss_X2Y = tf.reduce_mean(tf.abs(self.F(GeneratedY)-trainX))
            cycle_consistent_loss_Y2X = tf.reduce_mean(tf.abs(self.G(GeneratedX)-trainY))
            cycle_consistent = cycle_consistent_loss_X2Y+cycle_consistent_loss_Y2X

            fake_Y_perceptual = self.vgg(GeneratedY)
            real_Y_perceptual = self.vgg(trainY)
            fake_X_perceptual = self.vgg(GeneratedX)
            real_X_perceptual = self.vgg(trainX)
            reconstruction_loss_X2Y = 0
            reconstruction_loss_Y2X = 0
            for i in range(7):
                reconstruction_loss_X2Y += 0.14*tf.reduce_mean(tf.abs(fake_Y_perceptual[i]-real_Y_perceptual[i]))
                reconstruction_loss_Y2X += 0.14*tf.reduce_mean(tf.abs(fake_X_perceptual[i]-real_X_perceptual[i]))

            # reconstruction_loss_X2Y = tf.reduce_mean(tf.abs(GeneratedY-trainY))
            # reconstruction_loss_Y2X = tf.reduce_mean(tf.abs(GeneratedX-trainX))

            Dy_loss = self.gan_loss.DiscriminatorLoss(Dy_real_out,Dy_fake_out)
            Dx_loss = self.gan_loss.DiscriminatorLoss(Dx_real_out,Dx_fake_out)
            G_loss = self.gan_loss.GeneratorLoss(Dy_fake_out)+10.0*(cycle_consistent)+reconstruction_loss_X2Y
            F_loss = self.gan_loss.GeneratorLoss(Dx_fake_out)+10.0*(cycle_consistent)+reconstruction_loss_Y2X

        gradients_of_Dy = cycle_type.gradient(Dy_loss,self.Dy.trainable_variables)
        gradients_of_Dx = cycle_type.gradient(Dx_loss,self.Dx.trainable_variables)
        gradients_of_G = cycle_type.gradient(G_loss,self.G.trainable_variables)
        gradients_of_F = cycle_type.gradient(F_loss,self.F.trainable_variables)
        self.Dy_optimizer.apply_gradients(zip(gradients_of_Dy,self.Dy.trainable_variables))
        self.Dx_optimizer.apply_gradients(zip(gradients_of_Dx,self.Dx.trainable_variables))
        self.G_optimizer.apply_gradients(zip(gradients_of_G,self.G.trainable_variables))
        self.F_optimizer.apply_gradients(zip(gradients_of_F,self.F.trainable_variables))
        return G_loss,Dy_loss,F_loss,Dx_loss
    def train(self,epoches):
        self.ckpt.restore(self.manager.latest_checkpoint)
        my_step=int(self.ckpt.step)
        stop_flag = 0
        for _ in range(epoches):
            start = time.time()
            for trainX,trainY,maskX,maskY in self.train_set:
                my_step +=1
                self.ckpt.step.assign_add(1)
                step = int(self.ckpt.step)

                ###必要的超参数、变化的学习率都定义在这里 
                self.l = 10.0*(1.0/(step*0.1+1.0))
                self.wgp_shape = [trainY.shape[0],1,1,1,1] #3D 多一个通道
                ###
                if self.loss_name in ["WGAN","WGAN-GP"]:
                    for __ in range(1):
                        Dy_loss,Dx_loss = self.train_step_D(trainX,trainY,maskX,maskY)
                    for __ in range(1):
                        G_loss,F_loss = self.train_step_G(trainX,trainY,maskX,maskY)
                elif self.loss_name in ["Vanilla","LSGAN"]:
                    G_loss,Dy_loss,F_loss,Dx_loss = self.train_step(trainX,trainY,maskX,maskY)
                else:
                    raise ValueError("Inner Error")
                
                if step % 100 == 0:
                    save_path = self.manager.save()
                    print("Saved checkpoint for step {}: {}".format(step,save_path))
                    
                    self.G.save_weights(self.tmp_path+'/weights_saved/G.ckpt')
                    self.Dy.save_weights(self.tmp_path+'/weights_saved/Dy.ckpt')
                    self.F.save_weights(self.tmp_path+'/weights_saved/F.ckpt')
                    self.Dx.save_weights(self.tmp_path+'/weights_saved/Dx.ckpt')
                    
                    self.wirte_summary(step=step,
                                       seed=self.seed,
                                       G=self.G,
                                       F=self.F,
                                       G_loss=G_loss,
                                       Dy_loss=Dy_loss,
                                       F_loss=F_loss,
                                       Dx_loss=Dx_loss,
                                       out_path=self.out_path)

                    print ('Time to next 100 step {} is {} sec'.format(step,time.time()-start))
                    start = time.time()
                if step == 80200:
                    stop_flag = 1
                    break
            if stop_flag == 1:
                break
    def get_seed(self):
        seed_get = iter(self.test_set)
        testX,testY,maskX,maskY = next(seed_get)
        print(testX.shape,testY.dtype,maskX.dtype,maskY.shape)
        plt.imshow(testX[0,:,:,1,0],cmap='gray')
        plt.show()
        plt.imshow(testY[0,:,:,1,0],cmap='gray')
        plt.show()
        plt.imshow(maskX[0,:,:,1,0],cmap='gray')
        plt.show()
        plt.imshow(maskY[0,:,:,1,0],cmap='gray')
        plt.show()
        self.seed = testX,testY,maskX,maskY

    def wirte_summary(self,step,seed,G,F,G_loss,Dy_loss,F_loss,Dx_loss,out_path):
        testX,testY,maskX,maskY = seed
        GeneratedY = G(testX)
        GeneratedY = tf.multiply(GeneratedY,maskX)
        GeneratedX = F(testY)
        GeneratedX = tf.multiply(GeneratedX,maskY)
        testX=testX[:,:,:,1,:]
        testY=testY[:,:,:,1,:]
        maskX=maskX[:,:,:,1,:]
        maskY=maskY[:,:,:,1,:]
        GeneratedY=GeneratedY[:,:,:,1,:]
        GeneratedX=GeneratedX[:,:,:,1,:]
        plt.figure(figsize=(5,5))#图片大一点才可以承载像素
        plt.subplot(2,2,1)
        plt.title('real X')
        plt.imshow(testX[0,:,:,0],cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.title('fake Y')
        plt.imshow(GeneratedY[0,:,:,0],cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.title('fake X')
        plt.imshow(GeneratedX[0,:,:,0],cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.title('real Y')
        plt.imshow(testY[0,:,:,0],cmap='gray')
        plt.axis('off')
        plt.savefig(out_path+'/image_at_{}.png'.format(step))
        plt.close()
        img = Image.open(out_path+'/image_at_{}.png'.format(step))
        img = tf.reshape(np.array(img),shape=(1,500,500,4))

        with self.train_summary_writer.as_default():
            ##########################
            self.m_psnr_X2Y(tf.image.psnr(GeneratedY,testY,1.0,name=None))
            self.m_psnr_Y2X(tf.image.psnr(GeneratedX,testX,1.0,name=None)) 
            self.m_ssim_X2Y(tf.image.ssim(GeneratedY,testY,1, filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03)) 
            self.m_ssim_Y2X(tf.image.ssim(GeneratedX,testX,1, filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03)) 
            tf.summary.scalar('G_loss',G_loss,step=step)
            tf.summary.scalar('Dy_loss',Dy_loss,step=step)
            tf.summary.scalar('F_loss',F_loss,step=step)
            tf.summary.scalar('Dx_loss',Dx_loss,step=step)
            tf.summary.scalar('test_psnr_y', self.m_psnr_X2Y.result(), step=step) 
            tf.summary.scalar('test_psnr_x', self.m_psnr_Y2X.result(), step=step)
            tf.summary.scalar('test_ssim_y', self.m_ssim_X2Y.result(), step=step) 
            tf.summary.scalar('test_ssim_x', self.m_ssim_Y2X.result(), step=step)  
            tf.summary.image("img",img,step=step)

        ##########################
        self.m_psnr_X2Y.reset_states()
        self.m_psnr_Y2X.reset_states()
        self.m_ssim_X2Y.reset_states()
        self.m_ssim_Y2X.reset_states()

    def test(self):
        buf = self.manager.latest_checkpoint
        buf = buf[:-3]
        for index,temp_point in enumerate(["800","801","802"]):
            self.ckpt.restore(buf+temp_point)
            step = 0
            out_path = self.out_path+"/test"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            result_buf = []
            for i,(testX,testY,maskX,maskY) in enumerate(self.test_set):
                
                GeneratedY = self.G(testX)
                GeneratedY = tf.multiply(GeneratedY,maskX)
                GeneratedX = self.F(testY)
                GeneratedX = tf.multiply(GeneratedX,maskY)#测试时mask正好相反 因为只知道原来模态和原来模态的mask
                
                testX=testX[:,:,:,1,:]
                testY=testY[:,:,:,1,:]
                maskX=maskX[:,:,:,1,:]
                maskY=maskY[:,:,:,1,:]
                GeneratedY=GeneratedY[:,:,:,1,:]
                GeneratedX=GeneratedX[:,:,:,1,:]

                ABS_Y = tf.abs(GeneratedY-testY)
                ABS_X = tf.abs(GeneratedX-testX)
                black_board_rX = testX[0,:,:,0]
                black_board_Y = GeneratedY[0,:,:,0]
                black_board_absY = ABS_Y[0,:,:,0]
                black_board_X = GeneratedX[0,:,:,0]
                black_board_rY = testY[0,:,:,0]
                black_board_absX = ABS_X[0,:,:,0]
                step = i+1
                plt.figure(figsize=(10,10))#图片大一点才可以承载像素
                plt.subplot(3,2,1)
                plt.title('real X')
                plt.imshow(black_board_rX,cmap='gray')
                plt.axis('off')
                plt.subplot(3,2,2)
                plt.title('fake Y')
                plt.imshow(black_board_Y,cmap='gray')
                plt.axis('off')
                plt.subplot(3,2,3)
                plt.title('ABS Y')
                plt.imshow(black_board_absY,cmap='hot')
                plt.axis('off')
                plt.subplot(3,2,4)
                plt.title('ABS X')
                plt.imshow(black_board_absX,cmap='hot')
                plt.axis('off')
                plt.subplot(3,2,5)
                plt.title('fake X')
                plt.imshow(black_board_X,cmap='gray')
                plt.axis('off')
                plt.subplot(3,2,6)
                plt.title('real Y')
                plt.imshow(black_board_rY,cmap='gray')
                plt.axis('off')
                plt.savefig(out_path+'/image_at_{}.png'.format(step))
                plt.close()
                img = Image.open(out_path+'/image_at_{}.png'.format(step))
                img = tf.reshape(np.array(img),shape=(1,1000,1000,4))
                if (i+1) == 1:
                    np.save(out_path+"/out_Y.npy",black_board_Y)
                    np.save(out_path+"/out_X.npy",black_board_X)
                with self.train_summary_writer.as_default():
                ##########################
                    black_board_Y = tf.reshape(tf.constant(black_board_Y,dtype=tf.float32),shape=[1,128,128,1])
                    black_board_X = tf.reshape(tf.constant(black_board_X,dtype=tf.float32),shape=[1,128,128,1])
                    black_board_rY = tf.reshape(tf.constant(black_board_rY,dtype=tf.float32),shape=[1,128,128,1])
                    black_board_rX = tf.reshape(tf.constant(black_board_rX,dtype=tf.float32),shape=[1,128,128,1])
                    self.m_psnr_X2Y(tf.image.psnr(black_board_Y,black_board_rY,1.0,name=None))
                    self.m_psnr_Y2X(tf.image.psnr(black_board_X,black_board_rX,1.0,name=None)) 
                    self.m_ssim_X2Y(tf.image.ssim(black_board_Y,black_board_rY,1, filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03)) 
                    self.m_ssim_Y2X(tf.image.ssim(black_board_X,black_board_rX,1, filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03)) 
                    tf.summary.scalar('test_psnr_y', self.m_psnr_X2Y.result(), step=step) 
                    tf.summary.scalar('test_psnr_x', self.m_psnr_Y2X.result(), step=step)
                    tf.summary.scalar('test_ssim_y', self.m_ssim_X2Y.result(), step=step) 
                    tf.summary.scalar('test_ssim_x', self.m_ssim_Y2X.result(), step=step)  
                    tf.summary.image("img",img,step=step)
                    dx1,dy1 = tf.image.image_gradients(black_board_Y)
                    dx2,dy2 = tf.image.image_gradients(black_board_rY) 
                    dx_mean = tf.reduce_mean(tf.math.abs(dx1-dx2))
                    dy_mean = tf.reduce_mean(tf.math.abs(dy1-dy2))
                    IG_y = dy_mean+dx_mean
                    tf.summary.scalar('IG_y',IG_y, step=step)
                    dx1,dy1 = tf.image.image_gradients(black_board_X)
                    dx2,dy2 = tf.image.image_gradients(black_board_rX) 
                    dx_mean = tf.reduce_mean(tf.math.abs(dx1-dx2))
                    dy_mean = tf.reduce_mean(tf.math.abs(dy1-dy2))
                    IG_x = dy_mean+dx_mean
                    tf.summary.scalar('IG_x',IG_x, step=step)
                    tf.summary.image("img",img,step=step)
                    result_buf.append([i+1,self.m_psnr_X2Y.result().numpy(),self.m_psnr_Y2X.result().numpy(),
                                            self.m_ssim_X2Y.result().numpy(),self.m_ssim_Y2X.result().numpy(),
                                            IG_y.numpy(),IG_x.numpy()])
                ##########################
                self.m_psnr_X2Y.reset_states()
                self.m_psnr_Y2X.reset_states()
                self.m_ssim_X2Y.reset_states()
                self.m_ssim_Y2X.reset_states()

            import csv
            headers = ['Instance','test_psnr_y','test_psnr_x','test_ssim_y','test_ssim_x','IG_y','IG_x']
            rows = result_buf
            with open(out_path+'/result'+str(index)+'.csv','w')as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                f_csv.writerows(rows)
        
if __name__ == "__main__":
    a = 10
    b = float(a)
    print(a,b)
        
    