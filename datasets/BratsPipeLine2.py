"""
针对brats数据集
做包含预处理的数据管道(Python生成器)

每次优先读取npy 不存在则读取nii 同时保存npy

读取240*240 指定155中部切片
或者读取240*240*155 3D

以2D为例，进行中心附近剪裁，单个240*240切片 中心有效区域为196*144
在这个区域内选取若然随机的128*128切片 进行训练 确保训练的数据是有监督的 对其的
切片范围为中心的196-128 和144-128  即 68*16范围内去随机值

"""

import os 
import sys
from PIL import Image
import numpy as np 
import nibabel as nib
from scipy import ndimage
import random
base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from utils import CutPadding
import random
class UnpairedError(Exception):
    def __init__(self,path):
        self.err_msg = "There are not exixting paired samples! We can only find:"
        self.filename = path
class DataPipeLine():
    def __init__(self,path,target_size,patch_size,remake_flag=False,random_flag=False,test_flag=False):
        self.path = path
        self.datalist = self.__readDirFile(self.path,random_flag)
        self.target_size = target_size
        self.patch_size = patch_size
        self.dims = len(target_size)
        self.remake_flag = remake_flag
        self.random_flag = random_flag
        self.test_flag = test_flag
    def __readDirFile(self,path,random_flag=False):
        buf_A = []
        buf_B = []
        buf_A_mask_v0 = []
        buf_B_mask_v0 = []
        for (dirName, subdirList, fileList) in os.walk(path):
            try:
                for filename in fileList:
                    if "t1.nii" in filename.lower():  
                        buf_A.append(os.path.join(dirName,filename))
                    if "t2.nii" in filename.lower(): 
                        buf_B.append(os.path.join(dirName,filename))
                    if "mask_t1_v0.nii" in filename.lower():  
                        buf_A_mask_v0.append(os.path.join(dirName,filename))
                    if "mask_t2_v0.nii" in filename.lower(): 
                        buf_B_mask_v0.append(os.path.join(dirName,filename))
                if len(buf_A) > len(buf_B):
                    raise UnpairedError(buf_A.pop(-1))
                elif len(buf_A) < len(buf_B):
                    raise UnpairedError(buf_B.pop(-1))
                else:
                    pass
            except UnpairedError as error:
                print(error.err_msg)
                print(error.filename)
            else:# normal condition
                pass
            finally:# any way
                pass
        if random_flag:
            random.seed(1)
            random.shuffle(buf_A)
            random.seed(1)
            random.shuffle(buf_B)
            random.seed(1)
            random.shuffle(buf_A_mask_v0)
            random.seed(1)
            random.shuffle(buf_B_mask_v0)
            return list(zip(buf_A,buf_B,buf_A_mask_v0,buf_B_mask_v0))
        else:
            return list(zip(buf_A,buf_B,buf_A_mask_v0,buf_B_mask_v0))
    def read_file(self,path):
        if self.dims == 3:
            temp_path = path[:-3]+"npy"
            if (os.path.exists(temp_path)==True)and(self.remake_flag==False):
                return np.load(temp_path)
            else:
                return self.load_nii_file(path)
        elif self.dims == 2:
            temp_path = path[:-3]+"2D.npy"
            if (os.path.exists(temp_path)==True)and(self.remake_flag==False):
                return np.load(temp_path)
            else:
                return self.load_nii_file(path)
        else:
            raise ValueError
    def __read_nii_file(self,path):
        img = nib.load(path)
        img = np.array(img.dataobj[:,:,:])
        return img
    def __cut_nii_file(self,img):
        return CutPadding.cut_img_3D(img)
    def __save_nii2npy(self,img,path):
        if self.dims == 3:
            temp_path = path[:-3]+"npy"
        elif self.dims ==2:
            temp_path = path[:-3]+"2D.npy"
        else:
            raise ValueError
        np.save(temp_path,img)
        return img
    def __cut_np_array(self,array,target_shape=[128,128,128]):
        old_shape = array.shape
        buf = [0,0,0]
        for i in range(3):
            buf[i]=old_shape[i]//2-target_shape[i]//2
            #左半部右下标+1 减去目标点数的一半 获得新的起始点 10//2 -6//2 = 2 从下标2开始然后到下标2+6-1结束
        return array[buf[0]:buf[0]+target_shape[0],buf[1]:buf[1]+target_shape[1],buf[2]:buf[2]+target_shape[2]]            
    def __normalize(self,slice,dtype=np.float32):
        tmp = slice/slice.max()
        return tmp.astype(dtype)
    def get_test_ranges(self):
        """
        3D暂时不考虑
        选取若干128*128的区域,进行非随机的crop输出
        有效区域参数为196*144时
        x从48到下标191
        y从22到下标217
        选取x:48--175 64--191区域
        选取y:22--149 90--127区域
        """
        ranges_buf = []
        ranges_buf.append([[48,175+1],[22,149+1]])
        ranges_buf.append([[48,175+1],[90,217+1]])
        ranges_buf.append([[64,191+1],[22,149+1]])
        ranges_buf.append([[64,191+1],[90,217+1]])
        
        return ranges_buf
    def get_crop_ranges(self):
        """
        针对图像中心区域的随机剪裁
        默认参数是240*240 || 240*240*155尺寸
        有效区域参数为196*144 || 196*144*140尺寸
        3D的先写着 但是不一定最后用该参数
        """
        ranges_buf = [] 
        shift=3
        if self.dims == 3:
            x = (240-1)//2
            y = (240-1)//2
            z = 155//2
            x_region = 144
            y_region = 196
            z_region = 140
            x_range = x_region-self.patch_size[0]
            y_range = y_region-self.patch_size[1]
            z_range = z_region-self.patch_size[2]
            assert (x_range%2==0)and(y_range%2==0)and(z_range%2==0)
            a_half = self.patch_size[0]//2#最后得到的区间
            b_half = self.patch_size[1]//2
            c_half = self.patch_size[2]//2
            x_l = x-((x_range//2)-1)
            x_r = x+(x_range//2)#表示坐标的上界(可以取到) 不应该+1
            y_l = y-((y_range//2)-1)
            y_r = y+(y_range//2)
            z_l = z-((z_range//2)-1)
            z_r = z+(z_range//2)
            for _ in range(4):
                a = random.randint(x_l,x_r)
                b = random.randint(y_l,y_r)
                c = random.randint(z_l,z_r)
                ranges_buf.append([[a-(a_half-1),a+(a_half+1)],
                                   [b-(b_half-1),b+(b_half+1)],
                                   [c-(c_half-1),c+(c_half+1)]])#3D暂时不进行shift
        elif self.dims == 2:
            x = (240-1)//2
            y = (240-1)//2
            x_region = 144
            y_region = 196
            x_range = x_region-self.patch_size[0]
            y_range = y_region-self.patch_size[1]
            assert (x_range%2==0)and(y_range%2==0)
            a_half = self.patch_size[0]//2#最后得到的区间
            b_half = self.patch_size[1]//2
            x_l = x-((x_range//2)-1)
            x_r = x+(x_range//2)#表示坐标的上界(可以取到) 不应该+1
            y_l = y-((y_range//2)-1)
            y_r = y+(y_range//2)
            for _ in range(4):
                a = random.randint(x_l,x_r)
                b = random.randint(y_l,y_r)
                ranges_buf.append([[a-(a_half-1),a+(a_half+1)],
                                   [b-(b_half-1)+shift,b+(b_half+1)+shift]])#
        else:
            raise ValueError("Unsupported dims")
        return ranges_buf
    def load_nii_file(self,path):
        img = self.__read_nii_file(path)#读取3D源文件 
        # img = self.__cut_nii_file(img)#去除文件周围无意义的区域 3D去黑边
        #缩放到目标大小 最近邻插值
        if len(self.target_size)==2:
            # temp_targer_size = self.target_size[:]+[self.target_size[-1]]
            temp_targer_size = self.target_size[:]+[155]
        else:
            temp_targer_size = self.target_size[:]
        # ratio = [temp_targer_size[x]/img.shape[x] for x in range(3)]
        # resize_image = ndimage.interpolation.zoom(img,ratio, mode='nearest')
        # assert resize_image.shape==tuple(temp_targer_size)
        # resize_image[resize_image<0]=0#去除插值后出现的负像素
        resize_image = CutPadding.center_cut_3D(img=img,target_size=temp_targer_size)
        if self.dims == 3:
            resize_image = resize_image
        elif self.dims ==2:
            resize_image = resize_image[:,:,temp_targer_size[-1]//2]
        else:
            raise ValueError
        img_norm = self.__normalize(resize_image,dtype=np.float32)#归一化
        img_saved = self.__save_nii2npy(img_norm,path)#保存 并且返回保存的文件 将对2D 3D区别对待
        return img_saved
    def __iter__(self):
        #实现__iter__ 本身就是一个迭代器 但是没有call方法 不能被tensorflow from_generator识别 所以必须在实现一个一般的生成器函数
        for A,B in self.datalist:
            yield (self.read_file(A),self.read_file(B))
        return
    def generator(self):
        for A,B,A_m0,B_m0 in self.datalist:
            imgA = self.read_file(A)
            imgB = self.read_file(B)
            imgA_m0 = self.read_file(A_m0)
            imgB_m0 = self.read_file(B_m0)
            if self.dims == 3:
                if self.test_flag:
                    ranges_buf = self.get_test_ranges()
                else:
                    ranges_buf = self.get_crop_ranges()
                for item in ranges_buf:
                    buf = np.array(item,dtype=np.int32)
                    yield (imgA,imgB,imgA_m0,imgB_m0,buf)
                
            elif self.dims == 2:
                if self.test_flag:
                    ranges_buf = self.get_test_ranges()
                else:
                    ranges_buf = self.get_crop_ranges()
                for item in ranges_buf:
                    buf = np.array(item,dtype=np.int32)
                    yield (imgA,imgB,imgA_m0,imgB_m0,buf)
            else:
                raise ValueError("Unsupported dims")
        return 
    def chenk_saved_npy(self):
        #该方法直接进行一次全部迭代，将nii文件读取并且内容保存为预处理后的numpy矩阵 npy无压缩格式
        for i,(A,B) in enumerate(self):
            print(i+1,A.shape,B.dtype,
                    A.shape,B.dtype,
                    A.max(),B.min(),
                    A.max(),B.min())
    def save_png(self):
        from PIL import Image
        for i,(A,B) in enumerate(self.datalist):
            imgA = np.array(255*self.read_file(A),dtype=np.uint8)
            imgB = np.array(255*self.read_file(B),dtype=np.uint8)
            imgA = Image.fromarray(imgA)
            imgB = Image.fromarray(imgB)
            imgA.save(A[:-4]+".png")
            imgB.save(B[:-4]+".png")
            print(i)
        return

            
if __name__ == "__main__":
    # import tensorflow as tf 
    a = DataPipeLine(path="E:\\Datasets\\BraTS\\ToCrop\\",
                     target_size=[240,240],
                     patch_size=[128,128],
                     remake_flag=False,
                     random_flag=False)
    # a.chenk_saved_npy()
    import matplotlib.pyplot as plt
    for i,(A,B,A_m0,B_m0,buf) in enumerate(a.generator()):
        plt.figure(figsize=(5,5))#图片大一点才可以承载像素
        plt.subplot(2,2,1)
        plt.imshow(A,cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow(A_m0,cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(B,cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(B_m0,cmap='gray')
        plt.axis('off')
        print(buf)
        plt.show()
        
    # a = DataPipeLine("E:\\Datasets\\BraTS\\ToCrop\\",target_size=[128,128,155],remake_flag=True,random_flag=False)
    # a.chenk_saved_npy()
    # a.chenk_saved_npy()
    # dataset = tf.data.Dataset.from_generator(iter(a),output_types=tf.float32)\
    #         .batch(1)\
    #         .prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    # gen = DataPipeLine("E:\\Datasets\\BraTS\\MICCAI_BraTS2020_TrainingData",target_size=[64,64,64],update=True)
    # abc = gen.generator()
    # for i,(t1,t2) in enumerate(abc):
    #     print(i,t1.shape,t1.dtype,
    #             t2.shape,t2.dtype,
    #             t1.max(),t1.min(),
    #             t2.max(),t2.min())
