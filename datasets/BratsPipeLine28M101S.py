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
    def __init__(self,path,target_size,patch_size,remake_flag=False,random_flag=False,crop="crop_random"):
        self.path = path
        self.datalist = self.__readDirFile(self.path,random_flag)
        self.target_size = target_size
        self.patch_size = patch_size
        self.dims = len(target_size)
        self.remake_flag = remake_flag
        self.random_flag = random_flag
        self.crop = crop.lower()
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
            """
            打乱A B之间的对应关系
            """
            random_num1 = random.randint(0,200)
            random_num2 = random.randint(0,200)
            random.seed(random_num1)
            random.shuffle(buf_A)
            random.seed(random_num1)
            random.shuffle(buf_A_mask_v0)
            random.seed(random_num2)
            random.shuffle(buf_B)
            random.seed(random_num2)
            random.shuffle(buf_B_mask_v0)
            return list(zip(buf_A,buf_B,buf_A_mask_v0,buf_B_mask_v0))
        else:
            return list(zip(buf_A,buf_B,buf_A_mask_v0,buf_B_mask_v0))
    def re_rand(self):
        self.datalist = self.__readDirFile(self.path,random_flag=True)
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
    def get_centro_ranges(self,target_size,patch_size):
        ranges_buf = []
        shape = target_size
        for i in range(len(patch_size)):
            if shape[i]<patch_size[i]:
                raise ValueError("Unsupported target size")
            elif shape[i]==patch_size[i]:
                pass
            else:
                diff = shape[i]-patch_size[i]
                begin = diff//2
                end = diff-begin
                ranges_buf.append([begin,shape[i]-end])
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
        resize_image = CutPadding.center_crop_3D(img=img,target_size=temp_targer_size)
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
        length = len(self.datalist)
        for i,(A,B,A_m0,B_m0) in enumerate(self.datalist):
            imgA = self.read_file(A)
            imgB = self.read_file(B)
            imgA_m0 = self.read_file(A_m0)
            imgB_m0 = self.read_file(B_m0)
            if self.dims == 3:
                buf = None
                yield (imgA,imgB,imgA_m0,imgB_m0,buf)
            elif self.dims == 2:
                buf = None
                yield (imgA,imgB,imgA_m0,imgB_m0,buf)
            else:
                raise ValueError("Unsupported dims")
            if (i+1)==length:
                if self.random_flag:
                    self.re_rand()
                    print("lueluelue")        
        return None
    def generator(self):
        length = len(self.datalist)
        for i,(A,B,A_m0,B_m0) in enumerate(self.datalist):
            imgA = self.read_file(A)
            imgB = self.read_file(B)
            imgA_m0 = self.read_file(A_m0)
            imgB_m0 = self.read_file(B_m0)
            if self.dims == 3:
                ranges_buf = self.get_centro_ranges(target_size=self.target_size,patch_size=self.patch_size)
                slice_begin = ranges_buf[-1][0]
                slice_end = ranges_buf[-1][1]
                for slice_index in range(slice_begin,slice_end):
                    slice_imgA=imgA[:,:,slice_index]
                    slice_imgB=imgB[:,:,slice_index]
                    slice_imgA_m0=imgA_m0[:,:,slice_index]
                    slice_imgB_m0=imgB_m0[:,:,slice_index]
                    buf = np.array(ranges_buf[0:2],dtype=np.int32)
                    yield (slice_imgA,slice_imgB,slice_imgA_m0,slice_imgB_m0,buf)
            elif self.dims == 2:
                raise ValueError("Unsupported 2 dims")
            else:
                raise ValueError("Unsupported dims")
            if (i+1)==28:
                if self.random_flag:
                    self.re_rand()
                    print("lueluelue")
                break
            
        return None
    def chenk_saved_npy(self):
        #该方法直接进行一次全部迭代，将nii文件读取并且内容保存为预处理后的numpy矩阵 npy无压缩格式
        for i,(A,B,A_m0,B_m0,buf) in enumerate(self):
            print(i+1,A.shape,B.dtype,
                    B_m0.shape,A_m0.dtype,
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
    import tensorflow as tf 
    a = DataPipeLine(path="G:\\Datasets\\BraTS\\ToCrop\\",
                     target_size=[240,240,155],
                     patch_size=[128,128,101],
                     remake_flag=False,
                     random_flag=True)
    # a.chenk_saved_npy()

    import matplotlib.pyplot as plt
    for i,(A,B,A_m0,B_m0,buf) in enumerate(a.generator()):
        if i == 10:
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
    for i,(A,B,A_m0,B_m0,buf) in enumerate(a.generator()):
        if i == 10:
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
