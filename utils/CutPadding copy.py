"""
工具类
其实里面都只是数值计算函数
"""
import numpy as np 
import random
def crop_cal(crop_dims,crop_nums,crop_width):
    """
    随机采样的两种算法
    第一个算法需要运气 即很好的取得随机值 但是会不一定取得到
    第二个算法需要算力 可取点太多时 无法实现
    
    """
def random_crop_cal_3D_better(mask,crop_width_list=[32,32,32],target_nums=27):
    """
    以n维图形的某个点为中心点 
    构建n维举行
    为了避免取到重复的点而反复随机 牺牲存储空间 维护一个可取点列表
    取出一个点后
    依据条件更新可取点列表 即删除列表中不可取的点
    """
    assert mask.shape == (240,240,155)
    assert mask.dtype == np.int16
    
    min_distance = 1.0*((16**2)+(16**2)+(16**2))
    points_count = 32*32*32

    points_buf = []
    for i in range(0+15,240-17):
        for j in range(0+15,240-17):
            for k in range(0+15,155-17):
                if mask[i,j,k]==0:
                    pass
                else:
                    points_buf.append([i,j,k])
    result_buf = []
    for _ in range(target_nums):
        index =  random.randint(0,len(points_buf)-1)
        i = points_buf[index][0]
        j = points_buf[index][1]
        k = points_buf[index][2]
        patch = mask[i-15:i+17,j-15:j+17,k-15:k+17]
        conut_1 = sum((patch.flatten())==1)
        rate_1 = conut_1/points_count
        if rate_1 >=0.95:
            result_buf.append(points_buf[index])
        else:
            del points_buf[index]
            continue
        for item in points_buf:
            temp = 1.0*((item[0]-result_buf[-1][0])**2+(item[1]-result_buf[-1][1])**2+(item[2]-result_buf[-1][2])**2)
            if temp<=min_distance:
                points_buf.remove(item)
    return result_buf
def random_crop_cal_3D(mask,crop_width_list=[32,32,32],target_nums=27):
    """
    不同于简单的随机剪裁
    我需要给定维度和剪裁后宽度后 得到若干坐标区间
    使得剪裁结果可以大致分布均匀同时具有随机性
    为了避免毫无意义的空剪裁，确保区间中心在总体的mask内
    只剪裁出偶数长度的区间
    因此以第dim_length//2个点为中心点
    向前驱(dim_length//2)-1个点  后继im_length//2个点 作为一个剪裁出的区间
    确保剪裁区间内 有效体素值占比达到90% 95%
    先确定若干合适的中心点(分布在有效区域，满足有效区域占比大于95%,且和已取得中心点距离之间存在一半宽度差距)
    这种方法极容易进入近乎死循环 可以考虑基于
    """
    assert mask.shape == (240,240,155)
    assert mask.dtype == np.int16
    points_buf = []
    min_distance = 1.0*((16**2)+(16**2)+(16**2))
    points_count = 32*32*32
    while(len(points_buf)<target_nums):
        i = random.randint(0+15,240-1-17)
        j = random.randint(0+15,240-1-17)
        k = random.randint(0+15,155-1-17)
        if mask[i,j,k]==0:
            continue
        if len(points_buf)==0:
            tmp = mask[i-15:i+17,j-15:j+17,k-15:k+17]
            conut_1 = sum((tmp.flatten())==1)
            rate_1 = conut_1/points_count
            if rate_1 >=0.95:
                points_buf.append([i,j,k])
            else:
                continue
        else:
            refind_flag = 0
            for item in points_buf:
                distence = 1.0*((item[0]-i)**2+(item[1]-j)**2+(item[2]-k)**2)
                if distence < min_distance:
                    refind_flag = 1
                    break
            if refind_flag == 1:
                refind_flag = 0
                continue
            else:
                tmp = mask[i-15:i+17,j-15:j+17,k-15:k+17]
                conut_1 = sum((tmp.flatten())==1)
                rate_1 = conut_1/points_count
                if rate_1 >=0.95:
                    points_buf.append([i,j,k])
                else:
                    continue
    return points_buf
def gen_hole_mask(img,pix_val=1):
    shape = img.shape
    dtype = img.dtype
    #部分样本不是int16 类型 强制mask为int16
    new_mask_img = img[:,:,:]
    new_mask_img[new_mask_img!=pix_val]=0
    new_mask_img[new_mask_img==pix_val]=1
    return new_mask_img
def gen_mask(img):
    shape = img.shape
    dtype = img.dtype
    #部分样本不是int16 类型 强制mask为int16
    new_mask_img = np.ones(shape=shape,dtype=np.int16)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i,j,k]<=0:
                    new_mask_img[i,j,k]=0
                else:
                    break
            for k in range(img.shape[2]-1,-1,-1):
                if img[i,j,k]<=0:
                    new_mask_img[i,j,k]=0
                else:
                    break
    return new_mask_img
def cut_img_3D(img):
    """
    将3D的数据进行剪裁，去除3D黑边
    img:numpy array
    return a cut img with same axis number but not a fixed one
    """
    # print(img.shape,img.dtype)
    buf=[]
    for i in range(img.shape[0]):
        temp = img[i,:,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[0]-1,-1,-1):
        temp = img[i,:,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[1]):
        temp = img[:,i,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[1]-1,-1,-1):
        temp = img[:,i,:]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[2]):
        temp = img[:,:,i]
        if(temp.sum()!=0):
            buf.append(i)
            break
    for i in range(img.shape[2]-1,-1,-1):
        temp = img[:,:,i]
        if(temp.sum()!=0):
            buf.append(i)
            break
    pw=1 # plus_width 前后增加的额外像素 防止3D图像缺失一小部分
    for i in range(3):
        if buf[2*i]-pw>=0:
            buf[2*i] -= pw
    for i in range(3):
        if buf[2*i+1]+pw<=(img.shape[i]-1):
            buf[2*i+1] += pw
    cut_img = img[buf[0]:buf[1]+1,buf[2]:buf[3]+1,buf[4]:buf[5]+1]
    # print(cut_img.shape) # buf 记录的是坐标下标 自身不涉及index+1 -1 认为考虑+-1
    max_length = max(cut_img.shape)
    zeros = np.zeros(shape=[1,cut_img.shape[1],cut_img.shape[2]],dtype=np.int16)
    letf_layers = max_length - cut_img.shape[0]
    for i in range(letf_layers//2):
        cut_img = np.concatenate((zeros,cut_img),axis=0)
    for i in range(letf_layers-letf_layers//2):
        cut_img = np.concatenate((cut_img,zeros),axis=0)
    # print(cut_img.shape)
    zeros = np.zeros(shape=[cut_img.shape[0],1,cut_img.shape[2]],dtype=np.int16)
    letf_layers = max_length - cut_img.shape[1]
    for i in range(letf_layers//2):
        cut_img = np.concatenate((zeros,cut_img),axis=1)
    for i in range(letf_layers-letf_layers//2):
        cut_img = np.concatenate((cut_img,zeros),axis=1)
    # print(cut_img.shape)
    zeros = np.zeros(shape=[cut_img.shape[0],cut_img.shape[1],1],dtype=np.int16)
    letf_layers = max_length - cut_img.shape[2]
    for i in range(letf_layers//2):
        cut_img = np.concatenate((zeros,cut_img),axis=2)
    for i in range(letf_layers-letf_layers//2):
        cut_img = np.concatenate((cut_img,zeros),axis=2)
    # print(cut_img.shape)
    # print(cut_img.min())
    # print(cut_img.max())
    return cut_img
def center_cut_3D(img,target_size=[160,160,128]):
    #img numpy array
    shape = img.shape
    range_buf = []
    for i in range(3):
        if shape[i]<target_size[i]:
            raise ValueError("target size in big than original size")
        tmp = shape[i]-target_size[i]
        cut_begin = tmp//2
        cut_end = tmp-cut_begin
        range_buf.append([cut_begin,cut_end])#极右原则 需要剪裁的是偶数时 左右均分 奇数时  末端剪裁更多
    
    cut_img = img[range_buf[0][0]:-range_buf[0][1],range_buf[1][0]:-range_buf[1][1],range_buf[2][0]:-range_buf[2][1]] 
    assert cut_img.shape == tuple(target_size)
    return cut_img

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pylab as plt
    import nibabel as nib
    from nibabel import nifti1
    from nibabel.viewers import OrthoSlicer3D

    # example_filename = 'E:\\Datasets\\BraTs\\ToCrop\\MICCAI_BraTS2020_TrainingData\\Training_001\\Training_001_t1.nii'

    # img0 = nib.load(example_filename)
    # img = np.array(img0.dataobj[:,:,:])
    # cut_img = gen_mask(img)
    # print(cut_img.shape,cut_img.dtype)
    # nib.save(img0, 'E:\\Datasets\\BraTs\\ToCrop\\MICCAI_BraTS2020_TrainingData\\Training_001\\test.nii')

    # import nibabel as nib
 
    # img1 = nib.load('E:\\Datasets\\BraTs\\ToCrop\\MICCAI_BraTS2020_TrainingData\\Training_001\\test.nii')

    # data = img1.get_data()
    # affine = img1.affine
    
    # print(img1)
    
    # nib.save(img1, 'E:\\Datasets\\BraTs\\ToCrop\\MICCAI_BraTS2020_TrainingData\\Training_001\\test1.nii')
    
    # new_image = nib.Nifti1Image(img, affine)
    # nib.save(new_image, "E:\\Datasets\\BraTs\\ToCrop\\MICCAI_BraTS2020_TrainingData\\Training_001\\test2.nii")
    # data = cut_img
    # affine = img0.affine
    # new_image = nib.Nifti1Image(data, affine)
    # nib.save(new_image, "E:\\Datasets\\BraTs\\ToCrop\\MICCAI_BraTS2020_TrainingData\\Training_001\\test2.nii")

    import os
    buf_A = []
    buf_B = []
    for (dirName, subdirList, fileList) in os.walk("E:\\Datasets\\BraTs\\ToCrop"):
        for filename in fileList:
            if "t1.nii" in filename.lower():  
                buf_A.append(os.path.join(dirName,filename))
            if "t2.nii" in filename.lower(): 
                buf_B.append(os.path.join(dirName,filename))
    for i,item in enumerate(buf_A):
        save_path =item[:-6]+"mask_t1_v0.nii"
        img0 = nib.load(item)
        img = np.array(img0.dataobj[:,:,:],dtype=np.int16)
        mask = gen_mask(img)
        print(i+1,mask.shape,mask.dtype)
        data = mask
        affine = img0.affine
        new_image = nib.Nifti1Image(data, affine)
        nib.save(new_image,save_path)
    for i,item in enumerate(buf_B):
        save_path =item[:-6]+"mask_t2_v0.nii"
        img0 = nib.load(item)
        img = np.array(img0.dataobj[:,:,:],dtype=np.int16)
        mask = gen_mask(img)
        print(i+1,mask.shape,mask.dtype)
        data = mask
        affine = img0.affine
        new_image = nib.Nifti1Image(data, affine)
        nib.save(new_image,save_path)
    # import random
    # for _ in range(100):
    #     print(random.randint(0,9))
        
    # import numpy as np
    # from collections import  Counter
    # data = np.array([1.1,2,3,4,4,5])
    # # Counter(data)  # {label:sum(label)}
    
    # #简单方法
    # print()
    # print(sum(data==4))
    # print(1.0**2)
                            # import os
                            # buf_seg = []
                            # for (dirName, subdirList, fileList) in os.walk("E:\\Datasets\\BraTs\\ToCrop"):
                            #     for filename in fileList:
                            #         if "seg.nii" in filename.lower():  
                            #             buf_seg.append(os.path.join(dirName,filename))
                            # for i,item in enumerate(buf_seg):
                            #     img0 = nib.load(item)
                            #     img = np.array(img0.dataobj[:,:,:])
                            #     save_path =item[:-7]+"mask_v1.nii"
                            #     mask = gen_hole_mask(img,pix_val=1)
                            #     print(i+1,mask.shape,mask.dtype)
                            #     data = mask
                            #     affine = img0.affine
                            #     new_image = nib.Nifti1Image(data, affine)
                            #     nib.save(new_image,save_path)

                            #     img = np.array(img0.dataobj[:,:,:])
                            #     save_path =item[:-7]+"mask_v2.nii"
                            #     mask = gen_hole_mask(img,pix_val=2)
                            #     print(i+1,mask.shape,mask.dtype)
                            #     data = mask
                            #     affine = img0.affine
                            #     new_image = nib.Nifti1Image(data, affine)
                            #     nib.save(new_image,save_path)

                            #     img = np.array(img0.dataobj[:,:,:])
                            #     save_path =item[:-7]+"mask_v4.nii"
                            #     mask = gen_hole_mask(img,pix_val=4)
                            #     print(i+1,mask.shape,mask.dtype)
                            #     data = mask
                            #     affine = img0.affine
                            #     new_image = nib.Nifti1Image(data, affine)
                            #     nib.save(new_image,save_path)
        # break
    # a = [1,2,3,4,5]
    # print(a[0:-1])
    


