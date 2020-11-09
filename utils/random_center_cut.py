path = "E:\\Datasets\\BraTs\\ToCrop\\MICCAI_BraTS2020_TrainingData\\Training_001\\Training_001_t1.png"
from PIL import Image
import numpy as np
img = Image.open(path)
img = np.array(img)
import matplotlib.pyplot as plt  
plt.imshow(img,cmap="gray")
plt.show()

randbuf=[]
import random

x = 239//2
y = 239//2
x_l =x-7
x_r = x+8+1
y_l =  y-(68//2 - 1)
y_r =  y+(68//2 + 1)
for i in range(15):

    
    a = random.randint(x_l,x_r)
    b = random.randint(y_l,y_r)
    tmp_img = img[a-63:a+64+1,b-63:b+64+1]
    randbuf.append(tmp_img)

for item in randbuf:
    plt.imshow(item,cmap="gray")
    plt.show()
