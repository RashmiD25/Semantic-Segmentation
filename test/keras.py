import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import cv2
import albumentations as A

import time
import os
from tqdm.notebook import tqdm

!pip install -q segmentation-models-pytorch
!pip install -q torchsummary

from torchsummary import summary
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_PATH = 'hackathon2022/data/train/images/'
MASK_PATH = 'hackathon2022/data/train/labels/'
img_list = os.listdir(img_path)
for i in range(10):
    img = cv2.imread(IMAGE_PATH+img_list[i])
    label = cv2.imread(MASK_PATH+img_list[i][:-4]+'.png')
    label = label[:,:,0]
    plt.imshow(img)
    plt.show()
    plt.imshow(label)
    plt.show()

    
n_classes = 3 

def create_df():
    name = []
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df = create_df()
print('Total Images: ', len(df))

# Total Images:  161

#split data
X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=19)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

print('Train Size   : ', len(X_train))
print('Val Size     : ', len(X_val))
print('Test Size    : ', len(X_test))

# Train Size   :  122
# Val Size     :  22
# Test Size    :  17

img = Image.open(IMAGE_PATH + df['id'][100] + '.tif')
mask = Image.open(MASK_PATH + df['id'][100] + '.png')
print('Image Size', np.asarray(img).shape)
print('Mask Size', np.asarray(mask).shape)


plt.imshow(img)
plt.imshow(mask, alpha=0.6)
plt.title('Picture with Mask Appplied')
plt.show()

# Image Size (1040, 1392)
# Mask Size (1040, 1392)
