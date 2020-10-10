import os, sys
import SimpleITK as sitk
import keras
import tensorflow as tf
import numpy as np
np.random.seed(2016)
from keras.preprocessing.image import img_to_array
import random as rn
rn.seed(12345)
from keras import backend as K
from imutils import paths
import cv2
import scipy
import os
import glob
import math
import pickle
import datetime
from keras.layers.core import Reshape
from keras.layers.recurrent import LSTM,GRU
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import vgg16 as keras_vgg16
from keras.layers import BatchNormalization
from keras.models import Model
from sklearn import preprocessing
import efficientnet.keras as efn
img_width, img_height = 224,224
import gc
import math
import pandas as pd
import shutil
from test_crop import generate_mask, generate_bounding_box, dcm2png8bits
from keras.models import load_model
img_rows, img_cols, img_channel = 224, 224, 3


test_path = "/path_to_testdata/"
crop_path = "cropped/"
mask_path = "mask/"
png_path =  "png_dir/"

if not os.path.isdir(png_path):
    os.mkdir(png_path)

imagePaths = (test_path)
Files=os.listdir(imagePaths)
num = len(Files)
print("Info: Num_Inputs =:", num)


dcm2png8bits(test_path, png_path)

generate_mask(png_path, num, mask_path)

generate_bounding_box(png_path,
                       mask_path,
                      'bounding_box_covid.csv', crop_path)

Cropped=os.listdir(crop_path)
size = len(Cropped)
print("Info: Size_Cropped =:", size)

name=[]
data=[]
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(crop_path)))
# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    _,tail = os.path.split (imagePath)
    name.append(tail)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_rows, img_rows))
    image= image.astype('float32')
    image =  cv2.normalize(image,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX)

    image = img_to_array(image)
    data.append(image)

data = np.array(data)


model = load_model("res2-resnet-cleanraw-weights2-improvement-100-0.97.hdf5")
intermediate_output= np.zeros((size, 1), dtype=np.float16)
intermediate_output = model.predict(data)
covid= np.zeros((size, 1), dtype=np.float16)
for i in range(0,size):
   if intermediate_output[i] <= 0.5:
       covid[i]= 1
   else:
       covid[i]= 0

npa = np.asarray(name)
npa2 =npa.reshape(-1,1)
final = np.concatenate((npa2,intermediate_output,covid),axis=1)
df =pd.DataFrame(final)
df.columns =['Image_name','Score','COVID-19(1: POSITIVE, 0: NEGATIVE)']
df.to_csv('predictions.csv')
