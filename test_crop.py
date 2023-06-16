import tensorflow
from keras import backend as K
K.clear_session()
from keras.models import Model
from tensorflow.keras.layers import Input
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import Conv2D, Concatenate
from keras.layers import SeparableConv2D
from scipy.ndimage.interpolation import zoom
import statistics 
from skimage import img_as_ubyte
from skimage.segmentation import mark_boundaries
import numpy as np

#from tensorflow.keras.backend import tensorflow_backend
import keras.backend as tensorflow_backend

from keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D
import time
from scipy import interp
import cv2
import imutils
import pickle
import struct
import shutil
import numpy as np
import zlib
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import img_to_array
from keras.optimizers import Adam, SGD
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from keras.callbacks import CSVLogger
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

import itertools
from itertools import cycle
from sklearn.utils import class_weight
from keras.regularizers import l2
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pydicom
import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv
import random
from shutil import copyfile
from tensorflow.python.framework import ops
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import backend as K
from keras import applications
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
from matplotlib import pyplot as plt
from glob import glob
import skimage.io as io
import skimage.transform as trans
from PIL import Image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

#define different loss functions

def dice_loss(y_true, y_pred):
    num = 2 * np.sum(np.multiply(y_true, y_pred))
    den = np.sum(y_true) + np.sum(y_pred)
    return num/den

def bce_dice_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = np.sum(np.multiply(y_true, y_pred))
    union = np.sum(y_true+y_pred) - intersection
    return intersection/union   

#create a color dictionary
A = [128,128,128]
B = [128,0,0]
C = [192,192,128]
D = [128,64,128]
E = [60,40,222]
F = [128,128,0]
G = [192,128,128]
H = [64,64,128]
I = [64,0,128]
J = [64,64,0]
K = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([A, B, C, D, E, F, G, H, I, J, K, Unlabelled])

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],
                                        new_mask.shape[1]*new_mask.shape[2],
                                        new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1): 

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
        
def valGenerator(batch_size,val_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1): 

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        val_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        val_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    val_generator = zip(image_generator, mask_generator)
    for (img,mask) in val_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def testGenerator(test_path,target_size = (256,256),flag_multi_class = False,as_gray = True): 
    for filename in os.listdir(test_path):
        img = io.imread(os.path.join(test_path,filename),as_gray = as_gray) 
        img = img / 255.
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,test_path, flag_multi_class = False,num_class = 2):
    file_names = os.listdir(test_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]        
        #io.imsave(os.path.join(save_path,file_names[i]),img_as_uint(img))
        #print(img.shape)       
        img[img > 0.5] = 1
        img[img <= 0.5] = 0 
        #img.dtype='uint8'
        #img = cv2.resize(img,(256,256))

        img = cv2.normalize(img,None,   alpha=0,beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)        
        #print(img)
        cv2.imwrite( os.path.join(save_path, file_names[i]),img)
       

def generate_mask(test_path: str, num: int,save_path: str):
	model = load_model("unet.hdf5")
	#model.summary()

	data_gen_args = dict(rotation_range=10.,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=5,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    fill_mode='nearest') 

	testGene = testGenerator(test_path)
	results = model.predict_generator(testGene,num,verbose=1, workers=1, use_multiprocessing=False) #steps per epoch is the no. of samples in test image.
	saveResult(save_path, results, test_path)


#custom function to generate bounding boxes
def generate_bounding_box(image_dir: str, #containing images
                          mask_dir: str, #containing masks, images have same name as original images
                          dest_csv: str, crop_save_dir: str):

    if not os.path.isdir(mask_dir):
        raise ValueError("mask_dir not existed")
    if not os.path.isdir(crop_save_dir):
        os.mkdir(crop_save_dir)
    case_list = [f for f in os.listdir(mask_dir) if f.split(".")[-1] == 'png'] #all mask images are png files
    with open(dest_csv, 'w', newline='') as f:
        csv_writer = csv.writer(f)

        for j, case_name in enumerate(case_list):
            mask = cv2.imread(mask_dir + case_name)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            print(image_dir + case_name)
            image = cv2.imread(image_dir + case_name, cv2.COLOR_BGR2GRAY)
            #print(image.shape[0],image.shape[1])
            mask = cv2.resize(mask, (image.shape[1],image.shape[0])) #original images are resized to 256 x 256
            #cv2.imwrite(mask_dir_res + case_name, mask)
            if mask is None or image is None:
                raise ValueError("The image can not be read: " + case_name)

            reduce_col = np.sum(mask, axis=1)
            reduce_row = np.sum(mask, axis=0)
            # many 0s add up to none zero, we need to threshold it
            reduce_col = (reduce_col >= 255)*reduce_col
            reduce_row = (reduce_row >= 255)*reduce_row
            first_none_zero = None
            last_none_zero = None

            last = 0
            for i in range(reduce_col.shape[0]):
                current = reduce_col[i]
                if last == 0 and current != 0 and first_none_zero is None:
                    first_none_zero = i

                if current != 0:
                    last_none_zero = i

                last = reduce_col[i]

            up = first_none_zero
            down = last_none_zero

            first_none_zero = None
            last_none_zero = None
            last = 0
            for i in range(reduce_row.shape[0]):
                current = reduce_row[i]
                if last == 0 and current != 0 and first_none_zero is None:
                    first_none_zero = i

                if current != 0:
                    last_none_zero = i

                last = reduce_row[i]

            left = first_none_zero
            right = last_none_zero


            if up is None or down is None or left is None or right is None:
                print( "Error: The border is not found: " + case_name)
       	       	print("SKIPPING...")
                continue
            
            # new coordinates for image which is 1 times of mask, mask images are 256 x 256, 
            #so need to multiply 1 times to get 256 x 256, and relaxing the borders by 5% on all directions
            up_down_loose = int(1 * (down - up + 1) * 0.05)
            image_up = 1 * up - up_down_loose
            if image_up < 0:
                image_up = 0
            image_down = 1*(down+1)+up_down_loose
            if image_down > image.shape[0] + 1:
                image_down = image.shape[0]

            left_right_loose = int(1 * (right - left) * 0.05)
            image_left = 1 * left - left_right_loose
            if image_left < 0:
                image_left = 0
            image_right = 1*(right + 1)+left_right_loose
            if image_right > image.shape[1] + 1:
                image_right = image.shape[1]
            #print(image_up, image_down, image_left, image_right)
            #box = extract_bboxes(mask);
            #print(box)
            #rect = cv2.boundingRect(mask)               # function that computes the rectangle of interest
            #print(rect)
            #crop = image[rect[0]:(rect[0]+rect[2]), rect[1]:(rect[1]+rect[3])]  # crop the image to the desired rectangle 

            crop = image[image_up: image_down, image_left: image_right]
            #crop = cv2.resize(crop, (256, 256)) #the cropped image is resized to 256 x 256
            
            if crop.shape[0] <= int(image.shape[0] / 2)   or crop.shape[1] <= int( image.shape[1] / 2 )  :
                print(" Error: The Segmentation Failed: " + case_name)
                print("SKIPPING...")
       	       	continue
            cv2.imwrite( (crop_save_dir + case_name), crop) # cropped images saved to crop directory

            # write new csv
            crop_width = image_right - image_left + 1
            crop_height = image_down - image_up + 1

            csv_writer.writerow([case_name,
                                 image_left,
                                 image_up,
                                 crop_width,
                                 crop_height]) #writes xmin, ymin, width, and height

            if j % 50 == 0:
                print(j, " images are processed!")




#function to convert dicom to png

def Topng8bits(input_dir: str, output_dir: str):
    """
    Becareful all output images are gray image with 8 bit
    :param input_dir: dcm file directory
    :param output_dir: save directory
    """
    if not os.path.isdir(input_dir):
        raise ValueError("Input dir is not found!")

    if not os.path.isdir(output_dir):
        raise ValueError("Out dir is not found!")

    img_list = [f for f in os.listdir(input_dir)
                if f.split('.')[-1] == 'dcm' or f.split('.')[-1] == 'jpeg' or f.split('.')[-1] == 'jpg' or f.split('.')[-1] == 'png']
    for n, f in enumerate(img_list):
        
        if f.split(".")[-1] == "dcm":
            dcm_file = input_dir + f
            ds = pydicom.dcmread(dcm_file)
            pixel_array_numpy = ds.pixel_array
            pixel_array_numpy = cv2.normalize(pixel_array_numpy,
                                              None,
                                              alpha=0,
                                              beta=255,
                                              norm_type=cv2.NORM_MINMAX,
                                              dtype=cv2.CV_8UC1)
            img_file = output_dir + f.replace('.dcm', '.png')
            cv2.imwrite(img_file, pixel_array_numpy)

        elif f.split(".")[-1] == "jpeg":
                image_file = input_dir + f
                pixel_array_numpy = cv2.imread(image_file)
                pixel_array_numpy = cv2.cvtColor(pixel_array_numpy, cv2.COLOR_BGR2GRAY)
                pixel_array_numpy = cv2.normalize(pixel_array_numpy,
                                              None,
                                              alpha=0,
                                              beta=255,
                                              norm_type=cv2.NORM_MINMAX,
                                              dtype=cv2.CV_8UC1)
                image_file = output_dir + f.replace('.jpeg', '.png')
                cv2.imwrite(image_file, pixel_array_numpy)
                
        elif f.split(".")[-1] == "jpg":
                image_file = input_dir + f
                pixel_array_numpy = cv2.imread(image_file)
                pixel_array_numpy = cv2.cvtColor(pixel_array_numpy, cv2.COLOR_BGR2GRAY)
                pixel_array_numpy = cv2.normalize(pixel_array_numpy,
                                              None,
                                              alpha=0,
                                              beta=255,
                                              norm_type=cv2.NORM_MINMAX,
                                              dtype=cv2.CV_8UC1)

                image_file = output_dir + f.replace('.jpg', '.png')
                cv2.imwrite(image_file, pixel_array_numpy)

        elif f.split(".")[-1] == "png":
                image_file = input_dir + f
                pixel_array_numpy = cv2.imread(image_file , -1)
                pixel_array_numpy = cv2.cvtColor(pixel_array_numpy, cv2.COLOR_BGR2GRAY)
                pixel_array_numpy = cv2.normalize(pixel_array_numpy,
                                              None,
                                              alpha=0,
                                              beta=255,
                                              norm_type=cv2.NORM_MINMAX,
                                              dtype=cv2.CV_8UC1)

                image_file = output_dir + f
                cv2.imwrite(image_file, pixel_array_numpy)

