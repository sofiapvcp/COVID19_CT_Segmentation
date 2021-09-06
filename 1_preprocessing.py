# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:00:16 2020

@author: Joana N. Rocha
"""

import glob
import cv2
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% SET PATHS

data_path = 'F:/COVID_Joana/Dados covid/COVID-19-20_v2/Train/'
save_to = 'F:/COVID_Joana/Dados covid/pre_processed_data/'
 
create_path(save_to)

#%% SPLIT DATA PER SET

img_list = glob.glob(data_path+"*_ct.nii.gz")
mask_list = glob.glob(data_path+"*_seg.nii.gz")

X_trainval, X_test, y_trainval, y_test = train_test_split(img_list, mask_list, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.3, random_state=42)

# uncomment to generate sample set (patient per set)
#X_train = [X_train[0]]
#X_val = [X_val[0]]
#X_test = [X_test[0]]
#y_train = [y_train[0]]
#y_val = [y_val[0]]
#y_test = [y_test[0]]

#%% PRE-PROCESSES IMAGE AND MASK LIST PER SET AND SAVE THEM

ROI_mask = np.zeros((512,512))
ROI_mask[74:437,74:437] = 1.0

path=save_to+'train/'
patient_id=0
patient_id = preprocess_set(path,X_train,y_train,patient_id,ROI_mask,crop=True)
path=save_to+'val/'
patient_id = preprocess_set(path,X_val,y_val,patient_id,ROI_mask,crop=True)
path=save_to+'test/'
patient_id = preprocess_set(path,X_test,y_test,patient_id,ROI_mask,crop=True)
