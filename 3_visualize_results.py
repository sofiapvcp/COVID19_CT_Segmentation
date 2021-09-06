# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:43:59 2020

@author: Joana N. Rocha
"""

import matplotlib.pyplot as plt
from shutil import rmtree
import numpy as np
import cv2
import os


#%% DEFINE PATH TO SAVE

saveto_path = 'F:/COVID_Joana/Dados covid/visualize_test_results/'
if os.path.isdir(saveto_path)==True:
    rmtree(saveto_path)
try:
    os.mkdir(saveto_path)
except OSError:
    print ("Creation of the directory %s failed" % saveto_path)
    
    
#%% CHECK ALL IMAGES IN FOLDER

test_img_path = r'F:/COVID_Joana/Dados covid/v4_all_positive/test/imgs/imgs/'
test_gt_path = r'F:/COVID_Joana/Dados covid/v4_all_positive/test/masks/masks/'
test_pred_path = r'F:/COVID_Joana/Dados covid/model_07-01-2021_07h52_correctpreds/predictions_07-01-2021_07h52/'

img_list = [file for file in os.listdir(test_pred_path) if file.endswith('.png')]

# Double check common files ========================================================
# gt_list = [file for file in os.listdir(test_gt_path) if file.endswith('.png')]
# print(len(gt_list))
# print(len(img_list))
# common = list(set(img_list).intersection(gt_list))
# print(len(common))
# ==================================================================================

#%% GET FINAL OVERLAYED PLOTS (GROUND TRUTH AND PREDICTION)

dim = (512,512)

for i in range(0,len(img_list)):
    print(str(img_list[i]))
    img = cv2.imread(test_img_path + img_list[i])
    img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
#    plt.figure()
#    plt.imshow(img, cmap='Greys')
#    plt.show()
    
    gt_mask = cv2.imread(test_gt_path + img_list[i],0)
    gt_mask = cv2.resize(gt_mask, dim, interpolation = cv2.INTER_NEAREST)
#    plt.imshow(cv2.cvtColor(gt_mask,cv2.COLOR_GRAY2RGB), cmap='Greys')
#    plt.show()
#    plt.close()
    
    pred_mask = cv2.imread(test_pred_path + img_list[i],0)
#    plt.imshow(cv2.cvtColor(pred_mask,cv2.COLOR_GRAY2RGB), cmap='Greys')
#    plt.show()
#    plt.close()
    
    # TP / green
    img[:,:,1][np.logical_and(gt_mask==255,pred_mask==255)] = 255
    
    # FN / red
    img[:,:,2][np.logical_and(gt_mask==255,pred_mask==0)] = 255
    
    # FP / yellow
    img[:,:,1:3][np.logical_and(gt_mask==0,pred_mask==255)] = 255
    
    # Save to folder
    filename = saveto_path + img_list[i]
    cv2.imwrite(filename, img)

