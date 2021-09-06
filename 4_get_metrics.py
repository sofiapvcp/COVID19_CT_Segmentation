# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 17:29:35 2021

@author: Joana Rocha
"""

from utils_segmentation_models import *
import numpy as np
import shutil
import cv2
import os

#%% DEFINE PATH TO SAVE

saveto_path = 'F:/COVID_Joana/Dados covid/get_metrics/'
if os.path.isdir(saveto_path)==True:
    shutil.rmtree(saveto_path)
try:
    os.mkdir(saveto_path)
except OSError:
    print ("Creation of the directory %s failed" % saveto_path)
    
    
#%% CHECK ALL IMAGES IN FOLDER

test_gt_path = r'F:/COVID_Joana/Dados covid/v4_all_positive/test/masks/masks/'
test_pred_path = r'F:/COVID_Joana/Dados covid/model_07-01-2021_07h52_correctpreds/predictions_07-01-2021_07h52/'
dim = (512,512)

img_list = [file for file in os.listdir(test_pred_path) if file.endswith('.png')]

# Double check common files =======================================================
# gt_list = [file for file in os.listdir(test_gt_path) if file.endswith('.png')]
# print(len(gt_list))
# print(len(img_list))
# common = list(set(img_list).intersection(gt_list))
# print(len(common))
# ==================================================================================


#%% GET EVALUATION METRICS

accs = []
recs = []
precs = []
acc_per_class0 = [] 
acc_per_class1 = [] 
dices = []
jaccs = []

for i in img_list:
    print(i)
    gt_mask = (cv2.imread(test_gt_path + i,0))/255
    gt_mask = cv2.resize(gt_mask, dim, interpolation = cv2.INTER_NEAREST)
    pred_mask = (cv2.imread(test_pred_path + i,0))/255
    
    if pred_mask.sum() > 0:
        acc, rec, prec, acc_per_class, dice, jacc = get_metrics(gt_mask.ravel(),pred_mask.ravel(),print=False)
        accs.append(acc)
        recs.append(rec)
        precs.append(prec)
        acc_per_class0.append(acc_per_class[0])
        acc_per_class1.append(acc_per_class[1])
        dices.append(dice)
        jaccs.append(jacc)
    
final_acc = np.mean(accs)
final_rec = np.mean(recs)
final_precs = np.mean(precs)
final_acc0 = np.mean(acc_per_class0)
final_acc1 = np.mean(acc_per_class1)
final_dice = np.mean(dices)
final_jacc = np.mean(jaccs)


print("Acc: ", final_acc)
print("Acc per class: " + str(final_acc0) + "/" + str(final_acc1))
print("Recall: ", final_rec)
print("Precision: ", final_precs)
print("Dice: ", final_dice)
print("Jaccard: ", final_jacc)
    
#%% SAVE SCORES TO TXT

file = open(saveto_path + 'FINALscores.txt','w')
file.write('\nAcc: ')
file.write(str(final_acc)) 
file.write('\nAcc per class: ')
file.write(str(final_acc0) + "/" + str(final_acc1))
file.write('\nRecall: ')
file.write(str(final_rec)) 
file.write('\nPrecision: ')
file.write(str(final_precs)) 
file.write('\nDice: ')
file.write(str(final_dice)) 
file.write('\nJaccard: ')
file.write(str(final_jacc)) 

file.write('\n')
file.write(str(test_pred_path)) 
file.close() 
    
    