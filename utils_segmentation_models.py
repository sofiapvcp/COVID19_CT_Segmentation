# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 08:20:44 2020

@author: Joana N. Rocha
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, auc, roc_curve   
from matplotlib.ticker import MaxNLocator


#%%

def get_model_plots(history,date,save=True,saveto_path=None):
    """ Plot the performance during training and validation. """
    plt.figure()
    
    dice = history.history['f1-score']
    val_dice = history.history['val_f1-score']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    learn_rate = history.history['lr']
    epochs = range(len(dice))
    
    #LEARNING CURVE
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(loss, label="Loss", color='blue')
    plt.plot(val_loss, label="Validation loss", color='red')
    plt.plot(np.argmin(val_loss), np.min(val_loss), marker="x", markersize=12, color="red", label="Best model")
    plt.xlabel("Epochs")
#    plt.xticks(np.arange(len(loss)), np.arange(1, len(loss)+1))
#    plt.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Loss")
    plt.legend()
    if save:
        plt.savefig(saveto_path + 'learning_curve_' + date + '.png')
        
    #DICE
    plt.figure(figsize=(8, 8))
    plt.plot(epochs, dice, 'b', label='Training dice')
    plt.plot(epochs, val_dice, 'r', label='Validation dice')
    plt.title('Training and validation DICE')
    plt.xlabel("Epochs")
    plt.legend()
    if save:
        plt.savefig(saveto_path + 'trainval_dice_' + date + '.png')
    
    
    #LOSS
    plt.figure(figsize=(8, 8))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    #plt.ylim([0,0.05])
    plt.legend()
    if save:
        plt.savefig(saveto_path + 'trainval_loss_' + date + '.png')
    
    
    #LEARNING RATE
    plt.figure()
    plt.plot(epochs, learn_rate, 'b')
    plt.title('Learning Rate')
    plt.xlabel("Epochs")
    plt.legend()
    if save:
        plt.savefig(saveto_path + 'learningrate_' + date + '.png')
    
    plt.show()
    plt.close()

#%%
""" Evaluation """ 

def get_eval(val_scores,test_scores,date,save=True,saveto_path=None):   
    if save:
        file = open(saveto_path + 'scores_' + date + '.txt','w')
        file.write('\n')
        file.write('\nVal scores: ')
        file.write(str(val_scores)) 
        file.write('\nTest scores: ')
        file.write(str(test_scores)) 
        file.write('\nTest sklearn-based scores: ')
        file.close() 

def jaccard_similarity_score(tn, fp, fn, tp):
    jacc = tp/(tp + fp + fn)
    return jacc
    
def get_metrics(y_test,y_pred,print=False):
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)    
    prec = precision_score(y_test, y_pred)
    dice1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)  
    acc_per_class = cm.diagonal()/cm.sum(axis=1)
    tn=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]
    tp=cm[1][1]
    jacc = jaccard_similarity_score(tn, fp, fn, tp)
    
    if print:
        print("Accuracy: ",acc)
        print("Accuracy per class: ",acc_per_class)
        print(cm)
        print("Recall: ", rec)
        print("Precision: ", prec)
        print("Dice: ", dice1)
        print("Jaccard: ", jacc)
        
    return acc, rec, prec, acc_per_class, dice1, jacc