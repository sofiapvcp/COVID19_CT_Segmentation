# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:07:25 2020

@author: Joana N. Rocha
"""

from datetime import datetime
import os
import cv2
from shutil import copyfile, rmtree
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------
    
def get_current_datetime():
    """Gets current date and time string"""
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%Hh%M")
    return dt_string

def new_directory(path):
    """Creates a new folder."""
    if os.path.isdir(path)==True:
        delete_path(path)
        print('Deleted old directory.')
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)

#---------------------------------------------------------------------------------
    
def create_path(path):
    """Creates a train, val and test folders."""
    if os.path.isdir(path)==True:
        delete_path(path)
        print('Deleted old directory.')
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        # Create train folder
        try:
            os.mkdir(os.path.join(path, 'train'))
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            try:
                os.mkdir(os.path.join(path, 'train', 'imgs'))
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                try:
                    os.mkdir(os.path.join(path, 'train', 'imgs', 'imgs'))
                except OSError:
                    print ("Creation of the directory %s failed" % path)
            try:
                os.mkdir(os.path.join(path, 'train', 'masks'))
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                try:
                    os.mkdir(os.path.join(path, 'train', 'masks', 'masks'))
                except OSError:
                    print ("Creation of the directory %s failed" % path)
        # Create val folder
        try:
            os.mkdir(os.path.join(path, 'val'))
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            try:
                os.mkdir(os.path.join(path, 'val', 'imgs'))
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                try:
                    os.mkdir(os.path.join(path, 'val', 'imgs', 'imgs'))
                except OSError:
                    print ("Creation of the directory %s failed" % path)
            try:
                os.mkdir(os.path.join(path, 'val', 'masks'))
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                try:
                    os.mkdir(os.path.join(path, 'val', 'masks', 'masks'))
                except OSError:
                    print ("Creation of the directory %s failed" % path)
        # Create test folder
        try:
            os.mkdir(os.path.join(path, 'test'))
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            try:
                os.mkdir(os.path.join(path, 'test', 'imgs'))
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                try:
                    os.mkdir(os.path.join(path, 'test', 'imgs', 'imgs'))
                except OSError:
                    print ("Creation of the directory %s failed" % path)
            try:
                os.mkdir(os.path.join(path, 'test', 'masks'))
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                try:
                    os.mkdir(os.path.join(path, 'test', 'masks', 'masks'))
                except OSError:
                    print ("Creation of the directory %s failed" % path)
    
#---------------------------------------------------------------------------------

def delete_path(path):
    """Deletes a directory."""
    rmtree(path)
    # print('Directory deleted.')
    
#---------------------------------------------------------------------------------

def load_nii(path,show_shape=False,plot_image=False):
    """Loads NIftI image."""
    nii_obj = nib.load(path)
    nii_data = nii_obj.get_fdata()
    if show_shape: 
        print(nii_data.shape)
    if plot_image:
        fig, ax = plt.subplots()
        im = ax.imshow(nii_data[:,:,145],cmap='gray',origin='lower')
        plt.show()
        plt.close()
    return nii_data

#---------------------------------------------------------------------------------

def normalize(npzarray,maxHU=400.,minHU=-1000.,to255=False,plot_image=False):
    """Truncates Hounsfield Units and normalizes to range."""
    npzarray = (npzarray - minHU) / (maxHU - minHU) #* 255.0
    # convert to 0-1 
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    # or 0-255 range
    if to255: npzarray *= 255.0
    if plot_image:
        fig, ax = plt.subplots()
        im = ax.imshow(npzarray,cmap='gray',origin='lower')
        plt.show()
        plt.close()
    return npzarray

#---------------------------------------------------------------------------------

def preprocess_set(path,data,labels,patient_id,ROI_mask,crop=False):
    """Implements the complete pre-processing stage, as described in the publication."""
    for i in range(0,len(data)): 
        # Load mask
        mask = load_nii(labels[i],plot_image=False)
        # Load image
        img = load_nii(data[i],plot_image=False)
        
        for slice_i in range(0,img.shape[2]):
            slice_mask = mask[:,:,slice_i]*255
            slice_img = img[:,:,slice_i]

            if slice_mask.sum() > 0:
                # normalize HUs
                slice_img_norm = normalize(slice_img,maxHU=400.,minHU=-1000.)

                # crop ROI
                if crop==False:
                    slice_img_norm = slice_img_norm*ROI_mask
                else:
                    slice_img_norm = slice_img_norm[74:437,74:437]
                    slice_mask = slice_mask[74:437,74:437]
                
                # apply CLAHE
                clahe = cv2.createCLAHE(clipLimit = 2) #, tileGridSize=(8, 8))
                slice_img_norm = np.array(slice_img_norm * 255, dtype = np.uint8)          
                final_slice_img = clahe.apply(slice_img_norm) 
                # normalize to range
                final_slice_img = (final_slice_img-final_slice_img.min())/(final_slice_img.max()-final_slice_img.min())
                final_slice_img *= 255
                
                # save
                cv2.imwrite(path + 'masks/masks/' + 'patient_' + "{0:0=3d}".format(patient_id) + '_slice_' + "{0:0=3d}".format(slice_i) + '.png', slice_mask)
                cv2.imwrite(path + 'imgs/imgs/' + 'patient_' + "{0:0=3d}".format(patient_id) + '_slice_' + "{0:0=3d}".format(slice_i) + '.png', final_slice_img)

        print('Patient ' + str(patient_id) + '.')
        patient_id += 1
    print('Dataset preprocessing complete.')
    return patient_id