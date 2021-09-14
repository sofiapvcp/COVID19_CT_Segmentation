#%% IMPORTS AND ACTIVATE GPU

import sys


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#import gc
from utils_segmentation_models import *
from utils import *
import random
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GroupKFold
import cv2
#import imageio
import segmentation_models
from keras.layers.convolutional import Conv2D


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%% SET SEEDS

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

#%% SET DATA DIRECTORIES, TIME STAMP AND SAVETO PATH

train_dir_img = r"C:\Users\SofiaPereira\Documents\Projects\COVID_segmentation\preprocessed_data\train\imgs\imgs"
train_dir_msk =r"C:\Users\SofiaPereira\Documents\Projects\COVID_segmentation\preprocessed_data\train\masks\masks"

val_dir_img = r"C:\Users\SofiaPereira\Documents\Projects\COVID_segmentation\preprocessed_data\train\imgs\imgs"
val_dir_msk =r"C:\Users\SofiaPereira\Documents\Projects\COVID_segmentation\preprocessed_data\train\masks\masks"



date = get_current_datetime()
saveto_path = "C:/Users/SofiaPereira/Documents/Projects/COVID_segmentation/" + date + '/'
new_directory(saveto_path)

scores=[]

#%% PERFORM 5-FOLD CROSS VALIDATION
# Groups are formed keeping original class distribution.

kf = GroupKFold(n_splits = 5)
i=0
df=pd.read_csv(r"C:\Users\SofiaPereira\Documents\Projects\COVID_segmentation\preprocessed_data\info.csv")

for train_index, val_index in kf.split(df,groups=df['Pat_id']):
    K.clear_session()
    file = open(saveto_path + 'finalCV_scores_fold_' + str(i) + '_' + date + '.txt','w')
    
#%% KERAS DATA GENERATORS (WITH DATA AUGMENTATION) AND SOME HYPER PARAMETERS

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            vertical_flip=True,
            horizontal_flip=True)
            
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    
    train_datagen_masks = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            vertical_flip=True,
            horizontal_flip=True)
            
    val_datagen_masks = ImageDataGenerator(rescale=1./255)
    test_datagen_masks = ImageDataGenerator(rescale=1./255)
    
    
    
    # Some hyper parameters
    im_height=512
    im_width=512
    BATCH_SIZE = 8
    training_idx = df.iloc[train_index]
    validation_idx = df.iloc[val_index]

    train_image_generator = train_datagen.flow_from_dataframe(training_idx, directory = train_dir_img,x_col = "Name",
                                                   batch_size = BATCH_SIZE,target_size=(im_height,im_width),
                                                   color_mode='grayscale',
                                                   class_mode = None, shuffle = False,seed=1)
    
    train_mask_generator = train_datagen_masks.flow_from_dataframe(training_idx, directory = train_dir_msk,x_col = "Name",
                                                   batch_size = BATCH_SIZE,target_size=(im_height,im_width),
                                                   color_mode='grayscale',
                                                   class_mode = None, shuffle = False,seed=1)
						     
    val_image_generator = val_datagen.flow_from_dataframe(validation_idx, directory = train_dir_img,x_col = "Name",
                                                   batch_size = BATCH_SIZE,target_size=(im_height,im_width),
                                                   color_mode='grayscale',
                                                   class_mode = None, shuffle = False,seed=1)
    
    val_mask_generator = val_datagen_masks.flow_from_dataframe(validation_idx, directory = train_dir_msk,x_col = "Name",
                                                   batch_size = BATCH_SIZE,target_size=(im_height,im_width),
                                                   color_mode='grayscale',
                                                   class_mode = None, shuffle = False,seed=1)
    


    train_generator = zip(train_image_generator, train_mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)

    # Uncomment to check image generator
    # =============================================================================
    # i = 0
    # for batch in train_datagen.flow_from_directory(train_dir_msk, batch_size = BATCH_SIZE,target_size=(im_height,im_width),color_mode='grayscale',class_mode=None,shuffle=False, save_to_dir=saveto_path):
    #     i += 1
    #     if i > 20: # save 20 images
    #         break  # otherwise the generator would loop indefinitely
    # =============================================================================

#%% BUILD AND COMPILE SEGMENTATION MODEL (WITH IMAGENET WEIGHTS)

    print('get u-net')
    
    input_img = Input((im_height, im_width, 1), name='img') 
    
    # Map N=1 channels data to 3 channels so that imagenet weights can be used
    N = 1 
    base_model = segmentation_models.Unet(backbone_name='resnet34', encoder_weights='imagenet')
    inp = Input((None, None, N))
    l1 = Conv2D(3, (1, 1))(inp) 
    out = base_model(l1)

#%% DEFINE ADDITIONAL HYPER PARAMETERS AND CALLBACKS

    
    NO_OF_TRAINING_IMAGES = len(os.listdir(train_dir_img))
    NO_OF_VAL_IMAGES = len(os.listdir(val_dir_img))
    NO_OF_TEST_IMAGES = len(os.listdir(test_dir_img))
    
    steps_train= np.floor(NO_OF_TRAINING_IMAGES/BATCH_SIZE)
    steps_val= np.floor(NO_OF_VAL_IMAGES/BATCH_SIZE)
    steps_test= np.floor(NO_OF_TEST_IMAGES /BATCH_SIZE)
    
    NO_OF_EPOCHS = 100
    weights_path = saveto_path + 'finalmodel_' + date + '_fold_' + str(i) + '.h5'
    
    callbacks = [
        EarlyStopping(patience=5,monitor="val_loss",verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.000001, verbose=1),
        ModelCheckpoint(weights_path, verbose=1, save_best_only=True, save_weights_only=True)
    ]
#%% FIT THE MODEL
    model = Model(inp, out, name='unet model')

    # Compile and show model summary
    model.compile(optimizer=Adam(lr=0.0001), 
              loss=segmentation_models.losses.dice_loss, 
              metrics=[segmentation_models.metrics.f1_score])
    model.summary()
    print('fit u-net')
    history=model.fit_generator(train_generator, epochs= NO_OF_EPOCHS, 
                                steps_per_epoch= steps_train,
                                validation_steps=steps_val, validation_data=val_generator, 
                                callbacks=callbacks)


#%% GENERATE MODEL PERFORMANCE PLOTS AND FOLD METRICS
    get_model_plots(history=history,date=date,save=True,saveto_path=saveto_path,fold=i)
    val_scores = model.evaluate_generator(val_generator,fold=i, steps=steps_val,verbose=1)
    print(val_scores)
    scores.append(val_scores)
    file.write('\nValidation scores after 2nd stage finetuning')
    file.write(str(val_scores))
    del model
    del history
    i += 1
    
print("fold scores:", scores)

#%% LOAD WEIGHTS OF THE BEST MODEL

losses=[x[0] for x in scores]
dices=[x[1] for x in scores]
best_fold_loss = losses.index(min(losses))
weights_path = saveto_path + 'finalmodel_' + date + '_fold_' + str(best_fold_loss) + '.h5'

print('get u-net (rebuild model)')

input_img = Input((im_height, im_width, 1), name='img') 

# Map N=1 channels data to 3 channels so that imagenet weights can be used
N = 1 
base_model = segmentation_models.Unet(backbone_name='resnet34', encoder_weights='imagenet')
inp = Input((None, None, N))
l1 = Conv2D(3, (1, 1))(inp) 
out = base_model(l1)
model = Model(inp, out, name='unet model')
model.compile(optimizer=Adam(lr=0.0001), 
              loss=segmentation_models.losses.dice_loss, 
              metrics=[segmentation_models.metrics.f1_score])
model.load_weights(weights_path)
model.summary()

#%% DEFINE GENERATORS FOR TEST DATA

test_datagen = ImageDataGenerator(rescale=1./255)
test_datagen_masks = ImageDataGenerator(rescale=1./255)

NO_OF_TEST_IMAGES = len(os.listdir(test_dir_img))
steps_test= np.floor(NO_OF_TEST_IMAGES /BATCH_SIZE)

test_dir_img = r"C:\Users\SofiaPereira\Documents\Projects\COVID_segmentation\preprocessed_data\test\imgs"
test_dir_msk = r"C:\Users\SofiaPereira\Documents\Projects\COVID_segmentation\preprocessed_data\test\masks"

test_image_generator = val_datagen.flow_from_directory(directory = test_dir_img,
                                                   batch_size = BATCH_SIZE,target_size=(im_height,im_width),
                                                   color_mode='grayscale',class_mode=None,shuffle = False,seed=1)

test_mask_generator = val_datagen.flow_from_directory(directory = test_dir_msk,
                                                   batch_size = BATCH_SIZE,target_size=(im_height,im_width),
                                                   color_mode='grayscale',class_mode=None,shuffle = False,seed=1)
    

test_generator = zip(test_image_generator, test_mask_generator)

#%% EVALUATE ON TEST SET
print('Evaluating. (Loss, dice)') 
test_scores = model.evaluate_generator(test_generator,steps=steps_test,verbose=1) 

print(test_scores)

#%% PREDICT TEST SET
print('Predicting.')
test_preds = model.predict_generator(test_generator,steps=steps_test,verbose=1)
test_preds = test_preds.reshape(test_preds.shape[0],test_preds.shape[1],test_preds.shape[2])
test_preds_T = (test_preds>0.5).astype(np.uint8)

#%% VISUALIZE 10 RANDOM SETS OF ORIGINAL IMG + GROUND TRUTH + PREDICITON TO CHECK RESULTS

test_filenames = test_mask_generator.filenames
dim = (im_width, im_height)

for i in range(0,10):
    random_img = random.randint(0, test_preds_T.shape[0]-1)
    print(test_filenames[random_img][5:])
    plt.figure()
    # image
    img = test_dir_img + '/imgs' + test_filenames[random_img][5:]
    gt_img = cv2.imread(img,0)
    gt_img = cv2.resize(gt_img, dim)
    plt.imshow(cv2.cvtColor(gt_img,cv2.COLOR_GRAY2RGB), cmap='Greys')
    plt.show()
    # ground truth
    gt_path = test_dir_msk + '/' + test_filenames[random_img]
    gt_mask = cv2.imread(gt_path,0)
    gt_mask = cv2.resize(gt_mask, dim)
    plt.imshow(cv2.cvtColor(gt_mask,cv2.COLOR_GRAY2RGB), cmap='Greys')
    plt.show()
    plt.close()
    # prediction
    plt.figure()
    plt.imshow(test_preds_T[random_img], cmap='binary_r',vmin=0, vmax=1)
    plt.show()
    plt.close()

#%% SAVE PREDICTIONS AND SCORES 

# Save predictions to folder
preds_path = saveto_path + 'predictions_' + date + '/'
new_directory(preds_path)

for i in range(0,test_preds_T.shape[0]):
    slice_i = test_filenames[i][-7:-4]
    patient_id = test_filenames[i][-17:-14]
#    print("patient: ", str(patient_id))
#    print("slice: ", slice_i)
    prediction_i = test_preds_T[i]*255.0
    cv2.imwrite(preds_path + 'patient_' + str(patient_id) + '_slice_' + str(slice_i) + '.png', prediction_i)

# Save scores to txt
get_eval(val_scores,test_scores,date,save=True,saveto_path=saveto_path)



































