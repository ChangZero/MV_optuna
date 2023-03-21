import os
import random
import numpy as np
import pandas as pd
import cv2
import imgaug.augmenters as iaa
from PIL import Image
import tensorflow as tf
from keras import backend as K
from tqdm import tqdm

def median_blur(image_path, target_size):
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    # med_pix = np.median(src)
    # contrast = np.clip(src + (src - med_pix) * 3, 0, 255).astype("uint8")
    # scaled = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX)
    # hist = cv2.equalizeHist(scaled)
    medianblur = cv2.medianBlur(src, ksize = 3).astype("uint8")
    # alpha = 1.0
    # sharp = np.clip((1.0+alpha/10)*hist - alpha/10*medianblur, 0, 255).astype(np.uint8)
    # sharp = sharp/255
    medianblur = medianblur/255
    return medianblur

def noise_drop(image_path, target_size):
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    # med_pix = np.median(src)
    # contrast = np.clip(src + (src - med_pix) * 3, 0, 255).astype("uint8")
    # scaled = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX)
    # hist = cv2.equalizeHist(scaled)
    aug = iaa.Dropout(p=(0, 0.2))(images = src).astype("uint8")
    aug = aug/255
    return aug

def his_equalized(image_path, target_size):
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    med_pix = np.median(src)
    contrast = np.clip(src + (src - med_pix) * 3, 0, 255).astype("uint8")
    scaled = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX)
    hist = cv2.equalizeHist(scaled)
    hist = hist/255
    return hist

def sobel_masking_y(image_path, target_size):
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    # med_pix = np.median(src)
    # contrast = np.clip(src + (src - med_pix) * 3, 0, 255).astype("uint8")
    # scaled = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX)
    # hist = cv2.equalizeHist(scaled)
    sobel_y = cv2.Sobel(src, -1, 0, 1, delta=128).astype("uint8")
    sobel_y = sobel_y/255
    return sobel_y

def original_image(image_path, target_size):
    src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src, dsize = (target_size[0], target_size[1]), interpolation = cv2.INTER_CUBIC)
    src = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)
    src = src/255
    return src

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def load_df(file_path):
    label_name = ["ACC", "REJ"]
    filelist = []
    categories = []
    for label in label_name:
        filenames = os.listdir(os.path.join(file_path, label))
        for filename in filenames:
            if label == 'ACC':
                categories.append(0)
            else:
                categories.append(1)
            filelist.append(filename)
    df = pd.DataFrame({
        'filename': filelist,
        'y_label': categories
    })
    return df

def load_x_data(data_path, methods, target_size, df):
    x_data = []
    # y_data = []
    for method in methods:
        b_x_data = []
        for filename, category in tqdm(zip(df['filename'], df['y_label']), desc="Data Loading", mininterval=0.01, ascii = ' ='):
            if category == 0:
                img = os.path.join(os.path.join(data_path, "ACC"), filename)
            else:
                img = os.path.join(os.path.join(data_path, "REJ"), filename)

            if method == "median_blur":
                data = median_blur(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "noise_drop":
                data = noise_drop(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "his_equalized":
                data = his_equalized(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "sobel_masking":
                data = sobel_masking_y(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            elif method == "origin":
                data = original_image(img, target_size).reshape(target_size[0], target_size[1], target_size[2])
            else:
                raise Exception("Invalid method")
    
            b_x_data.append(data)
        # y_data.append(category)
        x_data.append(b_x_data)
        
    x_data = np.array(x_data, dtype=np.float32)
    # y_data = np.array(y_data)
    return x_data[0], x_data[1], x_data[2]

def load_y_data(df):
    y_data = np.array(df['y_label'], dtype=np.float32)
    return y_data

# # prototype code testing function
# def load_df2(file_path):
#     label_name = ["ACC", "REJ"]
#     filelist = []
#     categories = []
#     acc_count = 0
#     rej_count = 0
#     for label in label_name:
#         filenames = os.listdir(os.path.join(file_path, label))
#         for filename in filenames:
#             if acc_count + rej_count == 40:
#                 break
#             if label == 'ACC':
#                 if acc_count == 20:
#                     continue
#                 categories.append(0)
#                 acc_count += 1
#             else:
#                 if rej_count == 20:
#                     continue
#                 categories.append(1)
#                 rej_count += 1
#             filelist.append(filename)
#     test_df = pd.DataFrame({
#         'filename': filelist,
#         'y_label': categories
#     })

#     return test_df



