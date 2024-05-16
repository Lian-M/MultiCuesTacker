# This is a program of Dataset generation for MCT model
# Almost same as the original version but uses AirSim produced bounding box files.(xmin, ymin, xmax, ymax)
# which predicts assignment relationship between detections and tracks along time steps.
"""
Read in Track.csv, BoundingBoxInfo_10drones_Cam0_wholeData.txt, image_list = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
Return: X_r, X_cb, X_ci and A_cubes

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from utils import *
import re
from torchvision import datasets, transforms
from PIL import Image
import pickle
import cv2
import random

torch.manual_seed(0)

# Build normalized radar detection dataframe: (T, N, 3)
# radar_data = pd.read_csv('Video_new/myData_Tracks.csv') # Read in Tracks.csv file produced by Matlab RadarToolBox
# radar_data = radar_data[['sys_msec', 'track_id', 'pos_x', 'pos_y', 'pos_z']]
# radar_shifted_df = prepare_dataframe_for_pretrain(radar_data) # (T, N, 3)

radardata_dir = 'C:/Users/evage/Desktop/Multisensory based method/Video_new_diff10drones/'
radar_data_temp = h5py.File(os.path.join(radardata_dir, 'AllDrones_XYZinRadarFrame_Cam0.mat')) 
radar_data = radar_data_temp['positionsinRadar'][:, :, :].transpose(2, 1, 0)

# Min-Max Normalize to [0, 1]
def normalize_2d_array(array):
    min_values = np.min(array, axis=(0, 1), keepdims=True)
    max_values = np.max(array, axis=(0, 1), keepdims=True)
    normalized_array = (array - min_values) / (max_values - min_values)
    return normalized_array

# radar_data = normalize_3d_array(radar_data)


# Build normalized bounding box dataframe: (T, N, 4)
folder_path = 'Video_new_diff10drones/BB'
original_w = 1920                  # resize original images 
original_h = 1080
all_bounding_boxes = read_bounding_boxes_new(folder_path, original_w, original_h) # in which Bounding Box has been normalized by W and H
B = np.shape(radar_data)[0]
N = np.shape(radar_data)[1]
bbox_center = np.zeros((B, N, 2))
bbox_center[:, :, 0] = (all_bounding_boxes[:, :, 0] + all_bounding_boxes[:, :, 2]) / 2
bbox_center[:, :, 1] = (all_bounding_boxes[:, :, 1] + all_bounding_boxes[:, :, 3]) / 2

# Build positive and negative samples, and label them
X1 = []
X2 = []
Label = []

# for b in range(B):
#     for n1 in range(N):
#         x1 = radar_data[b, n1, :]          # (B, N, 3)
#         if np.all(bbox_center[b, n1, :] == 0):
#             continue
#         else:
#             x2 = bbox_center[b, n1, :]
#             label = 1
#             X1.append(x1)                     # attach the positive sample pair
#             X2.append(x2)
#             Label.append(label)

#             while True:
#                 num = random.randint(0, N-1)
#                 if num != n1 & any(bbox_center[b, num, :]):
#                     break
#             x2 = bbox_center[b, num, :]
#             label = 0
#             X1.append(x1)                     # attach the negative sample pair
#             X2.append(x2)
#             Label.append(label)

for b in range(B):
    for n1 in range(N):
        x1 = radar_data[b, n1, :]          # (B, N, 3)
        x2 = bbox_center[b, n1, :]
        label = 1
        X1.append(x1)                     # attach the positive sample pair
        X2.append(x2)
        Label.append(label)

        while True:
            num = random.randint(0, N-1)
            if num != n1:
                break
        x2 = bbox_center[b, num, :]
        label = 0
        X1.append(x1)                     # attach the negative sample pair
        X2.append(x2)
        Label.append(label)

# for b in range(B):
#     for n1 in range(N):
#         x1 = radar_data[b, n1, :]          # (B, N, 3)
#         for n2 in range(N):
#             x2 = bbox_center[b, n2, :]
#             if n1 == n2:
#                 label = 1
#             else:
#                 label = 0
#             X1.append(x1)
#             X2.append(x2)
#             Label.append(label)

# Label = np.array(Label).reshape((np.shape(X1)[0], 1))  # (B*N*N, 1)
X1 = np.array(X1)        # (B*N*N, 3)
X2 = np.array(X2)        # (B*N*N, 2)
# noise1 = np.random.normal(loc=0.0, scale=0.1, size=(np.shape(X1)[0],3))
# noise2 = np.random.normal(loc=0.0, scale=0.05, size=(np.shape(X1)[0],2))
# X1_noise = X1 + noise1
# X2_noise = X2 + noise2

# X1 = np.concatenate((X1, X1_noise), axis=0)
# X2 = np.concatenate((X2, X2_noise), axis=0)
# Label = np.concatenate((Label, Label), axis=0)

# print(X1[:10,:])
# print(X2[:10,:])

X1 = normalize_2d_array(X1)
X2 = normalize_2d_array(X2)
Label = np.array(Label).reshape((np.shape(X1)[0], 1))  # (B*N*N, 1)

# print(Label[:10,:])
# print(X1[0,:])
# print(X2[0,:])
# print(Label[0,:])
# print(X1[1,:])
# print(X2[1,:])
# print(Label[1,:])

print(np.shape(X1))
print(np.shape(X2))
print(np.shape(Label))


np.save("Video_new_diff10drones/Datasets_for_Pretain_rcSiamese/X1.npy", X1)
np.save("Video_new_diff10drones/Datasets_for_Pretain_rcSiamese/X2.npy", X2)
np.save("Video_new_diff10drones/Datasets_for_Pretain_rcSiamese/Label.npy", Label)

