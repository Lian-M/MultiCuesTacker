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

torch.manual_seed(0)

# Build radar detection dataframe X_r: (B, N, 7, 3)

radardata_dir = 'C:/Users/evage/Desktop/Multisensory based method/Video_new_diff10drones/'
radar_data_temp = h5py.File(os.path.join(radardata_dir, 'AllDrones_XYZinRadarFrame_Cam0.mat')) 
radar_data = radar_data_temp['positionsinRadar'][:, :, :].transpose(2, 1, 0)  # (T, B, V) = (171, 10, 3)
lookback = 6 # It should be as same as the 'lookback' in MCT model
T = np.shape(radar_data)[0]
N = np.shape(radar_data)[1]
V = np.shape(radar_data)[2]

stack_r_in_batch = np.zeros((T - lookback, N, lookback+1, V)) # stacked state vectors shape as [(T-n_steps), N, n_steps+1, V]
for item in range(T - lookback):
    temp = radar_data[item:item + lookback+1, :, :]         # state vectors at past 6 frames and current frame [N, seq+1, v] = [11, 6+1, 6]
    stack_r_in_batch[item, :, :, :] = temp.transpose(1, 0, 2)

# Normalize to [-1, 1]
def normalize_4d_array(array):
    min_values = np.min(array, axis=(0, 1, 2, 3), keepdims=True)
    max_values = np.max(array, axis=(0, 1, 2, 3), keepdims=True)
    normalized_array = (array - min_values) / (max_values - min_values)
    return normalized_array

radar_shifted_df = normalize_4d_array(stack_r_in_batch)
# radar_shifted_df = stack_r_in_batch
# print(radar_shifted_df[0,2,-1,:])

# Build bounding box dataframe X_cb: (B, N, 7, 4)
folder_path = 'C:/Users/evage/Desktop/Multisensory based method/Video_new_diff10drones/BB'
original_w = 1920                  # resize original images 
original_h = 1080
all_bounding_boxes = read_bounding_boxes_new(folder_path, original_w, original_h) # return (T, N, 4), in which Bounding Box has been normalized by W and H
all_bounding_boxes = all_bounding_boxes.transpose(1, 0, 2) # (N, T, 4)

stack_BB_in_batch = np.zeros((T - lookback, N, lookback+1, 4)) # stacked BB state vectors
for item in range(T - lookback):
	stack_BB_in_batch[item, :, :, :] = all_bounding_boxes[:, item:item + lookback+1, :]   # BB state vectors at past 6 frames and current frame

stack_BB_in_batch = normalize_4d_array(stack_BB_in_batch)

# Divide source data and shuffle detection, BB and images at current frame, and build A_cube for labeling.
radar_shifted_df = torch.Tensor(radar_shifted_df).float()
stack_BB_in_batch = torch.Tensor(stack_BB_in_batch).float()


# Data Augmentation by multiple random seed at the same frame

X_r_past = torch.zeros(((T - lookback), N, 6, 3))
X_r_cur = torch.zeros(((T - lookback), N, 3))
X_cb_past = torch.zeros(((T - lookback), N, 6, 4))
X_cb_cur = torch.zeros(((T - lookback), N, 4))

for item in range(np.shape(radar_shifted_df)[0]): # [0, T-6)
    lookback_past = radar_shifted_df[item, :, 0:lookback, :]  # (N, 6, 3)
    current = radar_shifted_df[item, :, lookback, :]          # (N, 3)
    # r_random_seed = torch.randperm(np.shape(current)[0])      # This is the Ground Truth IDs of the current radar detections
    # current = current[r_random_seed, :]
    X_r_past[item, :, :, :] = lookback_past
    X_r_cur[item, :, :] = current

    BB_lookback_past = stack_BB_in_batch[item, :, 0:lookback, :]   # (N, 6, 4)
    BB_current = stack_BB_in_batch[item, :, lookback, :]           # (N, 4)
    # c_random_seed = torch.randperm(np.shape(BB_current)[0])        # This is the Ground Truth IDs of the current bounding boxes
    # BB_current = BB_current[c_random_seed, :]
    X_cb_past[item, :, :, :] = BB_lookback_past
    X_cb_cur[item, :, :] = BB_current


X_r_past = X_r_past.numpy()
X_r_cur = X_r_cur.numpy()
X_cb_past = X_cb_past.numpy()
X_cb_cur = X_cb_cur.numpy()



# Check for nan in data
# for i in range(np.shape(X_cb_cur)[0]):
#     for j in range(np.shape(X_cb_cur)[1]):
#         for k in range(np.shape(X_cb_cur)[2]):
#             # for w in range(np.shape(X_r_cur)[3]):
#                 if np.isnan(X_cb_cur[i, j, k]) == True:
#                     print(i, j, k)
#                 if np.isinf(X_cb_cur[i, j, k]) == True:
#                     print(i, j, k)


# print(np.shape(X_r_past))
# print(np.shape(X_r_cur))
# print(np.shape(X_cb_past))
# print(np.shape(X_cb_cur))
# print(np.shape(A_cubes_cur))

# print(X_r_past[0,0,:,:])
# print(X_r_cur[0,0,:])
# print(X_cb_past[0,0,:,:])
# print(X_cb_cur[0,0,:])
# print(A_cubes_cur[0,:,:,:])

np.save("Pretrain_Datasets/X_r_past.npy", X_r_past)
np.save("Pretrain_Datasets/X_r_cur.npy", X_r_cur)
np.save("Pretrain_Datasets/X_cb_past.npy", X_cb_past)
np.save("Pretrain_Datasets/X_cb_cur.npy", X_cb_cur)



