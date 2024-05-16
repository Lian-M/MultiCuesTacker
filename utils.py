# These are functions used in the Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from copy import deepcopy as dc
import os
import re
# setting random seed To facilitate replication of the experiment
#torch.manual_seed(0)
"""
def BB_Transformation(df):
# Transform the original (7 past + 1 current) Bounding Boxes into the (6 past +1 current) BB velocity and IOU values used in Camera Position and Velocity Models
	C_pv = np.zeros((df.shape[0], df.shape[1], df.shape[2]-1, 3)) # (B, N, 8, 4), 4——(cx,cy,w,h) to (B, N, 7, 3), 3——(cv_x,cv_y,IOU)
	for n in range(df.shape[1]):
		for t in range(df.shape[2]-1):			
				C_pv[:, n, t, 0] = df[:, n, t+1, 0] - df[:, n, t, 0]
				C_pv[:, n, t, 1] = df[:, n, t+1, 1] - df[:, n, t, 1]
				#IOU calculation
				x1 = max(df[:, n, t+1, 0], df[:, n, t, 0])
				y1 = max(df[:, n, t+1, 1], df[:, n, t, 1])
				x2 = min(df[:, n, t+1, 0] + df[:, n, t+1, 2], df[:, n, t, 0] + df[:, n, t, 2])
				y2 = min(df[:, n, t+1, 1] + df[:, n, t+1, 3], df[:, n, t, 1] + df[:, n, t, 3])
				intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    			area1 = df[:, n, t+1, 2] * df[:, n, t+1, 3]
    			area2 = df[:, n, t, 2] * df[:, n, t, 3]
    			union_area = area1 + area2 - intersection_area
    			IOU = intersection_area / union_area
				C_pv[:, n, t, 2] = IOU
"""

# Prepare dataframe for testing
def prepare_testframe_for_lstm(df, n_steps, ID):
	index = ID                                              # index of the track you want to extract
	N = 1                                                   # num of tracks in .csv file
	V = 3                                                   # num of motion and position variables
	T = round(max(df['sys_msec'])/min(df['sys_msec']))      # max length of tracks
	trans_df = np.zeros((N, T, V))                          # (11, 114, 6)
	for i in range(np.shape(df)[0]): # 0-1134
		for n in range(1, N+1):           # 1-11
			if df['track_id'][i] == index:
				t = round(df['sys_msec'][i]/100)
				trans_df[n-1, t-1, 0] = df['pos_x'][i]
				trans_df[n-1, t-1, 1] = df['pos_y'][i]
				trans_df[n-1, t-1, 2] = df['pos_z'][i]
				#trans_df[n-1, t-1, 3] = df['vel_x'][i]
				#trans_df[n-1, t-1, 4] = df['vel_y'][i]
				#trans_df[n-1, t-1, 5] = df['vel_z'][i]
	stack_in_time = np.zeros((N * (T - n_steps), n_steps+1, V)) # stacked state vectors shape as [N*(T-n_steps), n_steps+1, V]
	for item in range(T - n_steps):
		temp = trans_df[:, item:item + n_steps+1, :]         # state vectors at past 6 frames and current frame [b, seq+1, v] = [11, 6+1, 3]
		stack_in_time[N*item:N*(item+1), :, :] = temp
	return stack_in_time

# Expand n_steps length past dataframe for the detections at first frame, used instead of 'prepare_testframe_for_lstm' 
def expend_past_for_firstframe(df, n_steps, ID):
	index = ID                                              # index of the track you want to extract
	N = 1                                                   # num of tracks in .csv file
	V = 3                                                   # num of motion and position variables
	T = round(max(df['sys_msec'])/min(df['sys_msec']))      # max length of tracks
	trans_df = np.zeros((N, T, V))                          # (11, 114, 6)

	vel_init = np.zeros((N, 1, V))                    # extract initial velocity
	for i in range(np.shape(df)[0]): # 0-1134
		for n in range(1, N+1):           # 1-11
			if df['track_id'][i] == index and round(df['sys_msec'][i]/100) == 1:
				vel_init[n-1, 0, 0] = df['vel_x'][i]
				vel_init[n-1, 0, 1] = df['vel_y'][i]
				vel_init[n-1, 0, 2] = df['vel_z'][i]

	for i in range(np.shape(df)[0]): # 0-1134
		for n in range(1, N+1):           # 1-11
			if df['track_id'][i] == index:
				t = round(df['sys_msec'][i]/100)
				trans_df[n-1, t-1, 0] = df['pos_x'][i]
				trans_df[n-1, t-1, 1] = df['pos_y'][i]
				trans_df[n-1, t-1, 2] = df['pos_z'][i]
				#trans_df[n-1, t-1, 3] = df['vel_x'][i]
				#trans_df[n-1, t-1, 4] = df['vel_y'][i]
				#trans_df[n-1, t-1, 5] = df['vel_z'][i]
	expend_trans_df = np.zeros((N, T + n_steps, V))       # expend past 6 state vectors
	expend_trans_df[:, n_steps:, :] = trans_df
	
	tem = trans_df[:, 0, :]
	for i in range(1, n_steps+1):
		expend_trans_df[:, n_steps - i, :] = tem - 0.1*i*vel_init
	
	stack_in_time = np.zeros((N * T, n_steps+1, V))       # stacked state vectors shape as [N*T, n_steps+1, V]
	for item in range(T):
		temp = expend_trans_df[:, item:item + n_steps+1, :]         # state vectors at past 6 frames and current frame [b, seq+1, v] = [11, 6+1, 3]
		stack_in_time[N*item:N*(item+1), :, :] = temp
	return stack_in_time

# Prepare dataframe for LSTM
def prepare_dataframe_for_lstm(df, n_steps):
	N = max(df['track_id']);                                 # num of tracks in .csv file
	V = 6;                                                   # num of motion and position variables
	T = round(max(df['sys_msec'])/min(df['sys_msec']));      # max length of tracks, here it is 114 for Scenario3
	trans_df = np.zeros((N, T, V))                           # (11, 114, 6)
	for i in range(np.shape(df)[0]): # 0-1134
		for n in range(1, N+1):           # 1-11
			if df['track_id'][i] == n:
				t = round(df['sys_msec'][i]/100)
				trans_df[n-1, t-1, 0] = df['pos_x'][i]
				trans_df[n-1, t-1, 1] = df['pos_y'][i]
				trans_df[n-1, t-1, 2] = df['pos_z'][i]
				trans_df[n-1, t-1, 3] = df['vel_x'][i]
				trans_df[n-1, t-1, 4] = df['vel_y'][i]
				trans_df[n-1, t-1, 5] = df['vel_z'][i]
	stack_in_batch = np.zeros((T - n_steps, N, n_steps+1, V)) # stacked state vectors shape as [(T-n_steps), N, n_steps+1, V]
	for item in range(T - n_steps):
		temp = trans_df[:, item:item + n_steps+1, :]         # state vectors at past 6 frames and current frame [N, seq+1, v] = [11, 6+1, 6]
		stack_in_batch[item, :, :, :] = temp
	return stack_in_batch

# Prepare dataframe for pretrain
def prepare_dataframe_for_pretrain(df):
	N = max(df['track_id']);                                 # num of tracks in .csv file
	V = 3;                                                   # num of motion and position variables
	T = round(max(df['sys_msec'])/min(df['sys_msec']));      # max length of tracks
	trans_df = np.zeros((T, N, V))                           # (170, 10, 3)
	for i in range(np.shape(df)[0]):  
		for n in range(1, N+1):           
			if df['track_id'][i] == n:
				t = round(df['sys_msec'][i]/100)
				trans_df[t-1, n-1, 0] = df['pos_x'][i]
				trans_df[t-1, n-1, 1] = df['pos_y'][i]
				trans_df[t-1, n-1, 2] = df['pos_z'][i]
	return trans_df


# Prepare dataframe for Model (input & output for training)
def prepare_dataframe_for_model(df, n_steps):
	N = max(df['track_id']);                                 # num of tracks in .csv file, here it is 11 for Scenario3
	V = 3;                                                   # num of position variables
	T = round(max(df['sys_msec'])/100)                       # max length of original tracks, here it is 114 for Scenario3, '/100' indicates that the time step interval is 100 ms.
	T_expend = round(max(df['sys_msec'])/100) + n_steps;     # max length of expended tracks
	input_df = np.zeros((N, T_expend, V))                    # (11, 114+6-1, 3), (detection, time_expend, variable)
	output_df = np.zeros((T, N, N))                          # (114, 11, 11), (time, detection, track)
	for i in range(np.shape(df)[0]): # 0-1134
		for n in range(1, N+1):           # 1-11
			if df['track_id'][i] == n:
				t = round(df['sys_msec'][i]/100) + n_steps
				input_df[n-1, t-1, 0] = df['pos_x'][i]
				input_df[n-1, t-1, 1] = df['pos_y'][i]
				input_df[n-1, t-1, 2] = df['pos_z'][i]
	for i in range(np.shape(df)[0]): # 0-1134
		for n in range(1, N+1):           # 1-11
			if df['track_id'][i] == n and round(df['sys_msec'][i]/100) > 0:
				t_label = round(df['sys_msec'][i]/100)		
				#output_df[t_label-1, n-1] = df['track_id'][i]
				output_df[t_label-1, n-1, n-1] = 1
	#random_seed = torch.randperm(input_df.size(0))           # use random seed to shuffle the input and output in dimension N, to disorder the original id arrange(which is 'from small to big').   
	#input_df = input_df[random_seed]
	#output_df = output_df[:, random_seed]
	
	for t in range(2, T+1):
		#torch.manual_seed(t)
		#torch.manual_seed(0)
		random_seed = torch.randperm(N)                       # use random seed to shuffle the input and output in dimension N at each time step (from t=2 to t=N), which in order to stimulate random of detection sequence at each time step.
		input_temp = input_df[:, t+n_steps-1, :]              # (11, 3)  --- (ntrack, nvariables)
		input_df[:, t+n_steps-1, :] = input_temp[random_seed]
		output_temp = output_df[t-1, :, :]                    # (11, 11) --- (ntrack, ndetection)
		output_df[t-1, :, :] = output_temp[random_seed, :]
	
	return input_df, output_df








# Expend n_steps length past dataframe for ground truth data, 
# if lookback = 6, then n_steps = 5 since the detections at first frame doesn't need to be predicted, the prediction begain from the second frame.
def expend_gt_data(df, n_steps):
	init_track_num = max(df['track_id']); 
	# Extract initial position and velocity
	pv_init = np.zeros((init_track_num, 6))                    
	for i in range(np.shape(df)[0]): 
		for n in range(1, init_track_num+1):           
			if df['track_id'][i] == n and round(df['sys_msec'][i]/100) == 1:
				pv_init[n-1, 0] = df['pos_x'][i]
				pv_init[n-1, 1] = df['pos_y'][i]
				pv_init[n-1, 2] = df['pos_z'][i]
				pv_init[n-1, 3] = df['vel_x'][i]
				pv_init[n-1, 4] = df['vel_y'][i]
				pv_init[n-1, 5] = df['vel_z'][i]
	# Expand the position state of these 'init_track_num' tracks for previous n_steps frames
    #num_row_expended = n_steps * init_track_num	
	for i in range(n_steps):
		for n in range(1, init_track_num+1):
			new_row = pd.DataFrame({'sys_msec': -i*100,'track_id': n, 
				                    'pos_x': pv_init[n-1, 0]-0.1*(i+1)*pv_init[n-1, 3],
				                    'pos_y': pv_init[n-1, 1]-0.1*(i+1)*pv_init[n-1, 4],
				                    'pos_z': pv_init[n-1, 2]-0.1*(i+1)*pv_init[n-1, 5],
				                    'vel_x': pv_init[n-1, 3],
				                    'vel_y': pv_init[n-1, 4],
				                    'vel_z': pv_init[n-1, 5],}, index=[0])
			df = pd.concat([new_row, df], ignore_index=True)
	return df, init_track_num 



# Expend n_steps length past dataframe for the detections at first frame, 
# if lookback = 6, then n_steps = 5 since the detections at first frame doesn't need to be predicted, the prediction begain from the second frame.
def expend_raw_data(df, n_steps):
	df.insert(1,'track_id','')            # insert empty column 'track_id' in unassigned data, after 'sys_msec'.
	# Get the num of detections at first frame, and assume it equals to the num of initial tracks, and assign track_id to the detections at first frame.
	init_track_num = 0
	for i in range(np.shape(df)[0]):
		if round(df['sys_msec'][i]/100) == 1:
			init_track_num += 1
			df['track_id'][i] = init_track_num
	# Extract initial position and velocity
	pv_init = np.zeros((init_track_num, 6))                    
	for i in range(np.shape(df)[0]): 
		for n in range(1, init_track_num+1):           
			if df['track_id'][i] == n and round(df['sys_msec'][i]/100) == 1:
				pv_init[n-1, 0] = df['pos_x'][i]
				pv_init[n-1, 1] = df['pos_y'][i]
				pv_init[n-1, 2] = df['pos_z'][i]
				pv_init[n-1, 3] = df['vel_x'][i]
				pv_init[n-1, 4] = df['vel_y'][i]
				pv_init[n-1, 5] = df['vel_z'][i]
	# Expand the position state of these 'init_track_num' tracks for previous n_steps frames
    #num_row_expended = n_steps * init_track_num	
	for i in range(n_steps):
		for n in range(1, init_track_num+1):
			new_row = pd.DataFrame({'sys_msec': -i*100,'track_id': n, 
				                    'pos_x': pv_init[n-1, 0]-0.1*(i+1)*pv_init[n-1, 3],
				                    'pos_y': pv_init[n-1, 1]-0.1*(i+1)*pv_init[n-1, 4],
				                    'pos_z': pv_init[n-1, 2]-0.1*(i+1)*pv_init[n-1, 5],
				                    'vel_x': pv_init[n-1, 3],
				                    'vel_y': pv_init[n-1, 4],
				                    'vel_z': pv_init[n-1, 5],}, index=[0])
			df = pd.concat([new_row, df], ignore_index=True)
	return df, init_track_num 

# Get input data for LSTM at t_th predict operation of the n_th track, from expended_df 
# n: track_id, t: the index of prediction(start from 1)
def get_input_for_lstm(df, n, t):
	current_time = (t + 1) * 100               # time when the prediction happens
	temp = np.zeros((1, 5, 3))
	for i in range(np.shape(df)[0]):
		for j in range(1, 5+1):
			time_prob = current_time-j*100
			if df['track_id'][i] == n and round(df['sys_msec'][i]) == time_prob:
				temp[0, 5-j, 0] = df['pos_x'][i]
				temp[0, 5-j, 1] = df['pos_y'][i]
				temp[0, 5-j, 2] = df['pos_z'][i]
	return temp

# Combine network input and output into a time series dataset, which then can be divided into batches by 'DataLoader' function.
"""
class TimeSeriesDataset(Dataset):
	def __init__(self, X1, X2, X3, y):
		self.X1 = X1
		self.X2 = X2
		self.X3 = X3
		self.y = y

	def __len__(self):
		return len(self.X1)

	def __getitem__(self, i):
		return self.X1[i], self.X2[i], self.X3[i], self.y[i]
"""
class TimeSeriesDataset(Dataset):
	def __init__(self, X1, X2, X3):
		self.X1 = X1
		self.X2 = X2
		self.X3 = X3

	def __len__(self):
		return len(self.X1)

	def __getitem__(self, i):
		return self.X1[i], self.X2[i], self.X3[i]

def read_bounding_boxes(folder_path):
	all_bounding_boxes = []   
	for file_name in os.listdir(folder_path):
		if file_name.endswith('.txt'):
			t = int(re.findall(r'\d+', file_name)[-1])
			with open(os.path.join(folder_path, file_name), 'r') as f:
				lines = f.readlines()                
				for line in lines:
					ID, cx, cy, w, h = map(float, line.strip().split())
					all_bounding_boxes.append([cx, cy, w, h, ID+1, t])
                                     
	return np.array(all_bounding_boxes)

def read_bounding_boxes_new(folder_path, W, H):
	# return normalized bbox dataframe (T, N, 4)
	N = 10
	counter = 0
	#all_bounding_boxes = np.zeros((T, N, 4))   
	for file_name in os.listdir(folder_path):
		if file_name.endswith('.txt'):
			with open(os.path.join(folder_path, file_name), 'r') as f:
				lines = f.readlines()
				all_bounding_boxes = np.zeros((len(lines), N, 4))                 
				for line in lines:
					t = int(line.strip().split()[0])
					for n in range(N):
						x_min, y_min, x_max, y_max = float(line.strip().split()[n*4 + 6]), float(line.strip().split()[n*4 + 7]), float(line.strip().split()[n*4 + 8]), float(line.strip().split()[n*4 + 9])
						#x_min, y_min, x_max, y_max = float(line.strip().split()[n*4 + 2]), float(line.strip().split()[n*4 + 3]), float(line.strip().split()[n*4 + 4]), float(line.strip().split()[n*4 + 5])
						if int(x_min) != -1 & int(x_max) != -1 & int(y_min) != -1 & int(y_max) != -1:
							counter += 1
							#all_bounding_boxes[t-1, n, :] = [(x_min-1) / W, (y_min-1) / H, (x_max-1) / W, (y_max-1) / H]
							all_bounding_boxes[t-1, n, :] = [x_min, y_min, x_max, y_max]					
	print(counter)
	return all_bounding_boxes

# IOU computation
"""
def compute_iou(box1, box2):
	x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
	x3, y3, x4, y4 = box2[0], box2[1], box2[2], box2[3]
	x_min = torch.max(x1, x3)
	y_min = torch.max(y1, y3)
	x_max = torch.min(x2, x4)
	y_max = torch.min(y2, y4)
	intersection = torch.clamp((x_max - x_min), min=0) * torch.clamp((y_max - y_min), min=0)
	area_box1 = (x2 - x1) * (y2 - y1)
	area_box2 = (x4 - x3) * (y4 - y3)
	union = area_box1 + area_box2 - intersection
	iou = intersection / union
	return iou
"""
def compute_iou(box1, box2):
	if torch.all(box1 == 0) or torch.all(box2 == 0):
		iou = 0
	else:
		x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
		x3, y3, x4, y4 = box2[0], box2[1], box2[2], box2[3]
		x_min = torch.max(x1, x3)
		y_min = torch.max(y1, y3)
		x_max = torch.min(x2, x4)
		y_max = torch.min(y2, y4)
		intersection = torch.mul(torch.clamp(torch.sub(x_max, x_min), min=0), torch.clamp(torch.sub(y_max, y_min), min=0))
		area_box1 = torch.mul(torch.sub(x2, x1), torch.sub(y2, y1))
		area_box2 = torch.mul(torch.sub(x4, x3), torch.sub(y4, y3))
		union = torch.sub(torch.add(area_box1, area_box2), intersection)
		iou = torch.div(intersection, union)
	return iou


def compute_iou_array(box1, box2):
	if np.all(box1 == 0) or np.all(box2 == 0):
		iou = 0
	else:
		x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
		x3, y3, x4, y4 = box2[0], box2[1], box2[2], box2[3]
		x_min = max(x1, x3)
		y_min = max(y1, y3)
		x_max = min(x2, x4)
		y_max = min(y2, y4)
		intersection = np.clip((x_max - x_min), a_min=0, a_max=None) * np.clip((y_max - y_min), a_min=0, a_max=None)
		area_box1 = (x2 - x1) * (y2 - y1)
		area_box2 = (x4 - x3) * (y4 - y3)
		union = area_box1 + area_box2 - intersection
		iou = intersection / union
	return iou


def get_bbox_center(bbox):
	x_min, y_min, x_max, y_max = bbox
	center_x = (x_min + x_max) / 2
	center_y = (y_min + y_max) / 2
	return torch.tensor([center_x, center_y])


