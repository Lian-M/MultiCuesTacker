# This is a training program for the built LSTM-tracker and reinforcement learning based Matchmaker 
# which predicts assignment relationship between detections and tracks along time steps.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from utils import *
from modules import *
import os
from tqdm import tqdm, trange

torch.manual_seed(0)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper-parameters setting
batch_size = 1     # empirical value  

# Prepare dataset for training 
X1 = np.load("Video_new_diff10drones/Datasets_for_Pretain_rcSiamese/X1.npy")    # (B, 3)
X2 = np.load("Video_new_diff10drones/Datasets_for_Pretain_rcSiamese/X2.npy")    # (B, 2)
Label = np.load("Video_new_diff10drones/Datasets_for_Pretain_rcSiamese/Label.npy")  # (B, 1)

#used to shuffle the dataset for train and validation
# random_seed = torch.randperm(np.shape(X1)[0])      # This is the Ground Truth IDs of the current radar detections
# X1 = X1[random_seed, :]
# X2 = X2[random_seed, :]
# Label = Label[random_seed]

# split_index = int(np.shape(X1)[0] * 0.9)
# X1_train = X1[:split_index, :]
# X1_test = X1[split_index:, :]
# X2_train = X2[:split_index, :]
# X2_test = X2[split_index:, :]
# Label_train = Label[:split_index]
# Label_test = Label[split_index:]

train_rate = 0.8
validation_rate = 0.1
test_rate = 0.1
train_size = int(train_rate * np.shape(X1)[0]) 
validation_size = int(validation_rate * np.shape(X1)[0])
test_size = int(test_rate * np.shape(X1)[0])
X1_train, X1_validation, X1_test = torch.utils.data.random_split(X1, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))
X2_train, X2_validation, X2_test = torch.utils.data.random_split(X2, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))
Label_train, Label_validation, Label_test = torch.utils.data.random_split(Label, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))

X1_train = torch.tensor(X1_train, dtype=torch.float)
X1_validation = torch.tensor(X1_validation, dtype=torch.float)
X1_test = torch.tensor(X1_test, dtype=torch.float)
X2_train = torch.tensor(X2_train, dtype=torch.float)
X2_validation = torch.tensor(X2_validation, dtype=torch.float)
X2_test = torch.tensor(X2_test, dtype=torch.float)
Label_train = torch.tensor(Label_train, dtype=torch.float)
Label_validation = torch.tensor(Label_validation, dtype=torch.float)
Label_test = torch.tensor(Label_test, dtype=torch.float)

# train_dataset = TimeSeriesDataset(X1_train, X2_train, Label_train)
# test_dataset = TimeSeriesDataset(X1_test, X2_test, Label_test)
train_dataset = torch.utils.data.TensorDataset(X1_train, X2_train, Label_train)
validation_dataset = torch.utils.data.TensorDataset(X1_validation, X2_validation, Label_validation)
test_dataset = torch.utils.data.TensorDataset(X1_test, X2_test, Label_test)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = Siamese(3, 2) # radar detection and bbox similarity score calculation model
model.load_state_dict(torch.load('rbb_SimScore_weights.pt'))
model = model.to(device)

model.eval()
predictions = []
threshholded_predictions = []
for batch_index, batch in enumerate(test_loader):
	X1_batch, X2_batch, Label_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device)
	with torch.no_grad():
		output = model(X1_batch, X2_batch)
		# print(output)
		prediction = output.cpu().numpy()
		threshholded_prediction = np.where(prediction < 0.8, 0, 1)    # (16, 1)
		predictions.append(prediction)
		threshholded_predictions.append(threshholded_prediction)

predicted_array = np.array(threshholded_predictions).squeeze()
gt_array = np.array(Label_test).squeeze()
predicted_conf = np.array(predictions).squeeze()

plt.plot(predicted_conf, label='predict_score')
plt.plot(gt_array, label='gt_score')
plt.xlabel('Samples')
plt.ylabel('Similarity Score')
plt.legend()
plt.grid()
plt.show()

count = 0
for i in range(np.shape(gt_array)[0]):
	if predicted_array[i] == gt_array[i]:
		count += 1

acc = count / np.shape(gt_array)[0]

print(np.shape(gt_array)[0])
print(acc)