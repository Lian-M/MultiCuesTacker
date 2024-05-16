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
batch_size = 16     # empirical value
num_epochs = 1000     
learning_rate = 0.001

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
print(train_size, validation_size, test_size)

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for _, batch in enumerate(train_loader):
	X1_batch, X2_batch, Label_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device)
	print(X1_batch.shape, X2_batch.shape, Label_batch.shape)
	break

# Build Model
model = Siamese(3, 2) # radar detection and bbox similarity score calculation model
model = model.to(device)

loss_function = nn.BCELoss()
loss_function = loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Define Training and Validation (per epoch)
train_loss = []
val_loss = []

def train_one_epoch():
	#model.train(True)
	model.train()
	print(f'Epoch: {epoch +1}')
	running_loss = 0.0
	training_loss = 0.0

	for batch_index, batch in enumerate(train_loader):
		X1_batch, X2_batch, Label_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device)
		# print(X1_batch)
		# print(X2_batch)
		# print(Label_batch)
		output = model(X1_batch, X2_batch)
		loss = loss_function(output, Label_batch)

		training_loss += loss.item()
		running_loss += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch_index % 10 == 9:      # print every 10 batches
			avg_loss_across_batches = running_loss / 10
			print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1, avg_loss_across_batches))
			running_loss = 0.0
	train_loss.append(training_loss/len(train_loader))
	print()

def validate_one_epoch():
	#model.train(False)
	model.eval()
	running_loss = 0.0

	for batch_index, batch in enumerate(validation_loader):
		X1_batch, X2_batch, Label_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device)

		with torch.no_grad():
			output = model(X1_batch, X2_batch)
			loss = loss_function(output, Label_batch)
			running_loss += loss.item()

	avg_loss_across_batches = running_loss / len(validation_loader)
	val_loss.append(avg_loss_across_batches)

	print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
	print('***************************************************')
	print()

# Start Training
for epoch in trange(num_epochs, desc="Training", unit="epoch"):
	train_one_epoch()
	validate_one_epoch()
torch.save(model.state_dict(), 'rbb_SimScore_weights.pt')
#torch.save(model.state_dict(), 'radar_vel_weights.pt')
#torch.save(model.state_dict(), 'radar_pv_weights.pt')

# Show Train and Validation Loss curves
plt.plot(train_loss, label='Train_loss')
plt.plot(val_loss, label='Validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


model.eval()
predictions = []
for batch_index, batch in enumerate(test_loader):
	X1_batch, X2_batch, Label_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device)
	with torch.no_grad():
		output = model(X1_batch, X2_batch)
		print(output)
		prediction = output.cpu().numpy()
		threshholded_prediction = np.where(prediction < 0.5, 0, 1)    # (16, 1)
		predictions.append(threshholded_prediction)

predicted_array = np.array(predictions).squeeze()
gt_array = np.array(Label_test).squeeze()

count = 0
for i in range(np.shape(gt_array)[0]):
	if predicted_array[i] == gt_array[i]:
		count += 1

acc = count / np.shape(gt_array)[0]

print(np.shape(gt_array)[0])
print(acc)