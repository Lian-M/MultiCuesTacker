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
from model_adaptive import *
import os
from tqdm import tqdm, trange

torch.manual_seed(0)
#torch.cuda.reset_max_memory_allocated()
#torch.cuda.empty_cache()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper-parameters setting
ninp = [2, 1, 3]         # [ninp_cp, ninp_cv, ninp_rp]   
nhid = [30, 20, 40]       # [nhid_cp, nhid_cv, nhid_rp]           
nout = [2, 1, 3]         # [nout_cp, nout_cv, nout_rp]
nlayers = [2, 2, 2]         # [nlayers_cp, nlayers_cv, nlayers_rp]
lookback = 6        # how many past time steps considered to make current prediction. In previous study, 6 is a reasonable setting.
sigma = 0.8         # a threshold in range [0, 1] on Score_cpv in pre-association
batch_size = 8     # empirical value
num_epochs = 9   
learning_rate = 0.001

# Prepare dataset for training 
X_r_past = np.load("X_r_past.npy")    # (B, N, 6, 3)
X_r_cur = np.load("X_r_cur.npy")    # (B, N, 3)
X_cb_past = np.load("X_cb_past.npy")    # (B, N, 6, 4)
X_cb_cur = np.load("X_cb_cur.npy")    # (B, N, 4)
A_cubes_cur = np.load("A_cubes_cur.npy")    # (B, N, N, N)

#Check for nan in data
# for i in range(np.shape(X_r)[0]):
#     for j in range(np.shape(X_r)[1]):
#         for k in range(np.shape(X_r)[2]):
#             for w in range(np.shape(X_r)[3]):
#                 if np.isnan(X_r[i, j, k, w]) == True:
#                     print(i, j, k, w)

#used to shuffle the dataset for train, validation and test
train_rate = 0.8
validation_rate = 0.1
test_rate = 0.1
train_size = int(train_rate * np.shape(X_r_past)[0]) 
validation_size = int(validation_rate * np.shape(X_r_past)[0])
test_size = int(test_rate * np.shape(X_r_past)[0])
print(train_size, validation_size, test_size)

Xr_past_train, Xr_past_validation, Xr_past_test = torch.utils.data.random_split(X_r_past, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))
Xr_cur_train, Xr_cur_validation, Xr_cur_test = torch.utils.data.random_split(X_r_cur, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))
Xcb_past_train, Xcb_past_validation, Xcb_past_test = torch.utils.data.random_split(X_cb_past, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))
Xcb_cur_train, Xcb_cur_validation, Xcb_cur_test = torch.utils.data.random_split(X_cb_cur, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))
Acubes_train, Acubes_validation, Acubes_test = torch.utils.data.random_split(A_cubes_cur, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))

Xr_past_train = torch.tensor(Xr_past_train, dtype=torch.float)
Xr_past_validation = torch.tensor(Xr_past_validation, dtype=torch.float)
Xr_past_test = torch.tensor(Xr_past_test, dtype=torch.float)
Xr_cur_train = torch.tensor(Xr_cur_train, dtype=torch.float)
Xr_cur_validation = torch.tensor(Xr_cur_validation, dtype=torch.float)
Xr_cur_test = torch.tensor(Xr_cur_test, dtype=torch.float)
Xcb_past_train = torch.tensor(Xcb_past_train, dtype=torch.float)
Xcb_past_validation = torch.tensor(Xcb_past_validation, dtype=torch.float)
Xcb_past_test = torch.tensor(Xcb_past_test, dtype=torch.float)
Xcb_cur_train = torch.tensor(Xcb_cur_train, dtype=torch.float)
Xcb_cur_validation = torch.tensor(Xcb_cur_validation, dtype=torch.float)
Xcb_cur_test = torch.tensor(Xcb_cur_test, dtype=torch.float)
Acubes_train = torch.tensor(Acubes_train, dtype=torch.float)
Acubes_validation = torch.tensor(Acubes_validation, dtype=torch.float)
Acubes_test = torch.tensor(Acubes_test, dtype=torch.float)

train_dataset = torch.utils.data.TensorDataset(Xr_past_train, Xr_cur_train, Xcb_past_train, Xcb_cur_train, Acubes_train)
validation_dataset = torch.utils.data.TensorDataset(Xr_past_validation, Xr_cur_validation, Xcb_past_validation, Xcb_cur_validation, Acubes_validation)
test_dataset = torch.utils.data.TensorDataset(Xr_past_test, Xr_cur_test, Xcb_past_test, Xcb_cur_test, Acubes_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for _, batch in enumerate(train_loader):
	Xr_past_batch, Xr_cur_batch, Xcb_past_batch, Xcb_cur_batch, A_cubes_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
	print(Xr_past_batch.shape, Xr_cur_batch.shape, Xcb_past_batch.shape, Xcb_cur_batch.shape, A_cubes_batch.shape)
	break

# Build Model
model = MultiCuesTracker(ninp, nhid, nout, nlayers, lookback, sigma)
model = model.to(device)
loss_function = nn.MSELoss()
loss_function = loss_function.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.0001)

# Define Training and Validation (per epoch)
train_loss = []
val_loss = []

def train_one_epoch():
	model.train()
	print(f'Epoch: {epoch +1}')
	running_loss = 0.0
	training_loss = 0.0

	for batch_index, batch in enumerate(train_loader):
		Xr_past_batch, Xr_cur_batch, Xcb_past_batch, Xcb_cur_batch, A_cubes_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
		predicted_A_cubes = model(Xr_past_batch, Xr_cur_batch, Xcb_past_batch, Xcb_cur_batch)
		loss = loss_function(predicted_A_cubes, A_cubes_batch)

		training_loss += loss.item()
		running_loss += loss.item()
		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
		optimizer.step()

		if batch_index % 10 == 9:      # print every 10 batches
			avg_loss_across_batches = running_loss / 10
			print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1, avg_loss_across_batches))
			running_loss = 0.0
	train_loss.append(training_loss/len(train_loader))
	print()


def validate_one_epoch():
	model.eval()
	running_loss = 0.0
	for batch_index, batch in enumerate(validation_loader):
		Xr_past_batch, Xr_cur_batch, Xcb_past_batch, Xcb_cur_batch, A_cubes_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)

		with torch.no_grad():
			predicted_A_cubes = model(Xr_past_batch, Xr_cur_batch, Xcb_past_batch, Xcb_cur_batch)
			loss = loss_function(predicted_A_cubes, A_cubes_batch)
			running_loss += loss.item()

	avg_loss_across_batches = running_loss / len(validation_loader)
	val_loss.append(avg_loss_across_batches)

	print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
	print('***************************************************')
	print()
	
# Start Training
for epoch in trange(num_epochs, desc="Training", unit="epoch"):
	train_one_epoch()
	scheduler.step()
	validate_one_epoch()
torch.save(model.state_dict(), 'best_weights.pt')
# Show Train and Validation Loss curves

plt.plot(train_loss, label='Train_loss')
plt.plot(val_loss, label='Validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()









