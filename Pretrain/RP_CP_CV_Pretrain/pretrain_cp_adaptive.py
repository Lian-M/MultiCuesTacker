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
num_epochs = 30   
learning_rate = 0.001

# Prepare dataset for training 
X_cb_past = np.load("Pretrain_Datasets/X_cb_past.npy")    # (B, N, 6, 4)
X_cb_cur = np.load("Pretrain_Datasets/X_cb_cur.npy")    # (B, N, 4)

B = np.shape(X_cb_past)[0]
N = np.shape(X_cb_past)[1]
X_cb_past_reshape = np.zeros((B*N, 6, 4))
X_cb_cur_reshape = np.zeros((B*N, 4))
for b in range(B):
	X_cb_past_reshape[b*N:(b+1)*N, :, :] = X_cb_past[b, :, :, :]
	X_cb_cur_reshape[b*N:(b+1)*N, :] = X_cb_cur[b, :, :]

# X_cb_past = X_cb_past.reshape(-1, 6, 4) # (B*N, 6, 4)
# X_cb_cur = X_cb_cur.reshape(-1, 4)      # (B*N, 4)

x_center_temp = (X_cb_past_reshape[:, :, 0] + X_cb_past_reshape[:, :, 2]) / 2     # (B*N, 6)
y_center_temp = (X_cb_past_reshape[:, :, 1] + X_cb_past_reshape[:, :, 3]) / 2     # (B*N, 6)

x_center_temp = np.expand_dims(x_center_temp, axis=-1) # (B*N, 6, 1)
y_center_temp = np.expand_dims(y_center_temp, axis=-1) # (B*N, 6, 1)
PC_tempin = np.concatenate((x_center_temp, y_center_temp), axis=-1)   # (B*N, 6, 2)
X_cb_past_reshape1 = PC_tempin

x_center_temp_cur = (X_cb_cur_reshape[:, 0] + X_cb_cur_reshape[:, 2]) / 2     # (B*N, )
y_center_temp_cur = (X_cb_cur_reshape[:, 1] + X_cb_cur_reshape[:, 3]) / 2     # (B*N, )

x_center_temp_cur = np.expand_dims(x_center_temp_cur, axis=-1) # (B*N, 1)
y_center_temp_cur = np.expand_dims(y_center_temp_cur, axis=-1) # (B*N, 1)
PC_tempin_cur = np.concatenate((x_center_temp_cur, y_center_temp_cur), axis=-1)   # (B*N, 2)
X_cb_cur_reshape1 = PC_tempin_cur

# print(np.shape(X_cb_past))
# print(np.shape(X_cb_cur))
#used to shuffle the dataset for train, validation and test
train_rate = 0.8
validation_rate = 0.1
test_rate = 0.1
train_size = int(train_rate * np.shape(X_cb_past_reshape1)[0]) 
validation_size = int(validation_rate * np.shape(X_cb_past_reshape1)[0])
test_size = int(test_rate * np.shape(X_cb_past_reshape1)[0])
print(train_size, validation_size, test_size)

Xcb_past_train, Xcb_past_validation, Xcb_past_test = torch.utils.data.random_split(X_cb_past_reshape1, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))
Xcb_cur_train, Xcb_cur_validation, Xcb_cur_test = torch.utils.data.random_split(X_cb_cur_reshape1, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(42))

Xcb_past_train = torch.tensor(Xcb_past_train, dtype=torch.float)
Xcb_past_validation = torch.tensor(Xcb_past_validation, dtype=torch.float)
Xcb_past_test = torch.tensor(Xcb_past_test, dtype=torch.float)
Xcb_cur_train = torch.tensor(Xcb_cur_train, dtype=torch.float)
Xcb_cur_validation = torch.tensor(Xcb_cur_validation, dtype=torch.float)
Xcb_cur_test = torch.tensor(Xcb_cur_test, dtype=torch.float)


train_dataset = torch.utils.data.TensorDataset(Xcb_past_train, Xcb_cur_train)
validation_dataset = torch.utils.data.TensorDataset(Xcb_past_validation, Xcb_cur_validation)
test_dataset = torch.utils.data.TensorDataset(Xcb_past_test, Xcb_cur_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

for _, batch in enumerate(train_loader):
	Xcb_past_batch, Xcb_cur_batch = batch[0].to(device), batch[1].to(device)
	print(Xcb_past_batch.shape, Xcb_cur_batch.shape)
	break

# Build Model
model = LSTMtracker(ninp[0], nhid[0], nout[0], nlayers[0])
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
		Xcb_past_batch, Xcb_cur_batch = batch[0].to(device), batch[1].to(device)
		predicted_Xcb = model(Xcb_past_batch)
		loss = loss_function(predicted_Xcb, Xcb_cur_batch)

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
		Xcb_past_batch, Xcb_cur_batch = batch[0].to(device), batch[1].to(device)

		with torch.no_grad():
			predicted_Xcb = model(Xcb_past_batch)
			loss = loss_function(predicted_Xcb, Xcb_cur_batch)
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
torch.save(model.state_dict(), 'camera_p_weights.pt')
# Show Train and Validation Loss curves

plt.plot(train_loss, label='Train_loss')
plt.plot(val_loss, label='Validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()








