# This is the code for testing the trained RL based RMC model, which shows the Prediction, GroundTruth and Error Range of a track
# Use the variable 'ID' to specifiy the track wanted to be shown)
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
import time
from collections import Counter

torch.manual_seed(0)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters setting
ninp = [2, 1, 3]         # [ninp_cp, ninp_cv, ninp_rp]   
nhid = [30, 20, 40]       # [nhid_cp, nhid_cv, nhid_rp]           
nout = [2, 1, 3]         # [nout_cp, nout_cv, nout_rp]
nlayers = [2, 2, 2]         # [nlayers_cp, nlayers_cv, nlayers_rp]
lookback = 6        # how many past time steps considered to make current prediction. In previous study, 6 is a reasonable setting.
sigma = 0.8         # a threshold in range [0, 1] on Score_cpv in pre-association
batch_size = 1      # empirical value     

# Prepare test data for Model
X_r_past = np.load("demo_dataset/X_r_past.npy")    # (B, N, 6, 3)
X_r_cur = np.load("demo_dataset/X_r_cur.npy")    # (B, N, 3)
X_cb_past = np.load("demo_dataset/X_cb_past.npy")    # (B, N, 6, 4)
X_cb_cur = np.load("demo_dataset/X_cb_cur.npy")    # (B, N, 4)
A_cubes_cur = np.load("demo_dataset/A_cubes_cur.npy")    # (B, N, N, N)

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

for _, batch in enumerate(test_loader):
	Xr_past_batch, Xr_cur_batch, Xcb_past_batch, Xcb_cur_batch, A_cubes_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
	print(Xr_past_batch.shape, Xr_cur_batch.shape, Xcb_past_batch.shape, Xcb_cur_batch.shape, A_cubes_batch.shape)
	break

# Load model
model = MultiCuesTracker(ninp, nhid, nout, nlayers, lookback, sigma)
model.load_state_dict(torch.load('best_weights.pt'))
model = model.to(device)

# Achieve Prediction and GroundTruth
all_predicted_A = []    # (T-6, N, N, N)
start_time = time.time()
for batch_index, batch in enumerate(test_loader):
		Xr_past_batch, Xr_cur_batch, Xcb_past_batch, Xcb_cur_batch, A_cubes_batch = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device)
		model.eval()
		with torch.no_grad():
			predicted_A_cubes = model(Xr_past_batch, Xr_cur_batch, Xcb_past_batch, Xcb_cur_batch)
			A_array = predicted_A_cubes.cpu().numpy()
			all_predicted_A.append(A_array.squeeze())
print(np.shape(all_predicted_A))

all_predicted_array = np.array(all_predicted_A)
all_gt_array = np.array(Acubes_test)

# Build array to show the association_result and the ground truth relationship
association_predicted = np.zeros((np.shape(all_predicted_array)[1], np.shape(all_predicted_array)[0], 2)) # (N, T-6, 2), the last dimension saves the detection index and bounding box index of an identity at one time step
association_gt = np.zeros_like(association_predicted)  # (N, B, 2)

# NMS
for t in range(np.shape(all_predicted_array)[0]):
	for identity in range(np.shape(all_predicted_array)[1]):
		max_index_p = np.where(all_predicted_array[t, identity, :, :] == np.max(all_predicted_array[t, identity, :, :]))
		association_predicted[identity, t, 0] = max_index_p[0][0] + 1   # detection index of identity at t
		association_predicted[identity, t, 1] = max_index_p[1][0] + 1   # bounding box index of identity at t

		max_index_gt = np.where(all_gt_array[t, identity, :, :] == np.max(all_gt_array[t, identity, :, :]))
		association_gt[identity, t, 0] = max_index_gt[0][0] + 1         # detection index of identity at t
		association_gt[identity, t, 1] = max_index_gt[1][0] + 1         # bounding box index of identity at t
end_time = time.time()


# Calculate the evaluation metrics
N = np.shape(all_predicted_array)[1]
T = np.shape(all_predicted_array)[0]
residual_error = association_gt - association_predicted

radar_accuracy = 1- np.count_nonzero(residual_error[:, :, 0]) / N / T
bbox_accuracy = 1- np.count_nonzero(residual_error[:, :, 1]) / N / T
print('Accuracy of radar data association:', radar_accuracy)
print('Accuracy of bbox data association:', bbox_accuracy)

Xcb_cur_test = Xcb_cur_test.numpy()
Xr_cur_test = Xr_cur_test.numpy()
zero_bbox_count = 0
for b in range(np.shape(Xcb_cur_test)[0]):
	for n in range(np.shape(Xcb_cur_test)[1]):
		if np.all(Xcb_cur_test[b, n, :] == 0) == True:
			zero_bbox_count += 1
bbox_FP = zero_bbox_count
bbox_all = np.shape(Xcb_cur_test)[0] * np.shape(Xcb_cur_test)[1] - zero_bbox_count
print('Number of FP in bboxes:', bbox_FP)
print('Total number of gt bboxes:', bbox_all)

zero_rdet_count = 0
for b in range(np.shape(Xr_cur_test)[0]):
	for n in range(np.shape(Xr_cur_test)[1]):
		if np.all(Xr_cur_test[b, n, :] == 0) == True:
			zero_rdet_count += 1
rdet_FP = zero_rdet_count
rdet_all = np.shape(Xr_cur_test)[0] * np.shape(Xr_cur_test)[1] - zero_rdet_count
print('Number of FP in radar detections:', rdet_FP)
print('Total number of gt radar detections:', rdet_all)

bbox_FN = 0 # Since the model always give a bbox prediction, there is no possible for a FN happens
rdet_FN = 0 # Since the model always give a radar detection prediction, there is no possible for a FN happens

MT_bbox_count = 0
for n in range(N):
	if (np.count_nonzero(residual_error[n, :, 1]) / T) < 0.2:
		MT_bbox_count +=1
MT_bbox = MT_bbox_count / N
print('MT of bbox:', MT_bbox)
MT_rdet_count = 0
for n in range(N):
	if (np.count_nonzero(residual_error[n, :, 0]) / T) < 0.2:
		MT_rdet_count +=1
MT_rdet = MT_rdet_count / N
print('MT of rdet:', MT_rdet)

ML_bbox_count = 0
for n in range(N):
	if (np.count_nonzero(residual_error[n, :, 1]) / T) > 0.8:
		ML_bbox_count +=1
ML_bbox = ML_bbox_count / N
print('ML of bbox:', ML_bbox)
ML_rdet_count = 0
for n in range(N):
	if (np.count_nonzero(residual_error[n, :, 0]) / T) > 0.8:
		ML_rdet_count +=1
ML_rdet = ML_rdet_count / N
print('ML of rdet:', ML_rdet)


# IDSW_bbox = 0
# IDSW_rdet = 0
# for t in range(T):
# 	count_r = Counter(association_predicted[:, t, 0])
# 	sw_r = N - len(count_r)
# 	count_c = Counter(association_predicted[:, t, 1])
# 	sw_c = N - len(count_c)
# 	IDSW_bbox += sw_r
# 	IDSW_rdet += sw_c

IDSW_bbox = np.count_nonzero(residual_error[:, :, 1])
IDSW_rdet = np.count_nonzero(residual_error[:, :, 0])

print('IDSW_bbox:', IDSW_bbox)
print('IDSW_rdet:', IDSW_rdet)

# def f1_score(y_true, y_pred):
#     # 将y_true和y_pred转换为set类型，去重并排序
#     y_true = set(sorted(y_true))
#     y_pred = set(sorted(y_pred))
    
#     # 计算TP、FP、FN数量
#     fp = sum([1 for yp in y_pred if yp not in y_true])
#     fn = sum([1 for yt in y_true if yt not in y_pred])
#     tp = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp])
#     # 计算精确率和召回率
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
#     # 计算F1 score
#     f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
#     return f1
def f1_score(y_true, y_pred):
    
    error = y_true - y_pred
    idfp = np.count_nonzero(error)
    idfn = np.count_nonzero(error)
    idtp = len(error) - idfp

    id_precision = idtp / (idtp + idfp) 
    id_recall = idtp / (idtp + idfn) 
    
    # 计算F1 score
    f1 = (2 * id_precision * id_recall) / (id_precision + id_recall)
    
    return f1

F_r = 0
F_c = 0
for n in range(N):
	f1_r = f1_score(association_gt[n, :, 0], association_predicted[n, :, 0])
	f1_c = f1_score(association_gt[n, :, 1], association_predicted[n, :, 1])
	F_r += f1_r
	F_c += f1_c
IDF1_r = F_r / N
IDF1_c = F_c / N
print('IDF1_r:', IDF1_r)
print('IDF1_c:', IDF1_c)

MOTA_r = 1 - (rdet_FN + rdet_FP + IDSW_rdet)/ rdet_all
MOTA_c = 1 - (bbox_FN + bbox_FP + IDSW_bbox)/ bbox_all
print('MOTA_r:', MOTA_r)
print('MOTA_c:', MOTA_c)


frag_r_count = 0
for n in range(N):
	start = None
	for t in range(T):
		if residual_error[n, t, 0] != 0:
			if start is None:
				start = t
		else:
			if start is not None:
				frag_r_count += 1
				start = None
Frag_r = frag_r_count
print('Frag_r:', Frag_r)
frag_c_count = 0
for n in range(N):
	start = None
	for t in range(T):
		if residual_error[n, t, 1] != 0:
			if start is None:
				start = t
		else:
			if start is not None:
				frag_c_count += 1
				start = None
Frag_c = frag_c_count
print('Frag_c:', Frag_c)

Hz = 1 / ((end_time - start_time) / (T))
print('Hz:', Hz) 

# Visualize the association result
N = np.shape(all_predicted_array)[1]
colors = plt.cm.viridis(np.linspace(0, 1, np.shape(all_predicted_array)[1]))

for i in range(np.shape(all_predicted_array)[1]):
    plt.plot(association_predicted[i, :, 0], color=colors[i])

plt.xlabel('Time step')
plt.ylabel('Prediced index of radar detection')
plt.legend(['N={}'.format(i+1) for i in range(N)])
plt.grid()
plt.show()

for i in range(np.shape(all_predicted_array)[1]):
    plt.plot(association_predicted[i, :, 1], color=colors[i])

plt.xlabel('Time step')
plt.ylabel('Prediced index of bounding box')
plt.legend(['N={}'.format(i+1) for i in range(N)])
plt.grid()
plt.show()

for i in range(np.shape(all_predicted_array)[1]):
    plt.plot(association_gt[i, :, 0], color=colors[i])

plt.xlabel('Time step')
plt.ylabel('Ground truth index of radar detection')
plt.legend(['N={}'.format(i+1) for i in range(N)])
plt.grid()
plt.show()

for i in range(np.shape(all_predicted_array)[1]):
    plt.plot(association_gt[i, :, 1], color=colors[i])

plt.xlabel('Time step')
plt.ylabel('Ground truth index of bounding box')
plt.legend(['N={}'.format(i+1) for i in range(N)])
plt.grid()
plt.show()
