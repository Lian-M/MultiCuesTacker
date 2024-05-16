# The Multicues Tracker model
# Heaven dreams make life tolerable <Rudy>
import torch
from torch import nn
from modules import *
from torchvision.models import vgg16_bn
from utils import *
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

class MultiCuesTracker(nn.Module):
    def __init__(self, ninp, nhid, nout, nlayers, lookback, sigma):     
        """
        MCT, a LSTM based tracker which considers multisenory cues(radar and camera). Since there are problems in the cv/cp value calculation of original paper, we directly use (cx,cy) and (w, h) in Camera velocity and position model.(or we can say position and shape model!)

        ninp: a list of state variable's length in each LSTM tracker, as [ninp_cp, ninp_cv, ninp_a, ninp_rp, ninp_rv], in our context it should be [2, 2, 256, 3, 3].
        nhid: a list of hidden state's length in each LSTM tracker,as [nhid_cp, nhid_cv, nhid_a, nhid_rp, nhid_rv], in our context it should be [12, 18, 256, 4, 4] according to reference and performance anslysis in LSTMtracker. 
        nout: a list of output state's length in each LSTM tracker, as [nout_cp, nout_cv, nout_a, nout_rp, nout_rv], in our context it should be [2, 2, 256, 3, 3].
        nlayers: a list of the stacked layer number in each LSTMtracker, as [nlayers_cp, nlayers_cv, nlayers_a, nlayers_rp, nlayers_rv], hyperparameters optimized through testing. [2,2,2,2,2] at initial.
        lookback: how many past time steps MCT considered to make current prediction. In previous study, 6 is a reasonable setting.
        sigma: a threshold range in [0, 1] on Score_ in pre-association
        ---------------------------

        Input
        ----------
        X_r_past : 6 past frames(assigned) ——radar detection torch.tensor which is shaped as (batch_size, num_ini_IDs, lookback, variables_r) —— (B, N, 6, 3), for V_r includes position and velocity information as (x, y, z) 
        X_r_cur  : 1 current frame(uassigned) —— (B, N, 3)
        X_cb_past: 6 past frames(assigned) ——bounding boxes extracted from camera data by AirSim which is shaped as (batch_size, num_ini_IDs, lookback, variables_c) —— (B, N, 6, 4) for V_c includes bounding box information as (xmin, ymin, xmax, ymax)     
        X_cb_cur : 1 current frame(uassigned) —— (B, N, 4)
        As implemented, these inputs are data after pre-processing described in the paper.

        Returns(Output)
        -------
        Y : A torch.tensor of shape (batch_size, num_ini_IDs, num_ini_Dets, num_ini_BBs) —— (B, N, N, N) where (N, N, N) is a A_cube at one time step
            
            These are the assignment result predicted by MCT. 
            e.g. (b, i, m, n) denotes that at current time step the relationship between the m-th radar detection, the n-th bounding box and the i-th identity, if the element is 1, the 
            the m-th radar detection and the n-th bounding box belongs to the i-th identity, otherwise not.
        """
        super(MultiCuesTracker, self).__init__()
        # --- Hyperparameters ---        
        self.ninp = ninp                           # [ninp_cp, ninp_cv, ninp_a, ninp_rp, ninp_rv]
        self.nhid = nhid                           # [nhid_cp, nhid_cv, nhid_a, nhid_rp, nhid_rv]
        self.nout = nout                           # [nout_cp, nout_cv, nout_a, nout_rp, nout_rv]
        self.nlayers = nlayers                     # [nlayers_cp, nlayers_cv, nlayers_a, nlayers_rp, nlayers_rv]
        self.lookback = lookback
        self.sigma = sigma                         # a threshold on Score_rbb
        # --- Sub-networks ---
        self.Position_model_C = LSTMtracker(ninp[0], nhid[0], nout[0], nlayers[0]) # Camera Position Model cx,cy
        self.Position_model_C.load_state_dict(torch.load('camera_p_weights.pt'))
        # set_parameter_requires_grad(self.Position_model_C, True)

        self.Velocity_model_C = LSTMtracker(ninp[1], nhid[1], nout[1], nlayers[1]) # Camera Velocity Model iou
        self.Velocity_model_C.load_state_dict(torch.load('camera_v_weights.pt'))
        # set_parameter_requires_grad(self.Velocity_model_C, True)

        self.Position_model_R = LSTMtracker(ninp[2], nhid[2], nout[2], nlayers[2]) # Radar Position Model x,y,z 
        self.Position_model_R.load_state_dict(torch.load('radar_p_weights.pt'))
        # set_parameter_requires_grad(self.Position_model_R, True)      


        self.score_fusion_C = BottleneckResidualBlock(2,1) # fuse Score_cpv
        self.score_fusion_R = BottleneckResidualBlock(1,1) # fuse Score_rpv

        # self.score_fusion_C = BottleneckNet(2,1) # fuse Score_cpv
        # self.score_fusion_R = BottleneckNet(1,1) # fuse Score_rpv
                

        self.S_rbb= Siamese(3, 2)             # Camera matrix module which used to predict the similarity score between a radar detection(x, y, z) and a bounding box(cx, cy) 
        self.S_rbb.load_state_dict(torch.load('rbb_SimScore_weights.pt'))
        set_parameter_requires_grad(self.S_rbb, True)
        self.flatten = nn.Flatten()
        #self.Voxel_mapping = Mapping(1, 1)

    def forward(self, X_r_past, X_r_cur, X_cb_past, X_cb_cur):     # assigned means that be sequeued according to ID in dimension N    
        B, N = X_r_past.shape[0], X_r_past.shape[1]                # X_r_past:(B, N, 6, 3)

        # Predict current BB state of each identity
        x_center_temp = torch.div(torch.add(X_cb_past[:, :, :, 0], X_cb_past[:, :, :, 2]), 2)     # (B, N, 6)
        y_center_temp = torch.div(torch.add(X_cb_past[:, :, :, 1], X_cb_past[:, :, :, 3]), 2)     # (B, N, 6)

        PC_tempin = torch.cat((x_center_temp.unsqueeze(-1), y_center_temp.unsqueeze(-1)), dim=-1)   # (B, N, 6, 2)

        cur_BB_prediction_center = torch.zeros((B, N, 2), requires_grad=True).cuda()
        for n in range(N):
            cur_BB_prediction_center[:, n, :] = self.Position_model_C(PC_tempin[:, n, :, :]) # (B, 2)
               
        # Calculate Score_cp and Score_cv        
        x_center_cur = torch.div(torch.add(X_cb_cur[:, :, 0], X_cb_cur[:, :, 2]), 2)         # (B, N)
        y_center_cur = torch.div(torch.add(X_cb_cur[:, :, 1], X_cb_cur[:, :, 3]), 2)         # (B, N)
        cur_bb_xy = torch.cat((x_center_cur.unsqueeze(-1), y_center_cur.unsqueeze(-1)), dim=-1) # (B, N, 2)

        PC_curin = cur_bb_xy.repeat(N, 1, 1, 1).permute(1, 0, 2, 3)  # (B, N, 2)-->(B, N, N, 2) the first N is added by replication along identity axis
        Score_cp = torch.mean(torch.pow(torch.sub(cur_BB_prediction_center.repeat(N, 1, 1, 1).permute(1, 2, 0, 3), PC_curin), 2), dim=-1) # (B, N, N)
        # Score_cp = torch.ones((B, N, N), requires_grad=True).cuda()                
        
        # Predict current IOU state of each identity
        total_iou = torch.zeros((B, N, self.lookback-1), requires_grad=True).cuda() 
        for b in range(B):
            for n in range(N):
                for t in range(self.lookback-1):
                    total_iou[b, n, t] = compute_iou(X_cb_past[b, n, t, :], X_cb_past[b, n, t+1, :])

        total_iou_trans = total_iou.unsqueeze(-1)                                         # (B, N, 5)-->(B, N, 5, 1)
        cur_mask = torch.zeros((B, N), requires_grad=True).cuda() 
        for n in range(N):
            cur_mask[:, n] = self.Velocity_model_C(total_iou_trans[:, n, :, :]).squeeze(-1) # (B, 1)
        cur_mask = cur_mask.repeat(N, 1, 1).permute(1, 2, 0)   # (B, N)-->(B, N, N) the second N is added by replication along bbox id axis
      
        # Calculate assumed IOU between current bboxes and last frame's bboxes of each identity 
        pre_Score_cv = torch.zeros((B, N, N), requires_grad=True).cuda()
        for b in range(B):
            for i in range(N):
                for j in range(N):
                    pre_Score_cv[b, i, j] = compute_iou(X_cb_past[b, i, -1, :], X_cb_cur[b, j, :])
                    
        # Calculate square error between predicted iou of each identity and assumed iou which is between current bboxes and last frame's bboxes of each identity
        Score_cv = torch.pow(torch.sub(cur_mask, pre_Score_cv), 2)    # (B, N, N)
        # Score_cv = torch.ones((B, N, N), requires_grad=True).cuda()

        # Calculated Score_cpv based on Score_cp and Score_cv
        cpv_fusion = torch.cat((Score_cp.unsqueeze(-1), Score_cv.unsqueeze(-1)), dim=-1).permute(0, 3, 1, 2) # (B, 2, N, N)
        Score_cpv = self.score_fusion_C(cpv_fusion).squeeze()
               
        #print(Score_cp[0,:,:])
        #print(Score_cv[0,:,:])
        #print(self.Score_cpv[:,:])
          
        # Predict current radar state of each identity
        cur_radar_prediction = torch.zeros((B, N, 3), requires_grad=True).cuda() 
        for n in range(N):
            cur_radar_prediction[:, n, :] = self.Position_model_R(X_r_past[:, n, :, :]) # (B, 3)
    
        #print(cur_radar_prediction[0,:,:])
        
        # Calculate Score_rp and Score_rv between current radar prediction of each identity and current radar detection
        Score_rp = torch.mean(torch.pow(torch.sub(cur_radar_prediction.repeat(N, 1, 1, 1).permute(1, 2, 0, 3), X_r_cur.repeat(N, 1, 1, 1).permute(1, 0, 2, 3)), 2), dim=-1) # (B, N, N)
                
        Score_r = self.score_fusion_R(Score_rp.unsqueeze(1)).squeeze() # (B, N, N)
        # Score_r = torch.ones((B, N, N), requires_grad=True).cuda()     
        #print(Score_rp[0,:,:])

        # Achieve correlation between current radar detections and bboxes   
        Score_rbb = torch.zeros((B, N, N), requires_grad=True).cuda() 
        for i in range(N):
            for j in range(N):                     
                Score_rbb[:, i, j] = self.S_rbb(X_r_cur[:, i, :], cur_bb_xy[:, j, :]).squeeze() # (Batch, 1)
        # Score_rbb = torch.ones((B, N, N), requires_grad=True).cuda() 
        
        # Use threshold sigma to prune brunches in Score_rbb
        # sigma_mask0 = torch.zeros_like(Score_rbb)
        # Score_rbb = torch.where(Score_rbb < self.sigma, sigma_mask0, Score_rbb)
        
        #Use GNN to prune brunches in Score_rbb
        # dis_mat = (1 - Score_rbb).cpu().detach().numpy()
        # mask_gnn = np.zeros((B, N, N))
        # for b in range(B):
        #     row_index, col_index = linear_sum_assignment(dis_mat[b, :, :])
        #     for n in range(N):
        #         mask_gnn[b, row_index[n], col_index[n]] = 1
        # mask_gnn = torch.tensor(mask_gnn, dtype=torch.float).cuda()
        # Score_rbb = torch.mul(mask_gnn, Score_rbb)

               
        # Final Association: A_cube calculation
        # self.A_cube = torch.zeros((B, N, N, N), requires_grad=True).cuda() # (identity, radar_detection, bounding_box),relationship between radar_detection, bounding_box and identity at current frame     

        A_cube = torch.mul(Score_r.repeat(N, 1, 1, 1).permute(1, 2, 3, 0), Score_cpv.repeat(N, 1, 1, 1).permute(1, 2, 0, 3))
        A_cube = torch.mul(Score_rbb.repeat(N, 1, 1, 1).permute(1, 0, 2, 3), A_cube)

        #ffusion_rc = torch.cat((self.flatten(self.Score_rpv), self.flatten(self.Score_cpv)), dim=1)
        #self.A_cube = self.SVM(ffusion_rc).reshape(B, N, N, N)
        """
        # NMS for A_cube
        for b in range(B):
            for i in range(N):
                if torch.sum(self.A_cube[b, i, :, :] > 0).item() != 0:
                    self.mask = (self.A_cube[b, i, :, :] == torch.max(self.A_cube[b, i, :, :])).to(dtype=torch.int32)
                    self.A_cube[b, i, :, :] = torch.mul(self.mask, self.A_cube[b, i, :, :])
        """

        """
        # relationship between radar_detection, bounding_box and identity at current frame
        for i in range(N):
            slice_for_target_i = A_cube[i,:,:]
            max_value, max_index = torch.max(slice_for_target_i, dim=0)
            row, col = torch.where(slice_for_target_i == max_value)
            print("The index of radar detection for identity %d is %d", (i, row+1))
            print("The index of bounding box for identity %d is %d", (i, col+1))
        """
        return A_cube

    # def loss_function(self, A_cube_hat, A_cube):
    #     # --- compute losses ---
    #     MSE = torch.nn.MSELoss()
    #     #BCE = torch.nn.BCELoss()
        
        
    #     # self.compress_r, _ = torch.max(A_cube, dim=2)
    #     # self.loss_cpv = MSE(self.Score_cpv, self.compress_r)

    #     # self.compress_c, _ = torch.max(A_cube, dim=3)
    #     # self.loss_rpv = MSE(self.Score_rp, self.compress_c)       
    #     self.loss_cube = MSE(A_cube_hat, A_cube.float())
    #     #self.loss_cube = torch.mean((A_cube_hat - A_cube.float())**2)
                
    #     #loss =self.loss_cube + self.loss_cpv / (self.loss_cpv / self.loss_cube).detach() + self.loss_a / (self.loss_a / self.loss_cube).detach() + self.loss_rpv / (self.loss_rpv / self.loss_cube).detach()
    #     #loss =self.loss_cube + self.loss_cpv / (self.loss_cpv / self.loss_cube).detach() + self.loss_rpv / (self.loss_rpv / self.loss_cube).detach()
    #     loss = self.loss_cube
    #     return loss


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
