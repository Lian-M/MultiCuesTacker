# These are the modules used in the Model
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg16_bn

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class LSTMtracker(nn.Module):
    """
    A network which predicts the current state of each identity.
    Input: (Batch, lookback, variable), lookback is the length of window in time dimension, variable is the number of variables observed at each time step.
    Output: (Batch, variabel), the output is the variables at current time step, hence the time dimension is 1 which is neglected.
    """   
    def __init__(self, input_size, hidden_size, output_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Appearance(nn.Module):
    """
    A network which predicts the current state of each identity.
    Input: appearance features extracted by VGG16. x--(Batch, lookback, 256), lookback is the length of window in time dimension.
                                                   y--(Batch, 256), 256 is the length of feature extracted by VGG16
    Output: (Batch, 1), the Appearance Score at current time step.
    """   
    def __init__(self, input_size, hidden_size, output_size, num_stacked_layers): # output_size should be set to 256 in the Model
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size) 
        self.fc2 = nn.Linear(output_size*2, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0)

    def forward(self, x, y):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        out_x, _ = self.lstm(x, (h0, c0))
        out_x = self.dropout(out_x[:, -1, :])
        out_x = self.fc1(out_x) # (B, 256)
        #out_y = self.fc1(y[:, -1, :]) # (B, 256)
        out_y = self.fc1(y) # (B, 256)
        out = torch.cat((out_x, out_y), dim=1).detach() # (B, 512)
        out = self.fc2(out)               # (B, 1)
        out = self.sigmoid(out)
        out = self.dropout(out)
        return out

class Siamese(nn.Module):
    """
    原理正确，但需要大量gt数据进行预训练才能用在MCT中
    A network which predicts the similarity score between a current radar detection and a current bbox.
    Input: the xyz coordinate of a current radar detection  x1--(Batch, 3)
           the xy center coordinate of a current bbox       x2--(Batch, 2)
    Output: score--(Batch, 1), a similarity score in range of [0, 1].
    """
    def __init__(self, rfeature_size, bbfeature_size): # rfeature_size is the length of radar detection position feature, bbfeature_size is the length of bb center feature
        super().__init__()
        
        self.fc1 = nn.Linear(rfeature_size, 8)
        self.BN1 = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, bbfeature_size)
        self.BN2 = nn.BatchNorm1d(2)

        self.fc11 = nn.Linear(bbfeature_size, 4)
        self.BN3 = nn.BatchNorm1d(4)
        self.fc22 = nn.Linear(4, 8)
        self.BN4 = nn.BatchNorm1d(8)


        self.fc3 = nn.Linear(8, 4)
        self.BN5 = nn.BatchNorm1d(4)
        self.fc4 = nn.Linear(4, 1)
        self.BN6 = nn.BatchNorm1d(1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        out1 = self.fc1(x1)
        #out1 = self.BN1(out1)
        out1 = self.relu(out1)     
        out1 = self.fc2(out1)    # fit a camera matrix to transform world xyz to image plane xy
        #out1 = self.BN2(out1)
        out1 = self.relu(out1)

        out1 = self.fc11(out1)
        #out1 = self.BN3(out1)
        out1 = self.relu(out1)
        out1 = self.fc22(out1)
        #out1 = self.BN4(out1)
        out1 = self.relu(out1)


        out2 = self.fc11(x2)
        #out2 = self.BN3(out2)
        out2 = self.relu(out2)
        out2 = self.fc22(out2)
        #out2 = self.BN4(out2)
        out2 = self.relu(out2)

        fout = torch.abs(out1-out2)
        fout = self.fc3(fout)
        #fout = self.BN5(fout)
        fout = self.relu(fout)
        fout = self.fc4(fout)
        fout = self.BN6(fout)
        fout = self.sigmoid(fout) # calculate the similarity between this tranformed xy to the exist xy
        #cosine_similarity = F.cosine_similarity(out, x2)
        return fout

class LSTM_Siamese(nn.Module):
    """
    原理正确，但需要大量gt数据进行预训练才能用在MCT中
    A network which predicts the similarity score between a image track(past) and a current image.
    Input: Images of a track(past) and a current image. x1--(Batch, lookback, 3, W, H), images of an identity at num_lookback past frames.
                                                        x2--(Batch, 3, W, H), a image of one detection at current frame. 
    Output: score--(Batch, 1), similarity score between a image track(past) and a current image.
    """   
    def __init__(self, input_size, hidden_size, output_size, num_stacked_layers): # output_size should be set to 256 in the Model
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(256, 256) 
        self.fc2 = nn.Linear(256, 1)


        self.VGG_16 = vgg16_bn(weights=True)
        self.VGG_16.classifier = nn.Sequential(*list(self.VGG_16.classifier.children())[:-6])
        self.VGG_16.classifier[0] = nn.Linear(in_features=self.VGG_16.classifier[0].in_features, out_features=256)

        # self.VGG_16 = VGG16(256)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.BN = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        batch_size = x1.size(0)
        lookback_w = x1.size(1)
        
        out_x1 = torch.cat((self.VGG_16(x1[:,0,:,:,:]).unsqueeze(1), self.VGG_16(x1[:,1,:,:,:]).unsqueeze(1), self.VGG_16(x1[:,2,:,:,:]).unsqueeze(1), self.VGG_16(x1[:,3,:,:,:]).unsqueeze(1), self.VGG_16(x1[:,4,:,:,:]).unsqueeze(1), self.VGG_16(x1[:,5,:,:,:]).unsqueeze(1)), dim=1)
        out_x2 = self.VGG_16(x2)  # (Batch, 256)
        
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        out_x11, _ = self.lstm(out_x1, (h0, c0))

        out_x11 = self.fc1(out_x11[:, -1, :]) # (B, 256)
        out_x11 = self.relu(out_x11)
        out_x22 = self.fc1(out_x2) # (B, 256)
        out_x22 = self.relu(out_x22)


        #out = torch.cat((out_x11, out_x22), dim=1).detach() # (B, 512)
        out = torch.abs(out_x11-out_x22)
        out = self.fc2(out)               # (B, 1)
        out = self.BN(out)
        out = self.sigmoid(out)
        return out

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckResidualBlock, self).__init__()

        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(3, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.sigmoid(out)
        return out

class BottleneckNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckNet, self).__init__()

        # 1x1 conv
        self.BNN1 = BottleneckResidualBlock(in_channels, 1)
        self.BNN2 = BottleneckResidualBlock(1, 1)
        self.BNN3 = BottleneckResidualBlock(1, out_channels)

    def forward(self, x):
        out = self.BNN1(x)
        out = self.BNN2(out)
        out = self.BNN3(out)

        return out

class Conv_score(nn.Module):
    def __init__(self, m, n):
        super(Conv_score, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=m, out_channels=16, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=n, kernel_size=1)
        self.BN1 = nn.BatchNorm2d(16)
        self.BN2 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_1(x)             # (B, 2, N, N)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.BN2(out)
        out = self.sigmoid(out)

        return out

class FC_score(nn.Module):
    def __init__(self, m, n):
        super(FC_score, self).__init__()
        self.fc1 = nn.Linear(m, 512)
        self.fc2 = nn.Linear(512, n)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)             # (B, 2, N, N)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out
class Mapping(nn.Module):
    def __init__(self, m, n):
        super(Mapping, self).__init__()
        self.conv1 = nn.Conv3d(m, 8, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=1, stride=1)
        self.conv3 = nn.Conv3d(16, 8, kernel_size=1, stride=1)
        self.conv4 = nn.Conv3d(8, n, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.sigmoid(out)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class Attention_Score(nn.Module):
    def __init__(self):
        super(Attention_Score, self).__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # 输入2通道，输出16通道，3x3卷积，步长为2，padding为1
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 输入16通道，输出32通道，3x3卷积，步长为2，padding为1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7),                      # 输入32通道，输出64通道，7x7卷积
        )
        # CBAM
        self.CBAM = CBAM()

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),             # 输入64通道，输出32通道，7x7卷积转置
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 输入32通道，输出16通道，3x3卷积转置，步长为2，padding为1，输出padding为1
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输入16通道，输出1通道，3x3卷积转置，步长为2，padding为1，输出padding为1
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class SVM(nn.Module):
    def __init__(self, m, n):
        super(SVM, self).__init__()
        self.linear1 = nn.Linear(m, 128)
        self.linear2 = nn.Linear(128, n)
        self.dropout = nn.Dropout(0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.sigmoid(out)
        out = self.dropout(out)
        return out

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return nn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer

class VGG16(nn.Module):
    # input format: [N, C, H, W]
    def __init__(self, n_classes=256):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        self.layer4 = vgg_conv_block([256,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        self.layer5 = vgg_conv_block([512,512,512], [512,512,512], [3,3,3], [1,1,1], 2, 2)
        # self.flatten = nn.Flatten()
        # FC layers
        self.layer6 = vgg_fc_layer(7*7*512, 256)
        # self.layer6 = vgg_fc_layer(7*7*512, 4096)
        # self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        # self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        #print(np.shape(out))
        out = self.layer6(out)
        #out = self.layer7(out)
        #out = self.layer8(out)
        #print(np.shape(out))
        return out

# class VGG16(nn.Module):
#     # input format: [N, C, H, W]
#     def __init__(self, n_classes=256):
#         super(VGG16, self).__init__()

#         # Conv blocks (BatchNorm + ReLU activation added in each block)
#         self.clayer1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.BN1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         #self.maxpooling1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

#         self.clayer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.BN2 = nn.BatchNorm2d(64)
#         self.maxpooling2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

#         self.clayer3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.BN3 = nn.BatchNorm2d(128)
#         #self.maxpooling3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

#         self.clayer4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.BN4 = nn.BatchNorm2d(128)
#         self.maxpooling4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

#         self.flatten = nn.Flatten()
#         # FC layers
#         self.FC = nn.Linear(56*56*128, 256)


#     def forward(self, x):
#         out = self.clayer1(x)
#         out = self.BN1(out)
#         out = self.relu(out)
#         #out = self.maxpooling1(out)

#         out = self.clayer2(out)
#         out = self.BN2(out)
#         out = self.relu(out)
#         out = self.maxpooling2(out)

#         out = self.clayer3(out)
#         out = self.BN3(out)
#         out = self.relu(out)
#         #out = self.maxpooling3(out)

#         out = self.clayer4(out)
#         out = self.BN4(out)
#         out = self.relu(out)
#         out = self.maxpooling4(out)

#         out = self.flatten(out)
#         out = self.FC(out)

#         return out