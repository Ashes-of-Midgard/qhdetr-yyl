import torch
import torch.nn as nn
import torch.nn.functional as F
from .lsq_plus import *
from ._quan_base_plus import *

class AMP(nn.Module):   # c,h,w -> q,h,w
    def __init__(self, in_channels=256, q=300, groups=32, tau=1.2):
        super(AMP, self).__init__()
        self.conv1 = Conv2dLSQ(in_channels, in_channels, kernel_size=5, padding=2, nbits_w=4)
        self.conv2 = Conv2dLSQ(in_channels, in_channels, kernel_size=3, padding=1, nbits_w=4)
        self.conv3 = Conv2dLSQ(in_channels, q, kernel_size=3, padding=1, nbits_w=4)
        self.gn1 = nn.GroupNorm(groups, in_channels)
        self.gn2 = nn.GroupNorm(groups, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tau = tau 

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.conv3(x)
        # 对空间维度进行softmax归一化
        x = x * self.tau
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0), x.size(1), -1)  # (batch_size, q, h*w)
        x = F.softmax(x, dim=2)  # 对h*w维度进行softmax
        x = x.view(x.size(0), x.size(1), h, w)  # 还原回(batch_size, q, h, w)
        return x
    
class WP(nn.Module):
    def __init__(self):
        super(WP, self).__init__()
        self.A_act = ActLSQ(nbits_a=4,in_features=8)
        self.F_act = ActLSQ(nbits_a=4,in_features=8)

    def forward(self, F, A):
        batch_size, c, h, w = F.size()
        q = A.size(1)
        
        F = F.view(batch_size, c, h * w).permute(0, 2, 1)  # (batch_size, h*w, c)
        A = A.view(batch_size, q, h * w)  # (batch_size, q, h*w)

        A = self.A_act(A)
        F = self.A_act(F)
        F_P = torch.bmm(A, F)  # (batch_size, q, c)
        
        return F_P
    
class CR(nn.Module):
    def __init__(self, in_channels=256):
        super(CR, self).__init__()
        self.fc1 = LinearLSQ(in_channels, in_channels, nbits_w=4, bias=True)
        self.fc2 = LinearLSQ(in_channels, in_channels, nbits_w=4, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
    
class SAPM(nn.Module):
    def __init__(self, in_channels=256, q=300, groups=32, tau=1.2):
        super(SAPM, self).__init__()
        self.amp = AMP(in_channels, q, groups, tau)
        self.wp = WP()
        self.cr = CR(in_channels)
        self.F_C_act = ActLSQ(nbits_a=4,in_features=8)
        self.F_P_act = ActLSQ(nbits_a=4,in_features=8)

    def forward(self, F):
        A = self.amp(F)
        F_P = self.wp(F, A)
        F_C = self.cr(F_P)

        F_C = self.F_C_act(F_C)
        F_P = self.F_P_act(F_P)

        F_O = F_C * F_P
        return F_O
    
class SAPM_Deformable(nn.Module):
    def __init__(self, in_channels=256, q=300, groups=32, tau=1.2, num_scales=4):
        super(SAPM_Deformable, self).__init__()
        self.num_scales = num_scales
        self.amps = nn.ModuleList([AMP(in_channels, q, groups, tau) for _ in range(num_scales)])
        self.wp = WP()
        self.cr = CR(in_channels)
        self.F_C_act = ActLSQ(nbits_a=4,in_features=8)
        self.F_P_act = ActLSQ(nbits_a=4,in_features=8)

    def forward(self, multi_scale_features):
        pooled_features = []
        
        for i in range(self.num_scales):
            F = multi_scale_features[i]
            A = self.amps[i](F)
            F_P = self.wp(F, A)
            pooled_features.append(F_P)
        
        # Average pooled features from different scales
        F_P_avg = torch.mean(torch.stack(pooled_features), dim=0)
        
        F_C = self.cr(F_P_avg)

        F_C = self.F_C_act(F_C)
        F_P_avg = self.F_P_act(F_P_avg)

        F_O = F_C * F_P_avg
        return F_O

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            LinearLSQ(n, k, nbits_w=4, bias=True) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SACO(nn.Module):
    def __init__(self, in_channels=256, q=300, num_heads=8, roi_size=7, num_decoder_layers=6):
        super(SACO, self).__init__()
        self.sapm_global = SAPM(in_channels, q)
        self.sapm_local = SAPM(in_channels, q)
        self.roi_align = RoIAlign((roi_size, roi_size), spatial_scale=1.0, sampling_ratio=-1)
        self.cross_attention = MultiHeadCrossAttention(in_channels, num_heads)
        self.box_head = BoxHead(in_channels, 4)  # For bounding box prediction
        self.num_decoder_layers = num_decoder_layers

    def forward(self, F_E, initial_queries, positional_queries, rois):
        # Initial global SAPM for content query initialization
        Q_c0 = self.sapm_global(F_E)
        
        # Initial content query
        Q = initial_queries + Q_c0 + positional_queries

        for _ in range(self.num_decoder_layers):
            # Cross attention
            Q = self.cross_attention(Q, F_E, F_E)
            
            # Predict bounding boxes
            B = self.box_head(Q)

            # RoI-Align to extract local features
            rois = B  # Use the predicted bounding boxes as RoIs
            rois = rois.view(-1, 4)  # Ensure the correct shape for RoI-Align

            local_features = self.roi_align(F_E, rois)

            # Local SAPM to refine content query
            Q_c_local = self.sapm_local(local_features)

            # Update content query
            Q = Q + Q_c_local
        
        return Q
    
def convert_to_rois(outputs_coord, feature_shape):
    batch_size, q, _ = outputs_coord.shape
    height, width = feature_shape

    # 将中心坐标和宽高转换为左上角和右下角坐标
    center_x = outputs_coord[..., 0] * width
    center_y = outputs_coord[..., 1] * height
    half_w = outputs_coord[..., 2] * width / 2
    half_h = outputs_coord[..., 3] * height / 2

    x1 = center_x - half_w
    y1 = center_y - half_h
    x2 = center_x + half_w
    y2 = center_y + half_h

    # 确保坐标在特征图范围内
    x1 = torch.clamp(x1, 0, width)
    y1 = torch.clamp(y1, 0, height)
    x2 = torch.clamp(x2, 0, width)
    y2 = torch.clamp(y2, 0, height)

    # 拼接坐标到形状为 [batch_size, 5]
    rois = torch.stack([x1, y1, x2, y2], dim=-1)
    batch_idx = torch.arange(batch_size, device=outputs_coord.device).view(-1, 1, 1).expand(batch_size, q, 1)
    rois = torch.cat((batch_idx, rois), dim=-1).view(batch_size * q, 5)

    return rois