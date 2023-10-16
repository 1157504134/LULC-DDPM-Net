"""
Squeeze and Excitation Module
*****************************

Collection of squeeze and excitation classes where each can be inserted as a block into a neural network architechture

    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_

"""

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)
        # print(f"ChannelSpatialSELayer:{output_tensor.size()}")
        return output_tensor


class SELayer(Enum):
    """
    Enum restricting the type of SE Blockes available. So that type checking can be adding when adding these blockes to
    a neural network::

        if self.se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])

        elif self.se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif self.se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
    """
    NONE = 'NONE'
    CSE = 'CSE'
    SSE = 'SSE'
    CSSE = 'CSSE'

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.conv2s = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.conv2s.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x,weights=None):

        for i, conv in enumerate(self.conv2s):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v




class BilinearAttention(nn.Module):
    def __init__(self, channels):
        super(BilinearAttention, self).__init__()
        self.channels = channels
        # Q_transform
        self.theta = nn.Sequential(nn.Conv2d(channels, channels // 8, kernel_size=1),
                                   nn.BatchNorm2d(channels // 8))
        # K_transform
        self.phi = nn.Sequential(nn.Conv2d(channels, channels // 8, kernel_size=1),
                                 nn.BatchNorm2d(channels // 8))
        # V_transform
        self.g = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1),
                               nn.BatchNorm2d(channels // 2))
        # softmax to calculate weights
        self.softmax = nn.Softmax(dim=-1)
        # final transform
        self.final_transform = nn.Sequential(nn.Conv2d(channels // 2, channels, kernel_size=1),
                                             nn.BatchNorm2d(channels))

    def forward(self, x, fea):
        b, _, h, w = x.size()

        # apply transformations
        theta = self.theta(x).view(b, -1, h * w)
        phi = self.phi(fea).view(b, -1, h * w)
        g = self.g(fea).view(b, -1, h * w)

        # calculate weights
        attention = self.softmax(torch.bmm(theta.transpose(1, 2), phi))

        # calculate final results
        out = torch.bmm(g, attention.transpose(1, 2)).view(b, -1, h, w)
        out = self.final_transform(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, src):
        attn_output, _ = self.attention(x, src, src)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        mlp_output = self.mlp(x)
        x = x + self.dropout(mlp_output)
        x = self.norm2(x)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_heads=8):
        super(FeatureFusion, self).__init__()
        self.proj_x = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.proj_y = nn.Conv2d(out_channels, out_channels, kernel_size=1)  # Assuming y matches out_channels
        self.transformer = TransformerBlock(out_channels, num_heads)

    def forward(self, x, y):
        B, C, H, W = x.shape
        x = self.proj_x(x).flatten(2).permute(2, 0, 1)
        y = self.proj_y(y).flatten(2).permute(2, 0, 1)
        z = self.transformer(x, y)
        z = z.permute(1, 2, 0).reshape(B, 6, H, W)
        return z

# Using the module in your DeepLab model
# ...
# self.feature_fusion = FeatureFusion(in_channels=6)  # Use 6 for your input channel size
# ...
# Using the module in your DeepLab model
# ...
# self.feature_fusion = FeatureFusion()
# # ...
#
# def forward(self, x, feat):
#     # ...
#     fused_feat = self.feature_fusion(x, feat)
#     # ...
#

if __name__ == "__main__":
    g = nn.Conv2d(320,3,kernel_size=(1,1))
    x = torch.rand(1,320,25,25)
    model = ChannelSpatialSELayer(320,320)
    x= model(x)
    x= g(x)
    print(x.size())