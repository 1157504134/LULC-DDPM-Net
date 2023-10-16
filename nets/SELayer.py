import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FusedFeatures(nn.Module):
    def __init__(self, num_features):
        super(FusedFeatures, self).__init__()
        self.se_weights = SELayer(num_features)

    def forward(self, x_feat_list, H, W):
        fused_feat = sum([F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=True) for feat in x_feat_list])
        fused_feat = self.se_weights(fused_feat)
        return fused_feat

# # Your network part
# class YourNetwork(nn.Module):
#     def __init__(self):
#         super(YourNetwork, self).__init__()
#         self.fused_features = FusedFeatures(128)  # assuming all these features are reduced to 128 channels
#         # then put fused_features in the place where original fusion happened

#     def forward(self, x):
#         # Assuming feat128_256, feat256_128, feat512_64, feat1024_32, feat1024_16 are computed somewhere
#         inputs = [feat128_256, feat256_128, feat512_64, feat1024_32, feat1024_16]
#         fused = self.fused_features(inputs, H, W)
#         fused = torch.relu(fused.float())
#         fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=True)
#         return fused