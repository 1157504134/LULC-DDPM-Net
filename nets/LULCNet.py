import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from nets.se import ChannelSpatialSELayer, BilinearAttention
import os
import torchvision.transforms as transforms
from fightingcv_attention.attention.SKAttention import SKAttention


import torch
from sklearn.cluster import KMeans
def compress_images(images: torch.Tensor, n_clusters: int=5):
    # Get the device (cuda or cpu) of images
    device = images.device

    # Move images tensor to cpu and convert to numpy
    images_np = images.permute(0, 2, 3, 1).detach().cpu().numpy()

    batch_size, height, width, _ = images_np.shape

    # Reshape the images tensor into a (N, 3) vector, where N is the number of pixels and 3 represents the RGB channels
    images_reshaped = images_np.reshape(-1, 3)

    # Create K-means model with explicit setting of n_init
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)

    # Fit K-means on the pixel values
    kmeans.fit(images_reshaped)

    # Get the labels for each pixel
    labels = kmeans.labels_

    # Get the cluster centers' colors
    cluster_centers = kmeans.cluster_centers_

    # Assign each pixel its corresponding cluster center color based on the clustering result
    compressed_images = cluster_centers[labels].astype('uint8')

    # Reshape the compressed image back to the original image shape
    compressed_images = compressed_images.reshape(batch_size, height, width, 3)

    # Convert back to tensor, send back to the input device (cuda or cpu), 
    # and permute back to (batch_size, channels, height, width)
    compressed_images = torch.tensor(compressed_images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    return compressed_images.cuda()

# 归一化
def normalize_tensor(t):
    return (t - t.min()) / (t.max() - t.min())
# 归一化
def normalize_tensor255(tensor):
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    normalized = (tensor - min_val) / (max_val - min_val)
    normalized = (normalized * 255).to(torch.uint8)
    return normalized/255.0




# 通道减少
# class ChannelReducer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ChannelReducer, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         x = x.float()
#         return self.conv(x)

class GatedFusion(nn.Module):
    def __init__(self, channel_num):
        super(GatedFusion, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, channel_num, 1, 1))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, feat):
        return x * torch.sigmoid(self.weights) + feat * torch.sigmoid(1 - self.weights)


class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelReducer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.float()

        attention = torch.mean(x, dim=(2, 3), keepdim=True)  # 通过平均池化计算通道注意力权重
        attention = self.sigmoid(attention)  # 通过sigmoid激活函数将注意力权重归一化到0-1范围
        x = x * attention  # 通道注意力权重与输入特征相乘

        return self.conv(x)


class FusionNet(nn.Module):
    def __init__(self, in_channels,outchannel, dropout_rate=0.1):
        super(FusionNet, self).__init__()

        self.ChannelSpatialSELayer = ChannelSpatialSELayer(in_channels)
        self.ChannelSpatialSELayer_x = ChannelSpatialSELayer(3)

        # 通道数转换
        self.channel_reducer = ChannelReducer(in_channels, outchannel)
        self.sigmod = nn.Sigmoid()

        # 特征融合卷积层
        self.fusion_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm2d = nn.BatchNorm2d(outchannel)

        # 引入Dropout
        self.dropout = nn.Dropout(dropout_rate)

        self.conv_cat = nn.Conv2d(2 * 3, 3, 1, 1, padding=0, bias=True)
        self.fusion_layer = GatedFusion(3)

    def forward(self, x, feat):
        # feat = self.ChannelSpatialSELayer(feat)
        # 通道数转换
        feat = self.channel_reducer(feat)
        # 特征融合
        fused_feat = self.fusion_conv(feat)
        fused_feat = self.ChannelSpatialSELayer_x(fused_feat)
        fused_feat = normalize_tensor(fused_feat)

        return fused_feat


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x

    # -----------------------------------------#


#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class Cat(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Cat, self).__init__()

        self.inchannel = inchannel
        self.outchannel = outchannel

        self.cat_conv = nn.Sequential(
            nn.Conv2d( self.inchannel, self.outchannel,1),
            nn.BatchNorm2d(self.outchannel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.cat_conv(x)
        return x



class LULCNet(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(LULCNet, self).__init__()
        if backbone == "xception":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            # ----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

        self.ChannelSpatialSELayer = ChannelSpatialSELayer(3)
        # self.conv_se_layer = ConvSELayer(8832, 3)
        self.ChannelSpatialSELayer_8832 = ChannelSpatialSELayer(8832)
        # self.fusion_net = FusionNet(in_channels=384)
        self.CatX = Cat(256*2,256)
        self.Catf = Cat(48*2,48)
        self.ConvYaosuo = nn.Conv2d(384, 3, 3)
        self.Conv128_3_3 = nn.Sequential(
            nn.Conv2d(128,64,3,1),
            nn.Conv2d(64,3,1,1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
             nn.Dropout(0.1),

        )
        self.Conv128_3 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1),
            nn.Conv2d(64, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
        )
        self.Conv256_3 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
            nn.Conv2d(128, 64, 1, 1),
            nn.Conv2d(64, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.Conv512_3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 128, 1, 1),
            nn.Conv2d(128, 64, 1, 1),
            nn.Conv2d(64, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.Conv1024_3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1),
            nn.Conv2d(512, 256, 3, 1),
            nn.Conv2d(256, 128, 3, 1),
            nn.Conv2d(128, 64, 3, 1),
            nn.Conv2d(64, 3, 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.SKAttention = SKAttention(channel=3,reduction=8)
        self.Conv36_6 = nn.Sequential(
            nn.Conv2d(36, 6, 1, 1),
        )
        self.batch3=nn.BatchNorm2d(3)
        self.relu=nn.ReLU(inplace=True)
        self.Conv256_128 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.Conv512_128 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.Conv1024_128 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1),
            nn.Conv2d(512, 256, 3, 1),
            nn.Conv2d(256, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.Conv128_128 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
           
        )
        # feat 通道整合
        self.channelcat = nn.Sequential(
               nn.Conv2d(128, 64, 3, 1),
               nn.BatchNorm2d(64),
               nn.LeakyReLU(inplace=True),
               nn.Conv2d(64,3,1,1),
               nn.BatchNorm2d(3),
               nn.Tanh()
        )
        
        self.conv2d15 = nn.Sequential(
            nn.Conv2d(6,3,1,1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
        self.SKAttention256 = SKAttention(256)
        
        self.cat_x = nn.Sequential(
            nn.Conv2d(256*2,256,1)
        )
        self.cat_low_level_features = nn.Sequential(
            nn.Conv2d(48*2,48,1)
        )
    
        
        
    # 获取低层特征
    def EXtractLowFea(self,x):
        H, W = x.size(2), x.size(3)
        x = self.SKAttention(x)
         # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        low_level_features = self.shortcut_conv(low_level_features)
        # -----------------------------------------
        # 48通道
        return low_level_features
        
    def EXtract(self,x):
        H, W = x.size(2), x.size(3)
         # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        # -----------------------------------------

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
    # 转存图片
    def save_tensor_as_png(self,folder_path, tensor):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 获取batchsize
        batchsize = tensor.size(0)
        
        # 将Tensor转换为PIL图像
        to_pil = transforms.ToPILImage()
        
        # 保存每张图像为PNG文件
        for i in range(batchsize):
            img = to_pil(tensor[i])
            img.save(os.path.join(folder_path, f'image{i+1}.png'))

    
    def Encoder(self,x):
        H, W = x.size(2), x.size(3)
         # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
       
        x = self.aspp(x)
        x = self.SKAttention256(x)
        low_level_features = self.shortcut_conv(low_level_features)
        # -----------------------------------------
        return x,low_level_features
        
    def Decoder(self,x,H,W,low_level_features):
         # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x 
  
        
    

    def forward(self, x, feat):
        g= x
        H, W = x.size(2), x.size(3)
        # 特征融合

        feat256_128 = self.Conv256_3(feat[1])
        feat256_128 = F.interpolate(feat256_128, size=(H, W), mode='bilinear', align_corners=True)
        fea = feat256_128
        fea = normalize_tensor255(fea)
        # # for i in range(1,8):
        #     # kmeans
        #     # feam=  compress_images(fea,i)
        #     # fea = self.conv2d15(fea)
        #     # feam = normalize_tensor255(feam)
        #     # self.save_tensor_as_png("./imgs_feat/fea_"+str(i),feam)
        # kmeans
        # fea=  compress_images(fea,5)
        # fea = self.conv2d15(fea)
        # fea = normalize_tensor255(fea)
        # self.save_tensor_as_png("./imgs_feat/fea",fea)
        # self.save_tensor_as_png("./imgs_feat/g",g)
        
        fea  =self.SKAttention(fea)
        # self.save_tensor_as_png("./imgs_feat/Attionfea",fea)
        x  =self.SKAttention(x)
        # self.save_tensor_as_png("./imgs_feat/Attionx",x)
        # 提取
        fea,low_level_featuresfea = self.Encoder(fea)
        x,low_level_featuresx= self.Encoder(x)
        # 拼接层
        low_level_featuresx = torch.cat((low_level_featuresx,low_level_featuresfea),1)
        
        low_level_featuresx =  self.cat_low_level_features(low_level_featuresx)
        
        x = self.Decoder(x,H,W,low_level_featuresx)
        
        return x


