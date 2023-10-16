import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
# from nets.SELayer import ConvSELayer
def feat_processing(feat):
    # 对feat列表中的每个特性图谱进行上采样到256x256，并在通道（索引1）维度上进行拼接
    feat_upsampled = [F.interpolate(f, size=(256, 256), mode='bilinear', align_corners=True) for f in feat]
    feat_combined = torch.cat(feat_upsampled, dim=1)
    
    return feat_combined

def Torchadd(l1, l2, l3):
        return l1 + l2 + l3
def get_Tenosr(diffusion,opt,image,m = 0):

    diffusion.feed_data(image)
    F_img = []
    # steps = [50,100,400]
    steps =[100]
    for t in steps :
        # print(t)
        fe,fd = diffusion.get_IMG_feats(t=t)
        # print('fe.shape: ', len(fe),t)
        # print('fd.shape: ', len(fd),t)
        if opt['model_cd']['feat_type'] == "dec":
            F_img.append(fd)
            del fd
            # Uncommet the following line to visualize features from the diffusion model
            # for level in range(0, len(fd)):
            #     print_feats(opt=opt, train_data=imgs, feats_A=fd, level=level, t=t)
            
        else:
            F_img.append(fe)
        del fe
    # print(len(F_img[0][0]))
    # print(len(F_img[1][0]))
    # print(len(F_img[2][0]))
  
    feat_tensor1 =F_img[0]
    # feat_tensor2 = F_img[1]
    # feat_tensor3 = F_img[2]
    feat_1 =Torchadd(feat_tensor1[0],feat_tensor1[1],feat_tensor1[2]) #128 256 256
    feat_2= Torchadd(feat_tensor1[3], feat_tensor1[4], feat_tensor1[5]) #256 128 128
    feat_3= Torchadd(feat_tensor1[6], feat_tensor1[7], feat_tensor1[8]) #512 64 64
    feat_4 = Torchadd(feat_tensor1[9], feat_tensor1[10], feat_tensor1[11])
    feat_5 = Torchadd(feat_tensor1[12], feat_tensor1[13], feat_tensor1[14])#1024 16 16

    # feat_2 = torch.cat((feat_tensor2[0], feat_tensor2[1], feat_tensor2[2]), 1)
    # feat_3 = torch.cat((feat_tensor3[0], feat_tensor3[1], feat_tensor3[2]), 1)
    feat = [feat_1,feat_2,feat_3,feat_4,feat_5]
    return feat



