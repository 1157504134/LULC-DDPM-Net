import os
import torch.nn as nn
import torch
from nets.LULC_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
from tqdm import tqdm
import numpy as np
from utils.utils import get_lr
from utils.utils_metrics import f_score
from misc.print_diffuse_feats import print_feats
import core.metrics as Metrics
from nets.get_tensor import get_Tenosr


import torch
import torch.nn as nn

class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelReducer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
class FusionNet(nn.Module):
    def __init__(self, in_channels):
        super(FusionNet, self).__init__()

        # 通道数转换
        self.channel_reducer = ChannelReducer(in_channels, 3)

        # 特征融合卷积层
        self.fusion_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, feat):
        # 通道数转换
        feat = self.channel_reducer(feat)

        # 特征融合
        fused_feat = self.fusion_conv(feat)
        fused_feat = self.relu(fused_feat)

        # 特征融合后与输入特征x相加
        output = x + fused_feat

        return output


def get_Tenosr222(diffusion,opt,image,m = 0):
    diffusion.feed_data(image)
    F_img = []
    # steps = [50,100,400]
    steps = [100]
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
    feat_vectors = []
    for i in range(3):
        # feat_vectors.append(F_img[0][i])
        feat_vectors.append(F_img[0][i])
        # feat_vectors.append(F_img[2][i])
        
    batch_size = image.size(0)
    print(batch_size)

    feat_size = torch.Size([batch_size, 3, 128, 256, 256])
    feat_tensor = torch.zeros(feat_size)
    for i in range(len(feat_vectors)):
        feat_tensor[:, i, :, :, :] = feat_vectors[i]   
    # print('feat_tensor.shape: ', feat_tensor.shape)
    # print("x_size: ", x_size[1])
    # print("feat_size: ", feat_size[1])
    # x = imgs
    # # 处理特征向量
    # fusion_net = FusionNet(in_channels=384)
    # fusion_net = fusion_net.cuda(local_rank)
    feat_tensor = feat_tensor.view(feat_tensor.size(0), -1, feat_tensor.size(3), feat_tensor.size(4))
    return feat_tensor



# 将cuda转为cpu
def move_tensors_to_cpu(tensors):
    cpu_tensors = []
    for tensor in tensors:
        if tensor.device.type == 'cuda' and tensor.device.index == 0:
            tensor = tensor.cpu().detach().numpy()
        cpu_tensors.append(tensor)
    return cpu_tensors

def get_list_dimensions(lst):
    dimensions = []
    while isinstance(lst, list):
        dimensions.append(len(lst))
        if len(lst) > 0:
            lst = lst[0]
        else:
            break
    return dimensions
def fit_one_epoch(diffusion,opt,model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir,local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                # # batch=3
                # # size(3,3,256,256)
                # # 扩散特征提取
                # # 把数据放在diffusion中
                # diffusion.feed_data(imgs)
                # # print('imgs.shape: ', imgs.shape)
                # F_img = []
                # # print( opt['model_cd']['t'])
                # steps = [50,100,400]
                # # steps = [100]
                # for t in steps :
                #     # print(t)
                #     fe,fd = diffusion.get_IMG_feats(t=t)
                #     # print('fe.shape: ', len(fe),t)
                #     # print('fd.shape: ', len(fd),t)
                #     if opt['model_cd']['feat_type'] == "dec":
                #         F_img.append(fd)
                #         del fd
                      
                #         # Uncommet the following line to visualize features from the diffusion model
                #         # for level in range(0, len(fd)):
                #         #     print_feats(opt=opt, train_data=imgs, feats_A=fd, level=level, t=t)
                     
                #     else:
                #        F_img.append(fe)
                #     del fe
                # # print(len(F_img[0]))      #15
                # # print(len(F_img[1]))      #15  
                # # print(len(F_img[2]))      #15
                # # print(len(F_img))         #3
                # # for i  in range(len(F_img)):
                # #     print(i)
                # #     for j in range(len(F_img[i])):
                # #         print(F_img[i][j].size())
                
                
          
                
                
                
                # feat_vectors = []
                # for i in range(3):
                #     # feat_vectors.append(F_img[0][i])
                #     feat_vectors.append(F_img[0][i])
                #     # feat_vectors.append(F_img[2][i])
                    
                # batch_size = imgs.size(0)
          
                # feat_size = torch.Size([batch_size, 3, 128, 256, 256])
                # feat_tensor = torch.zeros(feat_size)
                # for i in range(len(feat_vectors)):
                #     feat_tensor[:, i, :, :, :] = feat_vectors[i]   
                # # print('feat_tensor.shape: ', feat_tensor.shape)
                # # print("x_size: ", x_size[1])
                # # print("feat_size: ", feat_size[1])
                # # x = imgs
                # # # 处理特征向量
                # # fusion_net = FusionNet(in_channels=384)
                # # fusion_net = fusion_net.cuda(local_rank)
                # feat_tensor = feat_tensor.view(feat_tensor.size(0), -1, feat_tensor.size(3), feat_tensor.size(4))
                # feat_tensor = feat_tensor.cuda(local_rank)
                # del F_img
                # del feat_vectors
                # # output = fusion_net(x, feat_tensor)
           
                # # print('x.shape: ', x.shape)
                # # print('output.shape: ', output.shape)
                
                # # 将处理后的特征向量与输入x拼接
    
                
                feat_tensor = get_Tenosr(diffusion,opt,imgs)
                feat_tensor = feat_tensor
                  
                
                    
                
                


                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs, feat_tensor)
            #----------------------#
            #   计算损失
            #----------------------#
            # 组合loss
            if focal_loss:
                # 使用三个loss 平均
                # 这个参数是根据数据集自己调的
                # 
                loss1 = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                loss2 = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                loss3 =   main_dice = Dice_loss(outputs, labels)
                loss = loss1 + loss2 +loss3
            else:
                # 单纯的使用CELoss
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                # DiceLOss与CEloss组合
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs,feat_tensor)
                #----------------------#
                #   计算损失
                #----------------------#
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    _f_score = f_score(outputs, labels)
                    
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train(imgs, feat_tensor)
            #----------------------#
            #   计算损失
            #----------------------#
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice
            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'f_score'   : val_f_score / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train,diffusion,opt)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        del feat_tensor