from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.DINet import DINet
from models.Syncnet import SyncNetPerception
from utils.training_utils import get_scheduler, update_learning_rate,GANLoss
from config.config import DINetTrainingOptions
from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
from dataset.dataset_DINet_clip_xjx256 import DINetDataset


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

import cv2
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    '''
            clip training code of DINet
            in the resolution you want, using clip training code after frame training
            
        '''
    # load config
    opt = DINetTrainingOptions().parse_args()
    
    writer = SummaryWriter('./logs/'  + opt.result_path.split('/')[-1])
    
    # random.seed(opt.seed)
    # np.random.seed(opt.seed)
    # torch.cuda.manual_seed(opt.seed)
    # load training data
    train_data_ = "./asserts/training_data_youji2/training_json.json"
    train_data = DINetDataset(train_data_,opt.augment_num,opt.mouth_region_size)
    training_data_loader = DataLoader(dataset=train_data,  batch_size=opt.batch_size, shuffle=True,drop_last=True)
    train_data_length = len(training_data_loader)
    # init network
    net_g = DINet(opt.source_channel,opt.ref_channel,opt.audio_channel).cuda()
    net_dI = Discriminator(opt.source_channel ,opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_dV = Discriminator(opt.source_channel * 5, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()
    net_lipsync = SyncNetPerception(opt.pretrained_syncnet_path).cuda()
    
    # parallel
    net_g = nn.DataParallel(net_g)
    net_g = convert_model(net_g)
    net_dI = nn.DataParallel(net_dI)
    net_dV = nn.DataParallel(net_dV)
    net_vgg = nn.DataParallel(net_vgg)
        
    # 冻结前半部分网络的参数并输出名称
    for name, param in net_g.named_parameters():
        if 'out_conv' not in name:  # 通过名称检查是否是前半部分网络的参数
            param.requires_grad = False
            print(f'Frozen: {name}')
            
        if 'out_conv' in name:  # 通过名称检查是否是前半部分网络的参数
            # pdb.set_trace()  
            param.requires_grad = True
            print(f'Train --- : {name}')

    # setup optimizer
    # optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, net_g.parameters()), lr=opt.lr_g)
    optimizer_dI = optim.Adam(net_dI.parameters(), lr=opt.lr_dI)
    optimizer_dV = optim.Adam(net_dV.parameters(), lr=opt.lr_dI)
    ## load frame trained DInet weight
    print('loading frame trained DINet weight from: {}'.format(opt.pretrained_frame_DINet_path))
    checkpoint = torch.load(opt.pretrained_frame_DINet_path)
    net_g.load_state_dict(checkpoint['state_dict']['net_g'])
    net_dI.load_state_dict(checkpoint['state_dict']['net_dI'])                     ############  new added 
    net_dV.load_state_dict(checkpoint['state_dict']['net_dV'])                     ############  new added 
    
    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    criterionMSE = nn.BCELoss().cuda()
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    net_dV_scheduler = get_scheduler(optimizer_dV, opt.non_decay, opt.decay)
    
    # set label of syncnet perception loss    
    # # batchsize, 8, 8 for 256*256 clip
    # # batchsize, 4, 4 for 128*128 clip
    real_tensor = torch.ones(3, 8, 8, dtype=torch.float).cuda()
    # start train
    for epoch in range(opt.start_epoch, opt.non_decay+opt.decay+1):
        # net_g.train()
        running_sync_loss = 0.
        if epoch >= 5:
            sycnet_loss_weight = 0.05
        else:
            sycnet_loss_weight = 0.0
            
        for iteration, data in enumerate(training_data_loader):
            # forward
            source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full = data
            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            source_clip_mask = torch.cat(torch.split(source_clip_mask, 1, dim=1), 0).squeeze(1).float().cuda()
            reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_full = deep_speech_full.float().cuda()
            fake_out = net_g(source_clip_mask,reference_clip,deep_speech_clip)
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            source_clip_half = F.interpolate(source_clip, scale_factor=0.5, mode='bilinear')

            
            # vis network output img
            if iteration % 100 == 0:
                fake_frame = fake_out[0, :, :, :].squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
                real_frame = source_clip[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy() * 255
                fake_img_name = "./vis_clip_256_youji2/epoch" + str(epoch).zfill(4) + '_' + str(iteration).zfill(5) + '_fake.jpg'
                real_img_name = "./vis_clip_256_youji2/epoch" + str(epoch).zfill(4) + '_' + str(iteration).zfill(5) + '_real.jpg'
                cv2.imwrite(fake_img_name, fake_frame[:, :, ::-1])
                cv2.imwrite(real_img_name, real_frame[:, :, ::-1])
            
            # (1) Update DI network
            optimizer_dI.zero_grad()
            _,pred_fake_dI = net_dI(fake_out)
            loss_dI_fake = criterionGAN(pred_fake_dI, False)
            _,pred_real_dI = net_dI(source_clip)
            loss_dI_real = criterionGAN(pred_real_dI, True)
            # Combined DI loss
            loss_dI = (loss_dI_fake + loss_dI_real) * 0.5
            loss_dI.backward(retain_graph=True)
            optimizer_dI.step()

            # (2) Update DV network
            optimizer_dV.zero_grad()
            condition_fake_dV = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
            _, pred_fake_dV = net_dV(condition_fake_dV)
            loss_dV_fake = criterionGAN(pred_fake_dV, False)
            condition_real_dV = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)
            _, pred_real_dV = net_dV(condition_real_dV)
            loss_dV_real = criterionGAN(pred_real_dV, True)
            # Combined DV loss
            loss_dV = (loss_dV_fake + loss_dV_real) * 0.5
            loss_dV.backward(retain_graph=True)
            optimizer_dV.step()

            # (2) Update DINet
            _, pred_fake_dI = net_dI(fake_out)
            _, pred_fake_dV = net_dV(condition_fake_dV)
            optimizer_g.zero_grad()
            # compute perception loss
            perception_real = net_vgg(source_clip)
            perception_fake = net_vgg(fake_out)
            perception_real_half = net_vgg(source_clip_half)
            perception_fake_half = net_vgg(fake_out_half)
            loss_g_perception = 0
            for i in range(len(perception_real)):
                loss_g_perception += criterionL1(perception_fake[i], perception_real[i])
                loss_g_perception += criterionL1(perception_fake_half[i], perception_real_half[i])
            loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception
            # # gan dI loss
            loss_g_dI = criterionGAN(pred_fake_dI, True)
            # # gan dV loss
            loss_g_dV = criterionGAN(pred_fake_dV, True)
            ## sync perception loss
            fake_out_clip = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
            # fake_out_clip = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)
            fake_out_clip_mouth = fake_out_clip[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
            train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]
            sync_score = net_lipsync(fake_out_clip_mouth, deep_speech_full)
            
            # # batchsize, 8, 8 for 256*256 clip
            # # batchsize, 4, 4 for 128*128 clip
            loss_sync = criterionMSE(torch.sigmoid(sync_score).view(-1, 8, 8), real_tensor)
            # combine all losses
            loss_g =   loss_g_perception + loss_g_dI +loss_g_dV + loss_sync * sycnet_loss_weight
            loss_g.backward()
            optimizer_g.step()
            
            running_sync_loss += loss_sync.item()

            print(
                "===> Epoch[{}]({}/{}):  Loss_DI: {:.4f} Loss_GI: {:.4f} Loss_DV: {:.4f} Loss_GV: {:.4f} Loss_perception: {:.4f} compute_Loss_sync: {:.6f}  real_loss_sync: {:.6f}   running_loss_sync: {:6f}  lr_g = {:.7f} ".format(
                    epoch, iteration, len(training_data_loader), float(loss_dI), float(loss_g_dI),float(loss_dV), float(loss_g_dV), float(loss_g_perception), float(loss_sync), float(loss_sync * sycnet_loss_weight), float(running_sync_loss / (iteration+1)), 
                    optimizer_g.param_groups[0]['lr']))

            writer.add_scalar("Loss_Gan_D_image", float(loss_dI), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("Loss_Gan_G_image", float(loss_g_dI), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("Loss_Gan_D_video", float(loss_dV), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("Loss_Gan_G_video", float(loss_g_dV), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("Loss_perception", float(loss_g_perception), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("compute_Loss_sync", float(loss_sync), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("real_Loss_sync", float(loss_sync * sycnet_loss_weight), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("lr_g", float(optimizer_g.param_groups[0]['lr']), epoch*len(training_data_loader)+iteration)

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        update_learning_rate(net_dV_scheduler, optimizer_dV)
        
#         writer.add_scalar("Loss_DI_epoch", float(loss_dI), epoch)
#         writer.add_scalar("Loss_GI_epoch", float(loss_g_dI), epoch)
#         writer.add_scalar("Loss_DV_epoch", float(loss_dV), epoch)
#         writer.add_scalar("Loss_Gan_G_video", float(loss_g_dV), epoch)
#         writer.add_scalar("Loss_perception_epoch", float(loss_g_perception), epoch)
#         writer.add_scalar("Loss_sync_epoch", float(loss_sync), epoch)
#         writer.add_scalar("lr_g_epoch", float(optimizer_g.param_groups[0]['lr']), epoch)
#         writer.add_scalar("running_sync_loss_epoch", float(running_sync_loss / len(training_data_loader)), epoch)   
        
        # checkpoint
        if epoch %  opt.checkpoint == 0:
            if not os.path.exists(opt.result_path):
                os.mkdir(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'netG_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net_g': net_g.state_dict(),'net_dI': net_dI.state_dict(),'net_dV': net_dV.state_dict()},
                'optimizer': {'net_g': optimizer_g.state_dict(), 'net_dI': optimizer_dI.state_dict(), 'net_dV': optimizer_dV.state_dict()}
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))