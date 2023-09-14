from models.Syncnet import SyncNetPerception,SyncNet
from config.config import DINetTrainingOptions
from sync_batchnorm import convert_model

from torch.utils.data import DataLoader
from dataset.dataset_DINet_syncnet import DINetDataset

from utils.training_utils import get_scheduler, update_learning_rate,GANLoss

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":

    # load config
    opt = DINetTrainingOptions().parse_args()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # init network
    
    writer = SummaryWriter('./logs/syncnet_train')
    
    net_lipsync = SyncNet(15,29,128).cuda()
    import pdb
    pdb.set_trace()
    
    print('loading checkpoint for syncnet training: {}'.format("./asserts/syncnet_256mouth.pth"))
    checkpoint = torch.load("./asserts/syncnet_256mouth.pth")
    net_lipsync.load_state_dict(checkpoint['state_dict']['net'])

    criterionMSE = nn.BCELoss().cuda()
    # criterionMSE = nn.MSELoss().cuda()
    # set scheduler
    # set label of syncnet perception loss
    real_tensor = torch.tensor(1.0).cuda()
    
    # setup optimizer
   # optimizer_s = optim.Adam(net_lipsync.parameters(), lr=opt.lr_g)
    optimizer_s = optim.Adamax(net_lipsync.parameters(), lr=opt.lr_g)
    
    # set scheduler
    net_s_scheduler = get_scheduler(optimizer_s, opt.non_decay, opt.decay)

    
    # load training data
    train_data = DINetDataset(opt.train_data,opt.augment_num,opt.mouth_region_size)
    training_data_loader = DataLoader(dataset=train_data,  batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=12)
    train_data_length = len(training_data_loader)
    
    # # load training data
    # test_data = DINetDataset(opt.test_data,opt.augment_num,opt.mouth_region_size)
    # test_data_loader = DataLoader(dataset=test_data,  batch_size=1, shuffle=True,drop_last=True,num_workers=12)
    # test_data_length = len(test_data_loader)
    
    min_loss = 100
    # start train
    for epoch in range(opt.start_epoch, opt.non_decay+opt.decay+1):
        net_lipsync.train()
        running_sync_loss = 0.
        for iteration, data in enumerate(training_data_loader):
            # forward

            optimizer_s.zero_grad()
            source_clip, deep_speech_full, y = data
            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            source_clip = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1).cuda()
            deep_speech_full = deep_speech_full.float().cuda()

            y = y.cuda()
            ## sync perception loss
            source_clip_mouth = source_clip[:, :, train_data.radius:train_data.radius + train_data.mouth_region_size,
            train_data.radius_1_4:train_data.radius_1_4 + train_data.mouth_region_size]
            sync_score = net_lipsync(source_clip_mouth, deep_speech_full)
            # print(source_clip_mouth.shape, deep_speech_full.shape, sync_score.shape, y.shape)
            loss_sync = criterionMSE(torch.sigmoid(sync_score).view(-1, 8, 8), y)
            
            loss_sync.backward()
            optimizer_s.step()
            writer.add_scalar("loss_sync", float(loss_sync), epoch*len(training_data_loader)+iteration)
            running_sync_loss += loss_sync.item()

            print(
                "===> Epoch[{}]({}/{}):  Loss_Sync: {:.4f} lr_g = {:.7f}  running_sync_loss = {:.4f}".format(
                    epoch, iteration, len(training_data_loader), float(loss_sync) ,
                    optimizer_s.param_groups[0]['lr'], running_sync_loss/(iteration+1)))

        update_learning_rate(net_s_scheduler, optimizer_s)
        writer.add_scalar("running_sync_loss", float(running_sync_loss / len(training_data_loader)), epoch)

        # checkpoint
        if epoch %  opt.checkpoint == 0 :
            if not os.path.exists(opt.result_path):
                os.makedirs(opt.result_path)
            model_out_path = os.path.join(opt.result_path, 'netS_model_epoch_{}.pth'.format(epoch))
            states = {
                'epoch': epoch + 1,
                'state_dict': {'net': net_lipsync.state_dict()},
                'optimizer': {'net': optimizer_s.state_dict()}
            }
            torch.save(states, model_out_path)
            print("Checkpoint saved to {}".format(epoch))
        if epoch == 1000:
            break
