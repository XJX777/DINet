from models.Discriminator import Discriminator
from models.VGG19 import Vgg19
from models.DINet import DINet
from models.Syncnet import SyncNetPerception 
from models.color_syncnet import SyncNet_color  as SyncNet
from utils.training_utils import get_scheduler, update_learning_rate,GANLoss
from config.config import DINetTrainingOptions
from sync_batchnorm import convert_model
from torch.utils.data import DataLoader
from dataset.dataset_DINet_clip_colorsync import DINetDataset


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F



from torch.utils.tensorboard import SummaryWriter
import cv2



def _load(checkpoint_path):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    return model

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

if __name__ == "__main__":
    '''
            clip training code of DINet
            in the resolution you want, using clip training code after frame training
            
        '''
    # load config
    opt = DINetTrainingOptions().parse_args()
    
    writer = SummaryWriter('./logs/'  + opt.result_path.split('/')[-1])

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    # load training data
    train_data = DINetDataset(opt.train_data,opt.augment_num,opt.mouth_region_size)
    training_data_loader = DataLoader(dataset=train_data,  batch_size=opt.batch_size, shuffle=True,drop_last=True)
    train_data_length = len(training_data_loader)
    # init network
    opt.audio_channel = 256 #######
    net_g = DINet(opt.source_channel,opt.ref_channel,opt.audio_channel).cuda()
    net_dI = Discriminator(opt.source_channel ,opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_dV = Discriminator(opt.source_channel * 5, opt.D_block_expansion, opt.D_num_blocks, opt.D_max_features).cuda()
    net_vgg = Vgg19().cuda()
    
    # import pdb
    # pdb.set_trace()
    # for sync net
    # net_lipsync = SyncNetPerception(opt.pretrained_syncnet_path).cuda()
    net_lipsync = SyncNet().cuda()
    for p in net_lipsync.parameters():
        p.requires_grad = False
    # pdb.set_trace()
    net_lipsync = load_checkpoint(opt.pretrained_syncnet_path, net_lipsync)
    net_lipsync = net_lipsync.cuda()
    
    # parallel
    net_g = nn.DataParallel(net_g)
    net_g = convert_model(net_g)
    net_dI = nn.DataParallel(net_dI)
    net_dV = nn.DataParallel(net_dV)
    net_vgg = nn.DataParallel(net_vgg)
    # setup optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr_g)
    optimizer_dI = optim.Adam(net_dI.parameters(), lr=opt.lr_dI)
    optimizer_dV = optim.Adam(net_dV.parameters(), lr=opt.lr_dI)
    ## load frame trained DInet weight
    print('loading frame trained DINet weight from: {}'.format(opt.pretrained_frame_DINet_path))
    checkpoint = torch.load(opt.pretrained_frame_DINet_path)
    net_g.load_state_dict(checkpoint['state_dict']['net_g'])
    # set criterion
    criterionGAN = GANLoss().cuda()
    criterionL1 = nn.L1Loss().cuda()
    criterionMSE = nn.MSELoss().cuda()
    # set scheduler
    net_g_scheduler = get_scheduler(optimizer_g, opt.non_decay, opt.decay)
    net_dI_scheduler = get_scheduler(optimizer_dI, opt.non_decay, opt.decay)
    net_dV_scheduler = get_scheduler(optimizer_dV, opt.non_decay, opt.decay)
    # set label of syncnet perception loss
    real_tensor = torch.tensor(1.0).cuda()
    # start train
    for epoch in range(opt.start_epoch, opt.non_decay+opt.decay+1):
        net_g.train()
        for iteration, data in enumerate(training_data_loader):
            # forward
            source_clip,source_clip_mask, reference_clip,deep_speech_clip,deep_speech_full, sync_mel, sync_y = data
            
            source_clip = torch.cat(torch.split(source_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            source_clip_mask = torch.cat(torch.split(source_clip_mask, 1, dim=1), 0).squeeze(1).float().cuda()
            reference_clip = torch.cat(torch.split(reference_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_clip = torch.cat(torch.split(deep_speech_clip, 1, dim=1), 0).squeeze(1).float().cuda()
            deep_speech_full = deep_speech_full.float().cuda()
            
            fake_out = net_g(source_clip_mask,reference_clip,deep_speech_clip)
            fake_out_half = F.avg_pool2d(fake_out, 3, 2, 1, count_include_pad=False)
            source_clip_half = F.interpolate(source_clip, scale_factor=0.5, mode='bilinear')
            
            # crop sync input fake data
            fake_out_tmp = fake_out.clone()
            _, _, h_fake, w_fake = fake_out_tmp.shape
            fake_out_tmp = fake_out_tmp[:, :, int(h_fake * 0.380) : int(h_fake * 0.818), :]
            fake_out_tmp = torch.split(fake_out_tmp, opt.batch_size, dim=0)
            sync_fake_input = [[] for i in range(opt.batch_size)]
            for ref_id in range(5):
                for batch_id in range(opt.batch_size):
                    tmp_img = fake_out_tmp[ref_id][batch_id, :, :, :].permute(1, 2, 0).detach().cpu().numpy()[:, :, ::-1]
                    tmp_img = cv2.resize(tmp_img, (96, 48))
                    sync_fake_input[batch_id].append(tmp_img)
            batch_all_person_data = []
            for batch_single_person_data in sync_fake_input:
                batch_single_person_array = np.array(batch_single_person_data)
                batch_single_person_array = np.concatenate(batch_single_person_array, axis=2)
                batch_single_person_array = batch_single_person_array.transpose(2, 0, 1)
                batch_all_person_data.append(batch_single_person_array)
            sync_fake_img_tensor = torch.FloatTensor(batch_all_person_data)
            
            # prepare syncnet input data
            sync_mel = sync_mel.cuda()
            sync_fake_img_tensor = sync_fake_img_tensor.cuda()
            sync_y = sync_y.cuda()
            
            # vis network output img
            if iteration % 100 == 0:
                fake_frame = fake_out[0, :, :, :].squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
                real_frame = source_clip[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy() * 255
                fake_img_name = "./vis_img_clip/epoch" + str(epoch).zfill(4) + '_' + str(iteration).zfill(5) + '_fake.jpg'
                real_img_name = "./vis_img_clip/epoch" + str(epoch).zfill(4) + '_' + str(iteration).zfill(5) + '_real.jpg'
                cv2.imwrite(fake_img_name, fake_frame[:, :, ::-1])
                cv2.imwrite(real_img_name, real_frame[:, :, ::-1])
            
            # (1) Update DI network
            optimizer_dI.zero_grad()
            _,pred_fake_dI = net_dI(fake_out)
            loss_dI_fake = criterionGAN(pred_fake_dI, False)
            _,pred_real_dI = net_dI(source_clip)
            loss_dI_real = criterionGAN(pred_real_dI, True)
            # Combined DI loss
            # loss_dI = (loss_dI_fake + loss_dI_real) * 0.5               # GAN Loss D(true + Fake)  using  Discriminator-image-model
            loss_dI = (loss_dI_fake + loss_dI_real) * 0.5 * 1.5              # GAN Loss D(true + Fake)  using  Discriminator-image-model
            loss_dI.backward(retain_graph=True)
            optimizer_dI.step()

            # (2) Update DV network
            
            pdb.set_trace()
            
            optimizer_dV.zero_grad()
            condition_fake_dV = torch.cat(torch.split(fake_out, opt.batch_size, dim=0), 1)
            _, pred_fake_dV = net_dV(condition_fake_dV)
            loss_dV_fake = criterionGAN(pred_fake_dV, False)
            condition_real_dV = torch.cat(torch.split(source_clip, opt.batch_size, dim=0), 1)
            _, pred_real_dV = net_dV(condition_real_dV)
            loss_dV_real = criterionGAN(pred_real_dV, True)
            # Combined DV loss
            loss_dV = (loss_dV_fake + loss_dV_real) * 0.5 * 1.5              # GAN Loss D(true + Fake) using  Discriminator-video-model
            # loss_dV = (loss_dV_fake + loss_dV_real) * 0.5              # GAN Loss D(true + Fake) using  Discriminator-video-model
            loss_dV.backward(retain_graph=True)
            optimizer_dV.step()

            # (2) Update DINet
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
            loss_g_perception = (loss_g_perception / (len(perception_real) * 2)) * opt.lamb_perception        # perception-Loss  using  DINet(net_g)

            _, pred_fake_dI = net_dI(fake_out)
            _, pred_fake_dV = net_dV(condition_fake_dV)            
            # # gan dI loss
            loss_g_dI = criterionGAN(pred_fake_dI, True)                 # GAN Loss G  using Discriminator-image-model
            # # gan dV loss
            loss_g_dV = criterionGAN(pred_fake_dV, True)                 # GAN Loss G  using Discriminator-video-model
            
            # ## color sync loss
            sync_a, sync_v = net_lipsync(sync_mel, sync_fake_img_tensor)
            loss_sync = cosine_loss(sync_a, sync_v, sync_y)
            
            # combine all losses
            loss_g =   loss_g_perception +  1.5 * (loss_g_dI +loss_g_dV) + loss_sync * 1.0
            # loss_g =   loss_g_perception +  loss_g_dI +loss_g_dV + loss_sync
            loss_g.backward()
            optimizer_g.step()

            print(
                "===> Epoch[{}]({}/{}):  Loss_Gan_D_image: {:.4f} Loss_Gan_G_image: {:.4f} Loss_Gan_D_video: {:.4f} Loss_Gan_G_video: {:.4f} Loss_perception: {:.4f} Loss_sync: {:.4f} lr_g = {:.7f} ".format(
                    epoch, iteration, len(training_data_loader), float(loss_dI), float(loss_g_dI),float(loss_dV), float(loss_g_dV), float(loss_g_perception),float(loss_sync),
                    optimizer_g.param_groups[0]['lr']))
            
            writer.add_scalar("Loss_Gan_D_image", float(loss_dI), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("Loss_Gan_G_image", float(loss_g_dI), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("Loss_Gan_D_video", float(loss_dV), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("Loss_Gan_G_video", float(loss_g_dV), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("Loss_perception", float(loss_g_perception), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("Loss_sync", float(loss_sync), epoch*len(training_data_loader)+iteration)
            writer.add_scalar("lr_g", float(optimizer_g.param_groups[0]['lr']), epoch*len(training_data_loader)+iteration)

        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_dI_scheduler, optimizer_dI)
        update_learning_rate(net_dV_scheduler, optimizer_dV)
        
        writer.add_scalar("Loss_DI_epoch", float(loss_dI), epoch)
        writer.add_scalar("Loss_GI_epoch", float(loss_g_dI), epoch)
        writer.add_scalar("Loss_DV_epoch", float(loss_dV), epoch)
        writer.add_scalar("Loss_Gan_G_video", float(loss_g_dV), epoch)
        writer.add_scalar("Loss_perception_epoch", float(loss_g_perception), epoch)
        writer.add_scalar("Loss_sync_epoch", float(loss_sync), epoch)
        writer.add_scalar("lr_g_epoch", float(optimizer_g.param_groups[0]['lr']), epoch)
        
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
