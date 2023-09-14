import torch
import numpy as np
import json
import random
import cv2

from torch.utils.data import Dataset


def get_data(json_name,augment_num):
    print('start loading data')
    with open(json_name,'r') as f:
        data_dic = json.load(f)
    data_dic_name_list = []
    for augment_index in range(augment_num):
        for video_name in data_dic.keys():
            data_dic_name_list.append(video_name)
    random.shuffle(data_dic_name_list)
    print('finish loading')
    return data_dic_name_list,data_dic


class DINetDataset(Dataset):
    def __init__(self,path_json,augment_num,mouth_region_size):
        super(DINetDataset, self).__init__()
        self.data_dic_name_list,self.data_dic = get_data(path_json,augment_num)
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)

    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])    
        try:      
            source_anchor = random.sample(range(video_clip_num), 1)[0]
            wrong_source_anchor = random.sample(range(video_clip_num), 1)[0]
        except:
            print(video_name,video_clip_num)
            video_name = self.data_dic_name_list[0]
            video_clip_num = len(self.data_dic[video_name]['clip_data_list'])
            source_anchor = random.sample(range(video_clip_num), 1)[0]
            wrong_source_anchor = random.sample(range(video_clip_num), 1)[0]
        while source_anchor == wrong_source_anchor:
            wrong_source_anchor  = random.sample(range(video_clip_num), 1)[0]
                  
        source_clip_list = []
        source_clip_mask_list = []
        deep_speech_list = []
        
       # if random.choice([True, False]):
        if (index & 1) == 0:
            # y = torch.ones(1).float()
            y = torch.ones(8, 8, dtype=torch.float)
            chosen = source_anchor
        else:
            # y = torch.zeros(1).float()
            y = torch.zeros(8, 8, dtype=torch.float)
            chosen = wrong_source_anchor

        source_image_path_list = self.data_dic[video_name]['clip_data_list'][chosen]['frame_path_list']
        for source_frame_index in range(2, 2 + 5):
            ## load source clip
            source_image_data = cv2.imread(source_image_path_list[source_frame_index])[:, :, ::-1]
            source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
            source_clip_list.append(source_image_data)
            
            ## load deep speech feature
            deepspeech_array = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'][
                                       source_frame_index - 2:source_frame_index + 3])
            deep_speech_list.append(deepspeech_array)

        source_clip = np.stack(source_clip_list, 0)
        deep_speech_clip = np.stack(deep_speech_list, 0)
        #deep_speech_clip = np.reshape(deep_speech_clip,(-1,1024))
        deep_speech_full = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'])

        # # 2 tensor
        source_clip = torch.from_numpy(source_clip).float().permute(0, 3, 1, 2)
        deep_speech_full = torch.from_numpy(deep_speech_full).permute(1, 0)
        deep_speech_clip = torch.from_numpy(deep_speech_clip).permute(2,0, 1)
        
        return source_clip ,deep_speech_full,y

    def __len__(self):
        return self.length