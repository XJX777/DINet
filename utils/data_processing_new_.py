import csv
import numpy as np
import random
import json



def load_landmark_openface(json_path):
    '''
    load openface landmark from .csv file
    '''
    # import pdb
    # pdb.set_trace()
    # 打开JSON文件
    with open(json_path, 'r') as file:
        # 使用json.load()加载文件内容
        data_all = json.load(file)
        
    all_landmark = []
    for row_index, item_data in enumerate(data_all):
        item_landmark = item_data['landmark']
        all_landmark.append(item_landmark)
    landmark_array = np.array(all_landmark)
    # pdb.set_trace()
    return landmark_array


def compute_crop_radius(video_size,landmark_data_clip,random_scale = None):
    '''
    judge if crop face and compute crop radius
    '''
    # import pdb
    # pdb.set_trace()
    video_w, video_h = video_size[0], video_size[1]
    landmark_max_clip = np.max(landmark_data_clip, axis=1)
    if random_scale is None:
        random_scale = random.random() / 10 + 1.05
    else:
        random_scale = random_scale
    # radius_h = (landmark_max_clip[:,1] - landmark_data_clip[:,29, 1]) * random_scale
    # radius_w = (landmark_data_clip[:,54, 0] - landmark_data_clip[:,48, 0]) * random_scale
    radius_h = (landmark_max_clip[:,1] - landmark_data_clip[:,45, 1]) * random_scale      # 29 to 45. # 54 to 90.  48 to 84.  33 to 49. 
    radius_w = (landmark_data_clip[:,90, 0] - landmark_data_clip[:,84, 0]) * random_scale 
    radius_clip = np.max(np.stack([radius_h, radius_w],1),1) // 2
    radius_max = np.max(radius_clip)
    radius_max = (np.int(radius_max/4) + 1 ) * 4
    radius_max_1_4 = radius_max//4
    # clip_min_h = landmark_data_clip[:, 29,
    #              1] - radius_max
    # clip_max_h = landmark_data_clip[:, 29,
    #              1] + radius_max * 2  + radius_max_1_4
    # clip_min_w = landmark_data_clip[:, 33,
    #              0] - radius_max - radius_max_1_4
    # clip_max_w = landmark_data_clip[:, 33,
    #              0] + radius_max + radius_max_1_4
    clip_min_h = landmark_data_clip[:, 45,
                 1] - radius_max
    clip_max_h = landmark_data_clip[:, 45,
                 1] + radius_max * 2  + radius_max_1_4
    clip_min_w = landmark_data_clip[:, 49,
                 0] - radius_max - radius_max_1_4
    clip_max_w = landmark_data_clip[:, 49,
                 0] + radius_max + radius_max_1_4
    if min(clip_min_h.tolist() + clip_min_w.tolist()) < 0:
        return False,None
    elif max(clip_max_h.tolist()) > video_h:
        return False,None
    elif max(clip_max_w.tolist()) > video_w:
        return False,None
    elif max(radius_clip) > min(radius_clip) * 1.5:
        return False, None
    else:
        return True,radius_max