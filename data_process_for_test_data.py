import glob
import os
import subprocess
import cv2
import numpy as np
import json

from utils.data_processing import load_landmark_openface,compute_crop_radius
from utils.deep_speech import DeepSpeech
from config.config import DataProcessingOptions


import torch
import torch.nn as nn
from models.VGG19 import Vgg19
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



def extract_face_feature(res_crop_face_dir, res_feature_dir):
    '''
        extract video frames from videos
    '''
    net_vgg = Vgg19().cuda()
    net_vgg = nn.DataParallel(net_vgg)
    
    mouth_region_size = 128
    radius = mouth_region_size//2
    radius_1_4 = radius//4
    
    feature_save_list = []
    if not os.path.exists(res_feature_dir):
        os.mkdir(res_feature_dir)
    for video_face_dir in os.listdir(res_crop_face_dir):
        print(video_face_dir)
        face_path_list = glob.glob(os.path.join(os.path.join(res_crop_face_dir, video_face_dir), '*.jpg'))
        for image_path in face_path_list:
            source_image_data = cv2.imread(image_path)[:, :, ::-1]
            source_image_data = cv2.resize(source_image_data, (160, 208))/ 255.0
            print(source_image_data.shape)
            
            source_mouth_data = source_image_data[radius:radius + mouth_region_size, radius_1_4:radius_1_4 + mouth_region_size, :]
            source_mouth_data = cv2.resize(source_mouth_data, (128, 128))
            source_mouth_data = torch.from_numpy(source_mouth_data).float().permute(2,0,1)
            source_mouth_data = source_mouth_data.float().cuda()
            perception_real = net_vgg(source_mouth_data)[4].cpu()
            np.save(os.path.join(res_feature_dir, image_path.split('/')[-1] + '.npy'), perception_real)
            feature_save_list.append(perception_real)
            
    print("finish extrace feature")

    
    # 将每个feature重新整理为一维向量
    flattened_features = [feature.view(-1).numpy() for feature in feature_save_list]

    # 将feature向量堆叠成一个NumPy数组
    feature_matrix = np.stack(flattened_features)
    print(feature_matrix.shape)

    # 使用K均值聚类将100个feature分为5个类别
    print("start k-means ")
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(feature_matrix)

    # 获取每个类别中心
    cluster_centers = kmeans.cluster_centers_
    print("finish k-means ")

    # 找到每个类别中心最近的feature的序号
    closest_feature_indices = []
    for center in cluster_centers:
        # 计算每个中心点与所有feature的距离，并找到最近的feature序号
        distances = np.linalg.norm(feature_matrix - center, axis=1)
        closest_feature_index = np.argmin(distances)
        closest_feature_indices.append(closest_feature_index)

    # 输出每个类别中心最近的feature的序号
    for i, index in enumerate(closest_feature_indices):
        print(f"Cluster {i}: Closest feature index = {index}")

            
    
            
    


def extract_video_frame(source_video_dir,res_video_frame_dir):
    '''
        extract video frames from videos
    '''
    print(source_video_dir)
    if not os.path.exists(source_video_dir):
        raise ('wrong path of video dir')
    if not os.path.exists(res_video_frame_dir):
        os.mkdir(res_video_frame_dir)
    video_path_list = glob.glob(os.path.join(source_video_dir, '*.mp4'))
    for video_path in video_path_list:
        video_name = os.path.basename(video_path)
        frame_dir = os.path.join(res_video_frame_dir, video_name.replace('.mp4', ''))
        csv_name  = video_path.replace('.mp4', '.csv').replace('split_video_25fps', 'split_video_25fps_landmark_openface')
        if not os.path.exists(csv_name):
            continue
            
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        else:
            continue
        print('extracting frames from {} ...'.format(video_name))
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        print(video_path, fps)
        if int(fps) != 25:
            raise ('{} video is not in 25 fps'.format(video_path))
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            result_path = os.path.join(frame_dir, str(i).zfill(6) + '.jpg')
            cv2.imwrite(result_path, frame)


def crop_face_according_openfaceLM(openface_landmark_dir,video_frame_dir,res_crop_face_dir,clip_length):
    '''
      crop face according to openface landmark
    '''
    if not os.path.exists(openface_landmark_dir):
        raise ('wrong path of openface landmark dir')
    if not os.path.exists(video_frame_dir):
        raise ('wrong path of video frame dir')
    if not os.path.exists(res_crop_face_dir):
        os.mkdir(res_crop_face_dir)
    landmark_openface_path_list = glob.glob(os.path.join(openface_landmark_dir, '*.csv'))
    for landmark_openface_path in landmark_openface_path_list:
        video_name = os.path.basename(landmark_openface_path).replace('.csv', '')
        crop_face_video_dir = os.path.join(res_crop_face_dir, video_name)
        if not os.path.exists(crop_face_video_dir):
            os.makedirs(crop_face_video_dir)
        else:
            print("already exits")
            continue
            
        print('cropping face from video: {} ...'.format(video_name))
        landmark_openface_data = load_landmark_openface(landmark_openface_path).astype(np.int)
        frame_dir = os.path.join(video_frame_dir, video_name)
        if not os.path.exists(frame_dir):
            print("this frame dont exists")
            continue
            # raise ('run last step to extract video frame')
        if len(glob.glob(os.path.join(frame_dir, '*.jpg'))) != landmark_openface_data.shape[0]:
            raise ('landmark length is different from frame length')
        frame_length = min(len(glob.glob(os.path.join(frame_dir, '*.jpg'))), landmark_openface_data.shape[0])
            
        end_frame_index = list(range(clip_length, frame_length, clip_length))
        video_clip_num = len(end_frame_index)
        
        for i in range(video_clip_num):
            first_image = cv2.imread(os.path.join(frame_dir, '000000.jpg'))
            video_h,video_w = first_image.shape[0], first_image.shape[1]
            crop_flag, radius_clip = compute_crop_radius((video_w,video_h),
                                    landmark_openface_data[end_frame_index[i] - clip_length:end_frame_index[i], :,:])
            if not crop_flag:
                continue
            radius_clip_1_4 = radius_clip // 4
            print('cropping {}/{} clip from video:{}'.format(i, video_clip_num, video_name))
            # res_face_clip_dir = os.path.join(crop_face_video_dir, str(i).zfill(6))
            res_face_clip_dir = crop_face_video_dir
            if not os.path.exists(res_face_clip_dir):
                os.mkdir(res_face_clip_dir)
            for frame_index in range(end_frame_index[i]- clip_length,end_frame_index[i]):
                source_frame_path = os.path.join(frame_dir,str(frame_index).zfill(6)+'.jpg')
                source_frame_data = cv2.imread(source_frame_path)
                frame_landmark = landmark_openface_data[frame_index, :, :]
                crop_face_data = source_frame_data[
                                    frame_landmark[29, 1] - radius_clip:frame_landmark[
                                                                            29, 1] + radius_clip * 2 + radius_clip_1_4,
                                    frame_landmark[33, 0] - radius_clip - radius_clip_1_4:frame_landmark[
                                                                                              33, 0] + radius_clip + radius_clip_1_4,
                                    :].copy()
                res_crop_face_frame_path = os.path.join(res_face_clip_dir, str(frame_index).zfill(6) + '.jpg')
                if os.path.exists(res_crop_face_frame_path):
                    os.remove(res_crop_face_frame_path)
                cv2.imwrite(res_crop_face_frame_path, crop_face_data)
                
        


if __name__ == '__main__':
    ##########  step1: extract video frames
    # extract_video_frame("./asserts/testing_data/split_video_25fps", "./asserts/testing_data/split_video_25fps_frame")

    ##########  step4: crop face images
    crop_face_according_openfaceLM("./asserts/testing_data/split_video_25fps_landmark_openface", "./asserts/testing_data/split_video_25fps_frame" ,"./asserts/testing_data/split_video_25fps_crop_face", 1)
    
    ##########  step: get crop face feature
    extract_face_feature("./asserts/testing_data/split_video_25fps_crop_face/", "./asserts/testing_data/split_video_25fps_crop_face_feature/")


