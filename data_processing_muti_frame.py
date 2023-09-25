import glob
import os
import subprocess
import cv2
import numpy as np
import json

from utils.data_processing import load_landmark_openface,compute_crop_radius
from utils.deep_speech import DeepSpeech
from config.config import DataProcessingOptions

import os
import glob
import cv2
import threading
import concurrent.futures


def extract_video_frame(video_path,res_video_frame_dir):
    '''
        extract video frames from videos
    '''
    # if not os.path.exists(source_video_dir):
    #     raise ('wrong path of video dir')
    # if not os.path.exists(res_video_frame_dir):
    #     os.mkdir(res_video_frame_dir)
    # video_path_list = glob.glob(os.path.join(source_video_dir, '*.mp4'))
    # for video_path in video_path_list:
    video_name = os.path.basename(video_path)
    frame_dir = os.path.join(res_video_frame_dir, video_name.replace('.mp4', ''))
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    print('extracting frames from {} ...'.format(video_name))
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        raise ('{} video is not in 25 fps'.format(video_path))
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        result_path = os.path.join(frame_dir, str(i).zfill(6) + '.jpg')
        cv2.imwrite(result_path, frame)

if __name__ == '__main__':
    opt = DataProcessingOptions().parse_args()
    
    # 如果 opt.extract_video_frame 为 True，使用线程池处理视频
    if opt.extract_video_frame:
        # 获取所有视频路径
        video_path_list = glob.glob(os.path.join(opt.source_video_dir, '*.mp4'))
        
        # 定义线程池，限制最大并发线程数为20
        max_threads = 20
        with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
            # 创建一个列表以批量提交视频处理任务
            futures = []
            for video_path in video_path_list:
                future = executor.submit(extract_video_frame, video_path, opt.video_frame_dir)
                futures.append(future)
                
            # 等待所有任务完成
            concurrent.futures.wait(futures)
                
        print("All video frames extraction is done.")
        
        
# if __name__ == '__main__':
#     opt = DataProcessingOptions().parse_args()

#     # 如果 opt.extract_video_frame 为 True，启动多线程处理
#     if opt.extract_video_frame:
#         # 获取所有视频路径
#         video_path_list = glob.glob(os.path.join(opt.source_video_dir, '*.mp4'))
#         print(video_path_list)
        
#         # 定义一个互斥锁，以确保多线程安全
#         lock = threading.Lock()

#         # 定义一个函数，用于处理每个视频的帧提取
#         def process_video(video_path):
#             try:
#                 extract_video_frame(video_path, opt.video_frame_dir)
#             except Exception as e:
#                 print(f"Error processing video {video_path}: {e}")
            
#         # 创建一个线程列表
#         threads = []
        
#         # 启动多线程处理每个视频
#         for video_path in video_path_list:
#             thread = threading.Thread(target=process_video, args=(video_path,))
#             threads.append(thread)
#             thread.start()
            
#         # 等待所有线程完成
#         for thread in threads:
#             thread.join()
            
#         print("All video frames extraction is done.")




