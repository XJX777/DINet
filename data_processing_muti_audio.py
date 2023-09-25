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
import concurrent.futures

def extract_audio(video_path, res_audio_dir):
    '''
    extract audio files from videos
    '''
    
    # if not os.path.exists(source_video_dir):
    #     raise ('wrong path of video dir')
    if not os.path.exists(res_audio_dir):
        os.mkdir(res_audio_dir)
    # video_path_list = glob.glob(os.path.join(source_video_dir, '*.mp4'))
    # for video_path in video_path_list:
    print('extract audio from video: {}'.format(os.path.basename(video_path)))
    audio_path = os.path.join(res_audio_dir, os.path.basename(video_path).replace('.mp4', '.wav'))
    cmd = 'ffmpeg -i {} -f wav -ar 16000 {}'.format(video_path, audio_path)
    subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    opt = DataProcessingOptions().parse_args()
    
    if opt.extract_audio:
        # 获取所有视频路径
        video_path_list = glob.glob(os.path.join(opt.source_video_dir, '*.mp4'))
        
        # 定义线程池，限制最大并发线程数为20
        max_threads = 20
        with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
            # 创建一个列表以批量提交视频处理任务
            futures = []
            for video_path in video_path_list:
                future = executor.submit(extract_audio, video_path, opt.audio_dir)
                futures.append(future)
                
            # 等待所有任务完成
            concurrent.futures.wait(futures)
                
        print("All video frames extraction is done.")
    
# if __name__ == '__main__':
#     opt = DataProcessingOptions().parse_args()

#     # 如果 opt.extract_audio 为 True，启动多线程处理
#     if opt.extract_audio:
#         # 获取所有视频路径
#         video_path_list = glob.glob(os.path.join(opt.source_video_dir, '*.mp4'))
#         print(video_path_list)
        
#         # 定义一个互斥锁，以确保多线程安全
#         lock = threading.Lock()

#         # 定义一个函数，用于处理每个视频的帧提取
#         def process_video(video_path):
#             try:
#                 print("opt.audio_dir: ", )
#                 extract_audio(video_path, opt.audio_dir)
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




