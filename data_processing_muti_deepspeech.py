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
    
def extract_deep_speech(wav_path,res_deep_speech_dir,deep_speech_model_path):
    '''
    extract deep speech feature
    '''
    if not os.path.exists(res_deep_speech_dir):
        os.mkdir(res_deep_speech_dir)
    DSModel = DeepSpeech(deep_speech_model_path)
    # wav_path_list = glob.glob(os.path.join(audio_dir, '*.wav'))
    # for wav_path in wav_path_list:
    print("Doing wav_path: ", wav_path)
    video_name = os.path.basename(wav_path).replace('.wav', '')
    res_dp_path = os.path.join(res_deep_speech_dir, video_name + '_deepspeech.txt')
    if os.path.exists(res_dp_path):
        os.remove(res_dp_path)
    print('extract deep speech feature from audio:{}'.format(video_name))
    ds_feature = DSModel.compute_audio_feature(wav_path)
    np.savetxt(res_dp_path, ds_feature)

    
if __name__ == '__main__':
    opt = DataProcessingOptions().parse_args()
    
    if opt.extract_deep_speech:
        # 获取所有视频路径
        audio_path_list = glob.glob(os.path.join(opt.audio_dir, '*.wav'))
        
        # 定义线程池，限制最大并发线程数为20
        max_threads = 20
        with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
            # 创建一个列表以批量提交视频处理任务
            futures = []
            for audio_path in audio_path_list:
                future = executor.submit(extract_deep_speech, audio_path, opt.deep_speech_dir,opt.deep_speech_model)
                futures.append(future)
                
            # 等待所有任务完成
            concurrent.futures.wait(futures)
                
        print("All video deepspeech extraction is done.")    

    
    
# if __name__ == '__main__':
#     opt = DataProcessingOptions().parse_args()

#     # 如果 opt.extract_audio 为 True，启动多线程处理
#     if opt.extract_deep_speech:
#         # 获取所有视频路径
#         audio_path_list = glob.glob(os.path.join(opt.audio_dir, '*.wav'))
#         print(audio_path_list)
        
#         # 定义一个互斥锁，以确保多线程安全
#         lock = threading.Lock()

#         # 定义一个函数，用于处理每个视频的帧提取
#         def process_video(audio_path):
#             try:
#                 # print("opt.audio_dir: ", )
#                 # extract_audio(video_path, opt.audio_dir)
#                 extract_deep_speech(audio_path, opt.deep_speech_dir,opt.deep_speech_model)
#             except Exception as e:
#                 print(f"Error processing video {audio_path}: {e}")
            
#         # 创建一个线程列表
#         threads = []
        
#         # 启动多线程处理每个视频
#         for audio_path in audio_path_list:
#             thread = threading.Thread(target=process_video, args=(audio_path,))
#             threads.append(thread)
#             thread.start()
            
#         # 等待所有线程完成
#         for thread in threads:
#             thread.join()
            
#         print("All video frames extraction is done.")
        




