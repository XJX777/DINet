import os
import glob
import subprocess
import numpy as np
import concurrent.futures

# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"


# video_folder_path = "/workspace/DINet/asserts/training_data/split_video_25fps/"

org_command = "python -u detect_align_vid.py --detect_model checkpoints/detection_0525_2.pth --track_model checkpoints/track_0521.pth --pose_landmark_model checkpoints/arkit_pose_land_viz_v2_best.pth --eyebrow240_model checkpoints/eyebrow240_v4_025_1024.pth --eye_scale 0.25 --eye_normal --mouth240_model checkpoints/mouth240_v4_025_fusion_1114.pth --mouth_scale 0.25 --mouth_normal --mouth_pooling 'Max' --mouth_cover_score 0.8 --dconfig cfg_det --tconfig cfg_track --d_thres 0.65 --t_thres 0.90 --viz_threshold 0.65 --eye_size 64 --mouth_size 64 --exp_mode apperence --e_scale 0.5 --exp_num 52 --exp_model checkpoints/arkitv1_s050_gamma1_bins100_best.pth --post_smooth_fmcap --action_num 16 --elabel 24 --pose --save --eyeball_score 0.3 --oneeuro 1 --post_smooth --landmark106 --post_smooth_240 --det_mode ldmk --vid "

# for video_item in os.listdir(video_folder_path):
#     command = org_command
#     if video_item.endswith('.mp4'):
#         # print(video_item)
#         command = command + video_folder_path + video_item
#         print(command)
#         os.system(command)

def get_video_landmark(video_path):
    command = org_command + video_path
    print(command)
    os.system(command)
    
    
        
if __name__ == '__main__':
    video_folder_path = "/workspace/DINet/asserts/training_data/split_video_25fps/"
    
    # 获取所有视频路径
    video_path_list = glob.glob(os.path.join(video_folder_path, '*.mp4'))

    # 定义线程池，限制最大并发线程数为20
    max_threads = 5
    with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
        # 创建一个列表以批量提交视频处理任务
        futures = []
        for video_path in video_path_list:
            future = executor.submit(get_video_landmark, video_path)
            futures.append(future)

        # 等待所有任务完成
        concurrent.futures.wait(futures)

    print("All video deepspeech extraction is done.")    

        


