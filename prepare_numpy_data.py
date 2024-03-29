from torchvision.transforms import v2
from torchvision import tv_tensors

#from torchvision import datapoints

import torch
import cv2
import numpy as np

import glob
import os

from tqdm import tqdm

def read_video_frames_opencv(path_to_video, frame_pos1, frame_pos2):
    cap = cv2.VideoCapture(path_to_video)
    if frame_pos2 <= frame_pos1:
        raise ValueError(f'frame_pos2 shoud be greater than frame_pos1')
    
    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_num = frame_pos2 - frame_pos1
    if frame_num > total_frame_num:
        frame_num = total_frame_num - frame_pos1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos1)
    frames_list = []
    for frame_idx in range(frame_num):
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    #return torch.tensor(np.array(frames_list).transpose(0, 3, 1, 2))
    #return tv_tensors.Video(np.array(frames_list).transpose(0, 3, 1, 2))
    return np.array(frames_list)#)#.transpose(0, 3, 1, 2)

if __name__ == '__main__':
    #paths_to_videos_list = glob.glob(r'/home/aggr/mikhail_u/DATA/trash_to_train_on_video/*.mp4')
    paths_to_videos_list = glob.glob(r'I:\AVABOS\DATASET_VERSION_1.0\phys\*\*\*.mp4')

    #path_to_numpy = r'/home/aggr/mikhail_u/DATA/trash_to_train_on_video_numpy224'
    #os.makedirs(path_to_numpy, exist_ok=True)

    video_lengths_list = []

    frame_cut_idx = 304 # 16*19

    for path_to_video in tqdm(paths_to_videos_list):
        
        path_to_root, name = os.path.split(path_to_video)
        name = name[:-4]
        path_to_save_numpy = os.path.join(path_to_root, name + '.npy')

        frames = read_video_frames_opencv(path_to_video)
        
        frame_num = len(frames)
        video_lengths_list.append(frame_num)

        if frame_num == 0:
            continue
        '''
        if frame_num > frame_cut_idx:
            frames = frames[:frame_cut_idx]
        elif frame_num < frame_cut_idx:
            pad_value = frame_cut_idx - frame_num
            rows = frames.shape[1]
            cols = frames.shape[2]
            channels = frames.shape[3]
            canvas = np.zeros(shape=(frame_cut_idx, rows, cols, channels), dtype=frames.dtype)
            canvas[:frame_num] = frames
            frames = canvas
        '''
        frames = frames.transpose(0, 3, 1, 2)

        frames = torch.as_tensor(frames)
        #print(frames.shape)
        #tv_frames = tv_tensors.Video(frames)
        #print(tv_frames.shape)
        #resizer = v2.Resize(size=(224, 224), antialias=True)
        #tv_frames = resizer(tv_frames)
        #print(tv_frames.shape)
        #frames = tv_frames.numpy()#.transpose(0, 2, 3, 1)

        np.save(path_to_save_numpy, frames)