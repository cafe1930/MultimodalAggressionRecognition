import os
import glob
import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm
if __name__=='__main__':
    #path_to_dataset = r'I:\AVABOS\DATSET_V0' 
    #path_to_dataset = r'/home/aggr/mikhail_u/DATA/DATSET_V0'
    path_to_dataset = r'/home/ubuntu/mikhail_u/DATSET_V0'
    #path_to_dataset = r'C:\Users\admin\python_programming\DATA\AVABOS\DATSET_V0'

    paths_to_videos_list = glob.glob(os.path.join(path_to_dataset, 'physical', 'video', '*.mp4'))
    for path_to_video in tqdm(paths_to_videos_list):
        path_to_root, video_name = os.path.split(path_to_video)
        name = '.'.join(video_name.split('.')[:-1])
        video_frames, audio_frames, meta_data = torchvision.io.read_video(path_to_video, output_format="TCHW")
        #video_frames = tv_tensors.Video(video_frames)
        video_frames = v2.functional.resize(video_frames, size=(128, 128))
        path_to_save = os.path.join(path_to_root, f'{name}.pt')
        torch.save(video_frames, path_to_save)
    exit()
    path_to_dataset = r'I:\AVABOS\DATSET_V0_train_test_split'
    paths_to_videos_list = glob.glob(os.path.join(path_to_dataset, '*', 'physical', 'video', '*.pt'))
    for path_to_video in tqdm(paths_to_videos_list):
        os.remove(path_to_video)