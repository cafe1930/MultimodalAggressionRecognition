import pandas as pd
import glob
import os
import shutil

import torch
import torchvision
from torchvision.transforms import v2

from tqdm import tqdm

if __name__ == '__main__':
    path_to_source_dir = r'i:\AVABOS\DATSET_V0.2'
    path_to_partitioned_dataset = r'i:\AVABOS\DATSET_V0_train_test_split'
    #path_to_combinations_info_table = r'I:\AVABOS\!combinations_info_table.csv'
    path_to_combinations_info_table = os.path.join(path_to_source_dir,'!combinations_info_table.csv')
    target_partition_idx = 512952

    paths_to_videos_list = glob.glob(os.path.join(path_to_source_dir, 'physical', 'video', '*.mp4'))
    
    paths_to_waves_list = glob.glob(os.path.join(path_to_source_dir, 'verbal', 'pt_waveform', '*.pt'))    
    paths_to_embeddings_list = glob.glob(os.path.join(path_to_source_dir, 'verbal', '*', '*.npy'))
    
    combinations_info_table = pd.read_csv(path_to_combinations_info_table)
    combinations_info_table['cluster__indices_combination'] = combinations_info_table['cluster__indices_combination'].apply(lambda x:eval(x)if isinstance(x, str) else x)
    combinations_info_table['rest_indices_combination'] = combinations_info_table['rest_indices_combination'].apply(lambda x:eval(x)if isinstance(x, str) else x)
    train_test_combinations = combinations_info_table.loc[target_partition_idx][['cluster__indices_combination', 'rest_indices_combination']]
    train_test_combinations = train_test_combinations.rename({'cluster__indices_combination':'train_clusters', 'rest_indices_combination':'test_clusters'}).to_dict()

    os.makedirs(path_to_partitioned_dataset, exist_ok=True)
    print('-------------------------------------------')
    print('COPY VIDEOS')
    for path_to_video in tqdm(paths_to_videos_list):
        split_path = path_to_video.split(os.sep)
        video_name = split_path[-1]
        name = '.'.join(video_name.split('.')[:-1])
        split_name = name.split('_')
        cluster = split_name[0]
        cluster = int(cluster.split('-')[-1])
        partition = 'train' if cluster in train_test_combinations['train_clusters'] else 'test'
        path_to_saving_dir = os.path.join(path_to_partitioned_dataset, partition, split_path[-3], split_path[-2])
        os.makedirs(path_to_saving_dir, exist_ok=True)
        path_to_copy = os.path.join(path_to_saving_dir, video_name)
        shutil.copy2(path_to_video, path_to_copy)

    print('-------------------------------------------')
    print('COPY PT WAVEFORMS')
    for path_to_wave in tqdm(paths_to_waves_list):
        split_path = path_to_wave.split(os.sep)
        wave_name = split_path[-1]
        name = '.'.join(wave_name.split('.')[:-1])
        split_name = name.split('_')
        cluster = split_name[0]
        cluster = int(cluster.split('-')[-1])
        partition = 'train' if cluster in train_test_combinations['train_clusters'] else 'test'
        path_to_saving_dir = os.path.join(path_to_partitioned_dataset, partition, split_path[-3], split_path[-2])
        os.makedirs(path_to_saving_dir, exist_ok=True)
        path_to_copy = os.path.join(path_to_saving_dir, wave_name)
        shutil.copy2(path_to_wave, path_to_copy)

    print('-------------------------------------------')
    print('COPY TEXT EMBEDDINGS')
    for path_to_embeddings in tqdm(paths_to_embeddings_list):
        split_path = path_to_embeddings.split(os.sep)
        embeddings_name = split_path[-1]
        name = '.'.join(embeddings_name.split('.')[:-1])
        split_name = name.split('_')
        cluster = split_name[0]
        cluster = int(cluster.split('-')[-1])
        partition = 'train' if cluster in train_test_combinations['train_clusters'] else 'test'
        path_to_saving_dir = os.path.join(path_to_partitioned_dataset, partition, split_path[-3], split_path[-2])
        os.makedirs(path_to_saving_dir, exist_ok=True)
        path_to_copy = os.path.join(path_to_saving_dir, embeddings_name)
        shutil.copy2(path_to_embeddings, path_to_copy)

    print('-------------------------------------------')
    print('CREATE PT VIDEOS')
    paths_to_videos_list = glob.glob(os.path.join(path_to_partitioned_dataset, '*', 'physical', 'video', '*.mp4'))
    for path_to_video in tqdm(paths_to_videos_list):
        path_to_root, video_name = os.path.split(path_to_video)
        name = '.'.join(video_name.split('.')[:-1])
        video_frames, audio_frames, meta_data = torchvision.io.read_video(path_to_video, output_format="TCHW")
        #video_frames = tv_tensors.Video(video_frames)
        video_frames = v2.functional.resize(video_frames, size=(128, 128))
        path_to_save = os.path.join(path_to_root, f'{name}.pt')
        torch.save(video_frames, path_to_save)