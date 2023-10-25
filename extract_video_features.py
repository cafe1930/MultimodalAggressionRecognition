import torch
from torch import nn
import numpy as np

from torchvision.transforms import v2

import glob
import os

from tqdm import tqdm

from sklearn import model_selection

from datasets import NumpyVideoExtractorDataset

from models import R3D_extractor, Swin3d_T_extractor, S3D_extractor


if __name__ == '__main__':
    start_ep = 1000
    finish_ep = 1250
    
    model_name = 'swin3dt'
    #extractor = R3D_extractor(frame_num=304, window_size=16).cuda()
    extractor = Swin3d_T_extractor(frame_num=304, window_size=16).cuda()
    #extractor = S3D_extractor(frame_num=304, window_size=16).cuda()
    extractor.eval()

    #paths_to_video_npy_list = glob.glob(r'I:\AVABOS\trash_to_train_on_video_numpy\*.npy')
    # read names from file due to their order is generated in Windows
    with open('train_names.txt', 'r', encoding='utf-8') as fd:
        train_names = fd.read()
    train_names = train_names.split('\n')

    path_to_dataset_root = r'C:\Users\admin\python_programming\DATA\AVABOS'
    path_to_npy_videos = os.path.join(path_to_dataset_root, 'trash_to_train_on_video_numpy')

    paths_to_video_npy_list = glob.glob(os.path.join(path_to_npy_videos, '*.npy'))
                                        
    train_videos_list, test_videos_list = model_selection.train_test_split(paths_to_video_npy_list, test_size=0.3, random_state=1)

    train_videos_list = [os.path.join(path_to_npy_videos, train_name) for train_name in train_names]

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.3), scale=(0.6, 1.1), shear=10),
        v2.RandomPerspective(distortion_scale=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NumpyVideoExtractorDataset(
        paths_to_data_list=train_videos_list,
        augmentation_transforms=train_transforms,
        device=torch.device('cuda')
    )

    test_dataset = NumpyVideoExtractorDataset(
        paths_to_data_list=test_videos_list,
        augmentation_transforms=test_transforms,
        device=torch.device('cuda')
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
    )

    
    path_to_save_test = os.path.join(path_to_dataset_root, f'video_sequences_{model_name}', 'test')
    os.makedirs(path_to_save_test, exist_ok=True)
    for labels, data in tqdm(test_dataloader):
        with torch.inference_mode():
            results = extractor(data)
        for label, features_seq in zip(labels, results):
            path_to_save = os.path.join(path_to_save_test, label)
            np.save(path_to_save, features_seq)

    print('----------------')
    print('Test sample DONE')
    print('----------------')
    #exit()

    for epoch_idx in range(start_ep, finish_ep):
        print(f'Epoch # {epoch_idx} of {finish_ep} epochs')
        
        #path_to_save_train = os.path.join(r'/home/aggr/mikhail_u/DATA/video_squences_swin3d_t/train', str(epoch_idx))
        path_to_save_train = os.path.join(path_to_dataset_root, f'video_sequences_{model_name}', 'train', str(epoch_idx))
        os.makedirs(path_to_save_train, exist_ok=True)
        for labels, data in tqdm(train_dataloader):
            with torch.inference_mode():
                results = extractor(data)

            for label, features_seq in zip(labels, results):
                path_to_save = os.path.join(path_to_save_train, label)
                np.save(path_to_save, features_seq)