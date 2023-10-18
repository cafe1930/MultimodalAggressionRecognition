import torch
from torch import nn
import numpy as np

from torchvision.transforms import v2

import glob
import os

from tqdm import tqdm

from sklearn import model_selection

from datasets import NumpyVideoExtractorDataset

from models import R3D_extractor


if __name__ == '__main__':
    start_ep = 500
    finish_ep = 750

    extractor = R3D_extractor(frame_num=304, window_size=16).cuda()
    extractor.eval()

    #paths_to_video_npy_list = glob.glob(r'I:\AVABOS\trash_to_train_on_video_numpy\*.npy')
    paths_to_video_npy_list = glob.glob(r'/home/ubuntu/DATA/trash_to_train_on_video_numpy/*.npy')
    train_videos_list, test_videos_list = model_selection.train_test_split(paths_to_video_npy_list, test_size=0.3, random_state=1)
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
        batch_size=32,
        shuffle=True, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
    )

    #for idx in tqdm(range(len(train_dataset))):
    #    label, data = train_dataset[idx]

    
    #path_to_save_test = r'I:\AVABOS\video_squences_r3d\test'
    #os.makedirs(path_to_save_test, exist_ok=True)
    #for labels, data in tqdm(test_dataloader):
    #    with torch.inference_mode():
    #        results = extractor(data)
    #    for label, features_seq in zip(labels, results):
    #        path_to_save = os.path.join(path_to_save_test, label)
    #        np.save(path_to_save, features_seq)

    print('----------------')
    print('Test sample DONE')
    print('----------------')
    
    for epoch_idx in range(start_ep, finish_ep):
        print(f'Epoch # {epoch_idx} of {finish_ep} epochs')
        path_to_save_train = os.path.join(r'/home/ubuntu/DATA/video_squences_r3d/train', str(epoch_idx))
        os.makedirs(path_to_save_train, exist_ok=True)
        for labels, data in tqdm(train_dataloader):
            with torch.inference_mode():
                results = extractor(data)

            for label, features_seq in zip(labels, results):
                path_to_save = os.path.join(path_to_save_train, label)
                np.save(path_to_save, features_seq)

    