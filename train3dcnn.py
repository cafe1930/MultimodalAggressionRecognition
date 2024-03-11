import torch
from torch import nn
import torchvision

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from sklearn import metrics

#from trainer import TorchSupervisedTrainer

import os

import pandas as pd

import glob

import random
import os
import numpy as np

import argparse

from torchvision.transforms import v2

from datasets import VideoDataset, VideoBboxesDataset, RandomAffineVideoBboxes, RandomPerspectiveVideoBboxes, RandomHorizontalFlipVideoBboxes, CreateBboxesMasks, NormalizeBboxes, ResizeBboxes
from trainer import TorchSupervisedTrainer
from models import VideoMultiNN, FeatureSequenceProcessing, VideoAverageFeatures, AverageFeatureSequence, MultiCrossEntropyLoss, R3DWithBboxes, R3D

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset',  required=True)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--path_to_checkpoint')
    parser.add_argument('--batch_size', required=True)
    parser.add_argument('--nn_input_size', nargs='+', type=int)
    parser.add_argument('--epoch_num')
    parser.add_argument('--test_size', type=float)

    sample_args = [
        '--path_to_dataset',
        r'I:\AVABOS\DATASET_VERSION_1.0\PHYS',
        '--class_num', '4',
        '--epoch_num', '100',
        '--batch_size', '8']
    
    
    args = parser.parse_args(sample_args)

    path_to_dataset_root = args.path_to_dataset
    resume_training = args.resume_training
    path_to_checkpoint = args.path_to_checkpoint
    epoch_num = int(args.epoch_num)
    class_num = args.class_num
    batch_size = int(args.batch_size)

        
    if resume_training == True:
        if path_to_checkpoint is None:
            raise ValueError('--path_to_checkpoint flag must be specified if --resume_training flag')
 
    #datasize_size = len(names_list) // 2

    paths_to_train_dirs_list = glob.glob(os.path.join(path_to_dataset_root, 'train', '*'))
    paths_to_test_dirs_list = glob.glob(os.path.join(path_to_dataset_root, 'test', '*'))

    train_bboxes_transform = v2.Compose([
                        ResizeBboxes((112, 112)),
                        RandomPerspectiveVideoBboxes(distortion_scale=0.2),
                        RandomAffineVideoBboxes(degrees = 4, translate = (0.2, 0.2), scale=(0.8, 1.2), shear=(-5, 5, -5, 5)),
                        RandomHorizontalFlipVideoBboxes(p=0.5),
                        CreateBboxesMasks(),
                        NormalizeBboxes(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                        )

    test_bboxes_transform = v2.Compose([
                        NormalizeBboxes(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ResizeBboxes((112, 112))]
                        )
    
    train_transform = v2.Compose([
        v2.Resize((112, 112)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.3), scale=(0.6, 1.1), shear=10),
        v2.RandomPerspective(distortion_scale=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = v2.Compose([
        v2.Resize((112, 112)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_bboxes_dataset = VideoBboxesDataset(paths_to_train_dirs_list, train_bboxes_transform, 32, 'cuda')
    test_bboxes_dataset = VideoBboxesDataset(paths_to_test_dirs_list, test_bboxes_transform, 32, 'cuda')

    train_dataset = VideoDataset(paths_to_train_dirs_list, train_transform, 32, 'cuda')
    test_dataset = VideoDataset(paths_to_test_dirs_list, test_transform, 32, 'cuda')

    train_bboxes_dataloader = torch.utils.data.DataLoader(
        train_bboxes_dataset,
        batch_size=8,
        shuffle=True, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
    )
    test_bboxes_dataloader = torch.utils.data.DataLoader(
        test_bboxes_dataset,
        batch_size=8,
        shuffle=False, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
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
    
    device = torch.device('cuda:0')
    #device = torch.device('cpu')
    
    # имя модели соответствует имени экстрактора признаков
    model_name = 'R3D'

    model = R3D(4)
    

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    criterion = nn.CrossEntropyLoss()
    #criterion = MultiCrossEntropyLoss()

    metrics_dict = {
        'loss': None,
        'accuracy': metrics.accuracy_score,
        'precision': {'metric': metrics.precision_score, 'kwargs': {'average': None}},
        'recall':{'metric': metrics.recall_score, 'kwargs': {'average': None}},
        'f1-score':{'metric': metrics.f1_score, 'kwargs': {'average': None}},
        'UAR': {'metric': metrics.recall_score, 'kwargs': {'average': 'macro'}},
    }

    metrics_to_dispaly = ['loss', 'accuracy', 'UAR']

        

    if resume_training:
        trainer = torch.load(path_to_checkpoint)
        #trainer.train_loader.dataset.path_to_dataset = path_to_dataset
        #trainer.train_loader.dataset.path_to_dataset = path_to_dataset
    else:
        trainer = TorchSupervisedTrainer(
            model=model,
            model_name=model_name,
            train_loader=train_dataloader,
            test_loader=test_dataloader,
            metrics_dict=metrics_dict,
            metrics_to_display=metrics_to_dispaly,
            criterion=criterion,
            optimizers_list=[optimizer],
            checkpoint_criterion='UAR',
            device=device
            )


    #print(trainer.start_epoch)
    #exit()
    # Запуск обучения
    trainer.train(epoch_num-trainer.start_epoch)

    #print(segmentation_trainer.testing_log_df['mean IoU'])
    #epoch_idx = trainer.testing_log_df['accuracy'].astype(float).argmax()

    #best_acc = trainer.testing_log_df['accuracy'].astype(float).max()

    #print('Best Accuracy for {} is {} on {} epoch'.format(model_name, best_acc, epoch_idx))