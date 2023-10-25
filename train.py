import torch
from torch import nn
import torchvision

from sklearn.model_selection import train_test_split

from models import resnet


from sklearn import metrics

#from trainer import TorchSupervisedTrainer

import os

import pandas as pd

import glob

import random
import os
import numpy as np

import argparse

from datasets import SegmentationDataset
from trainer import SegmentationTrainer



import warnings
warnings.filterwarnings("ignore")


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
        '../mice_holes_dataset',
        #r'D:\develop\mice_holes_dataset',
        '--class_num', '2',
        '--batch_size', '8',
        '--epoch_num', '100',
        '--nn_input_size', '112', '112',
        '--test_size', '0.5']

    args = parser.parse_args(sample_args)

    path_to_dataset = args.path_to_dataset
    nn_input_size = args.nn_input_size
    resume_training = args.resume_training
    path_to_checkpoint = args.path_to_checkpoint
    epoch_num = int(args.epoch_num)
    test_size = args.test_size
    class_num = args.class_num
    
    if resume_training == True:
        if path_to_checkpoint is None:
            raise ValueError('--path_to_checkpoint flag must be specified if --resume_training flag')
        

    names_list = sorted([f[:-4] for f in os.listdir(path_to_dataset) if f.endswith('.npy')])#[:50]

    train_names, test_names = train_test_split(names_list, test_size=test_size, random_state=0)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    train_transforms = A.Compose(
        [
            v2.Resize(*nn_input_size),
            v2.HorizontalFlip(),
            v2.VerticalFlip(),
            v2.Affine(scale=(0.7, 1.3), rotate=(-90,90), shear=(-15, 15)),
            #A.GlassBlur(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    test_transforms = v2.Compose(
        [
            v2.Resize(*nn_input_size),
            #A.Rotate(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    
    #datasize_size = len(names_list) // 2

    batch_size=16

    train_dataset = entationDataset(path_to_dataset=path_to_dataset, instance_names_list=train_names, transforms=train_transforms)
    test_dataset = ntationDataset(path_to_dataset=path_to_dataset, instance_names_list=test_names, transforms=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    

    device = torch.device('cuda:0')
    #device = torch.device('cpu')

    model = R3D(in_channels=3, class_num=2)
    model_name = 'R3D'

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    criterion = nn.CrossEntropyLoss(weight=weights)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    #smp.losses.FocalLoss(mode='multiclass')

    labels = np.arange(0, class_num)
    metrics_dict = {
        'loss': None,
        'accuracy': ConfusionAccuracy(),
        'precision': 
    }

    metrics_to_dispaly = ['loss', 'accuracy']

    video_classification_trainer = TorchSupervisedTrainer(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        test_loader=test_loader,
        metrics_dict=metrics_dict,
        metrics_to_display=metrics_to_dispaly,
        criterion=criterion,
        optimizers_list=[optimizer],
        checkpoint_criterion='mean IoU',
        device=device)

    if resume_training:
        segmentation_trainer = torch.load(path_to_checkpoint)
        segmentation_trainer.train_loader.dataset.path_to_dataset = path_to_dataset
        segmentation_trainer.train_loader.dataset.path_to_dataset = path_to_dataset

    # Запуск обучения
    segmentation_trainer.train(epoch_num)

    #print(segmentation_trainer.testing_log_df['mean IoU'])
    epoch_idx = segmentation_trainer.testing_log_df['accuracy'].astype(float).argmax()

    best_acc = segmentation_trainer.testing_log_df['accuracy'].astype(float).max()

    print('Best Accuracy for {} is {} on {} epoch'.format(model_name, best_acc, epoch_idx))


