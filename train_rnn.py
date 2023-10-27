import torch
from torch import nn
import torchvision

from sklearn.model_selection import train_test_split

from models import RNN, VideoAverageFeatures
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

from datasets import RnnFeaturesDataset
from trainer import RNN_trainer

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
    '''
    sample_args = [
        '--path_to_dataset',
        r'E:\AVABOS\video_squences_r3d',
        #r'D:\develop\mice_holes_dataset',
        '--class_num', '2',
        '--epoch_num', '2000',
        '--batch_size', '128',
        '--resume_training',
        '--path_to_checkpoint', r'saving_dir\25.10.2023, 20-55-22 (R3D_GRU_1Layer)\R3D_GRU_1Layer_current_ep-1502.pt']
    '''
    sample_args = [
        '--path_to_dataset',
        r'I:\AVABOS\video_squences_r3d',
        '--class_num', '2',
        '--epoch_num', '2000',
        '--batch_size', '128']
    
    
    args = parser.parse_args(sample_args)

    path_to_dataset_root = args.path_to_dataset
    resume_training = args.resume_training
    path_to_checkpoint = args.path_to_checkpoint
    epoch_num = int(args.epoch_num)
    class_num = args.class_num
    
    if resume_training == True:
        if path_to_checkpoint is None:
            raise ValueError('--path_to_checkpoint flag must be specified if --resume_training flag')
 
    #datasize_size = len(names_list) // 2

    batch_size=128

    path_to_train_data_root = os.path.join(path_to_dataset_root, 'train', '0')
    path_to_test_data_root = os.path.join(path_to_dataset_root, 'test')

    train_dataset = RnnFeaturesDataset(path_to_data_root=path_to_train_data_root)
    test_dataset = RnnFeaturesDataset(path_to_data_root=path_to_test_data_root)

    #print(train_dataset[0])
    #print(test_dataset[-1])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)    
    
    device = torch.device('cuda:0')
    #device = torch.device('cpu')
    '''
    model = RNN(
        rnn_type=nn.GRU,
        rnn_layers_num=1,
        input_dim=512,
        hidden_dim=512,
        class_num=2)
    '''
    model = VideoAverageFeatures(
        input_dim=512,
        class_num=2
    )
    model_name = 'R3D_avg'

    #for l, d in tqdm(train_loader):
    #   pass
    #print(d.shape, l.shape)
    '''
    for l, d in tqdm(test_loader):
        pass
    print(d.shape, l.shape)
    out = model(d)
    print(out.shape)
    exit()
    '''

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    criterion = nn.CrossEntropyLoss()

    metrics_dict = {
        'loss': None,
        'accuracy': metrics.accuracy_score,
        'UAR': {'metric': metrics.recall_score, 'kwargs': {'average': 'macro'}},
    }

    metrics_to_dispaly = ['loss', 'accuracy', 'UAR']

    

    if resume_training:
        trainer = torch.load(path_to_checkpoint)
        #trainer.train_loader.dataset.path_to_dataset = path_to_dataset
        #trainer.train_loader.dataset.path_to_dataset = path_to_dataset
    else:
        trainer = RNN_trainer(
        model=model,
        model_name=model_name,
        train_loader=train_loader,
        test_loader=test_loader,
        metrics_dict=metrics_dict,
        metrics_to_display=metrics_to_dispaly,
        criterion=criterion,
        optimizers_list=[optimizer],
        checkpoint_criterion='UAR',
        device=device,
        train_dataset=RnnFeaturesDataset)


    #print(trainer.start_epoch)
    #exit()
    # Запуск обучения
    trainer.train(epoch_num-trainer.start_epoch)

    #print(segmentation_trainer.testing_log_df['mean IoU'])
    epoch_idx = trainer.testing_log_df['accuracy'].astype(float).argmax()

    best_acc = trainer.testing_log_df['accuracy'].astype(float).max()

    print('Best Accuracy for {} is {} on {} epoch'.format(model_name, best_acc, epoch_idx))