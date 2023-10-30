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

from datasets import RnnFeaturesDataset
from trainer import RNN_trainer
from models import VideoMultiNN, FeatureSequenceProcessing, VideoAverageFeatures, AverageFeatureSequence, MultiCrossEntropyLoss

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
        '--epoch_num', '10',
        '--batch_size', '128']
    
    
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
    
    
    # описание архитектур всех обучаемых рекуррентных нейросетей для обертки FeatureSequenceProcessing
    input_size = 512 # определяется размером вектора признаков, получаемого с экстрактора
    hidden_size = 512 # не будет работать, если будет более двух слоев rnn
    rnn_dict = {
        'LSTM_1L': {
            'model': nn.LSTM,
            'kwargs': {
                'input_size':input_size,
                'hidden_size':hidden_size,
                'num_layers':1,
                'bias':True,
                'batch_first':True,
                'dropout':0,
                'bidirectional':False,
                'proj_size':0
            }
        },
        'GRU_1L': {
            'model': nn.GRU,
            'kwargs': {
                'input_size':input_size,
                'hidden_size':hidden_size,
                'num_layers':1,
                'bias':True,
                'batch_first':True,
                'dropout':0,
                'bidirectional':False
            }
        },
        'Avg_features': {
            'model': AverageFeatureSequence,
            'kwargs': {'hidden_size':hidden_size}
        }
    }
    
    # словарь с моделями-обертками FeatureSequenceProcessing, включающими, помимо RNN, еще и выходной классификатор
    models_dict = {}
    for name, model in rnn_dict.items():
        models_dict[name] = FeatureSequenceProcessing(model, class_num=2)

    # имя модели соответствует имени экстрактора признаков
    model_name = 'R3D'

    model = VideoMultiNN(models_dict=models_dict)
    #print(model)
    #print(model(torch.randn(1, 19, input_size)))

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

    #criterion = nn.CrossEntropyLoss()
    criterion = MultiCrossEntropyLoss()

    metrics_dict = {
        'loss': None,
        'accuracy': metrics.accuracy_score,
        'UAR': {'metric': metrics.recall_score, 'kwargs': {'average': 'macro'}},
    }

    metrics_to_dispaly = ['loss', 'accuracy', 'UAR']

    # DEBUG
    '''
    optimizer.zero_grad()
    bs = 1
    pred = model(torch.randn(bs, 19, input_size).to(device))
    target = torch.randint(0, class_num, (bs, )).to(device)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    print(loss)
    exit()
    '''
        

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