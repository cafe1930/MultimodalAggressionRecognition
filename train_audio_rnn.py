import torch
from torch import nn

import torchaudio
from sklearn.model_selection import train_test_split

import pickle

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

from datasets import RnnFeaturesDataset, AudioDatasetPt
from trainer import AudioRNN_trainer
from models import (
    AudioMultiNN,
    FeatureSequenceProcessing,
    VideoAverageFeatures,
    AverageFeatureSequence,
    MultiCrossEntropyLoss,
    Wav2vecExtractor,
    Wav2vec2Extractor)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset',  required=True)
    parser.add_argument('--model_name', required=True)
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
        r'I:\AVABOS\audio_data',
        '--model_name', 'wav2vec2',
        '--class_num', '2',
        '--epoch_num', '2000',
        '--batch_size', '16']
    
    
    args = parser.parse_args(sample_args)

    path_to_dataset_root = args.path_to_dataset
    resume_training = args.resume_training
    path_to_checkpoint = args.path_to_checkpoint
    epoch_num = int(args.epoch_num)
    class_num = args.class_num
    batch_size = int(args.batch_size)
    model_name = args.model_name
    
    if resume_training == True:
        if path_to_checkpoint is None:
            raise ValueError('--path_to_checkpoint flag must be specified if --resume_training flag')
 
    #datasize_size = len(names_list) // 2
    device = torch.device('cuda:0')

    path_to_train_data_root = os.path.join(path_to_dataset_root, 'train')
    path_to_test_data_root = os.path.join(path_to_dataset_root, 'test')

    # имя модели соответствует имени экстрактора признаков
    #model_name = 'wav2vec1'
    # подготовка модели
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    sample_rate = bundle.sample_rate

    #train_dataset = RnnFeaturesDataset(path_to_data_root=path_to_train_data_root)
    train_dataset = AudioDatasetPt(path_to_train_data_root, sample_rate, 10, torch.device('cpu'))
    test_dataset = AudioDatasetPt(path_to_test_data_root, sample_rate, 10, torch.device('cpu'))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #print(train_dataset[0][0].dtype)
    #exit()

    if model_name == 'wav2vec1':
        # Wav2Vec1.0
        extractor_dict = {model_name: Wav2vecExtractor(torch.jit.load('wav2vec_feature_extractor_jit.pt'))}
        input_size = 512 # определяется размером вектора признаков, получаемого с экстрактора
    elif model_name == 'wav2vec2':
        # Wav2Vec 2.0
        extractor_dict = {model_name: Wav2vec2Extractor(bundle.get_model())}
        input_size = 768 # определяется размером вектора признаков, получаемого с экстрактора
    else:
        raise ValueError('Allowed model names are: wav2vec1, wav2vec1')
    
    # описание архитектур всех обучаемых рекуррентных нейросетей для обертки FeatureSequenceProcessing
    
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
            'kwargs': {'hidden_size':input_size}
        }
    }
    
    # словарь с моделями-обертками FeatureSequenceProcessing, включающими, помимо RNN, еще и выходной классификатор
    models_dict = {}
    for name, model in rnn_dict.items():
        models_dict[name] = FeatureSequenceProcessing(model, class_num=2)

    model = AudioMultiNN(models_dict=models_dict, extractor_dict=extractor_dict)

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

    trainer = AudioRNN_trainer(
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
    #trainer.load_checkpoint(r'saving_dir\30.10.2023, 03-13-05 (wav2vec2)\wav2vec2_current_ep-5')
    #print(trainer.path_to_best_checkpoint)
    #exit()
    
    #trainer.train(epoch_num-trainer.start_epoch)
    #exit()
    

    if resume_training:
        # здесь почему-то нельзя сохранить всю конфигурацию класса trainer из-за модели
        # поэтому восстанавливаем вручную
        trainer.load_checkpoint(path_to_checkpoint)
        
    
        
    #print(trainer.__dict__)
    
    
    #exit()

    #print(trainer.start_epoch)
    #exit()
    # Запуск обучения
    trainer.train(epoch_num-trainer.start_epoch)
    '''
    trainer = AudioRNN_trainer(
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
    trainer.load_checkpoint()
    '''

    #print(segmentation_trainer.testing_log_df['mean IoU'])
    #epoch_idx = trainer.testing_log_df['accuracy'].astype(float).argmax()

    #best_acc = trainer.testing_log_df['accuracy'].astype(float).max()

    #print('Best Accuracy for {} is {} on {} epoch'.format(model_name, best_acc, epoch_idx))