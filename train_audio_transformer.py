import torch
from torch import nn
import torchaudio
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

from datasets import PtAudioDataset, AppendZeroValues, WavAudioDataset
from trainer import TorchSupervisedTrainer
from models import TransformerSequenceProcessor, Wav2vec2Extractor, Wav2vecExtractor, CNN1D

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset',  required=True)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--path_to_checkpoint')
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--nn_input_size', nargs='+', type=int)
    parser.add_argument('--epoch_num', type=int)
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--max_audio_len', type=int)
    
    sample_args = [
        '--path_to_dataset',
        #r'C:\Users\admin\python_programming\DATA\AVABOS\DATSET_V0_train_test_split',
        r'I:\AVABOS\DATSET_V0_train_test_split',
        '--class_num', '2',
        '--epoch_num', '100',
        '--batch_size', '32',
        '--max_audio_len', '80000'
        #'--max_audio_len', '150000'
        ]
    
    args = parser.parse_args(sample_args)

    path_to_dataset_root = args.path_to_dataset
    resume_training = args.resume_training
    path_to_checkpoint = args.path_to_checkpoint
    epoch_num = int(args.epoch_num)
    class_num = args.class_num
    batch_size = int(args.batch_size)
    max_audio_len = args.max_audio_len
        
    if resume_training == True:
        if path_to_checkpoint is None:
            raise ValueError('--path_to_checkpoint flag must be specified if --resume_training flag')
        
    device = torch.device('cuda:0')
 
    paths_to_train_audios_list = glob.glob(os.path.join(path_to_dataset_root, 'train', 'verbal', 'pt_waveform', '*.pt'))
    paths_to_test_audios_list = glob.glob(os.path.join(path_to_dataset_root, 'test', 'verbal', 'pt_waveform', '*.pt'))
    #paths_to_train_audios_list = glob.glob(os.path.join(path_to_dataset_root, 'train', 'verbal', 'pt_waveform', '*.wav'))
    #paths_to_test_audios_list = glob.glob(os.path.join(path_to_dataset_root, 'test', 'verbal', 'pt_waveform', '*.wav'))

    #bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE
    sample_rate = bundle.sample_rate
    #print(sample_rate)
    

    train_transform = v2.Compose([
        AppendZeroValues(target_size=[max_audio_len]),
        #v2.ToDtype(torch.float32, scale=True)
    ])

    test_transform = v2.Compose([
        AppendZeroValues(target_size=[max_audio_len]),
        #v2.ToDtype(torch.float32, scale=True)
    ])

    # пример аугментация со спектрограммами
    class MakeRGBSpectrogram(nn.Module):
        def forward(self, x):
            return torch.stack([x, x, x], dim=0)
    train_transform = v2.Compose([
        AppendZeroValues(target_size=[max_audio_len]),
        torchaudio.transforms.Spectrogram(n_fft=512, wkwargs={'device':device}),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=80),
        torchaudio.transforms.TimeMasking(time_mask_param=80),
        MakeRGBSpectrogram()
        ])
    test_transform = v2.Compose([
        AppendZeroValues(target_size=[max_audio_len]),
        torchaudio.transforms.Spectrogram(n_fft=512, wkwargs={'device':device}),
        MakeRGBSpectrogram()
        ])

    train_dataset = PtAudioDataset(paths_to_train_audios_list, train_transform, 'cuda')
    test_dataset = PtAudioDataset(paths_to_test_audios_list, test_transform, 'cuda')
    #train_dataset = WavAudioDataset(paths_to_train_audios_list, train_transform, 'cuda')
    #test_dataset = WavAudioDataset(paths_to_test_audios_list, test_transform, 'cuda')
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
    )

    #for data, labels in tqdm(train_dataloader):
    #    break
    #print(data.shape)
    #exit()
    
    
    
    #device = torch.device('cpu')
    
    # имя модели соответствует имени экстрактора признаков
    model_name = 'audio vgg'

    

    #audio_extractor = Wav2vec2Extractor(bundle.get_model())
    #audio_extractor = Wav2vecExtractor(torch.jit.load('wav2vec_feature_extractor_jit.pt'))

    '''
    model = TransformerSequenceProcessor(
        extractor_model=audio_extractor,
        transformer_layer_num=2,
        transformer_head_num=8,
        hidden_size=1280,
        class_num=class_num
        )
    '''
    #model = CNN1D(class_num=2)
    model = torchvision.models.vgg11_bn(weights=torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1)
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

    metrics_to_dispaly = ['loss', 'accuracy', 'UAR', 'recall', 'precision']

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