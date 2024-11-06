import torch
from torch import nn
import torchaudio

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from sklearn import metrics

#from trainer import TorchSupervisedTrainer

import os

import pandas as pd
import json
import glob

import random
import os
import numpy as np

import argparse

from torchvision.transforms import v2

from datasets import PtAudioDataset, AppendZeroValues, MultimodalDataset
from trainer import MultimodalTrainer
from models import TransformerSequenceProcessor, MultimodalModel, CNN1D, Swin3d_T_extractor, OutputClassifier, MultiModalCrossEntropyLoss, Wav2vec2Extractor, Wav2vecExtractor, AudioCnn1DExtractorWrapper

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset',  required=True)
    parser.add_argument('--path_to_intersections_csv')
    parser.add_argument('--path_to_train_test_split_json')
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--path_to_checkpoint')
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--nn_input_size', nargs='+', type=int)
    parser.add_argument('--epoch_num', type=int)
    parser.add_argument('--test_size', type=float)
    parser.add_argument('--max_audio_len', type=int)
    parser.add_argument('--max_embeddings_len', type=int)
    parser.add_argument('--video_frames_num', type=int)
    parser.add_argument('--video_window_size', type=int)
    
    sample_args = [
        '--path_to_dataset',
        r'/home/aggr/mikhail_u/DATA/DATSET_V0',
        #r'C:\Users\admin\python_programming\DATA\AVABOS\DATSET_V0_train_test_split',
        #r'I:\AVABOS\DATSET_V0',
        '--path_to_intersections_csv',
        r'/home/aggr/mikhail_u/DATA/DATSET_V0/time_intervals_combinations_table.csv',
        #r'i:\AVABOS\DATSET_V0\time_intervals_combinations_table.csv',
        '--path_to_train_test_split_json',
        r'train_test_split.json',
        '--class_num', '2',
        '--epoch_num', '100',
        '--batch_size', '32',
        '--max_audio_len', '80000',
        '--max_embeddings_len', '48',
        '--video_frames_num', '128',
        '--video_window_size', '8'
        ]
    
    args = parser.parse_args(sample_args)

    path_to_dataset_root = args.path_to_dataset
    resume_training = args.resume_training
    path_to_intersections_csv = args.path_to_intersections_csv
    path_to_train_test_split_json = args.path_to_train_test_split_json

    path_to_checkpoint = args.path_to_checkpoint
    epoch_num = int(args.epoch_num)
    class_num = args.class_num
    batch_size = int(args.batch_size)
    max_audio_len = args.max_audio_len
    max_embeddings_len = args.max_embeddings_len
    video_frames_num = args.video_frames_num
    video_window_size = args.video_window_size

    modalities_list = [
        'audio',
        'text',
        'video'
        ]
        
    if resume_training == True:
        if path_to_checkpoint is None:
            raise ValueError('--path_to_checkpoint flag must be specified if --resume_training flag')

    time_interval_combinations_df = pd.read_csv(path_to_intersections_csv)

    with open(path_to_train_test_split_json) as fd:
        combinations_indices_dict = json.load(fd)
    
    train_time_interval_combinations_df =  []
    for cluster_id in combinations_indices_dict['train_clusters']:
        df = time_interval_combinations_df[time_interval_combinations_df['cluster_id']==cluster_id]
        train_time_interval_combinations_df.append(df)
    train_time_interval_combinations_df = pd.concat(train_time_interval_combinations_df, ignore_index=True)

    test_time_interval_combinations_df =  []
    for cluster_id in combinations_indices_dict['test_clusters']:
        df = time_interval_combinations_df[time_interval_combinations_df['cluster_id']==cluster_id]
        test_time_interval_combinations_df.append(df)
    test_time_interval_combinations_df = pd.concat(test_time_interval_combinations_df, ignore_index=True)

    #bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE
    sample_rate = bundle.sample_rate
    #print(sample_rate)

    train_video_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.3), scale=(0.6, 1.1), shear=10),
        v2.RandomPerspective(distortion_scale=0.2),
        v2.Resize((112, 112), antialias=True),
        AppendZeroValues(target_size=[video_frames_num, 3, 112, 112]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_video_transform = v2.Compose([
        v2.Resize((112, 112), antialias=True),
        AppendZeroValues(target_size=[video_frames_num, 3, 112, 112]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_audio_transform = v2.Compose([AppendZeroValues(target_size=[max_audio_len])])
    test_audio_transform = v2.Compose([AppendZeroValues(target_size=[max_audio_len])])
    train_text_transform = v2.Compose([AppendZeroValues(target_size=[max_embeddings_len, 768])])
    test_text_transform = v2.Compose([AppendZeroValues(target_size=[max_embeddings_len, 768])])

    train_transforms_dict = {
        'audio': train_audio_transform,
        'text': train_text_transform,
        'video': train_video_transform
    }
    train_transforms_dict = {k:v for k,v in train_transforms_dict.items()if k in modalities_list}
    test_transforms_dict = {
        'audio': test_audio_transform,
        'text': test_text_transform,
        'video': test_video_transform
    }
    test_transforms_dict = {k:v for k,v in test_transforms_dict.items()if k in modalities_list}
    train_dataset = MultimodalDataset(
        time_intervals_df=train_time_interval_combinations_df,
        path_to_dataset=path_to_dataset_root,
        modality_augmentation_dict=train_transforms_dict,
        actual_modalities_list=modalities_list,
        device='cuda',
        text_embedding_type='ru_conversational_cased_L-12_H-768_A-12_pt_v1_tokens'
        )
    test_dataset = MultimodalDataset(
        time_intervals_df=test_time_interval_combinations_df,
        path_to_dataset=path_to_dataset_root,
        modality_augmentation_dict=train_transforms_dict,
        actual_modalities_list=modalities_list,
        device='cuda',
        text_embedding_type='ru_conversational_cased_L-12_H-768_A-12_pt_v1_tokens'
        )
    
    #for i in tqdm(range(len(train_dataset))):
    #    res = train_dataset[i]
    #for i in tqdm(range(len(test_dataset))):
    #    res = test_dataset[i]
    #    if 'audio' in res and 'video' in res:
    #        break
    #print(res)
    #exit()

    #print(train_dataset[0])
    #print()
    #print(test_dataset[0])
    #exit()
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
    #for data in tqdm(train_dataloader):
    #    pass

    device = torch.device('cuda:0')
    #device = torch.device('cpu')
    
    # имя модели соответствует имени экстрактора признаков
    model_name = 'Separated Modalities'

    

    #audio_extractor = Wav2vec2Extractor(bundle.get_model())
    #audio_extractor = Wav2vecExtractor(torch.jit.load('wav2vec_feature_extractor_jit.pt'))
   
    #audio_extractor = nn.Sequential(CNN1D(class_num=2),nn.Linear(512, 768),nn.Dropout(0.3))
    audio_extractor = AudioCnn1DExtractorWrapper(hidden_size=768)
    audio_model = TransformerSequenceProcessor(
        extractor_model=audio_extractor,
        transformer_layer_num=2,
        transformer_head_num=8,
        hidden_size=768,
        class_num=class_num
        )
    text_model = TransformerSequenceProcessor(
        extractor_model=nn.Sequential(),
        transformer_layer_num=2,
        transformer_head_num=8,
        hidden_size=768,
        class_num=2
    )

    video_extractor = Swin3d_T_extractor(frame_num=video_frames_num, window_size=video_window_size)

    video_model = TransformerSequenceProcessor(
        extractor_model=video_extractor,
        transformer_layer_num=2,
        transformer_head_num=8,
        hidden_size=768,
        class_num=class_num
        )
       
    # определяем размерности векторов признаков для многомодальной обработки
    video_features_shape = video_model(torch.zeros([1, 3, video_frames_num, 112, 112]), ret_type='features').shape
    audio_features_shape = audio_extractor(torch.zeros([1, max_audio_len]), ret_type='features').shape
    text_features_shape = text_model(torch.zeros([1, max_embeddings_len, 768]), ret_type='features').shape
    modality_features_shapes_dict = {
        'audio':list(audio_features_shape)[1:],
        'text':list(text_features_shape)[1:],
        'video':list(video_features_shape)[1:]
    }
    modality_features_shapes_dict = {k:v for k,v in modality_features_shapes_dict.items() if k in modalities_list}
    modality_extractors_dict = {
        'audio':audio_extractor,
        'text':text_model,
        'video':video_model
    }
    modality_extractors_dict = {k:v for k,v in modality_extractors_dict.items() if k in modalities_list}
    modality_extractors_dict = nn.ModuleDict(modality_extractors_dict)
    
    modality_classifiers_dict = {
        'audio':OutputClassifier(768, 2),
        'text':OutputClassifier(768, 2),
        'video':OutputClassifier(768, 2)
    }
    modality_classifiers_dict = {k:v for k,v in modality_classifiers_dict.items() if k in modalities_list}
    modality_classifiers_dict = nn.ModuleDict(modality_classifiers_dict)
    model = MultimodalModel(
        modality_extractors_dict=modality_extractors_dict,
        modality_classifiers_dict=modality_classifiers_dict,
        modality_features_shapes_dict=modality_features_shapes_dict,
        hidden_size=768,
        class_num=2)
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())

    #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    #criterion = nn.CrossEntropyLoss()
    criterion = MultiModalCrossEntropyLoss(modalities_list)
    
    

    metrics_dict = {
        'loss': None,
        'accuracy': metrics.accuracy_score,
        'precision': {'metric': metrics.precision_score, 'kwargs': {'average': None}},
        'recall':{'metric': metrics.recall_score, 'kwargs': {'average': None}},
        'f1-score':{'metric': metrics.f1_score, 'kwargs': {'average': None}},
        'UAR': {'metric': metrics.recall_score, 'kwargs': {'average': 'macro'}},
    }

    metrics_to_dispaly = ['loss', 'accuracy', 'UAR', 'recall', 'precision', 'f1-score']

    if resume_training:
        trainer = torch.load(path_to_checkpoint)
        #trainer.train_loader.dataset.path_to_dataset = path_to_dataset
        #trainer.train_loader.dataset.path_to_dataset = path_to_dataset
    else:
        trainer = MultimodalTrainer(
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