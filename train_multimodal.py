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
import json
import glob

import random
import os
import numpy as np

import argparse

from torchvision.transforms import v2

from datasets import MultimodalPhysVerbDataset, AggrBatchSampler, AppendZeroValues, MultimodalDataset
from trainer import MultimodalTrainer
from models import AveragedFeaturesTransformerFusion, PhysVerbClassifier, PhysVerbModel, TransformerSequenceProcessor, EqualSizedTransformerModalitiesFusion, MultimodalModel, CNN1D, Swin3d_T_extractor, OutputClassifier, MultiModalCrossEntropyLoss, Wav2vec2Extractor, Wav2vecExtractor, AudioCnn1DExtractorWrapper

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset',  required=True)
    parser.add_argument('--path_to_intersections_csv')
    parser.add_argument('--path_to_train_test_split_json')
    parser.add_argument('--gpu_device_idx', type=int)
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
        r'/home/ubuntu/mikhail_u/DATASET_V0',
        #r'/home/aggr/mikhail_u/DATA/DATSET_V0',
        #r'C:\Users\admin\python_programming\DATA\AVABOS\DATSET_V0',
        #r'I:\AVABOS\DATSET_V0',
        '--path_to_intersections_csv',
        r'/home/ubuntu/mikhail_u/DATASET_V0/time_intervals_combinations_table.csv',
        #r'/home/aggr/mikhail_u/DATA/DATSET_V0/time_intervals_combinations_table.csv',
        #r'C:\Users\admin\python_programming\DATA\AVABOS\DATSET_V0\time_intervals_combinations_table.csv',
        #r'i:\AVABOS\DATSET_V0\time_intervals_combinations_table.csv',
        '--path_to_train_test_split_json',
        r'train_test_split.json',
        '--gpu_device_idx', '0',
        '--class_num', '2',
        '--epoch_num', '100',
        '--batch_size', '64',
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
    gpu_device_idx = args.gpu_device_idx

    # имя модели соответствует имени экстрактора признаков
    model_name = 'FullDs_V+T+fusion1L-focalloss'
    modality2aggr = {'video':'phys', 'text':'verb', 'audio':'verb'}
    modalities_list = [
        #'audio',
        'text',
        'video'
        ]
    aggr_types_list = set()
    for m in modalities_list:
        aggr_types_list.add(modality2aggr[m])

    aggr_types_list = list(aggr_types_list)

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
    # для выравнивания баланса классов (баланс смещен в сторону не агрессивного поведения)
    # удалим не агрессивные интервалы физ. поведения, которые не пересекаются с вербальным поведением
    #drop_no_aggr_filter = (train_time_interval_combinations_df['aggr_type']=='phys')&(train_time_interval_combinations_df['phys_aggr_label']=='NOAGGR')
    #train_time_interval_combinations_df = train_time_interval_combinations_df[~drop_no_aggr_filter]
    # DEBUG
    #train_time_interval_combinations_df = train_time_interval_combinations_df.loc[0:500]

    test_time_interval_combinations_df =  []
    for cluster_id in combinations_indices_dict['test_clusters']:
        df = time_interval_combinations_df[time_interval_combinations_df['cluster_id']==cluster_id]
        test_time_interval_combinations_df.append(df)
    test_time_interval_combinations_df = pd.concat(test_time_interval_combinations_df, ignore_index=True)
    # для выравнивания баланса классов (баланс смещен в сторону не агрессивного поведения)
    # удалим не агрессивные интервалы физ. поведения, которые не пересекаются с вербальным поведением
    #drop_no_aggr_filter = (test_time_interval_combinations_df['aggr_type']=='phys')&(test_time_interval_combinations_df['phys_aggr_label']=='NOAGGR')
    #test_time_interval_combinations_df = test_time_interval_combinations_df[~drop_no_aggr_filter]
    # DEBUG
    #test_time_interval_combinations_df = test_time_interval_combinations_df.loc[0:500]
    device = torch.device(f'cuda:{gpu_device_idx}')    
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

    # пример аугментация со спектрограммами
    '''
    class MakeRGBSpectrogram(nn.Module):
        def forward(self, x):
            return torch.stack([x, x, x], dim=0)
    train_audio_transform = v2.Compose([
        AppendZeroValues(target_size=[max_audio_len]),
        torchaudio.transforms.Spectrogram(n_fft=512, wkwargs={'device':device}),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=80),
        torchaudio.transforms.TimeMasking(time_mask_param=80),
        MakeRGBSpectrogram()
        ])
    test_audio_transform = v2.Compose([
        AppendZeroValues(target_size=[max_audio_len]),
        torchaudio.transforms.Spectrogram(n_fft=512, wkwargs={'device':device}),
        MakeRGBSpectrogram()
        ])
    '''
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

    train_dataset = MultimodalPhysVerbDataset(
        time_intervals_df=train_time_interval_combinations_df,
        path_to_dataset=path_to_dataset_root,
        modality_augmentation_dict=train_transforms_dict,
        actual_modalities_list=modalities_list,
        device=device,
        text_embedding_type='ru_conversational_cased_L-12_H-768_A-12_pt_v1_tokens'
        )
    test_dataset = MultimodalPhysVerbDataset(
        time_intervals_df=test_time_interval_combinations_df,
        path_to_dataset=path_to_dataset_root,
        modality_augmentation_dict=train_transforms_dict,
        actual_modalities_list=modalities_list,
        device=device,
        text_embedding_type='ru_conversational_cased_L-12_H-768_A-12_pt_v1_tokens'
        )
    
    #print(train_dataset[50][0][0][1].shape)
    #exit()
    
    train_batch_sampler = AggrBatchSampler(train_time_interval_combinations_df, batch_size=batch_size, shuffle=True)
    test_batch_sampler = AggrBatchSampler(test_time_interval_combinations_df, batch_size=batch_size, shuffle=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0
        #pin_memory=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=0
        #pin_memory=True
    )

    '''
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
    '''
    #for data in tqdm(train_dataloader):
    #    pass
    
    #device = torch.device('cpu')

    #torch.manual_seed(None)
    
    #audio_extractor = Wav2vec2Extractor(bundle.get_model())
    #audio_extractor = Wav2vecExtractor(torch.jit.load('wav2vec_feature_extractor_jit.pt'))
   
    #audio_extractor = nn.Sequential(CNN1D(class_num=2),nn.Linear(512, 768),nn.Dropout(0.3))
    audio_extractor = AudioCnn1DExtractorWrapper(hidden_size=768)
    #audio_extractor = torchvision.models.vgg11_bn(weights=torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1)
    #audio_extractor = nn.Sequential(audio_extractor.features,nn.AdaptiveAvgPool2d(1))

    #res = audio_extractor(torch.randn(1, 3, 257, 313))
    #print(res.shape)
    #exit()

    # изменяем размерность входного слоя
    #new_proj = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    #new_proj.weight = torch.nn.Parameter(torch.sum(audio_extractor.features[0][0].weight, dim=1).unsqueeze(1))
    #new_proj.bias = audio_extractor.features[0][0].bias
    #audio_extractor.features[0][0] = new_proj

    audio_model = TransformerSequenceProcessor(
        extractor_model=audio_extractor,
        transformer_layer_num=1,
        transformer_head_num=8,
        hidden_size=768,
        class_num=class_num
        )
    text_model = TransformerSequenceProcessor(
        extractor_model=nn.Sequential(),
        transformer_layer_num=1,
        transformer_head_num=8,
        hidden_size=768,
        class_num=2
    )

    # dummy class
    class E(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = Swin3d_T_extractor(frame_num=video_frames_num, window_size=video_window_size)
        def forward(self, x, ret_type='PIDOR EPTA'):
            return self.e(x)

    video_extractor = Swin3d_T_extractor(frame_num=video_frames_num, window_size=video_window_size)

    video_model = TransformerSequenceProcessor(
        extractor_model=video_extractor,
        transformer_layer_num=1,
        transformer_head_num=8,
        hidden_size=768,
        class_num=class_num
        )
       
    # определяем размерности векторов признаков для многомодальной обработки
    video_features_shape = video_model(torch.zeros([1, 3, video_frames_num, 112, 112])).shape
    audio_features_shape = audio_extractor(torch.zeros([1, max_audio_len])).shape
    text_features_shape = text_model(torch.zeros([1, max_embeddings_len, 768])).shape
    modality_features_shapes_dict = {
        'audio':list(audio_features_shape)[1:],
        'text':list(text_features_shape)[1:],
        'video':list(video_features_shape)[1:]
    }
    modality_features_shapes_dict = {k:v for k,v in modality_features_shapes_dict.items() if k in modalities_list}
    modality_extractors_dict = {
        'audio':audio_extractor,
        'text':nn.Sequential(),
        #'text':text_model,
        #'text':text_model,
        'video':video_extractor
        #'video':video_model
    }
    modality_extractors_dict = {k:v for k,v in modality_extractors_dict.items() if k in modalities_list}
    modality_extractors_dict = nn.ModuleDict(modality_extractors_dict)

    modality_fusion_module = EqualSizedTransformerModalitiesFusion(fusion_transformer_layer_num=1, fusion_transformer_hidden_size=768, fusion_transformer_head_num=8)
    #modality_fusion_module = AveragedFeaturesTransformerFusion(fusion_transformer_layer_num=1, fusion_transformer_hidden_size=768, fusion_transformer_head_num=8)
    '''
    aggr_classifiers_dict = {
        'phys':OutputClassifier(768, 2),
        'verb':OutputClassifier(768, 2),
    }
    aggr_classifiers_dict = {k:v for k,v in aggr_classifiers_dict.items() if k in aggr_types_list}
    aggr_classifiers_dict = nn.ModuleDict(aggr_classifiers_dict)
    '''
    aggr_classifiers = PhysVerbClassifier(
        modalities_list=modalities_list,
        class_num=2,
        input_audio_size=audio_features_shape[-1],
        input_text_size=text_features_shape[-1],
        input_video_size=video_features_shape[-1],
        verb_adaptor_out_dim=768
        )
    model = PhysVerbModel(
        modality_extractors_dict=modality_extractors_dict,
        modality_fusion_module=modality_fusion_module,
        classifiers=aggr_classifiers,
        modality_features_shapes_dict=modality_features_shapes_dict,
        hidden_size=768,
        class_num=2)
    
    #print(model.classifiers.classifiers_dict['phys'][3].weight)
    #exit()
    '''
    modality_classifiers_dict = {
        'audio':OutputClassifier(768, 2),
        'text':OutputClassifier(768, 2),
        'video':OutputClassifier(768, 2)
    }
    modality_classifiers_dict = {k:v for k,v in modality_classifiers_dict.items() if k in modalities_list}
    modality_classifiers_dict = nn.ModuleDict(modality_classifiers_dict)

    model = MultimodalModel(
        modality_extractors_dict=modality_extractors_dict,
        modality_fusion_module=modality_fusion_module,
        classifiers_dict=modality_classifiers_dict,
        modality_features_shapes_dict=modality_features_shapes_dict,
        hidden_size=768,
        class_num=2)
    '''
    model.to(device)
    '''
    for data, labels in tqdm(train_dataloader):
        break
    ret = model(data)
    print(ret)
    exit()
    '''
    optimizer = torch.optim.Adam(model.parameters())

    #for data, labels in train_dataloader:
    #    ret = model(data)
    
    #print(ret)
    #print(labels)
    #exit()

    #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

    #criterion = nn.CrossEntropyLoss()

    
    # вычисляем веса классов для физичекой и вербальной агрессии
    phys_aggr_filter = (train_time_interval_combinations_df['aggr_type'] == 'phys')
    verb_aggr_filter = (train_time_interval_combinations_df['aggr_type'] == 'verb')
    phys_verb_agr_filter = (train_time_interval_combinations_df['aggr_type'] == 'phys&verb')
    phys_aggr_df = train_time_interval_combinations_df[phys_aggr_filter]
    verb_aggr_df = train_time_interval_combinations_df[verb_aggr_filter]
    phys_verb_aggr_df = train_time_interval_combinations_df[phys_verb_agr_filter]
    all_phys_aggr_df = train_time_interval_combinations_df[phys_aggr_filter|phys_verb_agr_filter]
    all_verb_aggr_df = train_time_interval_combinations_df[verb_aggr_filter|phys_verb_agr_filter]

    verb_weights_series = 1-all_verb_aggr_df['verb_aggr_label'].value_counts()/len(all_verb_aggr_df)
    phys_weights_series = 1-all_phys_aggr_df['phys_aggr_label'].value_counts()/len(all_phys_aggr_df)

    verb_weights = torch.zeros([class_num], device=device)
    phys_weights = torch.zeros([class_num], device=device)

    verb_weights[0] = verb_weights_series['NOAGGR']
    verb_weights[1] = verb_weights_series['AGGR']

    phys_weights[0] = phys_weights_series['NOAGGR']
    phys_weights[1] = phys_weights_series['AGGR']
    '''
    print('ALL VERB:')
    print(verb_weights_series['AGGR'])
    print('ALL PHYS:')
    print(phys_weights_series['NOAGGR'])
    '''

    gamma_val = 2

    phys_focal_loss = torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='FocalLoss',
        alpha=phys_weights,
        gamma=gamma_val,
        reduction='mean',
        force_reload=False
    )

    verb_focal_loss = torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='FocalLoss',
        alpha=verb_weights,
        gamma=gamma_val,
        reduction='mean',
        force_reload=False
    )

    weighted_verb_cross_entropy_loss = nn.CrossEntropyLoss(verb_weights)
    weighted_phys_cross_entropy_loss = nn.CrossEntropyLoss(phys_weights)

    verb_cross_entropy_loss = nn.CrossEntropyLoss(verb_weights)
    phys_cross_entropy_loss = nn.CrossEntropyLoss(phys_weights)
    
    '''
    print('PHYS&VERB')
    print(phys_verb_aggr_df['phys_aggr_label'].value_counts())
    print('ONLY PHYS')
    print(phys_aggr_df['phys_aggr_label'].value_counts())
    print()

    phys_aggr_filter = (test_time_interval_combinations_df['aggr_type'] == 'phys')
    verb_aggr_filter = (test_time_interval_combinations_df['aggr_type'] == 'verb')
    phys_verb_agr_filter = (test_time_interval_combinations_df['aggr_type'] == 'phys&verb')
    phys_aggr_df = test_time_interval_combinations_df[phys_aggr_filter]
    verb_aggr_df = test_time_interval_combinations_df[verb_aggr_filter]
    phys_verb_aggr_df = test_time_interval_combinations_df[phys_verb_agr_filter]
    all_phys_aggr_df = test_time_interval_combinations_df[phys_aggr_filter|phys_verb_agr_filter]
    print('ALL PHYS:')
    print(all_phys_aggr_df['phys_aggr_label'].value_counts())
    print('PHYS&VERB')
    print(phys_verb_aggr_df['phys_aggr_label'].value_counts())
    print('ONLY PHYS')
    print(phys_aggr_df['phys_aggr_label'].value_counts())
    print()
    '''
    aggr_types_losses_dict = {
        'phys': phys_focal_loss,
        'verb': verb_focal_loss
    }
    aggr_types_losses_dict = {k: v for k, v in aggr_types_losses_dict.items() if k in aggr_types_list}
    criterion = MultiModalCrossEntropyLoss(modalities_losses_dict=aggr_types_losses_dict)
    
    metrics_dict = {
        'loss': None,
        'accuracy': metrics.accuracy_score,
        'precision': {'metric': metrics.precision_score, 'kwargs': {'average': None}},
        'recall':{'metric': metrics.recall_score, 'kwargs': {'average': None}},
        'f1-score':{'metric': metrics.f1_score, 'kwargs': {'average': None}},
        'UAR': {'metric': metrics.recall_score, 'kwargs': {'average': 'macro'}},
        'UAP': {'metric': metrics.precision_score, 'kwargs': {'average': 'macro'}},
        'UAF1': {'metric': metrics.f1_score, 'kwargs': {'average': 'macro'}}
    }

    metrics_to_dispaly = ['loss', 'accuracy', 'UAR', 'UAP', 'UAF1', 'recall', 'precision', 'f1-score']

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