import torch
from torch import nn
import torchvision

from sklearn.model_selection import train_test_split

from torchvision.models.segmentation import fcn

import albumentations as A
from albumentations.pytorch import ToTensorV2

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

from segmentation_autoencoder import *
from unet import *
from segnet import *

import segmentation_models_pytorch as smp


import warnings
warnings.filterwarnings("ignore")


class SegmentationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)['out']


class ConfusionMatrix:
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, true, pred):
        return metrics.confusion_matrix(true, pred, labels=self.labels)

class ConfusionIoU:
    def __init__(self, labels):
        self.labels = labels
    def __call__(self, confusion_matrix, **kwargs):
        iou = np.zeros(shape=(len(self.labels),))
        for label in self.labels:
            tp =confusion_matrix[label, label]
            #tn = np.sum(np.diag(confusion_matrix)) - tp
            fp = np.sum(confusion_matrix[label]) - tp
            fn = np.sum(confusion_matrix[:, label]) - tp

            if tp + fp + fn == 0:
                iou[label] = 0
            else:
                iou[label] = tp / (tp + fp + fn)
        return iou

class MeanConfuaionIou(ConfusionIoU):
    def __call__(self, confusion_matrix, **kwargs):
        return np.mean(super().__call__(confusion_matrix, **kwargs))


class ConfusionAccuracy:
    def __call__(self, confusion_matrix, **kwargs):
        return np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset',  required=True)
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
        '--batch_size', '16',#
        '--epoch_num', '2000',#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '--nn_input_size', '256', '256',
        '--test_size', '0.5']

    args = parser.parse_args(sample_args)

    path_to_dataset = args.path_to_dataset
    nn_input_size = args.nn_input_size
    resume_training = args.resume_training
    path_to_checkpoint = args.path_to_checkpoint
    epoch_num = int(args.epoch_num)
    test_size = args.test_size
    batch_size = int(args.batch_size)
    

    class_num = 2
    
    if resume_training == True:
        if path_to_checkpoint is None:
            raise ValueError('--path_to_checkpoint flag must be specified if --resume_trining flag')
        

    names_list = sorted([f[:-4] for f in os.listdir(os.path.join(path_to_dataset, 'images')) if f.endswith('.jpg')])[:10]

    train_names, test_names = train_test_split(names_list, test_size=test_size, random_state=0)

    train_transforms = A.Compose(
        [
            A.Resize(*nn_input_size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Affine(scale=(0.7, 1.3), rotate=(-90,90), shear=(-15, 15)),
            #A.GlassBlur(),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2()
        ]
    )

    test_transforms = A.Compose(
        [
            A.Resize(*nn_input_size),
            #A.Rotate(),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2()
        ]
    )
    
    #datasize_size = len(names_list) // 2

    device = torch.device('cuda:0')
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #device = torch.device('cpu')

    crossval_results_df = pd.DataFrame()
    for cv_iter_idx in range(5):
        for fold_idx in range(2):
            if fold_idx % 2 == 0:
                train_dataset = SegmentationDataset(path_to_dataset=path_to_dataset, instance_names_list=train_names, transforms=train_transforms)
                test_dataset = SegmentationDataset(path_to_dataset=path_to_dataset, instance_names_list=test_names, transforms=test_transforms)
            else:
                train_dataset = SegmentationDataset(path_to_dataset=path_to_dataset, instance_names_list=test_names, transforms=train_transforms)
                test_dataset = SegmentationDataset(path_to_dataset=path_to_dataset, instance_names_list=train_names, transforms=test_transforms)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            

            

            #model = fcn.fcn_resnet50(pretrained=True)
            #model.classifier = fcn.FCNHead(in_channels=2048, channels=class_num)
            #model = SegmentationWrapper(model=model)
            ##################################smodel = SegmentationAutoencoder(in_channels=3, class_num=2)
            #model = SegNet(torchvision.models.vgg16_bn(pretrained=True), class_num=2)
            #model = smp.Unet(encoder_name='vgg11', encoder_depth=3, decoder_channels=(128, 64, 32), classes=2)
            #model = smp.Unet(classes=1)
            #model = UNetHalf(in_channels=3, class_num=2)
            #model = UNetHalf(in_channels=3, class_num=2)
            model = SegmentationAutoencoderShallowHalfReduced(in_channels=3, class_num=2)
            model_name = f'segm_ae_shallow_half_{cv_iter_idx}iter_{fold_idx}fold'

            model.to(device)

            optimizer = torch.optim.Adam(model.parameters())

            print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')

            # вычислим веса классов
            #!!!!!!!!
            path_to_csv = os.path.join(path_to_dataset, 'overall_pixel_stats.csv')
            #stat_df = pd.read_csv(path_to_csv).drop(['Название изображения'], axis=1)*np.prod(nn_input_size)
            stat_df = pd.read_csv(path_to_csv).drop(['Название изображения'], axis=1)
            #print(stat_df)
            weights = (stat_df.sum(axis=0)).to_numpy()#[::-1]# столбцы перепутаны!
            #weights = 1-weights/weights.sum()
            weights = weights/weights.sum()
            weights = torch.FloatTensor(weights).to(device)
            #print(stat_df)
            #print(weights)
            '''
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                y_ = model(x)
                #print(y.shape, y_.shape)
                print(nn.BCEWithLogitsLoss(weights[1])(y_, y))

            exit()
            '''
            criterion = nn.CrossEntropyLoss(weight=weights)
            #criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
            #smp.losses.FocalLoss(mode='multiclass')

            labels = np.arange(0, class_num)
            metrics_dict = {
                'loss': None,
                'confusion_matrix': ConfusionMatrix(labels=labels),
                'accuracy': ConfusionAccuracy(),
                'class IoU': ConfusionIoU(labels=labels),
                'mean IoU': MeanConfuaionIou(labels=labels),
            }

            metrics_to_dispaly = ['loss', 'accuracy', 'mean IoU', 'class IoU']

            segmentation_trainer = SegmentationTrainer(
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

            #print(segmentation_trainer.testing_log_df)
            #exit()

            if resume_training:
                segmentation_trainer = torch.load(path_to_checkpoint)
                segmentation_trainer.train_loader.dataset.path_to_dataset = path_to_dataset
                segmentation_trainer.train_loader.dataset.path_to_dataset = path_to_dataset

            segmentation_trainer.train(epoch_num)



            epoch_idx = segmentation_trainer.testing_log_df['mean IoU'].argmax()
            best_mean_iou = segmentation_trainer.testing_log_df['mean IoU'].max()
            best_class_iou = segmentation_trainer.testing_log_df['class IoU']


            print('Best IoU for {} is {} on {} epoch'.format(model_name, best_mean_iou, epoch_idx))
            results_dict = {'cv_iter_idx': cv_iter_idx, 'fold_idx': fold_idx, 'best mean IoU': best_mean_iou, 'best class IoU': best_class_iou}

            crossval_results_df = pd.concat([crossval_results_df, pd.DataFrame([results_dict])], ignore_index=True)

    print(crossval_results_df)
    crossval_results_df.to_csv(f'saving_dir/{model_name}-CV.csv', index=False)


