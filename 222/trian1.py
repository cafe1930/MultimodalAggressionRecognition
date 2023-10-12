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


import random
import os
import numpy as np

import argparse

from datasets import SegmentationDataset
from trainer import SegmentationTrainer


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
    #path_to_dataset = r'H:\saratov_2020\dataset_512_256'
    #path_to_dataset = r'C:\Users\mokhail\python_programming\DATA\dataset_512_256'
    path_to_dataset = '/home/mikhail_u/dataset_256_256'
    nn_input_size = (256, 256)

    class_num = 6

    resume_training = True
    if resume_training == True:
        path_to_checkpoint = 'saving_dir/15.04.2022, 01-10-30 (fcn_resnet50_pretrained_512pix)/fcn_resnet50_pretrained_512pix_current_ep-2.pt'


    names_list = sorted([f[:-4] for f in os.listdir(os.path.join(path_to_dataset, 'images')) if f.endswith('.jpg')])#[:50]
    #random.seed(0)
    #random.shuffle(names_list)

    train_names, test_names = train_test_split(names_list, test_size=0.3, random_state=0)

    train_transforms = A.Compose(
        [
            A.Resize(*nn_input_size),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Affine(scale=(0.7, 1.3), rotate=(-90,90), shear=(-10, 10)),
            A.GlassBlur(),
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

    batch_size=32

    train_dataset = SegmentationDataset(path_to_dataset=path_to_dataset, instance_names_list=train_names, transforms=train_transforms)
    test_dataset = SegmentationDataset(path_to_dataset=path_to_dataset, instance_names_list=test_names, transforms=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda:0')

    model = fcn.fcn_resnet50(pretrained=True)
    model.classifier = fcn.FCNHead(in_channels=2048, channels=class_num)
    model = SegmentationWrapper(model=model)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    # вычислим веса классов
    #!!!!!!!!
    path_to_csv = os.path.join(path_to_dataset, 'overall_pixel_stats.csv')
    stat_df = pd.read_csv(path_to_csv).drop(['Название изображения'], axis=1)*np.prod(nn_input_size)
    weights = (stat_df.sum(axis=0)).to_numpy()
    #weights = 1-weights/weights.sum()
    weights = weights/weights.sum()
    weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    labels = np.arange(0, class_num)
    metrics_dict = {
        'loss': None,
        'confusion_matrix': ConfusionMatrix(labels=labels),
        'accuracy': ConfusionAccuracy(),
        'class IoU': ConfusionIoU(labels=labels),
        'mean IoU': MeanConfuaionIou(labels=labels),
    }

    metrics_to_dispaly = ['loss', 'accuracy', 'mean IoU']

    segmentation_trainer = SegmentationTrainer(
        model=model,
        model_name='fcn_resnet50_pretrained_512pix',
        train_loader=train_loader,
        test_loader=test_loader,
        metrics_dict=metrics_dict,
        metrics_to_display=['loss', 'accuracy', 'mean IoU'],
        criterion=criterion,
        optimizers_list=[optimizer],
        device=device)

    if resume_training:
        segmentation_trainer = torch.load(path_to_checkpoint)
        segmentation_trainer.train_loader.dataset.path_to_dataset = path_to_dataset
        segmentation_trainer.train_loader.dataset.path_to_dataset = path_to_dataset

    segmentation_trainer.train(100)
