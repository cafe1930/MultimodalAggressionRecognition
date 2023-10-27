import torchvision
from torchvision.transforms import v2
#import torchaudio
from torchvision import tv_tensors
import torch
from torch import nn
import cv2
import os
import glob
import numpy as np

from tqdm import tqdm

from sklearn import model_selection

class NumpyVideoExtractorDataset(torch.utils.data.Dataset):
    label_dict = {'AGGR': 1, 'NOAGGR': 0}
    def __init__(self, paths_to_data_list, augmentation_transforms, device):
        super().__init__()
        self.paths_to_data_list = paths_to_data_list
        #self.path_to_audios_root = path_to_audios_root
        self.augmentation_transforms = augmentation_transforms

        self.device = device
        #self.files_list = self.index_files()

    #def index_files(self):
    #    return [file_name for file_name in os.listdir(self.path_to_data_files) if file_name.endswith('.npy')]
    
    
    def get_label(self, idx):
        
        #A structure of a file name is xxx_._yyy_._LABEL.npy
        name = os.path.split(self.paths_to_data_list[idx])[-1]
        #return torch.as_tensor(self.label_dict[name.split('_._')[-1].split('.')[0]], dtype=torch.int64, device=self.device)
        return name

    def read_data_file(self, idx):
        #name = self.files_list[idx]
        #path_to_data_file = os.path.join(self.path_to_data_files, name)
        data = torch.as_tensor(np.load(self.paths_to_data_list[idx]), dtype=torch.float32)
        tv_data = tv_tensors.Video(data, device=self.device)
        return tv_data

    def __len__(self):
        return len(self.paths_to_data_list)
    
    def __getitem__(self, idx):
        data = self.read_data_file(idx)
        data = self.augmentation_transforms(data)
        label = self.get_label(idx)
        #return data
        return label, data.permute((1, 0, 2, 3))

class RnnFeaturesDataset(torch.utils.data.Dataset):
    label_dict = {'AGGR': 1, 'NOAGGR': 0}
    def __init__(self, path_to_data_root):
        self.path_to_data_root = path_to_data_root
        self.data_names_list = [n for n in os.listdir(path_to_data_root) if n.endswith('.npy')]
        #self.paths_to_data_list = [os.path.join(path_to_data_root, p) for p in os.listdir(path_to_data_root) if p.endswith('.npy')]

    def get_label(self, idx):
        
        #A structure of a file name is xxx_._yyy_._LABEL.npy
        #name = os.path.split(self.paths_to_data_list[idx])[-1]
        name = self.data_names_list[idx]
        return self.label_dict[name.split('_._')[-1].split('.')[0]]
        #return name

    def read_data_file(self, idx):
        #name = self.files_list[idx]
        #path_to_data_file = os.path.join(self.path_to_data_files, name)
        name = self.data_names_list[idx]
        path_to_data = os.path.join(self.path_to_data_root, name)
        data = torch.as_tensor(np.load(path_to_data), dtype=torch.float32)
        #data = torch.as_tensor(np.load(self.paths_to_data_list[idx]), dtype=torch.float32)
        #tv_data = tv_tensors.Video(data, device=self.device)
        return data
    
    def __len__(self):
        return len(self.data_names_list)
    
    def __getitem__(self, idx):
        data = self.read_data_file(idx)
        label = self.get_label(idx)
        return data, label    


if __name__ == '__main__':
    #torchvision.models.video.R3D_18_Weights.KINETICS400_V1
    model = model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
    model = nn.Sequential(*list(model.children())[:-1])#,nn.Flatten())
    model = model.cuda()
    #out = model(torch.randn(1, 3, 16, 112, 112))

    paths_to_video_npy_list = glob.glob(r'I:\AVABOS\trash_to_train_on_video_numpy\*.npy')
    train_videos_list, test_videos_list = model_selection.train_test_split(paths_to_video_npy_list, test_size=0.2, random_state=1)
    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.3), scale=(0.6, 1.1), shear=10),
        v2.RandomPerspective(distortion_scale=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NumpyVideoExtractorDataset(
        paths_to_data_list=train_videos_list,
        augmentation_transforms=train_transforms,
        device=torch.device('cuda')
    )

    test_dataset = NumpyVideoExtractorDataset(
        paths_to_data_list=test_videos_list,
        augmentation_transforms=test_transforms,
        device=torch.device('cuda')
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

    #for idx in tqdm(range(len(train_dataset))):
    #    label, data = train_dataset[idx]
    for labels, data in tqdm(train_dataloader):
        with torch.inference_mode():
            features_list = []
            for idx in range(0, data.size(2), 16):
                features_list.append(model(data[:,:,idx:idx+16,:,:]))

    print(labels)
    print(len(features_list))
    print(features_list[0].shape)

    


