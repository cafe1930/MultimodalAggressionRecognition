import torchvision
#import torchaudio
from torchvision import tv_tensors
import torch

import cv2
import os
import glob
import numpy as np

class NumpyDataset(torch.utils.data.Dataset):
    label_dict = {'AGGR': 1, 'NOAGGR': 0}
    def __init__(self, path_to_data_files, augmentation_transforms, device, data_type='VID'):
        super().__init__()
        self.path_to_data_files = path_to_data_files
        #self.path_to_audios_root = path_to_audios_root
        self.augmentation_transforms = augmentation_transforms
        self.data_type = data_type

        self.device = device
        self.files_list = self.index_files()

    def index_files(self):
        return [file_name for file_name in os.listdir(self.path_to_data_files) if file_name.endswith('.npy')]
    
    def get_label(self, idx):
        '''
        A structure of a file name is xxx_._yyy_._LABEL.npy
        '''
        name = self.files_list[idx]
        return self.label_dict[name.split('_._')[-1].split('.')[0]]
    
    def read_data_file(self, idx):
        name = self.files_list[idx]
        path_to_data_file = os.path.join(self.path_to_data_files, name)
        data = torch.as_tensor(np.load(path_to_data_file), dtype=torch.float32)
        tv_data = tv_tensors.Video(data, device=self.device)
        return tv_data

    def __len__(self):
        return len(self.files_list)
    
    def __getitem__(self, idx):
        data = self.read_data_file(idx)
        data = self.augmentation_transforms(data)
        label = self.get_label(idx)
        return label, data

if __name__ == '__main__':
    pass
