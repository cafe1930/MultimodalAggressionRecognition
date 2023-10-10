import torchvision
import torchaudio
import torch

import cv2
import os
import glob

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_videos_root, path_to_audios_root, augmentation_transforms):
        self.path_to_videos_root = path_to_videos_root
        self.path_to_audios_root = path_to_audios_root
        self.augmentation_transforms = augmentation_transforms

        self.files_list = []

    def read_video(self):
        

    def get_files_list(self):
        pass

    def __len__(self):
        return len(self.files_list)
    
    def __getitem__(self, idx):
        audio_data = None
        video_data = None
        label = None
        return label, audio_data, video_data

if __name__ == '__main__':
    pass
