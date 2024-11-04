import torchvision
import torchaudio
from torchvision.transforms import v2
#import torchaudio
from torchvision import tv_tensors
import torch
from torch import nn
import cv2
import os
import glob
import numpy as np

from models import R3DWithBboxes, R3D

from prepare_numpy_data import read_video_frames_opencv

from tqdm import tqdm

from sklearn import model_selection

class RandomAffineVideoBboxes(v2.RandomAffine):
    def forward(self, video_tensor, bboxes_tensor):
        channels ,frames_num, rows, cols = video_tensor.shape
                
        params = self._get_params([video_tensor])

        video_tensor = v2.functional.affine(video_tensor, center=self.center, **params)
        for frame_idx in range(0, bboxes_tensor.size(0)):
            bbox = bboxes_tensor[frame_idx,0]
            #print(bbox.sum())
            if bbox.sum()>0:
                tv_bbox = tv_tensors.BoundingBoxes(bbox, format='XYXY', canvas_size=(rows, cols))
                #print(tv_bbox)
                tv_bbox = v2.functional.affine(tv_bbox, center=self.center, **params)
                #print(tv_bbox)
                #print()
                bbox = tv_bbox.data
            bboxes_tensor[frame_idx, 0] = bbox

        return video_tensor, bboxes_tensor
    
class RandomPerspectiveVideoBboxes(v2.RandomPerspective):
    def forward(self, video_tensor, bboxes_tensor):
        channels ,frames_num, rows, cols = video_tensor.shape
                
        params = self._get_params([video_tensor])

        video_tensor = v2.functional.perspective(video_tensor, startpoints=None, endpoints=None, **params)
        for frame_idx in range(0, bboxes_tensor.size(0)):
            bbox = bboxes_tensor[frame_idx,0]
            #print(bbox.sum())
            if bbox.sum()>0:
                tv_bbox = tv_tensors.BoundingBoxes(bbox, format='XYXY', canvas_size=(rows, cols))
                #print(tv_bbox)
                tv_bbox = v2.functional.perspective(tv_bbox, startpoints=None, endpoints=None, **params)
                #print(tv_bbox)
                #print()
                bbox = tv_bbox.data
            bboxes_tensor[frame_idx, 0] = bbox

        return video_tensor, bboxes_tensor
    
class RandomHorizontalFlipVideoBboxes(v2.RandomHorizontalFlip):
    def forward(self, video_tensor, bboxes_tensor):
        if torch.rand(1) >= self.p:
            return video_tensor, bboxes_tensor
        channels ,frames_num, rows, cols = video_tensor.shape
                
        video_tensor = v2.functional.horizontal_flip(video_tensor)
        for frame_idx in range(0, bboxes_tensor.size(0)):
            bbox = bboxes_tensor[frame_idx,0]
            #print(bbox.sum())
            if bbox.sum()>0:
                tv_bbox = tv_tensors.BoundingBoxes(bbox, format='XYXY', canvas_size=(rows, cols))
                #print(tv_bbox)
                tv_bbox = v2.functional.horizontal_flip(tv_bbox)
                #print(tv_bbox)
                #print()
                bbox = tv_bbox.data
            bboxes_tensor[frame_idx, 0] = bbox

        return video_tensor, bboxes_tensor
    
class CreateBboxesMasks(nn.Module):
    def forward(self, video_tensor, bboxes):
        #print(bboxes.shape)
        channels, frames, rows, cols = video_tensor.shape
        canvas = np.zeros(shape=(frames, rows, cols), dtype=np.uint8)
        for frame_idx in range(min(bboxes.size(0), frames)):
            bbox = bboxes[frame_idx,0]
            #print(bbox.sum())
            if bbox.sum().item()>0:
                for b in bbox:
                    #frame = cv2.rectangle(canvas[frame_idx], pt1=tuple(b[:2]), pt2=tuple(b[2:]), color=(255,), thickness=-1)
                    pt1 = b[:2].tolist()
                    pt2 = b[2:].tolist()
                    try:
                        frame = cv2.rectangle(canvas[frame_idx], pt1, pt2, (255,), -1)
                    except:
                        print(f'bboxes {bboxes.shape}, canvas {canvas.shape}')
                canvas[frame_idx] = frame
            
        bbox_masks = torch.as_tensor(canvas, dtype=torch.float32).unsqueeze(0)
        return video_tensor, bbox_masks
    
class NormalizeBboxes(v2.Normalize):
    def forward(self, video_tensor, bboxes_tensor):
        #print(video_tensor.shape)
        video_tensor = super().forward(video_tensor.permute(1, 0, 2, 3))
        #return video_tensor.permute(1, 0, 2, 3), bboxes_tensor
        return video_tensor, bboxes_tensor

class ResizeBboxes(v2.Resize):
    def forward(self, video_tensor, bboxes_tensor):
        frames1, channels1, rows1, cols1 = video_tensor.shape
        frames1, channels1, rows2, cols2 = bboxes_tensor.shape
        video_tensor = super().forward(video_tensor)
        if rows1 == rows2 and cols1 == cols2:
            bboxes_tensor = super().forward(bboxes_tensor)
        else:
            for frame_idx in range(0, bboxes_tensor.size(0)):
                bbox = bboxes_tensor[frame_idx,0]
                #print(bbox.sum())
                if bbox.sum()>0:
                    tv_bbox = tv_tensors.BoundingBoxes(bbox, format='XYXY', canvas_size=(rows1, cols1))
                    #print(tv_bbox)
                    tv_bbox = super().forward(tv_bbox)
                    #print(tv_bbox)
                    #print()
                    bbox = tv_bbox.data
                bboxes_tensor[frame_idx, 0] = bbox
        return video_tensor, bboxes_tensor

class NumpyVideoExtractorDataset(torch.utils.data.Dataset):
    label_dict = {'AGGR': 1, 'NOAGGR': 0}
    def __init__(self, paths_to_data_list, augmentation_transforms, device):
        super().__init__()
        self.paths_to_data_list = paths_to_data_list
        #self.path_to_audios_root = path_to_audios_root
        self.augmentation_transforms = augmentation_transforms

        self.device = device
        #self.files_list = self.index_files()
    
    def get_label(self, idx):  
        #A structure of a file name is xxx_._yyy_._LABEL.npy
        name = os.path.split(self.paths_to_data_list[idx])[-1]
        return torch.as_tensor(self.label_dict[name.split('_._')[-1].split('.')[0]], dtype=torch.int64, device=self.device)
        #return name

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
        return data.permute((1, 0, 2, 3)), label
    
class PtVideoDataset(NumpyVideoExtractorDataset):
    def read_data_file(self, idx):
        data = torch.load(self.paths_to_data_list[idx])
        tv_data = tv_tensors.Video(data, device=self.device)
        return tv_data
    
    def get_label(self, idx):  
        #A structure of a file name is u_v_x_y_z_LABEL.npy
        name = os.path.split(self.paths_to_data_list[idx])[-1]
        name = '.'.join(name.split('.')[:-1])
        split_name = name.split('_')
        label = self.label_dict[split_name[-1]]
        return torch.as_tensor(label, dtype=torch.int64, device=self.device)


class AppendVideoZeroFrames(nn.Module):
    def __init__(self, target_frame_num:int):
        '''
        Append a number of zero-valued frames to a processing video
        A number of appending frames is defined as a difference between traget_frame_num and frame_num of imput video
        '''
        super().__init__()
        self.target_frame_num = target_frame_num

    def forward(self, video:torch.tensor):
        frames, channels, rows, cols = video.shape
        video_dtype = video.dtype
        appending_frame_num = self.target_frame_num - frames
        if appending_frame_num <= 0:
            return video[:self.target_frame_num]
        #if appending_frame_num >= self.target_frame_num:
        #    return video[:self.target_frame_num]
        return torch.cat([video, torch.zeros((appending_frame_num, channels, rows, cols), dtype=video_dtype)])
    

class AppendZeroValues(nn.Module):
    def __init__(self, target_size:torch.Size):
        '''
        Append a number of zero-valued features to a processing tensor
        A number of appending features is defined as a difference between target_size and size of imput tensor
        '''
        super().__init__()
        self.target_size = target_size


    def forward(self, input_tensor:torch.tensor):
        size_tensor = torch.tensor(input_tensor.shape)
        target_size_tensor = torch.tensor(self.target_size)
        dtype = input_tensor.dtype
        device = input_tensor.device

        diff = target_size_tensor - size_tensor
        
        features_to_append_num = diff[0]
        if features_to_append_num<=0:
            return input_tensor[:self.target_size[0]]

        appending_size = target_size_tensor
        appending_size[0] = features_to_append_num
        
        #if appending_frame_num >= self.target_frame_num:
        #    return video[:self.target_frame_num]
        #print(appending_size)
        #print(input_tensor.shape, torch.zeros(appending_size, dtype=dtype).shape)
        return torch.cat([input_tensor, torch.zeros(appending_size.tolist(), dtype=dtype, device=device)])


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

class AudioDatasetWav(torch.utils.data.Dataset):
    label_dict = {'AGGR': 1, 'NOAGGR': 0}
    def __init__(self, path_to_data_root, target_sample_rate, target_time_length, device):
        super().__init__()
        self.path_to_data_root = path_to_data_root
        self.target_sample_rate = target_sample_rate
        self.target_time_length = target_time_length
        self.device = device
        self.data_names_list = [n for n in os.listdir(path_to_data_root) if n.endswith('.wav')]

    def get_label(self, idx):
        #A structure of a file name is xxx_._yyy_._LABEL.npy
        #name = os.path.split(self.paths_to_data_list[idx])[-1]
        name = self.data_names_list[idx]
        return self.label_dict[name.split('_._')[-1].split('.')[0]]

    def read_data_file(self, idx):
        #name = self.files_list[idx]
        #path_to_data_file = os.path.join(self.path_to_data_files, name)
        name = self.data_names_list[idx]
        path_to_data = os.path.join(self.path_to_data_root, name)
        #data = torch.as_tensor(np.load(path_to_data), dtype=torch.float32)
        waveform, sample_rate = torchaudio.load(path_to_data)
        #waveform = waveform[0].to('cuda')
        if sample_rate != self.target_sample_rate:
            data = torchaudio.functional.resample(waveform[0].to(self.device), sample_rate, self.target_sample_rate)
        #data = torch.as_tensor(np.load(self.paths_to_data_list[idx]), dtype=torch.float32)
        #tv_data = tv_tensors.Video(data, device=self.device)
        if len(data)/self.target_time_length != self.target_sample_rate:
            target_samples = self.target_sample_rate*self.target_time_length - len(data)
            data = torch.cat([data, torch.zeros((target_samples,)).to(self.device)])
            
        return data#[0]
    
    def __len__(self):
        return len(self.data_names_list)
    
    def __getitem__(self, idx):
        data = self.read_data_file(idx)
        label = self.get_label(idx)
        return data, label
    
class PtAudioDataset(torch.utils.data.Dataset):
    label_dict = {'AGGR': 1, 'NOAGGR': 0}
    def __init__(self, paths_to_data_list, augmentation_transforms, device):
        super().__init__()
        self.paths_to_data_list = paths_to_data_list
        #self.path_to_audios_root = path_to_audios_root
        self.augmentation_transforms = augmentation_transforms

        self.device = device

    def read_data_file(self, idx):
        audio_data = torch.load(self.paths_to_data_list[idx]).to(self.device)
        return audio_data
    
    def get_label(self, idx):  
        #A structure of a file name is u_v_x_y_z_LABEL.npy
        name = os.path.split(self.paths_to_data_list[idx])[-1]
        name = '.'.join(name.split('.')[:-1])
        split_name = name.split('_')
        label = self.label_dict[split_name[-1]]
        return torch.as_tensor(label, dtype=torch.int64, device=self.device)
    
    def __len__(self):
        return len(self.paths_to_data_list)
    
    def __getitem__(self, idx):
        data = self.read_data_file(idx)
        data = self.augmentation_transforms(data)
        label = self.get_label(idx)
        #return data
        return data, label
    
class PtTextDataset(PtAudioDataset):
    def read_data_file(self, idx):
        audio_data = torch.as_tensor(np.load(self.paths_to_data_list[idx]), dtype=torch.float32, device=self.device)
        return audio_data
    
    
    

class NumpyVideoBboxesDataset2Classes(NumpyVideoExtractorDataset):
    label_dict = {'Нет':0, 'Захваты':1, 'Толчки': 1, 'Удары': 1}
    def get_label(self, idx):
        #A structure of a file name is xxx_._yyy!person,X!(t0, t1)!LABEL
        name = os.path.split(self.paths_to_data_list[idx])[-1]
        name = '.'.join(name.split('.')[:-1])
        label = name.split('!')[-1]
        label = self.label_dict[label]
        #return torch.as_tensor(self.label_dict[name.split('_._')[-1].split('.')[0]], dtype=torch.int64, device=self.device)
        return label
    
    def read_data_file(self, idx):
        #name = self.files_list[idx]
        #path_to_data_file = os.path.join(self.path_to_data_files, name)
        data = torch.as_tensor(np.load(self.paths_to_data_list[idx]), dtype=torch.float32)
        tv_data = tv_tensors.Video(data, device=self.device)
        return tv_data
    
class VideoBboxesDataset(torch.utils.data.Dataset):
    label_dict = {'Нет':0, 'Захваты':1, 'Толчки': 2, 'Удары': 3}
    def __init__(self, paths_to_data_list, augmentation_transforms, frame_num, device):
        super().__init__()
        self.paths_to_data_list = paths_to_data_list
        #self.path_to_audios_root = path_to_audios_root
        self.augmentation_transforms = augmentation_transforms
        self.device = device
        self.frame_num = frame_num
        #self.files_list = self.index_files()
    
    def get_label(self, idx):
        #A structure of a file name is xxx_._yyy!person,X!(t0, t1)!LABEL.npy
        name = os.path.split(self.paths_to_data_list[idx])[-1]
        #name = '.'.join(name.split('.')[:-1])
        label = name.split('!')[-1]
        label = self.label_dict[label]
        return label

    def read_data_file(self, idx):
        #name = self.files_list[idx]
        path_to_video_file = os.path.join(self.paths_to_data_list[idx], 'video.mp4')
        path_to_bboxes = os.path.join(self.paths_to_data_list[idx], 'bboxes.npy')

        bboxes = torch.as_tensor(np.load(path_to_bboxes))
        bboxes = bboxes[0:self.frame_num]

        data = read_video_frames_opencv(path_to_video_file, 0, self.frame_num)
        if len(data) < self.frame_num:
            frames, rows, cols, channels = data.shape
            canvas = np.zeros((self.frame_num, rows, cols, channels), dtype=np.uint8)
            canvas[:frames] = data
            data = canvas
        data = torch.as_tensor(data, dtype=torch.float32)
        tv_data = tv_tensors.Video(data, device=self.device)
        tv_data = tv_data.permute(3, 0, 1, 2)
        return tv_data, bboxes.to(self.device)

    def __len__(self):
        return len(self.paths_to_data_list)
    
    def __getitem__(self, idx):
        data, bboxes = self.read_data_file(idx)
        data, bboxes = self.augmentation_transforms(data, bboxes)
        label = self.get_label(idx)
        #return data
        return (data.permute((1, 0, 2, 3)), bboxes), label

class VideoDataset(VideoBboxesDataset):
    def read_data_file(self, idx):
        #name = self.files_list[idx]
        path_to_video_file = os.path.join(self.paths_to_data_list[idx], 'video.mp4')
        
        data = read_video_frames_opencv(path_to_video_file, 0, self.frame_num)
        if len(data) < self.frame_num:
            frames, rows, cols, channels = data.shape
            canvas = np.zeros((self.frame_num, rows, cols, channels), dtype=np.uint8)
            canvas[:frames] = data
            data = canvas
        data = torch.as_tensor(data, dtype=torch.float32)
        tv_data = tv_tensors.Video(data, device=self.device)
        tv_data = tv_data.permute(0, 3, 1, 2)
        #print(tv_data.shape)
        return tv_data
    
    def __getitem__(self, idx):
        data = self.read_data_file(idx)
        data = self.augmentation_transforms(data)
        label = self.get_label(idx)
        #return data
        return data.permute((1, 0, 2, 3)), label

    

if __name__ == '__main__':

    paths_to_train_dirs_list = glob.glob(r'I:\AVABOS\DATASET_VERSION_1.0\PHYS\train\*')
    paths_to_test_dirs_list = glob.glob(r'I:\AVABOS\DATASET_VERSION_1.0\PHYS\test\*')

    train_bboxes_transform = v2.Compose([
                        ResizeBboxes((112, 112)),
                        RandomPerspectiveVideoBboxes(distortion_scale=0.2),
                        RandomAffineVideoBboxes(degrees = 4, translate = (0.2, 0.2), scale=(0.8, 1.2), shear=(-5, 5, -5, 5)),
                        RandomHorizontalFlipVideoBboxes(p=0.5),
                        CreateBboxesMasks(),
                        NormalizeBboxes(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                        )

    test_bboxes_transform = v2.Compose([
                        NormalizeBboxes(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ResizeBboxes((112, 112))]
                        )
    
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.3), scale=(0.6, 1.1), shear=10),
        v2.RandomPerspective(distortion_scale=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_bboxes_dataset = VideoBboxesDataset(paths_to_train_dirs_list, train_bboxes_transform, 32, 'cuda')
    test_bboxes_dataset = VideoBboxesDataset(paths_to_test_dirs_list, test_bboxes_transform, 32, 'cuda')

    train_dataset = VideoDataset(paths_to_train_dirs_list, train_transform, 32, 'cuda')
    test_dataset = VideoDataset(paths_to_test_dirs_list, test_transform, 32, 'cuda')

    train_bboxes_dataloader = torch.utils.data.DataLoader(
        train_bboxes_dataset,
        batch_size=8,
        shuffle=True, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
    )
    test_bboxes_dataloader = torch.utils.data.DataLoader(
        test_bboxes_dataset,
        batch_size=8,
        shuffle=False, # Меняем на каждой эпохе порядок следования файлов
        num_workers=0
        #pin_memory=True
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
    cnn = R3DWithBboxes(4).cuda()
    for data, labels in tqdm(train_bboxes_dataloader):
        data = [d.cuda() for d in data]
        res = cnn(data)
        break
        
    
    print(res)

    exit()


    #torchvision.models.video.R3D_18_Weights.KINETICS400_V1
    model = model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
    model = nn.Sequential(*list(model.children())[:-1])#,nn.Flatten())
    model = model.cuda()
    #out = model(torch.randn(1, 3, 16, 112, 112))

    paths_to_video_npy_list = glob.glob(r'I:\AVABOS\DATASET_VERSION_1.0\phys\train\*')

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

    


