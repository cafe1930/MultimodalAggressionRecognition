from torchvision.models.video import r3d_18
import torchvision
import torch
from torch import nn

class PermuteModule(nn.Module):
    def forward(self, x):
        return x.permute(0, 4, 1, 2, 3)

class ExtractorBase(nn.Module):
    def __init__(self, frame_num, window_size):
        self.frame_num = frame_num
        self.window_size = window_size

    def forward(self, x):
        features_list = []
        for idx, i in enumerate(range(0, self.frame_num, self.window_size)):
            features = self.extractor(x[:,:,i:i+self.window_size,:,:])
            features_list.append(features)

        return torch.stack(features_list).permute((1, 0, 2)).detach().cpu().numpy()


class R3D_extractor(ExtractorBase):
    def __init__(self, frame_num, window_size):
        super().__init__(frame_num, window_size)
        #!!!!!
        model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
        self.extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

        # замораживаем веса (freeze weights), итерируя по параметрам нейронной сети
        # за это отвечает параметр requires_grad
        for p in self.extractor.parameters():
            p.requires_grad = False
        

class Swin3d_extractor(R3D_extractor):
    def __init__(self, frame_num, window_size):
        super().__init__(frame_num, window_size)
        model = torchvision.models.video.swin3d_t(weight=torchvision.models.video.Swin3D_T_Weights)
        self.extractor = nn.Sequential(*list(model.children())[:-2], PermuteModule(), nn.AdaptiveAvgPool3d(1), nn.Flatten())

        # замораживаем веса (freeze weights), итерируя по параметрам нейронной сети
        # за это отвечает параметр requires_grad
        for p in self.extractor.parameters():
            p.requires_grad = False    