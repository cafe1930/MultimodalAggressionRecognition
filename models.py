from torchvision.models.video import r3d_18
import torchvision
import torch
from torch import nn

class R3D_extractor(nn.Module):
    def __init__(self, frame_num, window_size):
        super().__init__()
        #!!!!!
        model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
        self.extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

        # замораживаем веса (freeze weights), итерируя по параметрам нейронной сети
        # за это отвечает параметр requires_grad
        for p in self.extractor.parameters():
            p.requires_grad = False
        
        self.frame_num = frame_num
        self.window_size = window_size
    
    def forward(self, x):
        features_list = []
        for idx, i in enumerate(range(0, self.frame_num, self.window_size)):
            features = self.extractor(x[:,:,i:i+self.window_size,:,:])
            features_list.append(features)

        return torch.stack(features_list).permute((1, 0, 2)).detach().cpu().numpy()




    