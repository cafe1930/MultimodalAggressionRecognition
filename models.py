from torchvision.models.video import r3d_18

import torch
from torch import nn

class R3D_avg(nn.Module):
    def __init__(self, frame_num, window_size, class_num):
        super.__init__()
        #!!!!!
        net = r3d_18(),
        net = nn.Sequential(*net[:-1])

        # замораживаем веса (freeze weights), итерируя по параметрам нейронной сети
        # за это отвечает параметр requires_grad
        for p in net.parameters():
            p.requires_grad = False

        self.extractor = nn.Sequential(
            net,
            nn.Linear(self.extractor_feature_num, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
            )

        self.sequence_processing = nn.AdaptiveAvgPool1d(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2)
            )
        
        self.frame_num = frame_num
        self.window_size = window_size
    
    def forward(self, x):
        features = torch.zeros((x.size(0), 256, self.frame_num//self.window_size))
        for idx, i in enumerate(range(0, self.frame_num, self.window_size)):
            #!!!!!!!
            features[..., idx] = self.extractor(x[:,i:i+self.window_size])

        features = self.sequence_processing(features)
        return self.classifier(features)




    