from torchvision.models.video import r3d_18
import torchvision
import torch
from torch import nn

class PermuteModule(nn.Module):
    def forward(self, x):
        return x.permute(0, 4, 1, 2, 3)

class ExtractorBase(nn.Module):
    def __init__(self, frame_num, window_size):
        super().__init__()
        self.frame_num = frame_num
        self.window_size = window_size
        self.extractor = self.prepare_model()

    def prepare_model(self):
        raise NotImplementedError(f'Method \'prepare_model\' in {self.__class__.__name__} should be implemented')

    def forward(self, x):
        features_list = []
        for idx, i in enumerate(range(0, self.frame_num, self.window_size)):
            features = self.extractor(x[:,:,i:i+self.window_size,:,:])
            features_list.append(features)

        return torch.stack(features_list).permute((1, 0, 2)).detach().cpu().numpy()
        #return torch.stack(features_list).detach().cpu()


class R3D_extractor(ExtractorBase):
    def prepare_model(self):
        model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

        # замораживаем веса (freeze weights), итерируя по параметрам нейронной сети
        # за это отвечает параметр requires_grad
        for p in model.parameters():
            p.requires_grad = False

        return model
        

class Swin3d_T_extractor(ExtractorBase):
    def prepare_model(self):
        model = torchvision.models.video.swin3d_t(weights=torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1)
        model = nn.Sequential(*list(model.children())[:-2], PermuteModule(), nn.AdaptiveAvgPool3d(1), nn.Flatten())

        # замораживаем веса (freeze weights), итерируя по параметрам нейронной сети
        # за это отвечает параметр requires_grad
        for p in model.parameters():
            p.requires_grad = False

        return model

class S3D_extractor(ExtractorBase):
    def prepare_model(self):
        model = torchvision.models.video.s3d(weights=torchvision.models.video.S3D_Weights.KINETICS400_V1)
        model = nn.Sequential(*list(model.children())[:-2], nn.AdaptiveAvgPool3d(1), nn.Flatten())

        # замораживаем веса (freeze weights), итерируя по параметрам нейронной сети
        # за это отвечает параметр requires_grad
        for p in model.parameters():
            p.requires_grad = False

        return model
    
class AudioExtractor(nn.Module):
    def __init__(self, frame_num, window_size):
        super().__init__()
        self.frame_num = frame_num
        self.window_size = window_size
        self.extractor = self.prepare_model()

    def prepare_model(self):
        raise NotImplementedError(f'Method \'prepare_model\' in {self.__class__.__name__} should be implemented')

    def forward(self, x):
        features_list = []
        for idx, i in enumerate(range(0, self.frame_num, self.window_size)):
            features = self.extractor(x[:,:,i:i+self.window_size,:,:])
            features_list.append(features)

        return torch.stack(features_list).permute((1, 0, 2)).detach().cpu().numpy()

class RNN(nn.Module):
    def __init__(self, rnn_type, rnn_layers_num, input_dim, hidden_dim, class_num):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        #self.rnn = nn.Sequential(
        #    rnn_type(input_dim, hidden_dim, batch_first=True),
        #    *[rnn_type(hidden_dim, hidden_dim, batch_first=True) for i in range(rnn_layers_num-1)]
        #    )
        
        self.rnn = nn.ModuleList(
            [rnn_type(input_dim, hidden_dim, batch_first=True)]+[rnn_type(hidden_dim, hidden_dim, batch_first=True) for i in range(rnn_layers_num-1)])

        # The linear layer that maps from hidden state space to tag space
        self.output_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, class_num)
        )

    def forward(self, sequence):
        #embeds = self.word_embeddings(sentence)
        #lstm_out, _ = self.rnn()
        for rnn in self.rnn:
            sequence, _ = rnn(sequence)
    
        tag_space = self.output_classifier(sequence[:,-1,:])
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space#, _

class VideoAverageFeatures(nn.Module):
    def __init__(self, input_dim, class_num):
        super().__init__()
        self.output_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, class_num)
        )

    def forward(self, x):
        return self.output_classifier(x.mean(dim=1))
    


if __name__ == '__main__':
    size = (10, 19, 1024)
    rnn = RNN(
        rnn_type=nn.GRU,
        rnn_layers_num=1,
        input_dim=1024,
        hidden_dim=512,
        class_num=2
    ).cuda()
    
    out = rnn(torch.randn(*size).cuda())
    print(out.shape)
