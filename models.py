from torchvision.models.video import r3d_18
import torchvision
import torch
from torch import nn

class PermuteModule(nn.Module):
    def forward(self, x):
        return x.permute(0, 4, 1, 2, 3)

class ExtractorBase(nn.Module):
    def __init__(self, frame_num, window_size):
        '''
        frame_num - общее количество кадров в видео
        window_size - размер обрабатываемого окна в кадрах
        '''
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

        return torch.stack(features_list).permute((1, 0, 2))#.detach().cpu().numpy()
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

        return torch.stack(features_list).permute((1, 0, 2))#.detach().cpu().numpy()

class AverageFeatureSequence(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        return x.mean(dim=1).unsqueeze(1), None # WTF?
    
class SequenceAverageFeatures(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        return x.mean(dim=1)#.unsqueeze(1)

class FeatureSequenceProcessing(nn.Module):
    def __init__(self, sequence_nn_dict, class_num):
        super().__init__()
        self.sequence_nn = sequence_nn_dict['model'](**sequence_nn_dict['kwargs'])
        self.hidden_size = sequence_nn_dict['kwargs']['hidden_size']

        # The linear layer that maps from hidden state space to tag space
        self.output_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, class_num)
        )

    def forward(self, sequence):
        sequence, _ = self.sequence_nn(sequence)
        preds = self.output_classifier(sequence[:,-1,:])
        return preds

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
    
class EmbeddingLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )
    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)
        output = self.embedding(x.view(batch_size*sequence_length, -1))
        return output.view(batch_size, sequence_length, -1)
    
class VideoMultiNN(nn.Module):
    def __init__(self, models_dict):
        super().__init__()
        #self.extractor_dict = nn.ModuleDict(extractor_dict)
        #self.extractor_dict.eval()

        #for name, model in rnn_dict.items():
        #    rnn_dict[name] = model['model'](**model['kwargs'])

        self.models_dict = nn.ModuleDict(models_dict)

        #self.embedding = EmbeddingLayer(imput_size, rnn_size)

        
    def get_models_names(self):
        return [name for name in self.models_dict.keys()]

    def forward(self, x):
        #return features
        output_dict = {}
        for name, model in self.models_dict.items():

            output_dict[name] = model(x) 
        return output_dict

class Wav2vecExtractor(nn.Module):
    def __init__(self, wav2vec_model):
        super().__init__()
        self.wav2vec = wav2vec_model
    
    def forward(self, x):
        features = self.wav2vec(x).permute(0, 2, 1)
        #print(features.shape)
        #print()
        #exit()

        return features

class Wav2vec2Extractor(Wav2vecExtractor):
    def forward(self, x):
        with torch.no_grad():
            features = self.wav2vec.extract_features(x)[0][0]
        #print(features.shape)
        #exit()

        return features 
    
#class Wav2Vec1Extractor(nn.Module):

    
class AudioMultiNN(nn.Module):
    def __init__(self, models_dict, extractor_dict):
        super().__init__()
        self.extractor_dict = nn.ModuleDict(extractor_dict)
        
        self.extractor_dict.eval()

        #for name, model in rnn_dict.items():
        #    rnn_dict[name] = model['model'](**model['kwargs'])

        self.models_dict = nn.ModuleDict(models_dict)

    def get_models_names(self):
        return [extractor_name for extractor_name in self.extractor_dict.keys()], [name for name in self.models_dict.keys()]

    def forward(self, x):
        with torch.no_grad():
            for name, model in self.extractor_dict.items():
                features = model(x)
                #features, _ = self.extractor.extract_features(x)

        #return features
        output_dict = {}
        for name, model in self.models_dict.items():
            output_dict[name] = model(features) 
        return output_dict

class LossesDict(dict):
    def backward(self):
        for name, loss in self.items():
            loss.backward()


class MultiCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output_dict, target):
        losses_dict = LossesDict()
        for name, preds in output_dict.items():
            losses_dict[name] = self.criterion(preds, target)

        return losses_dict
    
    
class AverageFeatureSequence(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        return x.mean(dim=1).unsqueeze(1), None


class MultiCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output_dict, target):
        losses_dict = LossesDict()
        for name, preds in output_dict.items():
            losses_dict[name] = self.criterion(preds, target)

        return losses_dict
    

class R3DWithBboxes(nn.Module):
    def __init__(self, class_num, alpha=0.4):
        super().__init__()
        self.alpha = alpha
        
        self.extractor = nn.ModuleDict()
        for name, module in torchvision.models.video.r3d_18().named_children():
            if 'layer' in name or 'stem' in name:
                self.extractor[name] = module
        self.output_classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, class_num)
        )

    def forward(self, data):
        frames, mask = data
        for name, module in self.extractor.items():
            if mask.shape[2:] != frames.shape[2:]:
                mask = nn.functional.interpolate(mask, frames.shape[2:])
            masked_frames = (1-self.alpha)*frames + self.alpha*mask
            frames = module(masked_frames)
        result = self.output_classifier(frames)

        return result
    
class R3D(R3DWithBboxes):
    def forward(self, frames):
        for name, module in self.extractor.items():
            frames = module(frames)
        result = self.output_classifier(frames)

        return result
    
class TransformerSequenceProcessor(nn.Module):
    def __init__(self, extractor_model, hidden_size, transformer_layer_num, transformer_head_num, class_num):
        super().__init__()
        self.feature_extractor = extractor_model
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=transformer_head_num, batch_first=True)
        self.transformer_squence_processing = nn.TransformerEncoder(
            transformer_layer,
            num_layers=transformer_layer_num,
            norm=nn.LayerNorm(hidden_size))
        
        self.average_features_sequence = SequenceAverageFeatures(hidden_size=hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, class_num)
            )

    def forward(self, x, ret_type='classifier'):
        with torch.no_grad():
            features = self.feature_extractor(x)
        transformer_sequence_features = self.transformer_squence_processing(features)
        avg_features = self.average_features_sequence(transformer_sequence_features)
        
        classifier_out = self.classifier(avg_features)
        if ret_type == 'classifier':
            return classifier_out
        elif ret_type == 'features':
            return transformer_sequence_features
        elif ret_type == 'all':
            return classifier_out, transformer_sequence_features


if __name__ == '__main__':
    video_extractor = Swin3d_T_extractor(frame_num=256, window_size=16).cuda()

    video = torch.randn((1, 3, 256, 112, 112)).cuda()
    print(video_extractor)
    #out = video_extractor(video)
    #print(out.shape)
