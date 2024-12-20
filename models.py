from torchvision.models.video import r3d_18
import torchvision
import torch
from torch import nn
import numpy as np

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
        with torch.no_grad():
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
        with torch.no_grad():
            features = self.wav2vec(x).permute(0, 2, 1)

        return features

class Wav2vec2Extractor(Wav2vecExtractor):
    def forward(self, x):
        with torch.no_grad():
            features = self.wav2vec.extract_features(x)[0][-1]

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
        items_num = len(self)
        for idx, (name, loss) in enumerate(self.items()):
            retain_graph = idx != (items_num-1)
            loss.backward(retain_graph=retain_graph)

class MultiModalCrossEntropyLoss(nn.Module):
    def __init__(self, modalities_losses_dict):
        super().__init__()
        self.criterion_dict = modalities_losses_dict
        self.modalities_list = list(modalities_losses_dict.keys())

    def forward(self, output_dict, target):
        losses_dict = LossesDict()
        #loss = 0
        for modality_names, modality_labels in target:
            batch_size = modality_labels.size(0)
            # шаблон - <modality_name>_EMPTY
            modality_name = modality_names[0].split('_')[0]
            modality_names = [n.split('_')[-1] for n in modality_names]
            modality_names = np.array(modality_names)
            not_empty_tensors = modality_names!='EMPTY'
            modality_names = modality_names[not_empty_tensors]
            if len(modality_names) > 0:
                if modality_name in self.modalities_list:
                    #output_dict[modality_name][~not_empty_tensors].detach_()
                    preds = output_dict[modality_name][not_empty_tensors]
                    labels = modality_labels[not_empty_tensors]
                    #print(preds)
                    #print()
                    #print(labels)
                    #loss += self.criterion_dict[modality_name](preds, labels)
                    losses_dict[modality_name] = self.criterion_dict[modality_name](preds, labels)
        #print(losses_dict)
        #print(target)

        #return loss
        return losses_dict

class AudioCnn1DExtractorWrapper(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.extractor = CNN1D(class_num=2).extractor
            
            self.adaptor = nn.Sequential(
                #nn.Flatten(),
                nn.Linear(512, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            
        def forward(self, x):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            h = self.extractor(x).permute(0, 2, 1)

            return self.adaptor(h)


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
        '''
        self.average_features_sequence = SequenceAverageFeatures(hidden_size=hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, class_num)
            )
        '''
    def forward(self, x):
        #with torch.no_grad():
        features = self.feature_extractor(x)
        return self.transformer_squence_processing(features)
        
        if ret_type=='classifier' or ret_type=='all':
            avg_features = self.average_features_sequence(transformer_sequence_features)
            classifier_out = self.classifier(avg_features)
        
        if ret_type == 'classifier':
            return classifier_out
        elif ret_type == 'features':
            return transformer_sequence_features
        elif ret_type == 'all':
            return classifier_out, transformer_sequence_features
        
class OutputClassifier(nn.Module):
    def __init__(self, input_features, class_num):
        super().__init__()
        self.classifier = nn.Sequential(
            SequenceAverageFeatures(hidden_size=input_features),
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, class_num)
        )
    def forward(self, x):
        return self.classifier(x)

class EqualSizedTransformerModalitiesFusion(nn.Module):
    def __init__(self,
                 fusion_transformer_layer_num,
                 fusion_transformer_hidden_size,
                 fusion_transformer_head_num):
        super().__init__()
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=fusion_transformer_hidden_size, nhead=fusion_transformer_head_num, batch_first=True,)
        self.modality_fusion_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=fusion_transformer_layer_num,
            norm=nn.LayerNorm(fusion_transformer_hidden_size))
        
    def forward(self, modalities_features_dict):
        
        modality_features_bounds_dict = {}
        prev_size = 0
        modalities_features_list = []
        sequence_lengths = []
        for modality_name, data in modalities_features_dict.items():
            #batch_size = data.size(1)
            seq_len = data.size(1)
            
            sequence_lengths.append({modality_name:seq_len})
            modality_features_bounds_dict[modality_name] = [prev_size, prev_size+seq_len]
            modalities_features_list.append(data)
            prev_size = prev_size+seq_len
        concat_features = torch.cat(modalities_features_list, dim=1)
        # если на вход поступил нулевой тензор, то это означает отсутствие модальности
        zero_features = concat_features.sum(dim=2)
        key_padding_mask = zero_features == 0

        # генерируем маску
        fused_features = self.modality_fusion_transformer(concat_features, src_key_padding_mask=key_padding_mask)
        #fused_features = self.modality_fusion_transformer(concat_features)
        #print(f'Fused features before:\t\t{fused_features.shape}, has NaN:{torch.isnan(fused_features).sum().item()>0}')
        #print(fused_features)
        #return fused_features, modality_features_bounds
        return {modality_name:fused_features[:,b0:b1] for modality_name, (b0, b1) in modality_features_bounds_dict.items()}
    
class CrossModalFusion(nn.Module):
    def __init__(self,
                 fusion_transformer_layer_num,
                 fusion_transformer_hidden_size,
                 fusion_transformer_head_num):
        super().__init__()
        
        transformer_layer_m1_to_m2 = nn.TransformerDecoderLayer(
            d_model=fusion_transformer_hidden_size, nhead=fusion_transformer_head_num, batch_first=True,)
        self.modality_fusion_m1_to_m2 = nn.TransformerDecoder(
            transformer_layer_m1_to_m2,
            num_layers=fusion_transformer_layer_num,
            norm=nn.LayerNorm(fusion_transformer_hidden_size))
        
        transformer_layer_m2_to_m1 = nn.TransformerDecoderLayer(
            d_model=fusion_transformer_hidden_size, nhead=fusion_transformer_head_num, batch_first=True,)
        self.modality_fusion_m1_to_m2 = nn.TransformerDecoder(
            transformer_layer_m1_to_m2,
            num_layers=fusion_transformer_layer_num,
            norm=nn.LayerNorm(fusion_transformer_hidden_size))
        
    def forward(self, modalities_features_dict):
        
        modality_features_bounds_dict = {}
        prev_size = 0
        modalities_features_list = []
        sequence_lengths = []
        for modality_name, data in modalities_features_dict.items():
            #batch_size = data.size(1)
            seq_len = data.size(1)
            
            sequence_lengths.append({modality_name:seq_len})
            modality_features_bounds_dict[modality_name] = [prev_size, prev_size+seq_len]
            modalities_features_list.append(data)
            prev_size = prev_size+seq_len
        concat_features = torch.cat(modalities_features_list, dim=1)
        # если на вход поступил нулевой тензор, то это означает отсутствие модальности
        zero_features = concat_features.sum(dim=2)
        key_padding_mask = zero_features == 0

        # генерируем маску
        fused_features = self.modality_fusion_transformer(concat_features, src_key_padding_mask=key_padding_mask)
        #fused_features = self.modality_fusion_transformer(concat_features)
        #print(f'Fused features before:\t\t{fused_features.shape}, has NaN:{torch.isnan(fused_features).sum().item()>0}')
        #print(fused_features)
        #return fused_features, modality_features_bounds
        return {modality_name:fused_features[:,b0:b1] for modality_name, (b0, b1) in modality_features_bounds_dict.items()}

class AveragedFeaturesTransformerFusion(EqualSizedTransformerModalitiesFusion):
            
    def forward(self, modalities_features_dict):

        # усреднение
        modalities_features_dict = {k:v.mean(dim=1).unsqueeze(1) for k, v in modalities_features_dict.items()}
        
        modality_features_bounds_dict = {}
        prev_size = 0
        modalities_features_list = []
        for modality_name, data in modalities_features_dict.items():
            size = data.size(1)
            modality_features_bounds_dict[modality_name] = [prev_size, prev_size+size]
            modalities_features_list.append(data)
            prev_size = prev_size+size

        concat_features = torch.cat(modalities_features_list, dim=1)

        zero_features = concat_features.sum(dim=2)
        key_padding_mask = zero_features == 0
        
        fused_features = self.modality_fusion_transformer(concat_features, src_key_padding_mask=key_padding_mask)
        #return fused_features, modality_features_bounds
        return {modality_name:fused_features[:,b0:b1] for modality_name, (b0, b1) in modality_features_bounds_dict.items()}

class MultimodalModel(nn.Module):
    def __init__(self, modality_extractors_dict, modality_fusion_module, classifiers, modality_features_shapes_dict, hidden_size, class_num):
        super().__init__()
        self.modality_extractors_dict = modality_extractors_dict
        self.modality_features_shapes_dict = modality_features_shapes_dict
        self.modality_fusion_module = modality_fusion_module#
        self.classifiers = classifiers

    def extract_features(self, input_data):
        modalities_features_dict = {}
        for modality_names, modality_batch in input_data:
            batch_size = modality_batch.size(0)
            # шаблон - <modality_name>_EMPTY; EMPTY означает, что модальность для выбранного экземпляра данных отсутствует
            modality_name = modality_names[0].split('_')[0]
            modality_names = [n.split('_')[-1] for n in modality_names]
            
            modality_names = np.array(modality_names)
            # выполняем фильтрацию пакетов (batches) c пустыми модальностями
            not_empty_tensors = modality_names!='EMPTY'
            modality_names = modality_names[not_empty_tensors]
            
            # создаем заглушку из нулей, чтобы обеспечить многомодальное обучение в случае отсутствия модальности
            modality_features_shape = self.modality_features_shapes_dict[modality_name]
            modality_features_shape = [batch_size] + modality_features_shape
            modality_features = torch.zeros(modality_features_shape, device=modality_batch.device)

            if len(modality_names) > 0:
                #modality_name = modality_names[0]
                if modality_name in self.modality_extractors_dict:
                    # выполняем извлечение признаков
                    features = self.modality_extractors_dict[modality_name](modality_batch[not_empty_tensors])
                    # ставим на места не пустых пакетов (batches) извлеченные признаки
                    modality_features[not_empty_tensors] = features
            modalities_features_dict[modality_name] = modality_features

        # modlalities_features_dict = {'modality_name': modality_features_tensor}
        return dict(sorted(modalities_features_dict.items())) 

    def forward(self, input_data):
        # извлечение признаков
        modalities_features_dict = self.extract_features(input_data)
        
        # выполняем слияние модальностей
        modalities_features_dict = self.modality_fusion_module(modalities_features_dict)
        #return modalities_features_dict
        # Выполнение классификации
        output_dict = {}
        for aggr_type in self.classifiers:
            output_dict[aggr_type] = self.classifiers[aggr_type](modalities_features_dict[aggr_type])

        return output_dict
    
    def get_output_names(self):
        return list(self.classifiers.keys())
    
class AudioTextAdaptor(nn.Module):
    def __init__(self, target_dim, audio_dim=None, text_dim=None, p_dropout=0.3):
        super().__init__()
        
        audio_text_adaptors_dict = {}
        if audio_dim is not None:
            audio_text_adaptors_dict['audio'] = nn.Sequential(
                nn.Linear(audio_dim, target_dim),
                nn.ReLU(),
                nn.Dropout(p_dropout),)

        if text_dim is not None:
            audio_text_adaptors_dict['text'] = nn.Sequential(
                nn.Linear(text_dim, target_dim),
                nn.ReLU(),
                nn.Dropout(p_dropout),)
        self.audio_text_adaptors_dict = nn.ModuleDict(audio_text_adaptors_dict)

    def forward(self, audio_text_features_dict, combination_type='concat'):
        # усредняем входные векторы признаков
        #audio_text_features_dict = {modality: features.mean(dim=1) for modality, features in audio_text_features_dict.items()}
        combined_modality = []
        for modality_name, features in audio_text_features_dict.items():
            result = self.audio_text_adaptors_dict[modality_name](features).mean(dim=1)
            #print(modality_name)
            #print(result.shape)
            #print(combined_modality)
            #print(audio_text_features_dict.keys())
            combined_modality.append(result)
            #print(combined_modality)
        # если мы на выход подаем усреденный вектор
        #combined_modality = torch.stack(combined_modality, dim=1).mean(dim=1)
        # если мы на выход подаем векторы двух модальностей
        #print(combined_modality)
        if combination_type == 'concat':
            combined_modality = torch.cat(combined_modality, dim=1)#.mean(dim=1)
        elif combination_type == 'sum':
            combined_modality = torch.stack(combined_modality, dim=1).sum(dim=1)
        elif combination_type == 'mean':
            combined_modality = torch.stack(combined_modality, dim=1).mean(dim=1)
        return combined_modality
    
class PhysVerbClassifier(nn.Module):
    #modality2aggr = {'video':'phys', 'text':'verb', 'audio':'verb'}
    #modality2aggr = {'video':'verb', 'text':'verb', 'audio':'verb'}
    def __init__(self, modalities_list, class_num, input_audio_size, input_text_size, input_video_size, verb_adaptor_out_dim, modality2aggr={'video':'verb', 'text':'verb', 'audio':'verb'}, p_droput=0.3):
        super().__init__()
        self.modalities_list = modalities_list
        self.class_num = class_num
        self.verb_adaptor_out_dim = verb_adaptor_out_dim
        adaptors_dict = {}
        classifiers_dict = {}
        self.modality2aggr = modality2aggr

        if 'video' in modalities_list:
            adaptors_dict['phys'] = SequenceAverageFeatures(hidden_size=input_video_size)
            classifiers_dict['phys'] = nn.Sequential(
                nn.Linear(input_video_size, 256),
                nn.ReLU(),
                nn.Dropout(p_droput),
                nn.Linear(256, class_num)
            )
        self.verb_modalities_sizes = {}
        if 'audio' in modalities_list:
            self.verb_modalities_sizes['audio_dim'] = input_audio_size
        if 'text' in modalities_list:
            self.verb_modalities_sizes['text_dim'] = input_text_size

        if len(self.verb_modalities_sizes) > 0:
            adaptors_dict['verb'] = AudioTextAdaptor(target_dim=verb_adaptor_out_dim, **self.verb_modalities_sizes)
        
        if len(self.verb_modalities_sizes) > 0:
            classifiers_dict['verb'] = nn.Sequential(
                nn.Linear(verb_adaptor_out_dim*len(self.verb_modalities_sizes), 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, class_num)
            )
        self.classifiers_dict = nn.ModuleDict(classifiers_dict)
        self.adaptors_dict = nn.ModuleDict(adaptors_dict)
        
    def forward(self, modalities_features_dict):
        output_dict = {}
        if 'video' in modalities_features_dict:
            video_features = modalities_features_dict['video']
            adapted_phys_features = self.adaptors_dict['phys'](video_features)
            output_dict['phys'] = self.classifiers_dict['phys'](adapted_phys_features)
        verb_features = {}
        if 'audio' in modalities_features_dict:
            verb_features['audio'] = modalities_features_dict['audio']
        if 'text' in modalities_features_dict:
            verb_features['text'] = modalities_features_dict['text']

        if len(verb_features) > 0:
            
            #for k, v in verb_features.items():
                #print()
                #print(k)
                #print(v.shape)
            adapted_verb_features = self.adaptors_dict['verb'](verb_features)
            #print('Adapted:')
            #print(adapted_verb_features.shape)
            output_dict['verb'] = self.classifiers_dict['verb'](adapted_verb_features)

        return output_dict


class PhysVerbClassifier(nn.Module):
    #modality2aggr = {'video':'phys', 'text':'verb', 'audio':'verb'}
    #modality2aggr = {'video':'verb', 'text':'verb', 'audio':'verb'}
    def __init__(self, modalities_list, class_num, modalities_adaptors_inout_sizes_dict, modality2aggr={'video':'phys', 'text':'verb', 'audio':'verb'}, p_droput=0.3):
        super().__init__()
        self.modalities_list = modalities_list
        self.modalities_inout_sizes_dict = modalities_adaptors_inout_sizes_dict
        self.class_num = class_num
        self.modalities_adaptors_inout_sizes_dict = modalities_adaptors_inout_sizes_dict
        self.modality2aggr = modality2aggr

        adaptors_dict, aggr_types_classifier_in_dims_dict = self.prepare_adaptors(modalities_list, modalities_adaptors_inout_sizes_dict, p_droput)
        
        classifiers_dict = self.prepare_classifiers(aggr_types_classifier_in_dims_dict, p_droput, class_num)

        self.classifiers_dict = nn.ModuleDict(classifiers_dict)
        self.adaptors_dict = nn.ModuleDict(adaptors_dict)
        #print('ADAPTORS_DICT')
        #print(self.adaptors_dict)
    
    def prepare_adaptors(self, modalities_list, modalities_adaptors_inout_sizes_dict, p_droput):
        adaptors_dict = {}
        aggr_types_classifier_in_dims_dict = {}
        for modality in modalities_list:
            in_features, out_features = modalities_adaptors_inout_sizes_dict[modality]
            adaptors_dict[modality] = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Dropout(p_droput),
                nn.ReLU(),
                SequenceAverageFeatures(hidden_size=out_features)
            )
            corr_aggr_type = self.modality2aggr[modality]
            try:
                aggr_types_classifier_in_dims_dict[corr_aggr_type] += out_features
            except:
                aggr_types_classifier_in_dims_dict[corr_aggr_type] = out_features
        
        return adaptors_dict, aggr_types_classifier_in_dims_dict
    
    def prepare_classifiers(self, aggr_types_classifier_in_dims_dict, p_droput, class_num):
        classifiers_dict = {}
        for aggr_type, classifier_input_dim in aggr_types_classifier_in_dims_dict.items():

            classifiers_dict[aggr_type] = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_input_dim//3),
                nn.ReLU(),
                nn.Dropout(p_droput),
                nn.Linear(classifier_input_dim//3, class_num)
            )
        return classifiers_dict

    def forward(self, modalities_features_dict):
        modalities_features_dict = dict(sorted(modalities_features_dict.items()))
        adapted_features_dict = {}
        for modality, features in modalities_features_dict.items():
            aggr_type = self.modality2aggr[modality]
            batch_size = features.size(0)
            adapted_features = self.adaptors_dict[modality](features)
            if aggr_type in adapted_features_dict:
                adapted_features = torch.cat([adapted_features_dict[aggr_type], adapted_features], dim=1)
                adapted_features_dict[aggr_type] = adapted_features
            else:
                adapted_features_dict[aggr_type] = adapted_features
        
        output_dict = {}
        for aggr_type, features in adapted_features_dict.items():
            output_dict[aggr_type] = self.classifiers_dict[aggr_type](features)

        return output_dict

class PhysVerbClassifierConcatFeatures(PhysVerbClassifier):
    def prepare_adaptors(self, modalities_list, modalities_adaptors_inout_sizes_dict, p_droput):
        adaptors_dict = {}
        
        for modality in modalities_list:
            in_features, out_features = modalities_adaptors_inout_sizes_dict[modality]
            adaptors_dict[modality] = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Dropout(p_droput),
                nn.ReLU(),
                SequenceAverageFeatures(hidden_size=out_features)
            )
        out_fetures_list = [v[1] for k, v in modalities_adaptors_inout_sizes_dict.items() if k in modalities_list]
        aggr_types_classifier_in_dims_dict = {}
        for aggr_type in self.modality2aggr.values():
            aggr_types_classifier_in_dims_dict[aggr_type] = sum(out_fetures_list)
        
        return adaptors_dict, aggr_types_classifier_in_dims_dict
    
    def forward(self, modalities_features_dict):
        modalities_features_dict = dict(sorted(modalities_features_dict.items()))
        adapted_features_list = []
        for modality, features in modalities_features_dict.items():
            aggr_type = self.modality2aggr[modality]
            batch_size = features.size(0)
            adapted_features = self.adaptors_dict[modality](features)
            adapted_features_list.append(adapted_features)
        adapted_features = torch.cat(adapted_features_list, dim=1)
        
        output_dict = {}
        for aggr_type in self.classifiers_dict:
            output_dict[aggr_type] = self.classifiers_dict[aggr_type](adapted_features)

        return output_dict
        

class PhysVerbClassifierAddFeatures(PhysVerbClassifier):
    def __init__(self, modalities_list, class_num, input_audio_size, input_text_size, input_video_size, verb_adaptor_out_dim, p_droput=0.3):
        super().__init__(modalities_list, class_num, input_audio_size, input_text_size, input_video_size, verb_adaptor_out_dim, p_droput)
        
        if 'verb' in self.classifiers_dict:
            self.classifiers_dict['verb'] = nn.Sequential(
                nn.Linear(verb_adaptor_out_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, class_num)
            )

    def forward(self, modalities_features_dict):
        #adapted_phys_features = None
        #adapted_verb_features = None
        output_dict = {}
        adapted_features_list = []
        if 'video' in modalities_features_dict:

            video_features = modalities_features_dict['video']
            
            adapted_phys_features = self.adaptors_dict['phys'](video_features)
            adapted_features_list.append(adapted_phys_features)
            
            #adapted_phys_features = torch.zeros()
        verb_features = {}
        if 'audio' in modalities_features_dict:
            verb_features['audio'] = modalities_features_dict['audio']
        if 'text' in modalities_features_dict:
            verb_features['text'] = modalities_features_dict['text']

        if len(verb_features) > 0:
            
            #for k, v in verb_features.items():
                #print()
                #print(k)
                #print(v.shape)
            
            adapted_verb_features = self.adaptors_dict['verb'](verb_features, combination_type='sum')
            #print('Adapted:')
            #print(adapted_verb_features.shape)
            adapted_features_list.append(adapted_verb_features)
            #output_dict['verb'] = self.classifiers_dict['verb'](adapted_verb_features)
        adapted_features_add = torch.stack(adapted_features_list, dim=1).sum(dim=1)
        
        output_dict['verb'] = self.classifiers_dict['verb'](adapted_features_add)
        output_dict['phys'] = self.classifiers_dict['phys'](adapted_features_add)

        return output_dict
   
class PhysVerbModel(nn.Module):
    #modality2aggr = {'video':'phys', 'text':'verb', 'audio':'verb'}
   
    def __init__(self, modality_extractors_dict, modality_fusion_module, classifiers, modality_features_shapes_dict, modality2aggr, hidden_size, class_num):
        super().__init__()
        self.modality_extractors_dict = modality_extractors_dict
        self.modality_features_shapes_dict = modality_features_shapes_dict
        self.modality_fusion_module = modality_fusion_module#
        self.classifiers = classifiers
        self.modality2aggr = modality2aggr
        #print('ХУЕПУТАЛО ЕБАНОЕ!')

    def extract_features(self, input_data):
        modalities_features_dict = {}
        for modality_names, modality_batch in input_data:
            batch_size = modality_batch.size(0)
            # шаблон - <modality_name>_EMPTY; EMPTY означает, что модальность для выбранного экземпляра данных отсутствует
            modality_name = modality_names[0].split('_')[0]
            modality_names = [n.split('_')[-1] for n in modality_names]
            
            modality_names = np.array(modality_names)
            # выполняем фильтрацию пакетов (batches) c пустыми модальностями
            not_empty_tensors = modality_names!='EMPTY'
            modality_names = modality_names[not_empty_tensors]
            
            # создаем заглушку из нулей, чтобы обеспечить многомодальное обучение в случае отсутствия модальности
            modality_features_shape = self.modality_features_shapes_dict[modality_name]
            modality_features_shape = [batch_size] + modality_features_shape
            modality_features = torch.zeros(modality_features_shape, device=modality_batch.device)

            if len(modality_names) > 0:
                #modality_name = modality_names[0]
                if modality_name in self.modality_extractors_dict:
                    # выполняем извлечение признаков
                    features = self.modality_extractors_dict[modality_name](modality_batch[not_empty_tensors])
                    # ставим на места не пустых пакетов (batches) извлеченные признаки
                    modality_features[not_empty_tensors] = features
            modalities_features_dict[modality_name] = modality_features

        # modlalities_features_dict = {'modality_name': modality_features_tensor}
        return dict(sorted(modalities_features_dict.items())) 

    def forward(self, input_data):
        # извлечение признаков
        modalities_features_dict = self.extract_features(input_data)
        
        # выполняем слияние модальностей
        modalities_features_dict = self.modality_fusion_module(modalities_features_dict)
        #return modalities_features_dict
        # Выполнение классификации
        '''
        output_dict = {}
        for aggr_type in self.classifiers:
            output_dict[aggr_type] = self.classifiers[aggr_type](modalities_features_dict[aggr_type])
        '''
        output_dict = self.classifiers(modalities_features_dict)
        return output_dict
    
    def get_output_names(self):
        return list(self.classifiers.keys())

    
    def get_output_names(self):
        return list(self.classifiers.classifiers_dict.keys())

    
class AudioTextualModel(nn.Module):
    def __init__(self, audio_extractor_model, text_extractor_model, hidden_size, class_num):
        super().__init__()
        self.audio_extractor = audio_extractor_model
        self.text_extractor = text_extractor_model
        #self.modality_fusion_module = EqualSizedModalitiesFusion(fusion_transformer_layer_num=2, fusion_transformer_hidden_size=768, fusion_transformer_head_num=2)
        self.modality_fusion_module = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.output_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, class_num)
            )

    def forward(self, x):
        data_dict = {}
        for modality_name, batch_tensor in x:
            data_dict[modality_name[0]] = batch_tensor
        
        #audio_features = self.audio_extractor(data_dict['audio'], ret_type='features')
        audio_features = self.audio_extractor(data_dict['audio'])
        text_features = self.text_extractor(data_dict['text'])

        #fused_audio, fused_text = self.modality_fusion_module([audio_features, text_features])

        #averaged_audio = fused_audio.mean(dim=1)
        #averaged_text = fused_text.mean(dim=1)
        averaged_audio = audio_features.mean(dim=1)
        averaged_text = text_features.mean(dim=1)
        concat_features = torch.cat([averaged_audio, averaged_text], dim=-1)

        fused_features = self.modality_fusion_module(concat_features)

        return self.output_classifier(fused_features)
        #return self.output_classifier(averaged_audio + averaged_text)
        #return self.output_classifier(averaged_text)

class CNN1D(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=160, stride=40, padding=160//2),
            #nn.Conv1d(1, 64, kernel_size=101, stride=30, padding=101//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4, 4),
            nn.Dropout1d(0.1),
            #################
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=3//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.MaxPool1d(4, 4),
            nn.Dropout1d(0.1),
            #################
            nn.Conv1d(64, 128, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Conv1d(128, 128, kernel_size=3, padding=3//2),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),

            nn.MaxPool1d(4, 4),
            nn.Dropout1d(0.1),
            #################
            nn.Conv1d(128, 256, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Conv1d(256, 256, kernel_size=3, padding=3//2),
            #nn.BatchNorm1d(256),
            #nn.ReLU(),

            nn.MaxPool1d(4, 4),
            nn.Dropout1d(0.1),
            #################

            nn.Conv1d(256, 512, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout1d(0.1),

            #nn.Conv1d(512, 512, kernel_size=3, padding=3//2),
            #nn.BatchNorm1d(512),
            #nn.ReLU()
            )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # это, похоже, и есть Lambda
            nn.Flatten(),
            nn.Dropout1d(0.2),
            nn.Linear(512, class_num)    
        )

    def forward(self, x):
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h = self.extractor(x)
        #return h
        return self.classifier(h)
        
'''
class CNN1D(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.extractor = nn.Sequential(
            #nn.Conv1d(1, 64, kernel_size=160, stride=40, padding=160//2),
            nn.Conv1d(1, 64, kernel_size=101, stride=30, padding=101//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            #nn.MaxPool1d(4, 4),
            nn.Dropout1d(0.1),
            #################
            nn.Conv1d(64, 64, kernel_size=31, stride=10, padding=30//2),
            nn.BatchNorm1d(64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=5//2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            #nn.MaxPool1d(4, 4),
            nn.Dropout1d(0.1),
            #################
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=5//2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=5//2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Conv1d(128, 128, kernel_size=3, padding=3//2),
            #nn.BatchNorm1d(128),
            #nn.ReLU(),

            nn.MaxPool1d(4, 4),
            nn.Dropout1d(0.1),
            #################
            nn.Conv1d(128, 256, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Conv1d(256, 256, kernel_size=3, padding=3//2),
            #nn.BatchNorm1d(256),
            #nn.ReLU(),

            nn.MaxPool1d(4, 4),
            nn.Dropout1d(0.1),
            #################

            nn.Conv1d(256, 512, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=3//2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout1d(0.1),

            #nn.Conv1d(512, 512, kernel_size=3, padding=3//2),
            #nn.BatchNorm1d(512),
            #nn.ReLU()
            )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), # это, похоже, и есть Lambda
            nn.Flatten(),
            nn.Dropout1d(0.2),
            nn.Linear(512, class_num)    
        )

    def forward(self, x):
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h = self.extractor(x)
        return h#self.classifier(h)
'''

if __name__ == '__main__':
    model = CNN1D(2)
    #out = model(torch.randn(1, 80000))
    #print(out.shape)
    extractor = model.extractor
    
    tensor = torch.randn(1, 1, 80000)
    for e in extractor:
        name = str(e)
        tensor = e(tensor)
        print(e)
        if 'Conv'in name:
            
            
            print(tensor.shape)
    print(model(tensor).shape)
