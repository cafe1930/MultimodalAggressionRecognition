import torch
from torch import nn

from tqdm import tqdm

import shutil
import os
import warnings

from datetime import datetime
import pickle
import pandas as pd

import random
import os
import numpy as np
import matplotlib.pyplot as plt

def reursively_to_device(data, device):
    if isinstance(data, list):
        reursively_to_device(data, device)
    elif isinstance(data, torch.tensor):
        data = data.to(device)


class TorchSupervisedTrainer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        metrics_dict: dict, # словарь с функциями, вычисляющими метрики
        metrics_to_display: list, # список метрик выводимых на экран в ходе обучения
        device: torch.device,
        criterion,
        optimizers_list: list, # список всех возможных оптимизаторов
        lr_schedulers_list=[None], # список всех возможных планировщиков скорости обучения, по умолчанию, без планировщиков
        saving_dir = 'saving_dir',
        checkpoint_criterion='loss'):
        '''
        model:  - оптимизируемая модель
        metrics_dict: dict
            - словарь, у которого ключи - это названия метрик, а значения - это функции вычисления метрик
              функция метрик должна соответствовать интерфейсу scikit-learn: metric_score(true_sample, predicted_sample)
              loss вычисляется при помощи средств pytorch, а не scikit-learn
        checkpoint_criterion='loss'
            - значение, по которому мы сохраняем веса модели
              если контролируем loss, то берем изначально большое значение критерия
              если контролируем точность, то берем ноль
        '''
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.batch_size = train_loader.batch_size

        self.device = device
        self.criterion = criterion
        self.optimizers_list = optimizers_list
        self.lr_schedulers_list = lr_schedulers_list
        self.metrics_dict = metrics_dict
        self.metrics_to_display = metrics_to_display

        # Эпоха, с которой мы начинаем обучение
        self.start_epoch = 0
        self.current_epoch = self.start_epoch

        #self.metrics_list = metrics_list
        # специфицируется для каждой задачи
        self.init_log()
        #!!!!!!!
        self.checkpoint_criterion = checkpoint_criterion
        self.best_criterion = self.define_best_criterion(checkpoint_criterion)
        

        self.train_samples_num = len(self.train_loader.dataset)
        self.test_samples_num = len(self.test_loader.dataset)

        self.path_to_current_checkpoint = None
        self.path_to_best_checkpoint = None
        
        if not os.path.isdir(saving_dir):
            os.mkdir(saving_dir)

        current_date_time = datetime.now().strftime('%d.%m.%Y, %H-%M-%S')
        self.saving_dir = os.path.join(saving_dir, current_date_time+f' ({model_name})')
        if not os.path.isdir(self.saving_dir):
            os.mkdir(self.saving_dir)
    
    def define_best_criterion(self, checkpoint_criterion):
        if checkpoint_criterion == 'loss':
            # если контролируем loss, то берем изначально большое значение критерия
            best_criterion = 0
            # функция сравнения текущего значения метрики с лучшим
            #self.is_best_result = lambda x,y: x<y
        else:
            # если контролируем другую метрику (accuracy, recall и т.д.), то берем изначально нулевое значение критерия
            best_criterion = 9999999
            # функция сравнения текущего значения метрики с лучшим
            #self.is_best_result = lambda x,y: x>y

        return best_criterion
    
    def init_log(self):
        self.training_log = pd.DataFrame(columns=self.metrics_dict.keys())
        self.testing_log = pd.DataFrame(columns=self.metrics_dict.keys())

    def train_step(self, batch):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        '''
        # отправляем данные на вычислительное устройство
        data, true_vals = batch[0], batch[1]
        #true_vals = true_vals.to(self.device)
        if isinstance(true_vals, (list, tuple)):
            #data = [d.to(self.device) for d in data]
            for i, tv in enumerate(true_vals):
                if isinstance(tv, list):
                    tv = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in tv]
                else:
                    true_vals[i] = tv.to(self.device)
        else:
            true_vals = true_vals.to(self.device)
        if isinstance(data, (list, tuple)):
            #data = [d.to(self.device) for d in data]
            for i, d in enumerate(data):
                if isinstance(d, list):
                    d = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in d]
                else:
                    data[i] = d.to(self.device)
        else:
            data = data.to(self.device)
        # стандартные операции:
        # 1. Обнуление градиентов для всех задействованных оптимизаторов
        for opt in self.optimizers_list:
            opt.zero_grad()

        #self.optimizer.zero_grad()
        # 2. Прямое распространение
        pred = self.model(data)
        # 3. Вычисление ошибки
        loss = self.criterion(pred, true_vals)
        
        # 4. Обратное распространение ошибки
        loss.backward()
        # 5. Обновление весов для всех задействованных оптимизаторов
        for opt in self.optimizers_list:
            opt.step()
        #self.optimizer.step()

        # Вычисление суммарной ошибки на батче
        #ret_loss = loss.item() * data.size(0)
        if isinstance(data, list):
            size = len(data[0])#.size(0)
        else:
            size = len(data)#.size(0)
        ret_loss = self.compute_batch_loss(loss, size)
        # получение результатов нейронной сети для последующей обработки
        pred_vals = self.nn_output_processing(pred)
        batch_results_dict = self.create_batch_results_dict(ret_loss, pred_vals, true_vals)
        return batch_results_dict
    
    def nn_output_processing(self, pred):
        '''
        ПЕРЕПИСЫВЕМАЯ ФУНКЦИЯ
        функция, выполняющая постобработку выхода нейронной сети
        '''
        _, pred_labels = torch.max(pred.data, dim=1)
        return pred_labels.detach().cpu().numpy()
        

    def compute_batch_loss(self, batch_loss, data_size):
        # специальный метод нужен для того, чтобы мочь обрабатывать множество независимых выходов...
        return batch_loss.item() * data_size
    
    def create_batch_results_dict(self, ret_loss, pred_vals, true_vals):
        return {'loss': ret_loss, 'true': true_vals.detach().cpu().numpy(), 'pred': pred_vals}

    def test_step(self, batch):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        '''
        # Шаг тестироавания
        # отправляем данные на вычислительное устройство
        data, true_vals = batch[0], batch[1]
        if isinstance(true_vals, (list, tuple)):
            #data = [d.to(self.device) for d in data]
            for i, tv in enumerate(true_vals):
                if isinstance(tv, list):
                    tv = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in tv]
                else:
                    true_vals[i] = tv.to(self.device)
        else:
            true_vals = true_vals.to(self.device)
        if isinstance(data, (list, tuple)):
            #data = [d.to(self.device) for d in data]
            for i, d in enumerate(data):
                if isinstance(d, list):
                    d = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in d]
                else:
                    data[i] = d.to(self.device)
        else:
            data = data.to(self.device)
        # Прямое распространение
        pred = self.model(data)
        # Прямое распространение
        loss = self.criterion(pred, true_vals)
        # Вычисление суммарной ошибки на батче
        #ret_loss = loss.item() * data.size(0)
        if isinstance(data, list):
            size = len(data[0])#.size(0)
        else:
            size = len(data)#.size(0)
        ret_loss = self.compute_batch_loss(loss, size)
        # получение результатов нейронной сети для последующей обработки
        pred_vals = self.nn_output_processing(pred)
        batch_results_dict = self.create_batch_results_dict(ret_loss, pred_vals, true_vals)
        return batch_results_dict

    
    def compute_epoch_loss(self, results_list):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        '''
        cummulative_loss = 0
        # В цикле накапливаем значения всех метрик
        for loss, true_vals, pred_vals in results_list:
            cummulative_loss += loss

        return cummulative_loss/len(results_list)


    def compute_epoch_results(self, epoch_results_list, mode):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        Здесь мы 'парсим' данные, полученные в ходе обучения или тестирования
        '''

        if mode =='train':
            dataset_size = self.train_samples_num
        elif mode == 'test':
            dataset_size = self.test_samples_num
        else:
            raise TypeError('mode should be either \'train\' or \'test\'')

        # process metrics
        true = []
        pred = []
        cummulative_loss = 0
        # В цикле накапливаем значения всех метрик
        for batch_results in epoch_results_list:
            cummulative_loss += batch_results['loss']
            true.append(batch_results['true'])
            pred.append(batch_results['pred'])

        # составляем массивы из полученных значений
        #true = np.array(true).reshape(-1)
        #pred = np.array(pred).reshape(-1)

        true = np.concatenate(true)#.reshape(-1)
        pred = np.concatenate(pred)#.reshape(-1)
        
        loss = cummulative_loss/dataset_size
        # Строка для вывода на экран
        # Словарь, который мы будем добавлять в лог
        log_results_dict = {}
        log_results_dict['loss'] = loss
        #log_results_dict['loss'] = self.compute_epoch_loss()

        for metric_name in self.metrics_dict.keys():
            if metric_name.lower() != 'loss':
                metric = self.metrics_dict[metric_name]
                if type(metric) is dict:
                    metric_func = metric['metric']
                    metric_kwargs = metric['kwargs']
                else:
                    metric_func = metric
                    metric_kwargs = {}
                # вычисляем метрику
                metric_value = metric_func(true, pred, **metric_kwargs)
                # добавляем метрику в словарь
                log_results_dict[metric_name] = metric_value

        return log_results_dict

    def compute_batch_results(self, batch_results):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        В случае классификации не очень важно дополнително обрабатывать выходы
        '''
        return batch_results

    def print_result(self, result_dict):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        '''
        metrics_string_to_print = ''
        for metric_name in self.metrics_to_display:
            metric_value = result_dict[metric_name]
            
            '''
            try:
                # если metric_value - это вектор, то выводим на экран среднее значение метрики
                size = len(metric_value)
                metric_value = np.mean(metric_value)
            except TypeError:
                pass
            '''

            try:
                iter(metric_value)
                metrics_string_to_print += f'{metric_name}: ['
                for m in metric_value:
                    if m < 0.001:
                        metrics_string_to_print += '{:.2e}, '.format(m)
                    else:
                        metrics_string_to_print += '{:.3f}, '.format(m)
                metrics_string_to_print += ']'
            except TypeError:
                if metric_value < 0.001:
                    # для краткости выводим очень малые чила в экспоненциальной записи
                    metrics_string_to_print += '{}: {:.2e}; '.format(metric_name, metric_value)
                else:
                    metrics_string_to_print += '{}: {:.3f}; '.format(metric_name, metric_value)
        # выводим метрики на экран
        print(metrics_string_to_print)

    def save_checkpoint(self, path_to_saving_weights):
        '''
        Сохранение весов при достижении лучшего результата
        '''
        # сохраняем весь класс целтиком, чтобы не париться
        try:
            torch.save(self, path_to_saving_weights)
        except:
            torch.save(self.model.state_dict(), path_to_saving_weights)

    def save_logs(self):
        '''
        Сохранение результатов обучения
        '''
        self.training_log.to_csv(os.path.join(self.saving_dir, 'train_log_.csv'), index=False)
        self.testing_log.to_csv(os.path.join(self.saving_dir, 'test_log_.csv'), index=False)

    def update_datasets(self):
        '''
        Функция нужна, если на каждой эпохе надо изменять датасеты
        '''
        pass

    def train(self, epoch_num):
        
        end_epoch = self.start_epoch+epoch_num
        for epoch_idx in range(self.start_epoch, end_epoch):
            self.current_epoch = epoch_idx
            print('Train epoch # {} of {} epochs...'.format(epoch_idx, end_epoch-1))
            #t0 = time.time()
            self.update_datasets()
            self.model.train()
            # список, куда будем записывать промежуточные результаты эпохи
            train_raw_result_list = []
            for batch in tqdm(self.train_loader):
                # обучение на одном батче
                train_results = self.compute_batch_results(self.train_step(batch))
                train_raw_result_list.append(train_results)
            
            # обновляем каждый планировщик скорости обучения из списка
            for lr_scheduler in self.lr_schedulers_list:
                # не обновляем планировщик, если его нет)
                if lr_scheduler is not None:
                    lr_scheduler.step()

            # парсим результаты на обучающей выборке
            train_results_dict = self.compute_epoch_results(train_raw_result_list, mode='train')

            # выводим обучающие результаты на экран
            self.print_result(train_results_dict)

            
            
            # сохраняем результат тренировочных метрик эпохи
            self.update_log('train', train_results_dict)

            # запускаем процедуру тестирования
            self.test_raw_result_list = self.test()
            # парсим результаты на тестовой выборке
            test_results_dict = self.compute_epoch_results(self.test_raw_result_list, mode='test')

            # выводим тестовые результаты на экран
            self.print_result(test_results_dict)

            # сохраняем результат тестировочных метрик эпохи
            self.update_log('test', test_results_dict)

            # сохраняем результаты обучения после каждой итерации
            self.save_logs()

            # обновляем стартовую эпоху, чтобы иметь возможность восстановить модель и продолжить обучение с
            #  той эпохи, с которой мы это обучение прекратили
            self.start_epoch = epoch_idx+1

            

            # save weights of current step
            self.prepare_current_checkpoint_path()

            
           
            self.save_checkpoint(self.path_to_current_checkpoint)

            # сохраняем лучшие веса - надо выполнить раньше save_checkpoint, чтобы сохранились пути до лучших весов
            self.save_best_weights(test_results_dict)

            
            
            

            # start testing procedure
            print('----------------------------------------------')
    
    def prepare_current_checkpoint_path(self):
        
        if self.path_to_current_checkpoint is not None:
            os.remove(self.path_to_current_checkpoint)

        current_weights_name = '{}_current_ep-{}.pt'.format(self.model_name, self.current_epoch)
        self.path_to_current_checkpoint = os.path.join(self.saving_dir, current_weights_name)

    def save_best_weights(self, test_results_dict):
        if self.checkpoint_criterion == 'loss':
            err = test_results_dict[self.checkpoint_criterion]
        else:
            err = 1 - test_results_dict[self.checkpoint_criterion]
        if err < self.best_criterion:
            print('BEST RESULTS HAS ACHIEVED, SAVING WEIGHTS')
            if self.path_to_best_checkpoint is not None:
                os.remove(self.path_to_best_checkpoint)
            
            best_weights_name = '{}_best_ep-{}.pt'.format(self.model_name, self.current_epoch)
            self.path_to_best_checkpoint = os.path.join(self.saving_dir, best_weights_name)
            # копируем текущие сохраняемые параметры
            print(f'currennt checkpoint: {self.path_to_current_checkpoint}, best checkpoint: {self.path_to_best_checkpoint}')
            shutil.copy2(self.path_to_current_checkpoint, self.path_to_best_checkpoint)

            #self.save_checkpoint(self.path_to_best_checkpoint)
            self.best_criterion = err#test_results_dict[self.checkpoint_criterion]


    def update_log(self, mode, results_dict):
        if mode.lower() == 'train':
            self.training_log = pd.concat([self.training_log, pd.DataFrame([results_dict])], ignore_index=True)
        elif mode.lower() == 'test':
            self.testing_log = pd.concat([self.testing_log, pd.DataFrame([results_dict])], ignore_index=True)

    def test(self):
        '''
        Процедура тестирования. Лучше ее вывести в отдельную функцию,
        чтобы иметь возможность потом выполнять тестирование на различных данных
        '''
        print('Testing procedure...', end=' ')
        #t0 = time.time()
        self.model.eval()
        with torch.no_grad():
            result_list = []
            for batch in tqdm(self.test_loader):
                test_results = self.compute_batch_results(self.test_step(batch))
                result_list.append(test_results)

        return result_list
    '''
    def infer(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            result_list = []
            for batch in tqdm(self.test_loader):
                test_results = self.compute_batch_results(self.test_step(batch))
                result_list.append(test_results)

        #t1 = time.time()
        #print('passed in {:.3f} seconds'.format(t1 - t0))
    '''
       
    def plot_train_process_results(self, metrics_list, train_test_sep=False, multiple_plots=False, save_plot=False):
        for metric_name in metrics_list:
            # если метрика содержит вектор значений, то мы на экран ее не выводим
            try:
                len(self.training_log[metric_name][0])
                warnings.warn('WARNING!\n\'{}\' metric will not be displayed due to it has multiple values.'.format(metric_name))
                metrics_list.remove(metric_name)
            except TypeError:
                if multiple_plots:
                    if train_test_sep:
                        title = '{} {} {}'.format(self.model_name, metric_name.capitalize(), 'train')
                        plt.figure(figsize=(6,6),facecolor='white')
                        plt.title(title)
                        plt.xlabel('Epochs')
                        plt.plot(self.training_log[metric_name])
                        plt.savefig(os.path.join(self.saving_dir, title+'.png'))
                        plt.show()
                        
                        title = '{} {} {}'.format(self.model_name, metric_name.capitalize(), 'test')
                        plt.figure(figsize=(6,6),facecolor='white')
                        plt.title(title)
                        plt.xlabel('Epochs')
                        plt.plot(self.testing_log[metric_name])
                        plt.savefig(os.path.join(self.saving_dir, title+'.png'))
                        plt.show()
                        
                    else:
                        title = '{} {}'.format(self.model_name, metric_name.capitalize())
                        plt.figure(figsize=(6,6),facecolor='white')
                        plt.title(title)
                        plt.xlabel('Epochs')
                        plt.plot(self.training_log[metric_name])
                        plt.plot(self.testing_log[metric_name])
                        plt.legend(['Train', 'Test'])
                        plt.savefig(os.path.join(self.saving_dir, title+'.png'))
                        plt.show()
                        
        if not multiple_plots:
            # Если мы печатаем раздельно для обучающей и для тестовой выборки
            if train_test_sep:
                fig, axs = plt.subplots(1, len(metrics_list)*2, figsize=(len(metrics_list)*2*6, 6),facecolor='white')
            else:
                fig, axs = plt.subplots(1, len(metrics_list), figsize=(len(metrics_list)*6, 6),facecolor='white')

            for plot_idx, metric_name in enumerate(metrics_list):
                if train_test_sep:
                    axs[plot_idx*2].set_title('{} {} {}'.format(self.model_name, metric_name.capitalize(), 'train'))
                    axs[plot_idx*2].set_xlabel('Epochs')
                    axs[plot_idx*2].plot(self.training_log[metric_name])
                    axs[plot_idx*2+1].set_title('{} {} {}'.format(self.model_name, metric_name.capitalize(), 'test'))
                    axs[plot_idx*2+1].set_xlabel('Epochs')
                    axs[plot_idx*2+1].plot(self.testing_log[metric_name])
                else:
                    axs[plot_idx].set_title('{} {}'.format(self.model_name, metric_name.capitalize()))
                    axs[plot_idx].set_xlabel('Epochs')
                    axs[plot_idx].plot(self.training_log[metric_name])
                    axs[plot_idx].plot(self.testing_log[metric_name])
                    axs[plot_idx].legend(['Train', 'Test'])

            if save_plot:
                path_to_save = os.path.join(self.saving_dir, '{}_training_process.png'.format(self.model_name))
                plt.savefig(path_to_save)


class SegmentationTrainer(TorchSupervisedTrainer):
    class_names_dict = {
        0: 'Не размечено',
        1: 'Пахотные земли (Озимые поля)',
        2: 'Залежные земли ИЛИ Сенокос',
        3: 'Залежные земли (кустарники)',
        4: 'Залежные земли (деревья)',
        5: 'Неиспользуемые участки, неудобья',
        6: 'Дороги',
        7: 'Области электропередач',
        8: 'Лесополосы',
        9: 'Заболоченные участки',
        10: 'Водные объекты',
        11: 'Площадь около строений',
        12: 'Строения',
        13: 'Прочие объекты (кладбище)'
        }

    classes_list = [key for key in class_names_dict.keys()]
    # создаем палитру
    palette = ['{:06X}'.format(i) for i in range(0, 16777215, 16777215//14)]
    palette = [(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)) for s in palette]


    def compute_batch_results(self, batch_results):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        В случае сегментации НАДО дополнително обрабатывать выходы
        '''
        true, pred = batch_results['true'].reshape(-1), batch_results['pred'].reshape(-1)

        ret_results = {}
        ret_results['loss'] = batch_results['loss']
        ret_results['confusion_matrix'] = self.metrics_dict['confusion_matrix'](true, pred)

        return ret_results

    def compute_epoch_results(self, epoch_results_list, mode):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        Здесь мы 'парсим' данные, полученные в ходе обучения или тестирования
        '''

        if mode =='train':
            dataset_size = self.train_samples_num
        elif mode == 'test':
            dataset_size = self.test_samples_num
        else:
            raise TypeError('mode should be either \'train\' or \'test\'')

        # process metrics
        cummulative_loss = 0
        cumulative_confusion = None
        # В цикле накапливаем значения всех метрик
        for batch_results in epoch_results_list:
            cummulative_loss += batch_results['loss']
            try:
                cumulative_confusion += batch_results['confusion_matrix']
            except TypeError:
                cumulative_confusion = batch_results['confusion_matrix']      
        
        loss = cummulative_loss/dataset_size
        # Строка для вывода на экран
        # Словарь, который мы будем добавлять в лог
        log_results_dict = {}
        log_results_dict['loss'] = loss
        log_results_dict['confusion_matrix'] = cumulative_confusion

        #log_results_dict['loss'] = self.compute_epoch_loss()

        for metric_name in self.metrics_dict.keys():
            
            if not (metric_name.lower() == 'loss' or metric_name.lower() == 'confusion_matrix'):
                metric = self.metrics_dict[metric_name]
                if type(metric) is dict:
                    metric_func = metric['metric']
                    metric_kwargs = metric['kwargs']
                else:
                    metric_func = metric
                    metric_kwargs = {}
                # вычисляем метрику
                metric_value = metric_func(cumulative_confusion, **metric_kwargs)
                # добавляем метрику в словарь
                log_results_dict[metric_name] = metric_value

        return log_results_dict

    def draw_masks_on_images(self, path_to_save, model_size):
        '''
        Пока реализовано только для обучающего/тестового набора
        '''
        raise NotImplementedError
    

class RNN_trainer(TorchSupervisedTrainer):
    class_names_dict = {
        0: 'NOAGGR',
        1: 'AGGR'
    }

    def define_best_criterion(self, checkpoint_criterion):        
        self.model_names_list = list(self.model.models_dict.keys())
        best_criterion_dict = {}
        for model_name in self.model_names_list:
            if checkpoint_criterion == 'loss':
                # если контролируем loss, то берем изначально большое значение критерия
                best_criterion_dict[model_name] = 9999999
                # функция сравнения текущего значения метрики с лучшим
                #self.is_best_result = lambda x,y: x<y
            else:
                # если контролируем другую метрику (accuracy, recall и т.д.), то берем изначально нулевое значение критерия
                best_criterion_dict[model_name] = 0
                # функция сравнения текущего значения метрики с лучшим
                #self.is_best_result = lambda x,y: x>y

        return best_criterion_dict

    def save_best_weights(self, test_results_dict):        
        for model_name, test_results in test_results_dict.items():
            best = self.best_criterion[model_name]
            err = test_results[self.checkpoint_criterion]
            
            if self.checkpoint_criterion == 'loss':
                if err < best:
                    print(f'Best results for {model_name} achieved, saving weights...')
                    self.save_model(model_name)
                    self.best_criterion[model_name] = err
            else:
                if err > best:
                    print(f'Best results for {model_name} achieved, saving weights...')
                    self.save_model(model_name)
                    self.best_criterion[model_name] = err
            '''
            if err < self.best_criterion:
                print('BEST RESULTS HAS ACHIEVED, SAVING WEIGHTS')
                if self.path_to_best_checkpoint is not None:
                    os.remove(self.path_to_best_checkpoint)
                
                best_weights_name = '{}_best_ep-{}.pt'.format(self.model_name, epoch_idx)
                self.path_to_best_checkpoint = os.path.join(self.saving_dir, best_weights_name)
                # копируем текущие сохраняемые параметры
                shutil.copy2(self.path_to_current_checkpoint, self.path_to_best_checkpoint)

                #self.save_checkpoint(self.path_to_best_checkpoint)
                self.best_criterion = err#test_results_dict[self.checkpoint_criterion]
            '''

    def save_model(self, model_name):
        if self.path_to_best_checkpoint is not None:
            if model_name in self.path_to_best_checkpoint:
                os.remove(self.path_to_best_checkpoint[model_name])
        else:
            self.path_to_best_checkpoint = {}
            
        best_weights_name = f'{model_name}_best_ep-{self.current_epoch}.pt'
        path_to_best_checkpoint = os.path.join(self.saving_dir, best_weights_name)
        self.path_to_best_checkpoint[model_name] = path_to_best_checkpoint

        torch.save(self.model.models_dict[model_name].state_dict(), path_to_best_checkpoint)
        

    def init_log(self):
        
        self.training_log = {}
        self.testing_log = {}
        for name in self.model.models_dict.keys():
            self.training_log[name] = pd.DataFrame(columns=self.metrics_dict.keys())
            self.testing_log[name] = pd.DataFrame(columns=self.metrics_dict.keys())

    def nn_output_processing(self, pred):
        '''
        ПЕРЕПИСЫВЕМАЯ ФУНКЦИЯ
        функция, выполняющая постобработку выхода нейронной сети
        pred - словарь с результатами выхода всех нейронных сетей
        '''
        pred_labels_dict = {}
        for name, pred in pred.items():
            _, pred_labels = torch.max(pred.data, dim=1)
            pred_labels_dict[name] = pred_labels.detach().cpu().numpy()

        return pred_labels_dict
    
    def compute_batch_loss(self, batch_loss, data_size):
        # специальный метод нужен для того, чтобы мочь обрабатывать множество независимых выходов...
        # batch_loss - словарь с значением ошибки для каждой нейронной сети
        batch_loss_dict = {}
        for name, loss in batch_loss.items():
            batch_loss_dict[name] = loss.item() * data_size
        return batch_loss_dict
    
    def compute_epoch_results(self, epoch_results_list, mode):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        Здесь мы 'парсим' данные, полученные в ходе обучения или тестирования
        '''

        if mode =='train':
            dataset_size = self.train_samples_num
        elif mode == 'test':
            dataset_size = self.test_samples_num
        else:
            raise TypeError('mode should be either \'train\' or \'test\'')

        # process metrics
        true = []
        #cummulative_loss = 0
        # В цикле накапливаем значения всех метрик
        models_cummulative_loss_dict = {}
        models_cunmulative_models_preds_dict = {}
        
        for batch_results in epoch_results_list:
            
            # обработка loss
            for model_name, loss in batch_results['loss'].items():
                try:
                    models_cummulative_loss_dict[model_name] += loss
                except KeyError:
                    models_cummulative_loss_dict[model_name] = loss
            # обработка выходов НС
            for model_name, pred in batch_results['pred'].items():
                try:
                    pred_arr = models_cunmulative_models_preds_dict[model_name]
                    models_cunmulative_models_preds_dict[model_name] = np.concatenate([pred_arr, pred])#.append(pred)
                except KeyError:
                    models_cunmulative_models_preds_dict[model_name] = pred
                
            true.append(batch_results['true'])

        

        true = np.concatenate(true)#.reshape(-1)
        #pred = np.concatenate(pred)#.reshape(-1)
                
        #loss = cummulative_loss/dataset_size
        # Словарь, который мы будем добавлять в лог
        log_results_dict = {}

        # compute losses for all the models
        for model_name, loss in models_cummulative_loss_dict.items():
            log_results_dict[model_name] = {'loss': models_cummulative_loss_dict[model_name] / dataset_size}
        
        
        # Строка для вывода на экран
        
        #log_results_dict['loss'] = models_cummulative_loss_dict
        #log_results_dict['loss'] = self.compute_epoch_loss()

        # compute metrics for all the models

        for model_name, pred in models_cunmulative_models_preds_dict.items():
            for metric_name in self.metrics_dict.keys():
                if metric_name.lower() != 'loss':
                    metric = self.metrics_dict[metric_name]
                    if type(metric) is dict:
                        metric_func = metric['metric']
                        metric_kwargs = metric['kwargs']
                    else:
                        metric_func = metric
                        metric_kwargs = {}
                    # вычисляем метрику
                    metric_value = metric_func(true, pred, **metric_kwargs)
                    # добавляем метрику в словарь под соответствующим именем модли
                    log_results_dict[model_name][metric_name] = metric_value

        return log_results_dict
    
    def print_result(self, result_dict):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        '''
        model_names = sorted(list(result_dict.keys()))

        
        metrics_df = pd.DataFrame(columns=model_names,index=self.metrics_to_display)
        #metrics_string_to_print = ''
        for model_name in model_names:
            #metrics_string_to_print += f'{model_name}:\t'
            model_metrics_dict = {}
            for metric_name in self.metrics_to_display:
                
                metric_value = result_dict[model_name][metric_name]
                model_metrics_dict[metric_name] = metric_value
                
            metrics_df[model_name] = model_metrics_dict
        # выводим метрики на экран
        
        
        print(metrics_df)
        print('------------------------------------------')


    def update_datasets(self):
        '''
        Функция нужна, если на каждой эпохе надо изменять датасеты
        '''
        # создаем доп. датасет для своей эпохи

        path_to_train_root = os.path.split(self.train_loader.dataset.path_to_data_root)[0]

        path_to_train_data = os.path.join(path_to_train_root, str(self.current_epoch))
        
        self.train_loader.dataset.path_to_data_root = path_to_train_data

    def update_log(self, mode, results_dict):
        
        for model_name, results in results_dict.items():
            if mode.lower() == 'train':
                self.training_log[model_name] = pd.concat([self.training_log[model_name], pd.DataFrame([results]).fillna(0)], ignore_index=True)
            elif mode.lower() == 'test':
                self.testing_log[model_name] = pd.concat([self.testing_log[model_name], pd.DataFrame([results]).fillna(0)], ignore_index=True)
        
    def save_logs(self):
        '''
        Сохранение результатов обучения
        '''
        for name in self.training_log.keys():
            
            self.training_log[name].to_csv(os.path.join(self.saving_dir, f'{name}_train_log.csv'), index=False)
            self.testing_log[name].to_csv(os.path.join(self.saving_dir, f'{name}_test_log_.csv'), index=False)

class MultimodalTrainer(RNN_trainer):

    def define_best_criterion(self, checkpoint_criterion):        
        self.model_names_list = list(self.model.get_output_names())
        best_criterion_dict = {}
        for model_name in self.model_names_list:
            if checkpoint_criterion == 'loss':
                # если контролируем loss, то берем изначально большое значение критерия
                best_criterion_dict[model_name] = 9999999
                # функция сравнения текущего значения метрики с лучшим
                #self.is_best_result = lambda x,y: x<y
            else:
                # если контролируем другую метрику (accuracy, recall и т.д.), то берем изначально нулевое значение критерия
                best_criterion_dict[model_name] = 0
                # функция сравнения текущего значения метрики с лучшим
                #self.is_best_result = lambda x,y: x>y

        return best_criterion_dict
    
    def create_batch_results_dict(self, ret_loss, pred_vals, true_vals):
        
        
        # обработка результатов
        modality_labels_dict = {}
        for modality_names, modality_labels_batch in true_vals:
            modality_name = modality_names[0].split('_')[0]
            modality_names = [n.split('_')[-1] for n in modality_names]
            
            modality_names = np.array(modality_names)
            not_empty_tensors = modality_names!='EMPTY'
            modality_names = modality_names[not_empty_tensors]
            if len(modality_names) > 0:
                modality_labels_batch = modality_labels_batch[not_empty_tensors].detach().cpu().numpy()
                modality_labels_dict[modality_name] = modality_labels_batch
                pred_vals[modality_name] = pred_vals[modality_name][not_empty_tensors]
            else:
                try:
                    pred_vals.pop(modality_name)
                except:
                    pass
                try:
                    ret_loss.pop(modality_name)
                except:
                    pass
        
        return {'loss': ret_loss, 'true': modality_labels_dict, 'pred': pred_vals}
    
    def compute_epoch_results(self, epoch_results_list, mode):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        Здесь мы 'парсим' данные, полученные в ходе обучения или тестирования
        '''
        
        if mode =='train':
            dataset_size = self.train_samples_num
        elif mode == 'test':
            dataset_size = self.test_samples_num
        else:
            raise TypeError('mode should be either \'train\' or \'test\'')

        # process metrics
        true = []
        #cummulative_loss = 0
        # В цикле накапливаем значения всех метрик
        models_cummulative_loss_dict = {}
        models_cunmulative_models_preds_dict = {}
        models_cummulative_true_dict = {}
        
        for batch_results in epoch_results_list:
            
            # обработка loss
            for model_name, loss in batch_results['loss'].items():
                try:
                    models_cummulative_loss_dict[model_name] += loss
                except KeyError:
                    models_cummulative_loss_dict[model_name] = loss
            # обработка выходов НС
            for model_name, pred in batch_results['pred'].items():
                try:
                    pred_arr = models_cunmulative_models_preds_dict[model_name]
                    models_cunmulative_models_preds_dict[model_name] = np.concatenate([pred_arr, pred])#.append(pred)
                except KeyError:
                    models_cunmulative_models_preds_dict[model_name] = pred
                
            for model_name, true_labels in batch_results['true'].items():
                try:
                    pred_arr = models_cummulative_true_dict[model_name]
                    models_cummulative_true_dict[model_name] = np.concatenate([pred_arr, true_labels])#.append(pred)
                except KeyError:
                    models_cummulative_true_dict[model_name] = true_labels
            #true.append()

        

        #true = np.concatenate(true)#.reshape(-1)
        #pred = np.concatenate(pred)#.reshape(-1)
                
        #loss = cummulative_loss/dataset_size
        # Словарь, который мы будем добавлять в лог
        log_results_dict = {}

        # compute losses for all the models
        for model_name, loss in models_cummulative_loss_dict.items():
            log_results_dict[model_name] = {'loss': models_cummulative_loss_dict[model_name] / dataset_size}
        
        
        # Строка для вывода на экран
        
        #log_results_dict['loss'] = models_cummulative_loss_dict
        #log_results_dict['loss'] = self.compute_epoch_loss()

        # compute metrics for all the models

        for model_name, pred in models_cunmulative_models_preds_dict.items():
            true = models_cummulative_true_dict[model_name]
            for metric_name in self.metrics_dict.keys():
                if metric_name.lower() != 'loss':
                    metric = self.metrics_dict[metric_name]
                    if type(metric) is dict:
                        metric_func = metric['metric']
                        metric_kwargs = metric['kwargs']
                    else:
                        metric_func = metric
                        metric_kwargs = {}
                    # вычисляем метрику
                    metric_value = metric_func(true, pred, **metric_kwargs)
                    # добавляем метрику в словарь под соответствующим именем модли
                    log_results_dict[model_name][metric_name] = metric_value

        return log_results_dict
    
    def save_model(self, model_name):
        if self.path_to_best_checkpoint is not None:
            if model_name in self.path_to_best_checkpoint:
                os.remove(self.path_to_best_checkpoint[model_name])
        else:
            self.path_to_best_checkpoint = {}
            
        best_weights_name = f'{model_name}_best_ep-{self.current_epoch}.pt'
        path_to_best_checkpoint = os.path.join(self.saving_dir, best_weights_name)
        self.path_to_best_checkpoint[model_name] = path_to_best_checkpoint
        try:
            torch.save(self.model, path_to_best_checkpoint)
        except:
            torch.save(self.model.state_dict(), path_to_best_checkpoint)

    def save_best_weights(self, test_results_dict):
        
        for model_name, test_results in test_results_dict.items():
            best = self.best_criterion[model_name]
            err = test_results[self.checkpoint_criterion]
            
            if self.checkpoint_criterion == 'loss':
                if err < best:
                    print(f'Best results for {model_name} achieved, saving weights...')
                    self.save_model(model_name)
                    self.best_criterion[model_name] = err
            else:
                if err > best:
                    print(f'Best results for {model_name} achieved, saving weights...')
                    self.save_model(model_name)
                    self.best_criterion[model_name] = err

    def init_log(self):
        self.training_log = {}
        self.testing_log = {}
        for name in self.model.get_output_names():
            self.training_log[name] = pd.DataFrame(columns=self.metrics_dict.keys())
            self.testing_log[name] = pd.DataFrame(columns=self.metrics_dict.keys())
    
    def update_datasets(self):
        pass

    


class AudioRNN_trainer(RNN_trainer):
    def update_datasets(self):
        pass

    def prepare_current_checkpoint_path(self):
        if self.path_to_current_checkpoint is not None:
            shutil.rmtree(self.path_to_current_checkpoint)

        current_checkpoint_name = '{}_current_ep-{}'.format(self.model_name, self.current_epoch)
        self.path_to_current_checkpoint = os.path.join(self.saving_dir, current_checkpoint_name)

    def save_checkpoint(self, path_to_current_checkpoint):
        '''
        Сохранение весов при достижении лучшего результата
        '''
        os.makedirs(path_to_current_checkpoint, exist_ok=True)
        # сохраняем только state_dict модели, т.к. весь класс сохранить нельзя
        torch.save(self.model.state_dict(), os.path.join(path_to_current_checkpoint, 'model.pt'))
        # save class attributes
        saving_dict = {}
        for attr_name, attr_val in self.__dict__.items():
            if attr_name != 'model':
                saving_dict[attr_name] = attr_val

        saving_dict['model'] = None

        with open(os.path.join(path_to_current_checkpoint,'trainer_params.pkl'), 'wb') as fd:
            pickle.dump(saving_dict, fd)

        # сохраняем атрибуты класса

    def load_checkpoint(self, path_to_checkpoint_dir):
        if os.path.isdir(self.saving_dir):
            shutil.rmtree(self.saving_dir)
        path_to_trainer_params = os.path.join(path_to_checkpoint_dir, 'trainer_params.pkl')
        with open(path_to_trainer_params, 'rb') as fd:
            loading_dict = pickle.load(fd)

        for attr_name, attr_val in loading_dict.items():
            if attr_name != 'model':
                self.__dict__[attr_name] = attr_val
        
        path_to_state_dict = os.path.join(path_to_checkpoint_dir, 'model.pt')
        self.model.load_state_dict(torch.load(path_to_state_dict, map_location=self.device))


