import torch
from torch import nn



#from trainer import TorchSupervisedTrainer

from tqdm import tqdm

import shutil
import os
import warnings

from datetime import datetime

import pandas as pd

import random
import os
import numpy as np
import matplotlib.pyplot as plt



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

        self.device = device
        self.criterion = criterion
        self.optimizers_list = optimizers_list
        self.lr_schedulers_list = lr_schedulers_list
        self.metrics_dict = metrics_dict
        self.metrics_to_display = metrics_to_display

        # Эпоха, с которой мы начинаем обучение
        self.start_epoch = 0
        #self.current_epoch = self.start_epoch

        #self.metrics_list = metrics_list
        # специфицируется для каждой задачи
        self.training_log_df = pd.DataFrame(columns=metrics_dict.keys())
        self.testing_log_df = pd.DataFrame(columns=metrics_dict.keys())
        #!!!!!!!
        self.checkpoint_criterion = checkpoint_criterion
        self.best_criterion = 999999
        '''
        if checkpoint_criterion == 'loss':
            # если контролируем loss, то берем изначально большое значение критерия
            self.best_criterion = 999999
            # функция сравнения текущего значения метрики с лучшим
            #self.is_best_result = lambda x,y: x<y
        else:
            # если контролируем другую метрику (accuracy, recall и т.д.), то берем изначально нулевое значение критерия
            self.best_criterion = 0
            # функция сравнения текущего значения метрики с лучшим
            #self.is_best_result = lambda x,y: x>y
        '''

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
    
    def nn_output_processing(self, pred):
        '''
        ПЕРЕПИСЫВЕМАЯ ФУНКЦИЯ
        функция, выполняющая постобработку выхода нейронной сети
        '''
        _, pred_labels = torch.max(pred.data, dim=1)
        return pred_labels


    def train_step(self, batch):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        '''
        # отправляем данные на вычислительное устройство
        data, true_vals = batch[0].to(self.device), batch[1].to(self.device)
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
        ret_loss = loss.item() * data.size(0)
        # получение результатов нейронной сети для последующей обработки
        pred_vals = self.nn_output_processing(pred)
        return {'loss': ret_loss, 'true': true_vals.detach().cpu().numpy(), 'pred': pred_vals.detach().cpu().numpy()}

    def test_step(self, batch):
        '''
        ПЕРЕПИСЫВАЕМАЯ ФУНКЦИЯ
        '''
        # Шаг тестироавания
        # отправляем данные на вычислительное устройство
        data, true_vals = batch[0].to(self.device), batch[1].to(self.device)
        # Прямое распространение
        pred = self.model(data)
        # Прямое распространение
        loss = self.criterion(pred, true_vals)
        # Вычисление суммарной ошибке на батче
        ret_loss = loss.item() * data.size(0)
        # получение меток
        pred_vals = self.nn_output_processing(pred)
        return {'loss': ret_loss, 'true': true_vals.cpu().numpy(), 'pred': pred_vals.cpu()}

    
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
        true = np.array(true).reshape(-1)
        pred = np.array(pred).reshape(-1)
        
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
        В случае классификации не очень важно обрабатывать дополнително обрабатывать выходы
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
        torch.save(self, path_to_saving_weights)

    def save_logs(self):
        '''
        Сохранение результатов обучения
        '''
        self.training_log_df.to_csv(os.path.join(self.saving_dir, 'train_log_.csv'))
        self.testing_log_df.to_csv(os.path.join(self.saving_dir, 'test_log_.csv'))

    def train(self, epoch_num):
        end_epoch = self.start_epoch+epoch_num
        for epoch_idx in range(self.start_epoch, end_epoch):
            print('Train epoch # {} of {} epochs...'.format(epoch_idx, end_epoch-1))
            #t0 = time.time()
            self.model.train()
            # список, куда будем записывать промежуточные результаты эпохи
            train_raw_result_list = []
            for batch in tqdm(self.train_loader):
                # обучение на одном батче
                train_results = self.compute_batch_results(self.train_step(batch))
                train_raw_result_list.append(train_results)

            #t1 = time.time()
            #print('passed in {:.3f} seconds'.format(t1 - t0))
            # парсим результаты на обучающей выборке
            train_results_dict = self.compute_epoch_results(train_raw_result_list, mode='train')

            # выводим обучающие результаты на экран
            self.print_result(train_results_dict)
            
            # сохраняем результат тренировочных метрик эпохи
            self.training_log_df = pd.concat([self.training_log_df, pd.DataFrame([train_results_dict])], ignore_index=True)
            
            # запускаем процедуру тестирования
            self.test_raw_result_list = self.test()
            # парсим результаты на тестовой выборке
            test_results_dict = self.compute_epoch_results(self.test_raw_result_list, mode='test')

            # выводим тестовые результаты на экран
            self.print_result(test_results_dict)

            # сохраняем результат тестировочных метрик эпохи
            self.testing_log_df = pd.concat([self.testing_log_df, pd.DataFrame([test_results_dict])], ignore_index=True)

            # сохраняем результаты обучения после каждой итерации
            self.save_logs()

            # обновляем стартовую эпоху, чтобы иметь возможность восстановить модель и продолжить обучение с
            #  той эпохи, с которой мы это обучение прекратили
            self.start_epoch = epoch_idx+1

            # save weights of current step
            if self.path_to_current_checkpoint is not None:
                os.remove(self.path_to_current_checkpoint)

            current_weights_name = '{}_current_ep-{}.pt'.format(self.model_name, epoch_idx)
            self.path_to_current_checkpoint = os.path.join(self.saving_dir, current_weights_name)
            self.save_checkpoint(self.path_to_current_checkpoint)

            # обновляем каждый планировщик скорости обучения из списка
            for lr_scheduler in self.lr_schedulers_list:
                # не обновляем планировщик, если его нет)
                if lr_scheduler is not None:
                    lr_scheduler.step()
            
            # сохраняем лучшие веса
            if self.checkpoint_criterion == 'loss':
                err = test_results_dict[self.checkpoint_criterion]
            else:
                err = 1 - test_results_dict[self.checkpoint_criterion]
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

            # start testing procedure
            print('----------------------------------------------')

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

        #t1 = time.time()
        #print('passed in {:.3f} seconds'.format(t1 - t0))
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
                len(self.training_log_df[metric_name][0])
                warnings.warn('WARNING!\n\'{}\' metric will not be displayed due to it has multiple values.'.format(metric_name))
                metrics_list.remove(metric_name)
            except TypeError:
                if multiple_plots:
                    if train_test_sep:
                        title = '{} {} {}'.format(self.model_name, metric_name.capitalize(), 'train')
                        plt.figure(figsize=(6,6),facecolor='white')
                        plt.title(title)
                        plt.xlabel('Epochs')
                        plt.plot(self.training_log_df[metric_name])
                        plt.savefig(os.path.join(self.saving_dir, title+'.png'))
                        plt.show()
                        

                        title = '{} {} {}'.format(self.model_name, metric_name.capitalize(), 'test')
                        plt.figure(figsize=(6,6),facecolor='white')
                        plt.title(title)
                        plt.xlabel('Epochs')
                        plt.plot(self.testing_log_df[metric_name])
                        plt.savefig(os.path.join(self.saving_dir, title+'.png'))
                        plt.show()
                        
                    else:
                        title = '{} {}'.format(self.model_name, metric_name.capitalize())
                        plt.figure(figsize=(6,6),facecolor='white')
                        plt.title(title)
                        plt.xlabel('Epochs')
                        plt.plot(self.training_log_df[metric_name])
                        plt.plot(self.testing_log_df[metric_name])
                        plt.legend(['Train', 'Test'])
                        plt.savefig(os.path.join(self.saving_dir, title+'.png'))
                        plt.show()
                        
        if not multiple_plots:
            # Если мы печатаем раздельно для обучающей и для тестовой выборки
            if train_test_sep:
                fig, axs = plt.subplots(1, len(metrics_list)*2, figsize=(len(metrics_list)*2*6, 6),facecolor='white')
            else:
                fig, axs = plt.subplots(1, len(metrics_list), figsize=(len(metrics_list)*6, 6),facecolor='white')

            #print(len(axs))
            for plot_idx, metric_name in enumerate(metrics_list):
                if train_test_sep:
                    axs[plot_idx*2].set_title('{} {} {}'.format(self.model_name, metric_name.capitalize(), 'train'))
                    axs[plot_idx*2].set_xlabel('Epochs')
                    axs[plot_idx*2].plot(self.training_log_df[metric_name])
                    axs[plot_idx*2+1].set_title('{} {} {}'.format(self.model_name, metric_name.capitalize(), 'test'))
                    axs[plot_idx*2+1].set_xlabel('Epochs')
                    axs[plot_idx*2+1].plot(self.testing_log_df[metric_name])
                else:
                    axs[plot_idx].set_title('{} {}'.format(self.model_name, metric_name.capitalize()))
                    axs[plot_idx].set_xlabel('Epochs')
                    axs[plot_idx].plot(self.training_log_df[metric_name])
                    axs[plot_idx].plot(self.testing_log_df[metric_name])
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
        #print('DEBUG!')
        #print(true, pred)
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
        
        #print(cummulative_loss)
        loss = cummulative_loss/dataset_size
        # Строка для вывода на экран
        # Словарь, который мы будем добавлять в лог
        log_results_dict = {}
        log_results_dict['loss'] = loss
        log_results_dict['confusion_matrix'] = cumulative_confusion

        #log_results_dict['loss'] = self.compute_epoch_loss()

        for metric_name in self.metrics_dict.keys():
            
            if not (metric_name.lower() == 'loss' or metric_name.lower() == 'confusion_matrix'):
                #print(metric_name)
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