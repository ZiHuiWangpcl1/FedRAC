import flgo
from my_algorithm import (CF_fedavg, FedRAC, FedRAC_ResNet, FedSAC, FedSAC_Resnet,
                          standalone, cffl, rffl,hffl, CF_qffl, FedAVE, IAFL, fedfv)
from flgo.algorithm import fedavg
import flgo.experiment.analyzer
import os
import numpy as np
from flgo.experiment.logger import BasicLogger
import time
import csv
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr
import time
import torch

# MLP
import flgo.benchmark.cifar10_classification.model.mlp as cifar10_mlp
import flgo.benchmark.svhn_classification.model.mlp as svhn_mlp
import flgo.benchmark.emnist_classification.model.mlp as emnist_mlp

# ResNet-18
from my_utils import imagenet_tiny_resnet18 as imagenet_tiny_resnet18

from my_utils.partition import POW, CLA, DirichletPartitioner


# CIFAR10
# task = './my_task/CIFAR10_POW'
# task = './my_task/CIFAR10_CLA'
# task = './my_task/CIFAR10_DIR(10.0)'

# SVHN
task = './my_task/SVHN_POW'
# task = './my_task/SVHN_CLA'
# task = './my_task/SVHN_DIR(10.0)'


gen_config = {
    'benchmark':
    # {'name':'my_benchmark.cifar10_classification'},
    {'name':'my_benchmark.svhn_classification'},
    # {'name':'my_benchmark.emnist_classification'},
    # {'name':'my_benchmark.imagenet_tiny_classification'},

    'partitioner':
    {
    # 'name':DirichletPartitioner,
    # 'para':{'num_clients':10,'alpha': 3.0}

    # 'name':DirichletPartitioner,
    # 'para':{'num_clients':10,'alpha': 7.0}

    'name': POW,
    'para': {'num_clients': 10, }

    # 'name': CLA,
    # 'para': {'num_clients': 10, }
     }
}

TensorWriter = SummaryWriter('./log/1001')

standalone_test = {
    'ImageNetTiny_POW': [0.2784, 0.3415, 0.3587, 0.3521, 0.3780, 0.3829, 0.3936, 0.3897, 0.4063, 0.4030],
    'Fashion_POW': [0.2784, 0.3415, 0.3587, 0.3521, 0.3780, 0.3829, 0.3936, 0.3897, 0.4063, 0.4030],
    'EMNIST_POW' :       [0.2784, 0.3415, 0.3587, 0.3521, 0.3780, 0.3829, 0.3936, 0.3897, 0.4063, 0.4030],
    'SVHN_POW' :          [0.3197, 0.4601, 0.5149, 0.5739, 0.6220, 0.6273, 0.6342, 0.6539, 0.6653, 0.6835],

    'CIFAR10_POW': [0.2784, 0.3415, 0.3587, 0.3521, 0.3780, 0.3829, 0.3936, 0.3897, 0.4063, 0.4030],
    'CIFAR10_CLA': [0.1000, 0.1702, 0.2221, 0.2563, 0.2770, 0.2879, 0.3019, 0.3274, 0.3557, 0.3665],
    'CIFAR10_DIR(3.0)': [0.3063, 0.2513, 0.2791, 0.2781, 0.2861, 0.2775, 0.2453, 0.2865, 0.3148, 0.3064],
    'CIFAR10_DIR(7.0)': [0.3720, 0.3781, 0.3167, 0.3868, 0.3331, 0.3518, 0.3458, 0.3759, 0.3640, 0.3783],
}



class MyLogger(BasicLogger):
    def initialize(self):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local data size)"""
        for c in self.participants:
            self.output['client_datavol'].append(len(c.train_data))
        self.acc_0, self.acc_1, self.acc_2, self.acc_3, self.acc_4, self.acc_5, self.acc_6, self.acc_7, self.acc_8, self.acc_9 = [], [], [], [], [], [], [], [], [], []

    """This logger only records metrics on validation dataset"""
    def log_once(self, round=None, corrs_agg={},*args, **kwargs):

        self.info('Current_time:{}'.format(self.clock.current_time))
        self.output['time'].append(self.clock.current_time)

        test_metric = self.coordinator.test()

        print("Testing Accuracy:", test_metric['accuracy'])
        print("Min of testing Accuracy:", np.min(test_metric['accuracy']))
        print("Max of testing Accuracy:", np.max(test_metric['accuracy']))

        valid_metrics = self.coordinator.global_test('valid')
        local_data_vols = [c.datavol for c in self.participants]
        total_data_vol = sum(local_data_vols)
        train_metrics = self.coordinator.global_test('train')

        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        for met_name, met_val in valid_metrics.items():
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        # output to stdout
        # self.show_current_output()

        standalone_test_acc = standalone_test[task.split('/')[-1]]

        corrs = pearsonr(standalone_test_acc, test_metric['accuracy'])
        print("corrs:", corrs[0])

        alpha = 0.5
        rate = 0
        rate_flag = ['0'] * 10

        for i in range(len(standalone_test_acc)):
            right_limit = ((1 - alpha) * (standalone_test_acc[i] / max(standalone_test_acc)) + alpha) * max(
                test_metric['accuracy'])

            if standalone_test_acc[i] < test_metric['accuracy'][i] and test_metric['accuracy'][i] <= right_limit:
                rate += 1
                rate_flag[i] = '+1'
            elif test_metric['accuracy'][i] <= standalone_test_acc[i]:
                rate = -10
                rate_flag[i] = '-10'
        rate = rate / len(standalone_test_acc)
        print("rate:", rate)
        print('rate_flag:', rate_flag)

        TensorWriter.add_scalar('Train Loss', self.output['train_loss'][-1], round)
        # TensorWriter.add_scalar('Valid Loss', self.output['valid_loss'][-1], round)
        TensorWriter.add_scalar('Max of test Accuracy', np.max(test_metric['accuracy']), round)
        TensorWriter.add_scalar('Min of test Accuracy', np.min(test_metric['accuracy']), round)
        TensorWriter.add_scalar('corrs', corrs[0], round)
        TensorWriter.add_scalar('rate', rate, round)

        corrs_agg[round] = corrs[0]
        corrs_agg = sorted(corrs_agg.items(), key = lambda kv:kv[1], reverse = True)
        if len(corrs_agg) >= 10 :
            max_corrs = corrs_agg[0:9:1]
            print("max_corrs:", max_corrs)

if __name__ == '__main__':
    # generate federated task if task doesn't exist
    if not os.path.exists(task): flgo.gen_task(gen_config, task_path=task)

    torch.set_num_threads(1)

    #--------------------------- MLP ---------------------------------

    # runner = flgo.init(task, algorithm = CF_fedavg,
    #                    option={'gpu':[1,], 'train_holdout': 0.1, 'log_file':True, 'pin_memory':True},
    #                    Logger=MyLogger, model=cifar10_mlp)
    # runner.run()

    runner = flgo.init(task, algorithm = FedRAC,
                       option={'gpu':[1,], 'train_holdout': 0.1, 'log_file':True, 'pin_memory':True},
                       Logger=MyLogger, model=FedRAC)
    runner.run()

    # --------------------------- ResNet ---------------------------------

    # runner = flgo.init(task, algorithm = CF_fedavg,
    #                    option={'gpu':[1,], 'train_holdout': 0.1, 'log_file':True, 'pin_memory':True},
    #                    Logger=MyLogger, model=imagenet_tiny_resnet18)
    # runner.run()

    # runner = flgo.init(task, algorithm=FedRAC_ResNet,
    #                    option={'gpu': [1, ], 'train_holdout': 0.1, 'log_file': True, 'pin_memory': True},
    #                    Logger=MyLogger, model=FedRAC_ResNet)
    # runner.run()

    # ------------ Tune parameters using grid search --------------------

    # option = {
    #     'learning_rate' : [0.001, 0.003, 0.01, 0.03],
    #     'batch_size' : [64,],
    #     'num_steps' : [1, 5, 10],
    #     'no_log_console' : True,
    #     'gpu' : [1],
    #     'train_holdout' : [0.1],
    #     'early_stop' : 100,
    # }
    #
    # import flgo.experiment.device_scheduler as ds
    # asc = ds.AutoScheduler([1])
    # # flgo.tune(task, CF_fedavg, option, Logger=MyLogger, model=cifar10_vit_base, scheduler=asc)
    # flgo.tune(task, CF_fedavg, option, Logger=MyLogger, model=cifar100_vit_base, scheduler=asc)
