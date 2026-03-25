from flgo.algorithm.fedbase import BasicServer, BasicClient
import copy
import torch
import collections
import numpy as np
from my_utils.myfflow import compute_grad_update, flatten, unflatten,\
                                mask_grad_update_by_order,add_gradient_updates,\
                                add_update_to_model
from flgo.utils.fmodule import FModule
import flgo.utils.fmodule as fmodule
from torch import nn
import torch.nn.functional as F 
from flgo.utils import fmodule
import torch.multiprocessing as mp
import time
import random
import csv

def _modeldict_mul(md1, md2):
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] * md2[layer]
    return res

def _model_mul(m1, m2):
    res = m1.__class__().to(m1.get_device())
    fmodule._modeldict_cp(res.state_dict(), _modeldict_mul(m1.state_dict(), m2.state_dict()))
    return res

class cifar_mlp(FModule):
    def __init__(self, dim_in=3*32*32, dim_hidden=200, dim_out=10):
        super(cifar_mlp, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(dim_hidden, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.map  = None

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        if self.map is not None:
            x = x * self.map[0]      # 第一层子模型进行更新，x=[_,200]
        x = self.fc1(x)
        x = self.relu(x)
        if self.map is not None:
            x = x * self.map[1]   
        return x
    
    def set_map(self, map=None):
        self.map = map

class SHVH_mlp(FModule):
    def __init__(self, dim_in=3*32*32, dim_hidden=200, dim_out=10):
        super(SHVH_mlp, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(dim_hidden, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.map = None

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        if self.map is not None:
            x = x * self.map[0]   
        x = self.fc1(x)
        x = self.relu(x)
        if self.map is not None:
            x = x * self.map[1]   
        return x

    def set_map(self, map=None):
        self.map = map

class emnist_mlp(FModule):
    def __init__(self, device=None):
        super(emnist_mlp, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 62)
        self.map = None

    def forward(self, x):
        x = self.get_embedding(x)  # 得到通过map嵌入后的结果
        x = self.fc3(x)
        return x

    def get_embedding(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])  # 展平输入
        x = self.fc1(x)
        x = F.relu(x)
        # 特征映射
        if self.map is not None:
            x = x * self.map[0]
        x = self.fc2(x)
        x = F.relu(x)
        # 特征映射
        if self.map is not None:
            x = x * self.map[1]
        return x

    def set_map(self, map=None):
        self.map = map

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        if 'CIFAR10' in object.option['task']:
            object.model = cifar_mlp().to(object.device)
        elif 'SVHN' in object.option['task']:
            object.model = SHVH_mlp().to(object.device)
        elif 'EMNIST' in object.option['task']:
            object.model = emnist_mlp().to(object.device)
        else:
            object.model = None
    else:
        object.model = None


standalone_test = {
    'EMNIST_POW' :       [0.2784, 0.3415, 0.3587, 0.3521, 0.3780, 0.3829, 0.3936, 0.3897, 0.4063, 0.4030],

    'CIFAR10_POW' :       [0.2819, 0.3426, 0.3600, 0.3544, 0.3791, 0.3897, 0.3975, 0.4010, 0.4050, 0.4068],
    'CIFAR10_CLA' :       [0.1000, 0.1716, 0.2230, 0.2569, 0.2767, 0.2897, 0.3071, 0.3319, 0.3565, 0.3690],
    'CIFAR10_DIR(1.5)' :  [0.1644, 0.2697, 0.2374, 0.3113, 0.3382, 0.2196, 0.2285, 0.2232, 0.2825, 0.2459],
    'CIFAR10_DIR(3.0)' :  [0.3076, 0.2543, 0.2832, 0.2830, 0.2856, 0.2798, 0.2459, 0.2905, 0.3213, 0.3079],
    'CIFAR10_DIR(5.0)' :  [0.3152, 0.3432, 0.3495, 0.3302, 0.3488, 0.3308, 0.3188, 0.3040, 0.3251, 0.3628],
    'CIFAR10_DIR(6.0)' :  [0.2639, 0.3539, 0.3179, 0.3404, 0.3596, 0.3502, 0.3232, 0.3284, 0.3983, 0.3294],
    'CIFAR10_DIR(7.0)' :  [0.3751, 0.3889, 0.3213, 0.3907, 0.3362, 0.3597, 0.3508, 0.3797, 0.3676, 0.3840],
    'CIFAR10_DIR(8.0)' :  [0.2989, 0.3692, 0.3667, 0.3799, 0.3906, 0.3538, 0.3736, 0.3492, 0.3610, 0.3668],
    'CIFAR10_DIR(9.0)' :  [0.3115, 0.3020, 0.3795, 0.4055, 0.3988, 0.3352, 0.3708, 0.3557, 0.3914, 0.3519],
    'CIFAR10_DIR(10.0)' : [0.3702, 0.3434, 0.3990, 0.3597, 0.3777, 0.3543, 0.3734, 0.3782, 0.3270, 0.3521],

    'SVHN_POW' :          [0.3195, 0.4602, 0.5132, 0.5774, 0.6203, 0.6272, 0.6334, 0.6531, 0.6661, 0.6833],
    'SVHN_CLA' :          [0.0669, 0.2397, 0.3542, 0.4127, 0.4703, 0.5101, 0.5460, 0.5710, 0.5898, 0.5913],
    'SVHN_DIR(1.5)' :     [0.2725, 0.3837, 0.3503, 0.3953, 0.3588, 0.4160, 0.3332, 0.3164, 0.3529, 0.3077],
    'SVHN_DIR(2.0)' :     [0.2702, 0.5863, 0.4255, 0.3279, 0.4140, 0.3937, 0.4240, 0.4311, 0.2674, 0.3566],
    'SVHN_DIR(3.0)' :     [0.6143, 0.4030, 0.6120, 0.6125, 0.5492, 0.4491, 0.4834, 0.5001, 0.4413, 0.4927],
    'SVHN_DIR(4.0)' :     [0.4985, 0.4252, 0.5056, 0.5629, 0.5549, 0.4613, 0.5067, 0.5269, 0.4544, 0.6095],
    'SVHN_DIR(5.0)' :     [0.3974, 0.5397, 0.5576, 0.6710, 0.6420, 0.5718, 0.5502, 0.5523, 0.5819, 0.5546],
    'SVHN_DIR(6.0)' :     [0.6067, 0.6398, 0.5800, 0.6033, 0.6238, 0.5176, 0.5520, 0.4780, 0.6408, 0.5798],
    'SVHN_DIR(7.0)' :     [0.5768, 0.5282, 0.5873, 0.6341, 0.5697, 0.4377, 0.6298, 0.6244, 0.4722, 0.6112],
    'SVHN_DIR(8.0)' :     [0.6420, 0.5901, 0.4962, 0.6341, 0.6424, 0.4421, 0.6507, 0.6374, 0.5712, 0.6593],
    'SVHN_DIR(9.0)' :     [0.5359, 0.5858, 0.6557, 0.5953, 0.6042, 0.5013, 0.6523, 0.6778, 0.6358, 0.6772],
    'SVHN_DIR(10.0)' :    [0.6486, 0.5671, 0.6353, 0.6377, 0.5905, 0.6597, 0.6023, 0.6670, 0.6498, 0.6699],
}

class Server(BasicServer):  
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'Beta': 80.0, 't':100, 'alpha': 0.4, 'k': 100,  'ce':'oracle', 'ne':10, 'train_on_all': True})
        self.optimizer_name = self.option['optimizer']
        self.learning_rate = self.option['learning_rate']
        self.weight_decay = self.option['weight_decay']
        self.momentum = self.option['momentum']
        self.map = [[torch.ones(200).to(self.device), torch.ones(200).to(self.device)] for i in range(10)]
        self.standalone_test_acc = torch.tensor(standalone_test[self.option['task'].split('/')[-1]]).to(self.device)
        print("self.standalone_test_acc", self.standalone_test_acc)

        self.freq_stats_file = 'FedRAC_80_neuron_frequency_stats.csv'

        with open(self.freq_stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['round'] + [f'neuron_{i}' for i in range(400)] + ['mean', 'variance'])

        tmp = torch.tensor([torch.exp((xi)*self.Beta) for xi in self.standalone_test_acc])
        self.client_contributions = tmp/tmp.max()*100
        self.sample_option = 'full'
        print("self.client_contributions", self.client_contributions)
        self.c = 0
        self.all_map_frenquency = torch.zeros(1, 400).to(self.device)



    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')
        self.gv.logger.info("--------------Initial Evaluation--------------")
        while self.current_round <= self.num_rounds:
            self.gv.clock.step()
            # iterate
            self.gv.logger.time_start('Eval Time Cost')

            updated = self.iterate()

            # using logger to evaluate the model if the model is updated
            if updated is True or updated is None:
                self.gv.logger.info("--------------Round {}--------------".format(self.current_round))
                # check log interval
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.log_once(round=self.current_round)  # 单次触发式日志方法
                    self.gv.logger.time_end('Eval Time Cost')
                # check if early stopping
                if self.gv.logger.early_stop(): break
                self.current_round += 1
            # decay learning rate
            self.global_lr_scheduler(self.current_round)
            # clear package buffer
            self.sending_package_buffer = [None for _ in self.clients]
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return

    def iterate(self):
        if 1< self.current_round <= self.t:
            model_glo = copy.deepcopy(self.model)
            map_glo = [torch.ones(200).to(self.device), torch.ones(200).to(self.device)]
            model_glo.set_map(map_glo)
            acc_glo = self.calculator.test(model_glo, self.validation_data, batch_size = self.option['test_batch_size'])['accuracy']
            contri_client_values, contri_client_indices = torch.sort(self.client_contributions)
            all_map_frequency_values, all_map_frequency_indices = torch.sort(self.all_map_frenquency)

            all_submodel_acc = []
            index_map = 0
            pack_step = 8

            a = time.time()

            for i in range(400//pack_step):

                map_c_s = [torch.zeros(400).to(self.device)]
                index_map += pack_step
                
                map_c_s[0][all_map_frequency_indices[0][0:index_map]] = 1 # 根据排序后的索引来设置map_c_s中对应位置为1
                map_tem = [map_c_s[0][:200], map_c_s[0][200:]]

                model_glo.set_map(map_tem)
                # tem_submodel_acc = self.calculator.test(model_glo, self.validation_data, batch_size = self.option['test_batch_size'])['accuracy']
                tem_submodel_acc = self.calculator.test(model_glo, self.validation_data, batch_size = 1024)['accuracy']
                all_submodel_acc.append(tem_submodel_acc)

            b = time.time()
            print("=======", b-a)

            indexs_all = [0 for i in range(len(self.standalone_test_acc))]
            index_tmp = 0

            for i in range(len(self.standalone_test_acc)):

                if i == 9:
                    indexs_all[i] = 400

                else:
                    al = 0.20 - 0.20/(torch.log(torch.tensor(0.3 * self.current_round)+ 3))
                    # al = acc_glo * 0.05 + 0.4
                    # al = self.alpha
                    min_acc = acc_glo * al
                    for s in range(400//pack_step-1, index_tmp, -1):
                        # if all_submodel_acc[s] <= acc_glo * contri_client_values[i] /100:
                        if all_submodel_acc[s] <= self.standalone_test_acc[contri_client_indices[i]]  + (acc_glo - min_acc) * contri_client_values[i] /100:
                            break
                        else:
                            continue

                    indexs_all[i] = pack_step*(s)
                    index_tmp = s

            for i in range(len(self.standalone_test_acc)):
                map_s = [torch.zeros(400).to(self.device)]

                map_s[0][all_map_frequency_indices[0][0:indexs_all[i]]] = 1
                map_tem = [map_s[0][:200], map_s[0][200:]]
                
                self.map[contri_client_indices[i]] = map_tem

                self.all_map_frenquency[0] += map_s[0]

            print("self.all_map_frenquency", self.all_map_frenquency[0])
            frequencies = self.all_map_frenquency[0].cpu().numpy()
            mean_freq = np.mean(frequencies)
            var_freq = np.var(frequencies)

            with open(self.freq_stats_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.current_round] + list(frequencies) + [mean_freq, var_freq])

            print(f"Round {self.current_round} - Mean: {mean_freq:.4f}, Var: {var_freq:.4f}")
            # ==========================================================

        elif self.current_round > self.t:
            model_glo = copy.deepcopy(self.model)
            map_glo = [torch.ones(200).to(self.device), torch.ones(200).to(self.device)]
            model_glo.set_map(map_glo)
            acc_glo = self.calculator.test(model_glo, self.validation_data, batch_size = self.option['test_batch_size'])['accuracy']

            contri_client_values, contri_client_indices = torch.sort(self.client_contributions)

            all_map_frequency_values, all_map_frequency_indices = torch.sort(self.all_map_frenquency)

            all_submodel_acc = []
            index_map = 0
            pack_step = 8

            a = time.time()

            for i in range(400//pack_step):
                map_c_s = [torch.zeros(400).to(self.device)]
                index_map += pack_step
                
                map_c_s[0][all_map_frequency_indices[0][0:index_map]] = 1 # 根据排序后的索引来设置map_c_s中对应位置为1
                map_tem = [map_c_s[0][:200], map_c_s[0][200:]]

                model_glo.set_map(map_tem)
                tem_submodel_acc = self.calculator.test(model_glo, self.validation_data, batch_size = self.option['test_batch_size'])['accuracy']

                all_submodel_acc.append(tem_submodel_acc)

            b = time.time()
            print("=======", b-a)

            indexs_all = [0 for i in range(len(self.standalone_test_acc))]
            index_tmp = 0

            for i in range(len(self.standalone_test_acc)):

                if i == 9:
                    indexs_all[i] = 400

                else:
                    al = 0.80 - 0.80/(torch.log(torch.tensor(0.3 * self.current_round)+ 3))
                    min_acc = acc_glo * al
                    for s in range(400//pack_step-1, index_tmp, -1):
                        if all_submodel_acc[s] <= self.standalone_test_acc[contri_client_indices[i]]  + (acc_glo - min_acc) * contri_client_values[i] /100:
                            break
                        else:
                            continue
                    indexs_all[i] = pack_step*(s)
                    index_tmp = s
            for i in range(len(self.standalone_test_acc)):
                map_s = [torch.zeros(400).to(self.device)]
                map_s[0][all_map_frequency_indices[0][0:indexs_all[i]]] = 1
                map_tem = [map_s[0][:200], map_s[0][200:]]
                self.map[contri_client_indices[i]] = map_tem
                self.all_map_frenquency[0] += map_s[0]

            print("self.all_map_frenquency", self.all_map_frenquency[0])

            frequencies = self.all_map_frenquency[0].cpu().numpy()
            mean_freq = np.mean(frequencies)
            var_freq = np.var(frequencies)

            with open(self.freq_stats_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.current_round] + list(frequencies) + [mean_freq, var_freq])

            print(f"Round {self.current_round} - Mean: {mean_freq:.4f}, Var: {var_freq:.4f}")

        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        model_updates = [mi-self.model for mi in models]
        update_tensors = [fmodule._model_to_tensor(ui) for ui in model_updates]
        model_masks = [torch.gt(torch.abs(update_i), 0).int() for update_i in update_tensors]  
        sum_model_masks = torch.where(torch.isinf(1.0/sum(model_masks)), torch.full_like(1.0/sum(model_masks), 0), 1.0/sum(model_masks))
        mask = fmodule._model_from_tensor(sum_model_masks, self.model.__class__)
        update = fmodule._model_sum(model_updates)

        final_update = _model_mul(mask, update)
        self.model = self.model + final_update

        return

    def pack(self, client_id, map, mtype=0, *args, **kwargs):
        m = copy.deepcopy(self.model)
        m.set_map(copy.deepcopy(self.map[client_id]))
        return {"model" : m}

    def validation_(self, model=None, flag='test'):
        """
        Evaluate the model on the test dataset owned by the server.
        Args:
            model (flgo.utils.fmodule.FModule): the model need to be evaluated
        Returns::
            metrics (dict): specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if model is None: model=self.model
        if flag == 'valid': dataset = self.validation_data
        else: dataset = self.train_data
        if dataset is None: return {}
        else:
            return self.calculator.test(model, dataset, batch_size = self.option['test_batch_size'], num_workers = self.option['num_workers'], pin_memory = self.option['pin_memory'])

    def global_test(self, dataflag='valid'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            round: the current communication round
            dataflag: choose train data or valid data to evaluate
        :return
            evals: the evaluation metrics of the global model on each client's dataset
            loss: the loss of the global model on each client's dataset
        """
        all_metrics = collections.defaultdict(list)
        model = self.model
        for i,c in enumerate(self.clients):
            model.set_map(self.map[i])
            client_metrics = c.test(model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)

        return all_metrics


    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            the metric and loss of the model on the test data
        """
        if model==None: model=self.model
        all_metrics = collections.defaultdict(list)
        for i in range(self.num_clients):
            model.set_map(self.map[i])
            if self.test_data:
                client_metrics = self.calculator.test(model, self.test_data, batch_size = self.option['test_batch_size'])

                for met_name, met_val in client_metrics.items():
                    all_metrics[met_name].append(met_val)
        return all_metrics



class Client(BasicClient):

    def initialize(self):
        if self.train_on_all:
            self.train_data.indices = self.train_data.indices + self.valid_data.indices
        return
