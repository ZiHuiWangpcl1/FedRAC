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
import torchvision

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=out_channels * BasicBlock.expansion),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=2, num_channels=out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class imagenet_tiny_resnet18(fmodule.FModule):
    """
    CIFAR-style ResNet18 for Tiny-ImageNet
    - 3x3 conv1, no maxpool
    - GroupNorm replaces all BatchNorm
    - Retains map/neuron_importance_list/activation for old code compatibility
    """
    def __init__(self, num_classes=200):
        super().__init__()
        self.in_channels = 64

        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=2, num_channels=64),
            nn.ReLU(inplace=True)
        )

        # residual layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 兼容旧接口
        self.map = None
        self.neuron_importance_list = []
        self.activaton = nn.Sigmoid()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # conv1
        x = self.conv1(x)
        if self.map is not None:
            x = x * torch.repeat_interleave(
                self.map[0], repeats=x.shape[-1]*x.shape[-2]
            ).reshape(len(self.map[0]), x.shape[-2], x.shape[-1])

        # residual layers
        x = self.layer1(x)
        if self.map is not None:
            x = x * torch.repeat_interleave(
                self.map[1], repeats=x.shape[-1]*x.shape[-2]
            ).reshape(len(self.map[1]), x.shape[-2], x.shape[-1])

        x = self.layer2(x)
        if self.map is not None:
            x = x * torch.repeat_interleave(
                self.map[2], repeats=x.shape[-1]*x.shape[-2]
            ).reshape(len(self.map[2]), x.shape[-2], x.shape[-1])

        x = self.layer3(x)
        if self.map is not None:
            x = x * torch.repeat_interleave(
                self.map[3], repeats=x.shape[-1]*x.shape[-2]
            ).reshape(len(self.map[3]), x.shape[-2], x.shape[-1])

        x = self.layer4(x)
        x = self.avgpool(x)
        if self.map is not None:
            x = x * torch.repeat_interleave(
                self.map[4], repeats=x.shape[-1]*x.shape[-2]
            ).reshape(len(self.map[4]), x.shape[-2], x.shape[-1])

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def set_map(self, map=None):
        self.map = map

class cifar10_resnet18(FModule):
    def __init__(self, num_classes=10):
        super(cifar10_resnet18, self).__init__()

        resnet18 = torchvision.models.resnet18()
        resnet18.fc = nn.Linear(512, 10)
        resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.model = resnet18

        self.map = None
        self.neuron_importance_list = []
        self.activaton = nn.Sigmoid()

    def _init_weights(self, m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.model.bn1(self.model.conv1(x)))
        if self.map is not None:  # x=[32,64,_,_]
            x = x * torch.repeat_interleave(self.map[0], repeats=x.shape[-1] * x.shape[-2]).reshape(len(self.map[0]), x.shape[-2], x.shape[-1])

        x = self.model.layer1(x)
        if self.map is not None:
            x = x * torch.repeat_interleave(self.map[1], repeats=x.shape[-1] * x.shape[-2]).reshape(len(self.map[1]), x.shape[-2], x.shape[-1])

        x = self.model.layer2(x)
        if self.map is not None:
            x = x * torch.repeat_interleave(self.map[2], repeats=x.shape[-1] * x.shape[-2]).reshape(len(self.map[2]), x.shape[-2], x.shape[-1])

        x = self.model.layer3(x)
        if self.map is not None:
            x = x * torch.repeat_interleave(self.map[3], repeats=x.shape[-1] * x.shape[-2]).reshape(len(self.map[3]), x.shape[-2], x.shape[-1])

        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        if self.map is not None:
            x = x * torch.repeat_interleave(self.map[4], repeats=x.shape[-1] * x.shape[-2]).reshape(len(self.map[4]), x.shape[-2], x.shape[-1])

        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        return x
    
    def set_map(self, map=None):
        self.map = map

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        if 'CIFAR10' in object.option['task']:
            object.model = cifar10_resnet18().to(object.device)
        elif 'ImageNetTiny' in object.option['task']:
            object.model = imagenet_tiny_resnet18().to(object.device)
        else:
            object.model = None
    else:
        object.model = None

# ResNet
standalone_test = {
    #ResNet18
    'ImageNetTiny_POW' :       [0.2692, 0.3426, 0.3741, 0.3901, 0.4156, 0.4340, 0.4257, 0.4548, 0.4535, 0.4693],
    'CIFAR10_POW' :       [0.2692, 0.3426, 0.3741, 0.3901, 0.4156, 0.4340, 0.4257, 0.4548, 0.4535, 0.4693],
    'CIFAR10_CLA' :       [0.1000, 0.1781, 0.2401, 0.2819, 0.2947, 0.3092, 0.3404, 0.3760, 0.4004, 0.4218],
    'CIFAR10_DIR(1.5)' :  [0.1680, 0.2715, 0.2577, 0.3359, 0.3684, 0.2267, 0.2410, 0.2449, 0.3007, 0.2571],
    'CIFAR10_DIR(2.0)' :  [0.3110, 0.2066, 0.3104, 0.2727, 0.3247, 0.2512, 0.3492, 0.3190, 0.3608, 0.3001],
    'CIFAR10_DIR(3.0)' :  [0.3185, 0.2626, 0.3145, 0.3075, 0.3027, 0.2931, 0.2580, 0.3279, 0.3371, 0.3364],
    'CIFAR10_DIR(4.0)' :  [0.3903, 0.3797, 0.4110, 0.3852, 0.4299, 0.3571, 0.3470, 0.3062, 0.3572, 0.3134],
    'CIFAR10_DIR(5.0)' :  [0.3752, 0.3912, 0.3939, 0.3556, 0.3942, 0.3629, 0.3464, 0.3291, 0.3620, 0.3907],
    'CIFAR10_DIR(6.0)' :  [0.2878, 0.3950, 0.3527, 0.3603, 0.3853, 0.3793, 0.3648, 0.3573, 0.4456, 0.3641],
    'CIFAR10_DIR(8.0)' :  [0.3148, 0.4136, 0.4223, 0.4102, 0.4246, 0.3935, 0.4173, 0.3746, 0.4039, 0.4183],

}

class Server(BasicServer):  
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'Beta': 10.0, 't':200, 'alpha': 0.4, 'k': 100,  'ce':'oracle', 'ne':10, 'train_on_all': True})
        self.optimizer_name = self.option['optimizer']
        self.learning_rate = self.option['learning_rate']
        self.weight_decay = self.option['weight_decay']
        self.momentum = self.option['momentum']
        self.map = [[torch.ones(64).to(self.device), torch.ones(64).to(self.device),torch.ones(128).to(self.device),torch.ones(256).to(self.device),torch.ones(512).to(self.device)] for i in range(10)] # 10个用户，10个map（构造初始化）
        self.standalone_test_acc = torch.tensor(standalone_test[self.option['task'].split('/')[-1]]).to(self.device)

        print("self.standalone_test_acc", self.standalone_test_acc)

        tmp = torch.tensor([torch.exp((xi)*self.Beta) for xi in self.standalone_test_acc])
        self.client_contributions = tmp/tmp.max()*100

        self.sample_option = 'full'
        print("self.client_contributions", self.client_contributions)
        self.c = 0
        self.all_map_frenquency = torch.zeros(1, 1024).to(self.device)

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
                    self.gv.logger.log_once(round=self.current_round)
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
            map_glo = [torch.ones(64).to(self.device), torch.ones(64).to(self.device),torch.ones(128).to(self.device),torch.ones(256).to(self.device),torch.ones(512).to(self.device)]
            model_glo.set_map(map_glo)
            acc_glo = self.calculator.test(model_glo, self.validation_data, batch_size = self.option['test_batch_size'])['accuracy']
            contri_client_values, contri_client_indices = torch.sort(self.client_contributions)
            all_map_frequency_values, all_map_frequency_indices = torch.sort(self.all_map_frenquency)
            all_submodel_acc = []
            index_map = 0

            a = time.time()

            for i in range(1024//32):

                map_c_s = [torch.zeros(1024).to(self.device)]
                index_map += 32

                map_c_s[0][all_map_frequency_indices[0][0:index_map]] = 1 # 根据排序后的索引来设置map_c_s中对应位置为1
                map_tem = [map_c_s[0][:64], map_c_s[0][64:128], map_c_s[0][128:256], map_c_s[0][256:512], map_c_s[0][512:1024]]

                model_glo.set_map(map_tem)
                # tem_submodel_acc = self.calculator.test(model_glo, self.validation_data, batch_size = self.option['test_batch_size'])['accuracy']
                tem_submodel_acc = self.calculator.test(model_glo, self.validation_data, batch_size = 1024)['accuracy']
                all_submodel_acc.append(tem_submodel_acc)

            b = time.time()
            # print("=======", b-a)

            indexs_all = [0 for i in range(len(self.standalone_test_acc))]
            index_tmp = 0

            print("sssssssss", all_submodel_acc)
            print("aaaaaaaaa", acc_glo)

            for i in range(len(self.standalone_test_acc)):

                if i == 9:
                    indexs_all[i] = 1024

                else:
                    al = 0.95 - 0.95/(torch.log(torch.tensor(0.3 * self.current_round)+ 3))
                    min_acc = acc_glo * al
                    for s in range(1024//32-1, index_tmp, -1):
                        if all_submodel_acc[s] <= self.standalone_test_acc[contri_client_indices[i]]  + (acc_glo - min_acc) * contri_client_values[i] /100:
                            break
                        else:
                            continue

                    indexs_all[i] = 32*(s)
                    index_tmp = s

            for i in range(len(self.standalone_test_acc)):
                map_s = [torch.zeros(1024).to(self.device)]

                map_s[0][all_map_frequency_indices[0][0:indexs_all[i]]] = 1
                map_tem = [map_s[0][:64], map_s[0][64:128], map_s[0][128:256], map_s[0][256:512], map_s[0][512:1024]]
                
                self.map[contri_client_indices[i]] = map_tem

                self.all_map_frenquency[0] += map_s[0]

        if self.current_round > self.t:
            model_glo = copy.deepcopy(self.model)
            map_glo = [torch.ones(64).to(self.device), torch.ones(64).to(self.device),torch.ones(128).to(self.device),torch.ones(256).to(self.device),torch.ones(512).to(self.device)]
            model_glo.set_map(map_glo)
            acc_glo = self.calculator.test(model_glo, self.validation_data, batch_size = self.option['test_batch_size'])['accuracy']
            contri_client_values, contri_client_indices = torch.sort(self.client_contributions)
            all_map_frequency_values, all_map_frequency_indices = torch.sort(self.all_map_frenquency)

            all_submodel_acc = []
            index_map = 0

            a = time.time()

            for i in range(1024//8):
                map_c_s = [torch.zeros(1024).to(self.device)]
                index_map += 8
                
                map_c_s[0][all_map_frequency_indices[0][0:index_map]] = 1
                map_tem = [map_c_s[0][:64], map_c_s[0][64:128], map_c_s[0][128:256], map_c_s[0][256:512], map_c_s[0][512:1024]]
                model_glo.set_map(map_tem)
                tem_submodel_acc = self.calculator.test(model_glo, self.validation_data, batch_size = self.option['test_batch_size'])['accuracy']

                all_submodel_acc.append(tem_submodel_acc)

            b = time.time()

            indexs_all = [0 for i in range(len(self.standalone_test_acc))]
            index_tmp = 0
            
            print("aaaaaaaaa", acc_glo)

            for i in range(len(self.standalone_test_acc)):

                if i == 9:
                    indexs_all[i] = 1024

                else:
                    alpha = 0.95
                    al = alpha*(1.0 - 1.0/(torch.log(torch.tensor(0.3 * self.current_round)+ 3)))
                    min_acc = acc_glo * al
                    for s in range(1024//8-1, index_tmp, -1):
                        if all_submodel_acc[s] <= self.standalone_test_acc[contri_client_indices[i]]  + (acc_glo - min_acc) * contri_client_values[i] /100:
                            break
                        else:
                            continue
                    indexs_all[i] = 8*(s)
                    index_tmp = s
                
                print("bbbbbbbbb", self.standalone_test_acc[contri_client_indices[i]]  + (acc_glo - min_acc) * contri_client_values[i] /100)

            for i in range(len(self.standalone_test_acc)):
                map_s = [torch.zeros(1024).to(self.device)]
                map_s[0][all_map_frequency_indices[0][0:indexs_all[i]]] = 1
                map_tem = [map_s[0][:64], map_s[0][64:128], map_s[0][128:256], map_s[0][256:512], map_s[0][512:1024]]
                self.map[contri_client_indices[i]] = map_tem
                self.all_map_frenquency[0] += map_s[0]

            print("self.all_map_frenquency", self.all_map_frenquency[0])

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
