import copy
import os
from collections import OrderedDict
import torch
import numpy as np
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fuf
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
import torchvision
import torch.nn as nn

class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate if self.training else input
        return output

class CIFAR10Model:
    class AugmentDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.transform = torchvision.transforms.Compose(
                [RandomCrop(size=(32, 32), padding=4), RandomHorizontalFlip(0.5)])

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, item):
            img, label = self.dataset[item]
            return self.transform(img), label

    class Model(fuf.FModule):
        def __init__(self, rate: float = 1.0, track=False):
            super().__init__()
            self.num_classes = 10
            self.encoder = nn.Sequential(
                nn.Conv2d(3, int(64*rate), 5, bias=True),
                Scaler(rate),
                nn.BatchNorm2d(int(64*rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(int(64*rate), int(64*rate), 5, bias=True),
                Scaler(rate),
                nn.BatchNorm2d(int(64 * rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(1),
                nn.Linear(int(1600*rate), int(384*rate), bias=True),
                Scaler(rate),
                nn.BatchNorm1d(int(384*rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.Linear(int(384*rate), int(192*rate), bias=True),
                Scaler(rate),
                nn.BatchNorm1d(int(192 * rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
            )
            self.head = nn.Linear(int(192*rate), 10, bias=True)

        def forward(self, x):
            x = self.encoder(x)
            return self.head(x)

    @classmethod
    def init_dataset(cls, object):
        if 'Client' in object.get_classname():
            object.train_data = cls.AugmentDataset(object.train_data)

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(object.rate ** object.p)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                if hasattr(object, '_track'):
                    return cls.Model(1.0, object._track)
                else:
                    return cls.Model(1.0)

class CIFAR100Model:
    class AugmentDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.transform = torchvision.transforms.Compose(
                [RandomCrop(size=(32, 32), padding=4), RandomHorizontalFlip(0.5)])

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, item):
            img, label = self.dataset[item]
            return self.transform(img), label

    class Model(fuf.FModule):
        def __init__(self, rate: float = 1.0, track=False):
            super().__init__()
            self.num_classes = 100
            self.encoder = nn.Sequential(
                nn.Conv2d(3, int(64*rate), 5, bias=True),
                Scaler(rate),
                nn.BatchNorm2d(int(64*rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(int(64*rate), int(64*rate), 5, bias=True),
                Scaler(rate),
                nn.BatchNorm2d(int(64 * rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(1),
                nn.Linear(int(1600*rate), int(384*rate), bias=True),
                Scaler(rate),
                nn.BatchNorm1d(int(384*rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
                nn.Linear(int(384*rate), int(192*rate), bias=True),
                Scaler(rate),
                nn.BatchNorm1d(int(192 * rate), momentum=None, track_running_stats=track),
                nn.ReLU(),
            )
            self.head = nn.Linear(int(192*rate), 100, bias=True)

        def forward(self, x):
            x = self.encoder(x)
            return self.head(x)

    @classmethod
    def init_dataset(cls, object):
        if 'Client' in object.get_classname():
            object.train_data = cls.AugmentDataset(object.train_data)

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(object.rate ** object.p)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                if hasattr(object, '_track'):
                    return cls.Model(1.0, object._track)
                else:
                    return cls.Model(1.0)

class DOMAINNETModel:
    class Model(fuf.FModule):
        def __init__(self, rate: float = 1.0, track=False, num_classes=10):
            super().__init__()
            self.rate = rate
            self.num_classes = num_classes
            self.features = nn.Sequential(
                OrderedDict([
                    ('conv1',  nn.Conv2d(3, int(rate*64), 11, stride=4, padding=2, bias=True, )),
                    ('scaler1', Scaler(rate)),
                    ('bn1', nn.BatchNorm2d(int(64*rate), momentum=None, track_running_stats=track)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv2', nn.Conv2d(int(rate*64), int(rate*192), 5, padding=2, bias=True, )),
                    ('scaler2', Scaler(rate)),
                    ('bn2', nn.BatchNorm2d(int(192*rate), momentum=None, track_running_stats=track)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                    ('conv3', nn.Conv2d(int(rate*192), int(rate*384), 3, padding=1, bias=True, )),
                    ('scaler3', Scaler(rate)),
                    ('bn3', nn.BatchNorm2d(int(384 * rate), momentum=None, track_running_stats=track)),
                    ('relu3', nn.ReLU(inplace=True)),

                    ('conv4', nn.Conv2d(int(rate*384), int(rate*256), 3, padding=1, bias=True, )),
                    ('scaler4', Scaler(rate)),
                    ('bn4', nn.BatchNorm2d(int(256 * rate), momentum=None, track_running_stats=track)),
                    ('relu4', nn.ReLU(inplace=True)),

                    ('conv5', nn.Conv2d(int(rate*256), int(rate*256), 3, padding=1, bias=True, )),
                    ('scaler5', Scaler(rate)),
                    ('bn5', nn.BatchNorm2d(int(256 * rate), momentum=None, track_running_stats=track)),
                    ('relu5', nn.ReLU(inplace=True)),
                    ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
                ])
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc1 = nn.Linear(int(rate*256 * 6 * 6), int(4096 * rate), bias=True, )
            self.scaler6 = Scaler(rate)
            self.bn6 = nn.BatchNorm1d(int(4096 * rate), momentum=None, track_running_stats=track)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(int(rate*4096), int(rate*4096), bias=True, )
            self.scaler7 = Scaler(rate)
            self.bn7 = nn.BatchNorm1d(int(4096 * rate), momentum=None, track_running_stats=track)
            self.head = nn.Linear(int(rate*4096), num_classes, bias=True, )

        def encoder(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.scaler6(x)
            x = self.bn6(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.scaler7(x)
            x = self.bn7(x)
            x = self.relu(x)
            return x

        def forward(self, x):
            x = self.encoder(x)
            x = self.head(x)
            return x

    @classmethod
    def init_dataset(cls, object):
        pass

    @classmethod
    def init_local_module(cls, object):
        if 'Client' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                return cls.Model(object.rate ** object.p)

    @classmethod
    def init_global_module(cls, object):
        if 'Server' in object.__class__.__name__:
            if not hasattr(object, '_model_class'):
                object._model_class = cls
                return
            else:
                if hasattr(object, '_track'):
                    return cls.Model(1.0, object._track)
                else:
                    return cls.Model(1.0)

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'rate':0.5, 'use_label_mask':True})
        self.rp = 1.0
        self._track = False
        self.model = self._model_class.init_global_module(self)
        self.client_capability = [c._capability for c in self.clients]
        for c in self.clients:
            c.pi = 0
            while self.rate**(2*c.pi)>c._capability:
                c.pi+=1
        self.client_level = [c.pi for c in self.clients]
        self.level_set = set(sorted(self.client_level))
        self.model_shapes = {}
        for pi in range(max(self.level_set)+1):
            tmp_md = self._model_class.Model(self.rate**pi).state_dict()
            self.model_shapes[pi] = {k:v.shape for k,v in tmp_md.items()}
        # only for testing
        self.track_model_shapes = {}
        for pi in self.level_set:
            tmp_md = self._model_class.Model(self.rate**pi, track=True).state_dict()
            self.track_model_shapes[pi] = {k:v.shape for k,v in tmp_md.items()}
        self.test_model = None

    def pack(self, client_id, mtype=0, *args, **kwargs):
        pi = self.client_level[client_id]
        if mtype==0:
            md = copy.deepcopy(self.model.state_dict())
            layer_shapes = self.model_shapes[pi]
        else:
            md = copy.deepcopy(self.test_model.state_dict())
            layer_shapes = self.track_model_shapes[pi]
        for k in md.keys():
            lshape = layer_shapes[k]
            for dim,l in enumerate(lshape):
                md[k] = md[k].narrow(dim, 0, l)
        return {'w': md}

    def aggregate(self, models: list, label_masks = None):
        mds = [mi.state_dict() for mi in models]
        full_model_shape = self.model_shapes[0]
        tmp_md = {k: torch.zeros(s) for k, s in full_model_shape.items()}
        mask = {k: torch.zeros(s) for k, s in full_model_shape.items()}
        for i,md in enumerate(mds):
            for k, v in md.items():
                s = v.shape
                cmd_md = 'tmp_md[k]['
                for d in s:
                    cmd_md += f':{d},'
                cmd_md = cmd_md[:-1]
                cmd_md += ']'
                target_weight = eval(cmd_md)

                cmd_mask = 'mask[k]['
                for d in s:
                    cmd_mask += f':{d},'
                cmd_mask = cmd_mask[:-1]
                cmd_mask += ']'
                target_mask = eval(cmd_mask)

                if not self.use_label_mask or 'head' not in k:
                    m = torch.ones_like(v)
                else:
                    lb_mask = label_masks[i]
                    m = torch.ones_like(v)
                    for lb in range(len(lb_mask)):
                        if lb_mask[lb] == 1:
                            v[lb] = 0
                            m[lb] = 0

                target_weight += v
                target_mask += m
        for k in tmp_md.keys():
            tmp_md[k]/=(mask[k]+1e-8)
        self.model.load_state_dict(tmp_md)
        return self.model

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients, mtype=0)
        self.model = self.aggregate(res['model'], res['label_mask'])
        # collect bn stats
        tmp_model = self._model_class.Model(1.0, track=True)
        tmp_md = tmp_model.state_dict()
        tmp_md.update(self.model.state_dict())
        tmp_model.load_state_dict(tmp_md)
        all_train_data = torch.utils.data.ConcatDataset([c.train_data for c in self.clients])
        train_loader = torch.utils.data.DataLoader(all_train_data, batch_size=self.clients[0].batch_size)
        tmp_model.train()
        with torch.no_grad():
            for i, batch_data in enumerate(train_loader):
                tmp_model(batch_data[0])
        self.test_model = tmp_model
        self.communicate([i for i in range(self.num_clients)], mtype=1)

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.p = self.rate**self.pi
        self.model = self._model_class.Model(self.p, track=True)
        self.train_model = self._model_class.Model(self.p)
        self.actions = {0:self.reply, 1:self.set_test_model}
        local_labels = list(set([d[-1] for d in self.train_data]))
        self.label_mask = torch.zeros(self.model.num_classes).to(self.device)
        self.label_mask[local_labels] = 1

    def unpack(self, received_pkg):
        w = received_pkg['w']
        self.train_model.load_state_dict(w)
        return self.train_model

    def pack(self, model, *args, **kwargs):
        return {'model':model, 'label_mask':self.label_mask.cpu()}

    def set_test_model(self, pkg):
        w = pkg['w']
        self.model.load_state_dict(w)

    @fuf.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            model.zero_grad()
            y = model(batch_data[0])
            if self.use_label_mask:
                y = y.masked_fill(self.label_mask==0, 0)
            loss = self.calculator.criterion(y, batch_data[-1])
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                                  max_norm=self.clip_grad)
            optimizer.step()
        return

def init_global_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_global_module(object)

def init_local_module(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_local_module(object)

def init_dataset(object):
    module_class = eval(os.path.split(object.option['task'])[-1].upper().split('_')[0]+'Model')
    return module_class.init_dataset(object)

