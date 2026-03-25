import torchvision
from flgo.benchmark.toolkits.cv.classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import flgo.benchmark
import os.path
import torch
import json
import numpy as np
import copy

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'EMNIST'), split='byclass'):
        super(TaskGenerator, self).__init__('emnist_classification', rawdata_path, torchvision.datasets.EMNIST, transforms)
        self.split = split
        self.additional_option = {'split': self.split}

    def load_data(self):
        train_data = torchvision.datasets.EMNIST(root=self.rawdata_path, split = self.split, download=True, train=True, transform=self.transform)
        self.train_data, self.validation = self.split_dataset(train_data, 0.1)
        self.test_data = torchvision.datasets.EMNIST(root=self.rawdata_path, split = self.split, download=True, train=False, transform=self.transform)

    def split_dataset(self, dataset, p=0.0):
        if p == 0: return dataset, None
        s1 = int(len(dataset) * p)
        s2 = len(dataset) - s1
        return torch.utils.data.random_split(dataset, [s2, s1])

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, torchvision.datasets.EMNIST, transforms)

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': list(range(len(generator.test_data))), 'validation_data': generator.validation.indices, 'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option}
        # feddata = {'client_names': client_names, 'server_data': list(range(len(generator.test_data))),  'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        if hasattr(generator.partitioner, 'local_perturbation'): feddata['local_perturbation'] = generator.partitioner.local_perturbation
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_data = self.builtin_class(root=self.feddata['rawdata_path'], download=True, train=True, transform=self.transform, **self.feddata['additional_option'])
        test_data = self.builtin_class(root=self.feddata['rawdata_path'], download=True, train=False, transform=self.transform, **self.feddata['additional_option'])
        test_data = self.TaskDataset(test_data, list(range(len(test_data))), None, running_time_option['pin_memory'])
        # rearrange data for server
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        validation_data = self.TaskDataset(train_data, self.feddata['validation_data'], None, running_time_option['pin_memory'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid, 'validation': validation_data}}
        # rearrange data for clients
        local_perturbation = self.feddata['local_perturbation'] if 'local_perturbation' in self.feddata.keys() else [None for _ in self.feddata['client_names']]
        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = None if  local_perturbation[cid] is None else [torch.tensor(t) for t in local_perturbation[cid]]
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'], cpert, running_time_option['pin_memory'])

            c = []
            for i in cdata.indices:
                b = cdata.dataset[i]
                c.append(b[1])
            d = np.unique(c, return_counts=True)
            print(d)

            cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_valid, cdata_test = self.split_dataset(cdata_valid, 0.5)
            else:
                cdata_test = None
            task_data[cname] = {'train':cdata_train, 'valid':cdata_valid, 'test': cdata_test}
        return task_data
