import os
import json
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
import random
from torch.utils.data import Subset, Dataset
import flgo.benchmark
from flgo.benchmark.toolkits.cv.classification import (
    BuiltinClassGenerator,
    BuiltinClassPipe,
    GeneralCalculator
)
import urllib.request
import zipfile
import shutil


def prepare_tiny_imagenet(root):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    extract_path = os.path.join(root, "tiny-imagenet-200")
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    if os.path.exists(train_dir) and os.path.exists(val_dir):
        return

    os.makedirs(root, exist_ok=True)
    if not os.path.exists(zip_path):
        print("Downloading Tiny-ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
    if not os.path.exists(extract_path):
        print("Extracting Tiny-ImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)

    shutil.move(os.path.join(extract_path, "train"), train_dir)

    os.makedirs(val_dir, exist_ok=True)
    val_images = os.path.join(extract_path, "val", "images")
    val_annotations = os.path.join(extract_path, "val", "val_annotations.txt")

    val_map = {}
    with open(val_annotations, "r") as f:
        for line in f:
            img, cls, *_ = line.strip().split("\t")
            val_map[img] = cls

    for cls in set(val_map.values()):
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    for img, cls in val_map.items():
        shutil.move(os.path.join(val_images, img), os.path.join(val_dir, cls, img))

    shutil.rmtree(extract_path)
    print("Tiny-ImageNet is ready.")


base_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975),
                                     (0.2302, 0.2265, 0.2262))
])

data_transforms = {
    'train': base_transform,
    'val': base_transform,
    'test': base_transform
}

class AugmentDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.TF = TF
        self.random = random

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        angle = self.random.uniform(-20, 20)
        img = self.TF.rotate(img, angle)
        if self.random.random() > 0.5:
            img = self.TF.hflip(img)
        return img, label

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def init_dataset(object):
    if 'Client' in object.get_classname():
        object.train_data = AugmentDataset(object.train_data)

TaskCalculator = GeneralCalculator
RAW_DATA_PATH = os.path.join(flgo.benchmark.path, "RAW_DATA", "ImageNetTiny")


# ========= TaskGenerator =========
class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=RAW_DATA_PATH):
        super().__init__(
            benchmark='imagenet_tiny_classification',
            rawdata_path=rawdata_path,
            builtin_class=torchvision.datasets.ImageFolder,
            transform=data_transforms['train']
        )

    def split_dataset(self, dataset, p=0.0):
        if p == 0.0:
            return dataset, None
        s1 = int(len(dataset) * p)
        s2 = len(dataset) - s1
        return torch.utils.data.random_split(dataset, [s2, s1])

    def load_data(self):
        prepare_tiny_imagenet(self.rawdata_path)

        train_path = os.path.join(self.rawdata_path, 'train')
        val_path = os.path.join(self.rawdata_path, 'val')

        train_data = self.builtin_class(train_path, transform=data_transforms['train'])
        self.train_data, self.validation = self.split_dataset(train_data, 0.1)

        self.test_data = self.builtin_class(val_path, transform=data_transforms['test'])


# ========= TaskPipe =========
class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super().__init__(task_name, torchvision.datasets.ImageFolder, data_transforms['train'])

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {
            'client_names': client_names,
            'server_data': list(range(len(generator.test_data))),
            'validation_data': generator.validation.indices,
            'rawdata_path': generator.rawdata_path,
            'additional_option': getattr(generator, 'additional_option', {})
        }
        for cid, cname in enumerate(client_names):
            feddata[cname] = {'data': generator.local_datas[cid]}
        if hasattr(generator.partitioner, 'local_perturbation'):
            feddata['local_perturbation'] = generator.partitioner.local_perturbation
        with open(os.path.join(self.task_path, 'data.json'), 'w') as f:
            json.dump(feddata, f)

    def load_data(self, running_time_option) -> dict:
        prepare_tiny_imagenet(self.feddata['rawdata_path'])

        train_path = os.path.join(self.feddata['rawdata_path'], 'train')
        val_path = os.path.join(self.feddata['rawdata_path'], 'val')

        train_data = self.builtin_class(train_path, transform=data_transforms['train'],
                                        **self.feddata.get('additional_option', {}))
        test_data = self.builtin_class(val_path, transform=data_transforms['test'],
                                       **self.feddata.get('additional_option', {}))

        test_data = self.TaskDataset(
            test_data,
            list(range(len(test_data))),
            None,
            running_time_option['pin_memory']
        )
        server_test, server_valid = self.split_dataset(test_data, running_time_option['test_holdout'])

        validation_data = self.TaskDataset(
            train_data,
            self.feddata['validation_data'],
            None,
            running_time_option['pin_memory']
        )

        task_data = {
            'server': {
                'test': server_test,
                'valid': server_valid,
                'validation': validation_data
            }
        }

        local_perturbation = self.feddata.get(
            'local_perturbation',
            [None for _ in self.feddata['client_names']]
        )

        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = (
                None if local_perturbation[cid] is None
                else [torch.tensor(t) for t in local_perturbation[cid]]
            )
            cdata = self.TaskDataset(
                train_data,
                self.feddata[cname]['data'],
                cpert,
                running_time_option['pin_memory']
            )

            labels = [cdata.dataset[i][1] for i in cdata.indices]
            print(f"[{cname}] class distribution:", np.unique(labels, return_counts=True))

            ctrain, cvalid = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout'] > 0 and running_time_option['local_test']:
                cvalid, ctest = self.split_dataset(cvalid, 0.5)
            else:
                ctest = None

            task_data[cname] = {
                'train': ctrain,
                'valid': cvalid,
                'test': ctest
            }

        return task_data
