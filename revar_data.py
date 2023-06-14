#data based utilities
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import pandas as pd
import os 
import copy
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms


class ImageNet100(Dataset):
    def __init__(self, data_dir="./ImageNet100/", 
                       label_file="Labels.json",
                       split="train",
                       num_examples=None,
                       transform=None):
        
        self.label_file = label_file
        self.transform = transform
        
        if split=="test":
            split = "val"

        self.imagefolder = os.path.join(data_dir, split)
        self.image_list, self.label_list = [], []
        count = 0
        self.label_dict = {}
        for curr_label in os.listdir(self.imagefolder):
            if curr_label not in self.label_dict.keys():
                self.label_dict[curr_label] = count
                count+=1
            for curr_ims in os.listdir(curr_label):
                self.label_list.append(self.label_dict[curr_label])
                self.image_list.append(os.path.join(self.imagefolder, curr_label, curr_ims))
        
        if num_examples is not None:
            self.label_list = self.label_list[:num_examples]

        self.num_classes=100
        self.targets = self.label_list
        
    def __len__(self):
        return len(self.label_list)
    
    def __getitem__(self, index):
        img = Image.open(self.image_list[index])
        if(self.transform):
            img = self.transform(img)
        return img, torch.tensor(self.label_list[index])


def build_dataloader(
        seed=1,
        dataset='cifar10',
        num_meta_total=1000,
        batch_size=100,
        data_dir=None,
        label_file=None,
        num_examples=None,
        domain_shift=False,
        unsup_adapt=False,
        num_meta_unsup=None,
        input_size=32,
        use_val_unsup=False,
):

    np.random.seed(seed)
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset_list = {
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'dr': Retina,
        'im100': ImageNet100,
    }

    if dataset in ["cifar10", "cifar100"]:

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = dataset_list[dataset](root='/home/anmolreddy/data/cifar/', train=True, download=False, transform=train_transforms)
        test_dataset = dataset_list[dataset](root='/home/anmolreddy/data/cifar/', train=False, download=False, transform=test_transforms)
        num_classes = len(train_dataset.classes)
        


    else:
        train_transforms = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = dataset_list[dataset](data_dir=data_dir, label_file=label_file, split="train", transform=train_transforms, num_examples=num_examples)
        if not domain_shift:
            test_dataset = dataset_list[dataset](data_dir=data_dir, label_file=label_file, split="test", transform=train_transforms, num_examples=num_examples)
        else:
            test_dataset = dataset_list[dataset](data_dir=data_dir, label_file=label_file, split="train", transform=test_transforms, num_examples=num_examples, domain_shift=domain_shift)
        num_classes=train_dataset.num_classes


    
    num_meta = int(num_meta_total / num_classes)

    index_to_meta = []
    index_to_train = []


    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(train_dataset.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])
        index_to_class_for_train = index_to_class[num_meta:]

        
        index_to_train.extend(index_to_class_for_train)
    
    meta_dataset = copy.deepcopy(train_dataset)
    train_dataset.data = train_dataset.data[index_to_train]
    train_dataset.targets = list(np.array(train_dataset.targets)[index_to_train])
    meta_dataset.data = meta_dataset.data[index_to_meta]
    meta_dataset.targets = list(np.array(meta_dataset.targets)[index_to_meta])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    

    if unsup_adapt:
        if use_val_unsup==True:
            meta_dataset_s = dataset_list[dataset](data_dir=data_dir, label_file=label_file, split="val", transform=test_transforms, num_examples=num_examples, domain_shift=domain_shift)
        else:
            meta_dataset_s = copy.deepcopy(test_dataset)
            index_to_meta = []
            index_to_test = []

            for class_index in range(num_classes):
                index_to_class = [index for index, label in enumerate(test_dataset.targets) if label == class_index]
                np.random.shuffle(index_to_class)
                index_to_meta.extend(index_to_class[:num_meta_unsup])
                index_to_class_for_test = index_to_class[num_meta_unsup:]

                
                index_to_test.extend(index_to_class_for_test)
            
            # print(len(index_to_test))
            # print(len(index_to_meta))
            
            test_dataset.data = test_dataset.data[index_to_test]
            test_dataset.targets = list(np.array(test_dataset.targets)[index_to_test])
            meta_dataset_s.data = meta_dataset_s.data[index_to_meta]
            meta_dataset_s.targets = list(np.array(meta_dataset_s.targets)[index_to_meta])

        print(test_dataset.data.shape[0])
        print(train_dataset.data.shape[0])
        print(meta_dataset_s.data.shape[0])
        print(meta_dataset.data.shape[0])

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
        meta_dataloader_s = DataLoader(meta_dataset_s, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        return train_dataloader, meta_dataloader, test_dataloader, meta_dataloader_s


    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    
    return train_dataloader, meta_dataloader, test_dataloader