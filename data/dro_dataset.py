import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class DRODataset(Dataset):
    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn, logger=None):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        group_array = []
        y_array = []
        total_size, tenth = len(self.dataset), len(self.dataset)//10
        if logger:
            logger.write("DRO dataset creation, len {}, tenth {}\n".format(total_size, tenth))
            logger.flush()
        
        i = 0
        for x,y,g in self:
            if logger and i % tenth == 0:
                logger.write("dataset processing {} tenth, at num {}\n".format(
                    i//tenth,
                    i)
                )
            logger.flush()
            i += 1
            group_array.append(g)
            y_array.append(y)
        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(y_array)
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()

    def __getitem__(self, idx):
        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x,y,g in self:
            return x.size()

    def get_loader(self, train, **kwargs):
        if not train: # Validation or testing
            shuffle = False
            sampler = None
        else: # Training but not reweighting
            shuffle = True
            sampler = None

        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        return loader
