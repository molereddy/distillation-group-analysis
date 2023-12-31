import os, csv, pickle
import argparse
import pandas as pd, numpy as np
import torch, time
import torch.nn as nn
import torchvision

from local_models import model_attributes, FeatResNet, SimKD
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args
from train import train

# python3 save_datasets.py -s confounder -d CelebA -t Blond_Hair -c Male
# python3 save_datasets.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2

def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, 
                        default='confounder', required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    parser.add_argument('-widx', '--worst_group_idx', type=int, default=2)
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--model', choices=model_attributes.keys(), default='resnet18')
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--logs_dir', default='./results')

    parser.add_argument('--batch_size', type=int, default=24)

    args = parser.parse_args()
    check_args(args)
    dataset_path = os.path.join(args.logs_dir, args.dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
    log_file_path = os.path.join(dataset_path, 'data_proc.log')
    logger = Logger(log_file_path, 'w')
    logger.write(log_file_path + '\n')
    log_args(args, logger)
    
    for seed in range(1):
        set_seed(seed)
        data_start_time = time.time()
        test_data = None
        train_data, val_data, test_data = prepare_data(args, train=True)
        with open(os.path.join(args.logs_dir, args.dataset, 
                               '_'.join([args.target_name] + list(map(str, args.confounder_names)) + ['dataset', f'{seed}.pkl'])), 'wb') as file:
            pickle.dump({'train_data': train_data, 'val_data': val_data, 'test_data': test_data}, file)
        logger.write("{:.2g} minutes for data processing\n".format((time.time()-data_start_time)/60))
        logger.flush()
    

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name



if __name__=='__main__':
    main()
