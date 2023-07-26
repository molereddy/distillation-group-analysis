import os, csv, pickle
import argparse
import pandas as pd
import torch, time
import torch.nn as nn
import torchvision

from models import model_attributes, FeatResNet, SimKD
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args
from train import train


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
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--logs_dir', default='./results')

    args = parser.parse_args()
    check_args(args)
    log_file_path = os.path.join(args.logs_dir, args.dataset, 'data_proc.log')
    logger = Logger(log_file_path, 'w')
    logger.write(log_file_path + '\n')
    log_args(args, logger)
    
    for seed in range(3):
        set_seed(seed)
        data_start_time = time.time()
        test_data = None
        train_data, val_data, test_data = prepare_data(args, train=True, logger=logger)
        with open(os.path.join(args.logs_dir, args.dataset, 'dataset_processed_data.pkl'), 'wb') as file:
            pickle.dump({'train_data': train_data, 'val_data': val_data, 'test_data': test_data}, file)
        logger.write("{:.2g} minutes for data processing\n".format((time.time()-data_start_time)/60))
        logger.flush()
    

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name



if __name__=='__main__':
    main()
