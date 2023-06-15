import os, csv
import argparse
import pandas as pd
import torch, time
import torch.nn as nn
import torchvision

from models import model_attributes
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args
from train import test


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    # Resume?
    parser.add_argument('--resume', default=False, action='store_true')
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    # Objective
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')

    # Model
    parser.add_argument( '--model_path', type=str)
    parser.add_argument( '--model', type=str, default="resnet18")
    parser.add_argument( '--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001) # 1e-3 for waterbirds and 1e-4 for celebA
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    # Optimization
    parser.add_argument('--batch_size', type=int, default=128)
    # Misc
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_every', default=50, type=int) # number of batches after which to log
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)

    args = parser.parse_args()
    check_args(args)
    
    args.log_dir = os.path.dirname(args.model_path)
    
    if os.path.exists(args.log_dir) and args.resume:
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'


    log_file_path = os.path.join(args.log_dir, 'test_{}.txt'.format(os.path.basename(args.model_path)))
    print(log_file_path)
    logger = Logger(log_file_path, mode)
    
    # Record args
    log_args(args, logger)
    set_seed(args.seed)
    
    # Data
    # Test data for label_shift_step is not implemented yet
    data_start_time = time.time()
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True, logger=logger)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)

    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':4, 'pin_memory':True}
    train_loader = train_data.get_loader(train=True, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, **loader_kwargs)
    
    logger.write("{:.2g} minutes for data processing\n".format((time.time()-data_start_time)/60))
    logger.flush()
    
    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)
    
    logger.flush()

    model = torch.load(args.model_path)
    model.eval()
    logger.write("model loaded\n")
    logger.flush()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, 'test.csv'))
        epoch_offset = df.loc[len(df)-1,'epoch']+1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset=0
    test_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode)
    test(model, criterion, data, logger, test_csv_logger, args)

    test_csv_logger.close()

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio



if __name__=='__main__':
    main()
