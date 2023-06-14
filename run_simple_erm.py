import os, csv
import argparse
import pandas as pd
import torch, time
import torch.nn as nn
import torchvision

from models import model_attributes
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
    parser.add_argument( '--model', choices=model_attributes.keys(), default='resnet50')
    parser.add_argument( '--teacher', choices=model_attributes.keys())

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=50) # 300 for waterbirds and 50 for celebA
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001) # 1e-3 for waterbirds and 1e-4 for celebA
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int) # number of batches after which to log
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)

    args = parser.parse_args()
    check_args(args)
    
    args.log_dir = os.path.join(args.log_dir, args.dataset)
    model_path_prefix = ""
    if args.teacher is not None:
        model_path_prefix += args.teacher + "_"
    model_path_prefix += args.model + "_{}".format(args.seed)
    if model_path_prefix == "": model_path_prefix = "base"
    args.log_dir = os.path.join(args.log_dir, model_path_prefix)
    
    if os.path.exists(args.log_dir) and args.resume:
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    log_file_path = os.path.join(args.log_dir, 'log.txt')
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
        train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)

    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':4, 'pin_memory':True}
    train_loader = train_data.get_loader(train=True, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, **loader_kwargs)
    
    print("{:.2g} minutes for data processing".format((time.time()-data_start_time)/60))
    
    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes

    log_data(data, logger)

    ## Initialize model
    print("loading model")
    if resume:
        model = torch.load(os.path.join(args.log_dir, 'last_model.pth')).to(device=args.device)
        d = train_data.input_size()[0]
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes).to(device=args.device)
        model.has_aux_logits = False
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT').to(device=args.device)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(weights='DEFAULT').to(device=args.device)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(weights='DEFAULT').to(device=args.device)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(weights='DEFAULT').to(device=args.device)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    else:
        raise ValueError('Model not recognized.')
    
    logger.flush()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, 'test.csv'))
        epoch_offset = df.loc[len(df)-1,'epoch']+1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset=0
    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, 'train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, 'test.csv'), train_data.n_groups, mode=mode)

    train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args, epoch_offset=epoch_offset)

    train_csv_logger.close()
    val_csv_logger.close()
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
