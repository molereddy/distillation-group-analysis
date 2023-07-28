import os, csv, pickle
import argparse
import pandas as pd
import torch, time
import torch.nn as nn
import torchvision

from models import model_attributes, FeatResNet, SimKD
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from data.dro_dataset import get_loader
from utils import set_seed, Logger, CSVBatchLogger, log_args
from train import train


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, default='confounder', required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    parser.add_argument('-widx', '--worst_group_idx', type=int, default=2)
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
    parser.add_argument('--model', choices=model_attributes.keys(), default='resnet50')
    parser.add_argument('--model_state', choices=['scratch', 'pretrained'], default='scratch')
    parser.add_argument("--teacher", type=str, choices=['resnet50', 'resnet50-pt', 'resnet50-ft'], help="teacher name")
    parser.add_argument('--teacher_type', choices=['best', 'last'], default='best')
    parser.add_argument('--method', type=str, choices=['KD', 'SimKD', 'ERM', 'JTT'], default='ERM')

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=160) # 160 for waterbirds and 75 for celebA
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001) # 1e-3 for waterbirds and 1e-4 for celebA
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--logs_dir', default='./results')
    parser.add_argument('--log_every', default=10, type=int) # number of batches after which to log
    parser.add_argument('--save_step', type=int)

    args = parser.parse_args()
    check_args(args)
    
    if (args.teacher is not None or 'resnet50' in args.model) and args.dataset == 'CUB':
        args.batch_size = 64
    
    if args.dataset == "CUB":
        args.n_epochs = 150
        args.lr = 1e-3
        args.log_every = (int(10 * 128 / args.batch_size)//10+1) * 10 # roughly 1280/batch_size
        args.widx = 2
    elif args.dataset == 'CelebA':
        args.n_epochs = 80
        args.lr = 5e-5
        if args.method == 'SimKD': args.lr *= 5
        args.log_every = (int(80 * 128 / args.batch_size)//10+1) * 30 # roughly 30720/batch_size
        args.widx = 3
    
    if args.save_step is None:
        args.save_step = args.n_epochs//2

    # set model, teacher and log file paths
    if args.model_state == 'scratch':
        model_state_name = args.model
    elif args.model_state == 'pretrained':
        model_state_name = args.model + '-pt'
    
    if args.teacher is not None and args.method != 'KD':
        teacher_logs_dir = os.path.join(args.logs_dir, args.dataset, args.teacher+'_'+str(args.seed))
        args.logs_dir = os.path.join(args.logs_dir, args.dataset, 
                                     '_'.join([args.teacher, args.method, model_state_name, str(args.seed)]))
    elif args.teacher is not None:
        # teacher pretrain state is given in args.teacher itself
        teacher_logs_dir = os.path.join(args.logs_dir, args.dataset, args.teacher+'_'+str(args.seed))
        args.logs_dir = os.path.join(args.logs_dir, args.dataset, 
                                     '_'.join([args.teacher, model_state_name, str(args.seed)]))
    else:
         args.logs_dir = os.path.join(args.logs_dir, args.dataset,  
                                 '_'.join([model_state_name, str(args.seed)]))
    
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)
    # resume training
    if os.path.exists(args.logs_dir) and args.resume:
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'
    
    ## Initialize logs
    log_file_path = os.path.join(args.logs_dir, 'train.log')
    logger = Logger(log_file_path, mode)
    logger.write(log_file_path + '\n')
    log_args(args, logger)
    set_seed(args.seed)
    
    # Data
    # Test data for label_shift_step is not implemented yet
    data_start_time = time.time()
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        # train_data, val_data, test_data = prepare_data(args, train=True)
        with open(os.path.join('./results', args.dataset, 
                               '_'.join([args.target_name] + list(map(str, args.confounder_names)) + \
                                   ['dataset', f'{args.seed}.pkl'])
                                ), 'rb') as file:
            data = pickle.load(file)
            train_data = data['train_data']
            val_data = data['val_data']
            test_data = data['test_data']
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)

    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':4, 'pin_memory':True}
    train_loader = get_loader(train_data, train=True, **loader_kwargs)
    val_loader = get_loader(val_data, train=False, **loader_kwargs)
    if test_data is not None:
        test_loader = get_loader(test_data, train=False, **loader_kwargs)
    
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

    models = {}
    ## Initialize model
    logger.write("-" * 50 + '\n')
    weights_dict = {} if args.model_state == 'scratch' else {'weights': 'DEFAULT'}
    if resume:
        model = torch.load(os.path.join(args.logs_dir, 'last_model.pth')).to(device=args.device)
        d = train_data.input_size()[0]
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes).to(device=args.device)
        model.has_aux_logits = False
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(**weights_dict).to(device=args.device)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes).to(device=args.device)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(**weights_dict).to(device=args.device)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes).to(device=args.device)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(**weights_dict).to(device=args.device)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes).to(device=args.device)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(**weights_dict).to(device=args.device)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes).to(device=args.device)
    else:
        raise ValueError('Model not recognized.')
    
    logger.write("loaded model\n")
    models['student'] = model
    logger.flush()

    # load teacher
    if args.teacher is not None:
        if 'resnet18' in args.teacher:
            teacher = torchvision.models.resnet18().to(device=args.device)
            d = teacher.fc.in_features
            teacher.fc = nn.Linear(d, n_classes).to(device=args.device)
        elif 'resnet50' in args.teacher:     
            teacher = torchvision.models.resnet50().to(device=args.device)
            d = teacher.fc.in_features
            teacher.fc = nn.Linear(d, n_classes).to(device=args.device)
        teacher_ckpt = torch.load(os.path.join(teacher_logs_dir, f'{args.teacher_type}_ckpt.pth.tar'))
        teacher.load_state_dict(teacher_ckpt['model'])
        teacher.eval()
        models['teacher'] = teacher
        logger.write(f"teacher loaded: {os.path.join(teacher_logs_dir, f'{args.teacher_type}_ckpt.pth.tar')}")
    
    if args.method == 'SimKD':
        models['teacher'] = FeatResNet(models['teacher'])
        models['student'] = FeatResNet(models['student'])
        models['teacher'].eval()
        models['student'].eval()
        
        data_samples = next(iter(data['train_loader']))[0][:2].to(device=args.device)
        t_n = models['teacher'](data_samples)[0][0].shape[1]
        s_n = models['student'](data_samples)[0][0].shape[1]
        model_simkd = SimKD(s_n=s_n, t_n=t_n, factor=2).to(device=args.device)
        model_simkd.train()
        models['simkd'] = model_simkd
        
    logger.flush()

    if resume:
        df = pd.read_csv(os.path.join(args.logs_dir, 'test.csv'))
        epoch_offset = df.loc[len(df)-1,'epoch']+1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset=0
    
    train_csv_logger = CSVBatchLogger(os.path.join(args.logs_dir, 'train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger =  CSVBatchLogger(os.path.join(args.logs_dir, 'val.csv'), train_data.n_groups, mode=mode)
    test_csv_logger =  CSVBatchLogger(os.path.join(args.logs_dir, 'test.csv'), train_data.n_groups, mode=mode)
    
    train(models, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args, epoch_offset=epoch_offset)
    
    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()
    
    logger.write("{:.2g}h for running\n".format((time.time()-data_start_time)/3600))

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio



if __name__=='__main__':
    main()
