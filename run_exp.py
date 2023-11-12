import os, csv, pickle
import argparse
import pandas as pd
import torch, time
import torch.nn as nn
import torchvision

from models import model_attributes, FeatResNet, SimKD, SemiResNet, Projector
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from data.dro_dataset import get_loader
from utils import set_seed, Logger, CSVBatchLogger, log_args, get_model
from train import train


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, default='confounder')
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
    parser.add_argument('--model', choices=['resnet18', 'resnet18-pt', 'resnet50', 'resnet50-pt',"bert","bert-base-uncased"\
                        "distilbert","distilbert-base-uncased"], default='resnet18-pt')
    parser.add_argument("--teacher", type=str, choices=['resnet50', 'resnet50-pt', 'resnet50-pt_JTT','resnet50-pt_group_DRO',\
                        "bert","bert-base-uncased"], help="teacher name")
    parser.add_argument('--teacher_type', choices=['best', 'last'], default='best')
    parser.add_argument('--method', type=str, choices=['KD', 'SimKD', 'ERM', 'JTT', 'DeTT', 'aux_wt'], default='ERM')
    parser.add_argument('--kd_alpha', type=float, default=1)

    # Optimization
    parser.add_argument('--n_epochs', type=int)
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
    parser.add_argument("--use_bert_params", type=int, default=1)
    parser.add_argument('--save_preds_at', type=list, help='when to save ERM predictions', default=[])
    parser.add_argument('--id_ckpt', type=int, help='which epoch to load id model for DeTT/JTT')
    parser.add_argument('--upweight', type=float, help='upweight factor for DeTT/JTT')
    
    parser.add_argument('--reweigh_at', type=int, default=1, help='when to reweight samples using aux')
    parser.add_argument('--alpha', type=int, help='')
    parser.add_argument('--beta', type=int, help='')
    
    
    

    args = parser.parse_args()
    
    if args.dataset == "CUB":
        args.target_name = "waterbird_complete95"
        args.confounder_names = ["forest2water2"]
        if args.n_epochs is None: args.n_epochs = 250
        if args.method == 'ERM':
            args.lr = 1e-3
            args.weight_decay = 1e-4
            args.save_preds_at = [0, 1, 2, 40, 60, 80]
        elif args.method == 'KD':
            args.lr = 5e-4
            args.weight_decay = 1e-1
        elif args.method == 'SimKD':
            args.lr = 1e-4
            args.weight_decay = 1e-1
        elif args.method == 'JTT':
            args.lr = 1e-5
            args.weight_decay = 1
            args.id_ckpt = 1
            args.upweight = 50
        elif args.method == 'DeTT':
            args.lr = 1e-5
            args.weight_decay = 1e-3
            args.id_ckpt = 1
            args.upweight = 50
        elif args.method == 'aux_wt':
            args.lr = 5e-4
            args.weight_decay = 1e-1
            if args.alpha is None:
                args.alpha = 0.05
                args.beta = 4
        else: 
            raise NotImplementedError
        args.log_every = (int(10 * 128 / args.batch_size)//10+1) * 10 # roughly 1280/batch_size
        args.widx = 2
    
    elif args.dataset == 'CelebA':
        args.target_name = "Blond_Hair"
        args.confounder_names = ["Male"]
        if args.n_epochs is None: args.n_epochs = 60
        if args.method == 'ERM':
            args.lr = 1e-4
            args.weight_decay = 1e-4
            args.save_preds_at = [0, 1, 2]
        elif args.method == 'KD':
            args.lr = 1e-5
            args.weight_decay = 1e-1
        elif args.method == 'SimKD':
            args.lr = 5e-5
            args.weight_decay = 1e-1
        elif args.method == 'JTT':
            args.lr = 1e-5
            args.weight_decay = 1e-1
            args.id_ckpt = 1
            args.upweight = 50
        elif args.method == 'DeTT':
            args.lr = 1e-5
            args.weight_decay = 1e-3
            args.id_ckpt = 1
            args.upweight = 50
        elif args.method == 'aux_wt':
            args.lr = 5e-4
            args.weight_decay = 1e-2
            args.alpha = 0.1
            args.beta = 3.5
        else: 
            raise NotImplementedError
        args.log_every = (int(80 * 128 / args.batch_size)//10+1) * 30 # roughly 30720/batch_size
        args.widx = 3

    elif args.dataset == 'MultiNLI':
        args.target_name = "gold_label_random"
        args.confounder_names = ["sentence2_has_negation"]
        args.save_step = 1
        args.n_epochs = 5
        if args.method == 'ERM':
            args.lr = 2e-5
            args.weight_decay = 0
            args.save_preds_at = [0, 1, 2]
        elif args.method == 'KD':
            args.lr = 2e-5
            args.weight_decay = 0
        elif args.method == 'JTT':
            args.lr = 1e-5
            args.weight_decay = 1e-1
            args.id_ckpt = 1
            args.upweight = 6
        elif args.method == 'aux_wt':
            args.lr = 2e-5
            args.weight_decay = 1e-2
            args.alpha = 0.2
            args.beta = 3
        else: 
            raise NotImplementedError
        
        args.log_every = 2000
        args.widx = 5

    
    elif args.dataset == 'jigsaw':
        args.target_name = "toxicity"
        args.confounder_names = ["identity_any"]
        args.save_step = 1
        args.n_epochs = 3
        if args.method == 'ERM':
            args.lr = 2e-5
            args.weight_decay = 0
            args.save_preds_at = [0, 1, 2]
        elif args.method == 'KD':
            args.lr = 2e-5
            args.weight_decay = 0
        elif args.method == 'JTT':
            args.lr = 1e-5
            args.weight_decay = 1e-1
            args.id_ckpt = 1
            args.upweight = 6
        elif args.method == 'aux_wt':
            args.lr = 2e-5
            args.weight_decay = 1e-2
            args.alpha = 0.05
            args.beta = 3
        else: 
            raise NotImplementedError
        args.log_every = 2000
        args.widx = 3
        
        
    if ((args.model.startswith("bert") or args.model.startswith("distilbert")) and args.use_bert_params): 
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    if args.model.startswith("bert") or args.model.startswith("distilbert"): # and args.model != "bert": 
        if args.use_bert_params:
            print("\n"*5, f"Using bert params", "\n"*5)
        else: 
            print("\n"*5, f"WARNING, Using {args.model} without using BERT HYPER-PARAMS", "\n"*5)

    if args.teacher is not None:
        args.batch_size = 64
    if args.save_step is None:
        args.save_step = args.n_epochs//10
    check_args(args)
    
    
    
    # set directory for storing results
    if args.method in ['KD', 'SimKD', 'DeTT']:
        teacher_logs_dir = os.path.join(args.logs_dir, args.dataset, args.teacher+'_'+str(args.seed))
        args.logs_dir = os.path.join(args.logs_dir, args.dataset, 
                                     '_'.join([args.teacher, args.method, arg.kd_alpha, args.model, str(args.seed)]))
    elif args.method == 'JTT':
         args.logs_dir = os.path.join(args.logs_dir, args.dataset,  
                                 '_'.join([args.model, args.method, str(args.seed)]))
    elif args.method == 'ERM':
         args.logs_dir = os.path.join(args.logs_dir, args.dataset,  
                                 '_'.join([args.model, str(args.seed)]))
    elif args.method == 'aux_wt':
        teacher_logs_dir = os.path.join(args.logs_dir, args.dataset, args.teacher+'_'+str(args.seed))
        args.logs_dir = os.path.join(args.logs_dir, args.dataset, 
                                     '_'.join([args.teacher, args.method, str(args.alpha), 
                                               str(args.beta), args.model, str(args.seed)]))
        
    
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)
    
    ## Initialize logs
    log_file_path = os.path.join(args.logs_dir, 'train.log')
    logger = Logger(log_file_path)
    log_args(args, logger)
    set_seed(args.seed)
    
    # Data
    # Test data for label_shift_step is not implemented yet
    data_start_time = time.time()
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        file_path = os.path.join('./results', args.dataset, 
                                 '_'.join([args.target_name] + list(map(str, args.confounder_names)) + 
                                          ['dataset', f'{args.seed}.pkl']))

        train_data, val_data, test_data = prepare_data(args, train=True)
        with open(file_path, 'wb') as file:
            data_to_save = {'train_data': train_data, 'val_data': val_data, 'test_data': test_data}
            pickle.dump(data_to_save, file)

        # load from a .pkl file (to make it faster) - comment out in general
        # with open(file_path, 'rb') as file:
        #     data = pickle.load(file)
        #     train_data = data['train_data']
        #     val_data = data['val_data']
        #     test_data = data['test_data']
    
    elif args.shift_type == 'label_shift_step': # never used
        train_data, val_data = prepare_data(args, train=True)

    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':4, 'pin_memory':True}
    train_loader = get_loader(train_data, train=True, **loader_kwargs)
    val_loader = get_loader(val_data, train=False, **loader_kwargs)
    if test_data is not None:
        test_loader = get_loader(test_data, train=False, **loader_kwargs)
    
    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes
    
    logger.write("{:.2g} minutes for data processing\n".format((time.time() - data_start_time)/60))
    logger.flush()

    log_data(data, logger)
    
    

    models = {}
    ## Initialize model
    logger.write("-" * 50 + '\n')
    student = get_model(args.model.replace('-pt', ''), 'pt' in args.model, n_classes)
    models['student'] = student.to(device=args.device)
    logger.flush()

    # load teacher
    if args.method in ['SimKD', 'KD', 'DeTT', 'aux_wt']:
        if 'group_DRO' in args.teacher:
            teacher = torch.load(os.path.join('/home/anmolreddy/projects/group_DRO/logs/',
                                              args.dataset, 'resnet50', 'best_model.pth'))
        else:
            teacher = get_model(args.teacher.split('-pt')[0], 'pt' in args.teacher, n_classes)
            teacher_ckpt = torch.load(os.path.join(teacher_logs_dir, f'{args.teacher_type}_ckpt.pth.tar'))
            teacher.load_state_dict(teacher_ckpt['model'])
        teacher.eval()
        models['teacher'] = teacher.to(device=args.device)
        logger.write(f"teacher loaded: {os.path.join(teacher_logs_dir, f'{args.teacher_type}_ckpt.pth.tar')}\n")
    
    if args.method in ['SimKD', 'DeTT']:
        if args.model.startswith("bert") or args.model.startswith("distilbert"):
            models['teacher'] = FeatBert(models['teacher'])
            models['student'] = FeatBert(models['student'])
        else:
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
    
    if args.method in ['JTT', 'DeTT']:
        saved_preds_df = pd.read_csv(os.path.join('results', args.dataset,
                                                   '_'.join([args.model, str(args.seed)]),
                                                   f'epoch-{args.id_ckpt}_predictions.csv')
                                     )
        wrong_idxs = saved_preds_df.loc[saved_preds_df['wrong_pred'] == 1, 'index'].values
        logger.write("upweighting {:.2f}% of the dataset from epoch-{}".format(100 * len(wrong_idxs)/len(train_data), args.id_ckpt))
        train_data.update_weights(wrong_idxs, args.upweight)
    
    if args.method == 'aux_wt':
        basic_block = True
        aux_net = SemiResNet(models['student'])
        sample_inputs = next(iter(data['train_loader']))[0].to(device=args.device)
        if basic_block:
            converter = BasicBlock(32 * 2**feature_level, 64 * 2**feature_level, stride=1).to(device=args.device)
            features = converter(aux_net(sample_inputs))
            d = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(features), 1).shape[1]
            projector = Projector(d, converter).to(device=args.device)
        else:
            features = aux_net(sample_inputs)
            d = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(features), 1).shape[1]
            projector = Projector(d).to(device=args.device)
        models['aux_net'] = aux_net
        models['projector'] = projector
    
    logger.flush()
    
    train_csv_logger = CSVBatchLogger(os.path.join(args.logs_dir, 'train.csv'), train_data.n_groups)
    val_csv_logger =  CSVBatchLogger(os.path.join(args.logs_dir, 'val.csv'), train_data.n_groups)
    test_csv_logger =  CSVBatchLogger(os.path.join(args.logs_dir, 'test.csv'), train_data.n_groups)
    
    train(models, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args)
    
    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()
    
    logger.write("{:.2g}h for running\n".format((time.time() - data_start_time)/3600))



def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio
    if args.method in ['KD', 'SimKD', 'DeTT', 'aux_wt']:
        assert args.teacher is not None
    if args.method in ['JTT', 'DeTT']:
        assert args.upweight is not None
        assert args.id_ckpt is not None


if __name__=='__main__':
    main()
