import os, copy, pickle
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy, save_checkpoint, plot_train_progress
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule

def run_epoch(epoch, model, optimizers, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=10, scheduler=None, teacher=None, target_group_idx=None):
    if is_training:
        model.train()
    else:
        model.eval()
    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            outputs = model(x)
            
            if teacher is None:
                loss_main = loss_computer.loss(outputs, y, g, is_training)
            else:
                teacher_logits = teacher(x)
                loss_main = loss_computer.loss_kd(outputs, y, teacher_logits, g, is_training)
            if is_training:
                for optimizer in optimizers:
                    optimizer.zero_grad()
                loss_main.backward()
                for optimizer in optimizers:
                    optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if not is_training:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            return loss_computer.log_stats(logger, is_training, target_group_idx=args.widx)
        
        elif loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            loss_computer.reset_stats()

def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset, teacher=None):
    model = model.cuda()

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        dataset=dataset['train_data'],
        adj=adjustments,
        normalize_loss=args.use_normalized_loss,
        min_var_weight=args.minimum_variational_weight)

    if args.finetune == 1:
        parameters_fc = model.fc.parameters()
        parameters_rest = filter(lambda p: p not in parameters_fc, model.parameters())
        optimizer = torch.optim.SGD(parameters_fc,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd)
        optimizer_slow = torch.optim.SGD(parameters_rest, lr=args.lr/100, momentum=0.9, weight_decay=args.wd/100)
        optimizers = [optimizer, optimizer_slow]
    else:
        optimizer = torch.optim.SGD(
           filter(lambda p: p.requires_grad, model.parameters()),
           lr=args.lr,
           momentum=0.9,
           weight_decay=args.wd)
        optimizers = [optimizer]
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[0],
            'min',
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08)
    else:
        scheduler = None

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0
    best_state_dict = copy.deepcopy(model.state_dict())
    midway = args.n_epochs//2

    best_acc = 0   
    is_last = False
    test_wg_accs, test_avg_accs, test_ub_accs = [], [], []
    
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, optimizers,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
            teacher=teacher)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            dataset=dataset['val_data'])
        run_epoch(
            epoch, model, optimizers,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False)

        # Test set; don't print to avoid peeking
        logger.write(f'\nTest:\n')
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                dataset=dataset['test_data'])
            avg_acc, ub_acc, wg_acc = run_epoch(
                epoch, model, optimizers,
                dataset['test_loader'],
                test_loss_computer,
                logger, test_csv_logger, args,
                is_training=False,
                target_group_idx=args.widx)
            curr_test_acc = test_loss_computer.avg_acc
            test_avg_accs.append(avg_acc)
            test_ub_accs.append(ub_acc)
            test_wg_accs.append(wg_acc)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizers[0].param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)
        if epoch == midway:
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

        if args.scheduler:
            val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        curr_val_acc = val_loss_computer.avg_acc

        if curr_val_acc >= best_val_acc or epoch == 0:
            best_val_acc = curr_val_acc
            best_test_acc = curr_test_acc
            is_best = True
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            is_best = False
        if epoch == args.n_epochs-1:
            is_last = True
        state = {'epoch': epoch,
                'val_acc': curr_val_acc,
                'test_acc': curr_test_acc,
                'model': copy.deepcopy(model.state_dict()),
                'best_epoch': best_epoch,
                'best_val_acc': best_val_acc,
                'best_test_acc': best_test_acc,
                'best_model' : best_state_dict,}
        save_checkpoint(state, logs_dir = args.logs_dir, is_best=is_best, save_freq=args.save_step, is_last=is_last)
        
        logger.write(f'Current validation acc: {curr_val_acc:.4f}\n')
        logger.write(f'Current {epoch} test avg acc: {avg_acc:.4f}, unbiased acc: {ub_acc:.4f}, worst acc: {wg_acc:.4f}\n')
        if is_best: logger.write(f'New best!\n')
        logger.write(f'Best epoch {best_epoch} of val acc {best_val_acc:.4f}: avg acc {best_test_acc:.4f}, unbiased acc: {test_ub_accs[best_epoch]:.4f}, worst acc: {test_wg_accs[best_epoch]:.4f}\n')        

        logger.write('\n')
        
    plot_train_progress(test_avg_accs, test_ub_accs, test_wg_accs, os.path.join(args.logs_dir, 'training_curves.png'))
    
    with open(os.path.join(args.logs_dir, 'train_history.pkl'), 'wb') as file:
        pickle.dump({'test_avg_accs': test_avg_accs,
                     'test_ub_accs': test_ub_accs,
                     'test_wg_accs': test_wg_accs
                    },
                    file)
    

def test(model, criterion, dataset, logger, test_csv_logger, args):
    model = model.cuda()
    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)
    logger.write(f'\nTest:\n')
    if dataset['test_data'] is not None:
        test_loss_computer = LossComputer(
            criterion,
            dataset=dataset['test_data'])
        run_epoch(
            0, model, optimizers,
            dataset['test_loader'],
            test_loss_computer,
            logger, test_csv_logger, args,
            is_training=False)
