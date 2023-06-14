import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=10, scheduler=None):
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

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
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
        gamma=args.gamma,
        adj=adjustments,
        normalize_loss=args.use_normalized_loss,
        min_var_weight=args.minimum_variational_weight)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08)
    else:
        scheduler = None

    best_val_acc = 0

    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            dataset=dataset['val_data'])
        run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                dataset=dataset['test_data'])
            run_epoch(
                epoch, model, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                None, test_csv_logger, args,
                is_training=False)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler:
            val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        if args.save_best:
            curr_val_acc = val_loss_computer.avg_acc
            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')
        logger.write('\n')
