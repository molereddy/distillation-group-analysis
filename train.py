import os, copy, pickle, types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from local_models import FeatResNet, Projector
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import AverageMeter, save_checkpoint, plot_train_progress, precision_recall
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule

def train_aux(models, loader, logger, args, num_aux_epochs=1):
    criterion = nn.CrossEntropyLoss(reduction='none')
    projector, aux_net = models['projector'], models['aux_net']
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, projector.parameters()), 
                            lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    for _, param in models['aux_net'].named_parameters():
        param.requires_grad = False
    projector.refresh()
    projector.train()
    
    total, corr = 0, 0
    for _ in range(num_aux_epochs):
        for batch in loader:
            x, y = batch[0].to(device=args.device), batch[1].to(device=args.device)
            yhat = projector(aux_net(x))
            
            _, predicted_labels = torch.max(yhat, 1)
            total += y.size(0)
            corr += (predicted_labels == y).sum().item()
            
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
    
    logger.write(f"Train accuracy of aux layer:{100 * corr/total:.2f}\n")

def reweigh_aux(models, loader, logger, args):
    projector, aux_net = models['projector'], models['aux_net']
    projector.eval()
    aux_net.eval()
    outputs, targets, indices, margins = [], [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x, y, _, ind = batch[0].to(device=args.device), batch[1], batch[2], batch[3]
            yhat = torch.softmax(projector(aux_net(x)), dim=1).to(device='cpu')
            output = torch.argmax(yhat, dim=1)
            
            outputs.append(output)
            targets.append(y)
            indices.append(ind)
            margins.append(torch.abs(yhat[:, 0] - yhat[:, 1]))
    
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)
    indices = np.concatenate(indices, axis=0)
    margins = np.concatenate(margins, axis=0)

    new_weights = np.exp(args.beta * margins[outputs != targets]**args.alpha)
    edited_indices = indices[outputs != targets]
    logger.write("re-weighting {:.2f}% of the dataset\n".format(100 * len(new_weights)/len(targets)))
    return edited_indices, new_weights


def run_epoch(epoch, models, optimizer, loader, loss_computer, \
              logger, csv_logger, args,
              is_training, show_progress=False, log_every=10, \
              scheduler=None, target_group_idx=None,
              train_dataset=None):
    
    if is_training:
        models['student'].train()
        if args.method in ['SimKD', 'DeTT']: models['simkd'].train()
        if ((args.model_type.startswith("bert") or args.model_type.startswith("distilbert"))and args.use_bert_params): # or (args.model_type == "bert"):
             models['student'].zero_grad()
    else:
        models['student'].eval()
        if args.method in ['SimKD', 'DeTT']: models['simkd'].eval()
    
    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    save_preds = args.method == 'ERM' and is_training and (epoch in args.save_preds_at)
    
    with torch.set_grad_enabled(is_training):
        n = len(prog_bar_loader)
        
        for batch_idx, batch in enumerate(prog_bar_loader):
            
            x = batch[0].to(device=args.device)
            y = batch[1].to(device=args.device)
            g = batch[2].to(device=args.device)
            data_idx = batch[3]
            
            if args.method in ['ERM', 'JTT']:
                if args.model_type.startswith("bert") :
                    input_ids = x[:, :, 0]
                    input_masks = x[:, :, 1]
                    segment_ids = x[:, :, 2]
                    outputs = models['student'](input_ids=input_ids,attention_mask=input_masks,token_type_ids=segment_ids,labels=y)[1] 
                elif args.model_type.startswith("distilbert"):
                    input_ids = x[:, :, 0]
                    input_masks = x[:, :, 1]
                    segment_ids = x[:, :, 2]
                    outputs = models['student'](input_ids=input_ids,attention_mask=input_masks,labels=y)[1] 
                else:
                    outputs = models['student'](x)
                
                loss_main = loss_computer.loss_erm(outputs, y, g, is_training, wt = None if args.method == 'ERM' else batch[4].to(device=args.device))
                    
            elif args.method in ['KD', 'dedier']:
                if args.model_type.startswith("bert"):
                    input_ids = x[:, :, 0]
                    input_masks = x[:, :, 1]
                    segment_ids = x[:, :, 2]
                    outputs = models['student'](input_ids=input_ids,attention_mask=input_masks,token_type_ids=segment_ids,labels=y)[1] 
                    teacher_logits = models['teacher'](input_ids=input_ids,attention_mask=input_masks,token_type_ids=segment_ids,labels=y)[1] 
                elif args.model_type.startswith("distilbert"):
                    input_ids = x[:, :, 0]
                    input_masks = x[:, :, 1]
                    segment_ids = x[:, :, 2]
                    outputs = models['student'](input_ids=input_ids,attention_mask=input_masks,labels=y)[1] 
                    teacher_logits = models['teacher'](input_ids=input_ids,attention_mask=input_masks,token_type_ids=segment_ids,labels=y,)[1] 
                else:
                    outputs = models['student'](x)
                    teacher_logits = models['teacher'](x)
                
                loss_main = loss_computer.loss_kd(outputs, y, teacher_logits, g, is_training, 
                                                  wt = None if args.method == 'KD' else batch[4].to(device=args.device))
            
            elif args.method in ['SimKD', 'DeTT']:
                sft_base = models['student'](x)[0][0]
                tft_base = models['teacher'](x)[0][0]
                tft_base = tft_base.detach()
                sft, tft, outputs = models['simkd'](sft_base, tft_base, models['teacher'].fc)
                loss_main = loss_computer.loss_mse(outputs, y, sft, tft, g, is_training, 
                                                   wt = None if args.method == 'SimKD' else batch[4].to(device=args.device))
            else:
                raise NotImplementedError
            
            if is_training:
                if ((args.model_type.startswith("bert") or args.model_type.startswith("distilbert")) and args.use_bert_params): 
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(models['student'].parameters(),
                                                   args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    models['student'].zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(models['student'], args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
                
            
            if batch_idx == 0 and save_preds:
                wrongness_flags = np.argmax(outputs.detach().cpu().numpy(), axis=1) != y.cpu().numpy()
                indices = data_idx.numpy()
                worst_group_flags = g.cpu().numpy() == args.widx
                predicted = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                
            elif save_preds:
                wrongness_flags = np.concatenate([
                    wrongness_flags,
                    np.argmax(outputs.detach().cpu().numpy(), axis=1) != y.cpu().numpy()
                ])
                indices = np.concatenate([indices, data_idx.numpy()])
                worst_group_flags = np.concatenate([
                    worst_group_flags, g.cpu().numpy() == args.widx
                ])
                predicted = np.concatenate([predicted, np.argmax(outputs.detach().cpu().numpy(), axis=1)])

        if save_preds:
            output_df = pd.DataFrame()
            output_df['index'] = indices
            output_df['wrong_pred'] = wrongness_flags
            output_df[f'worst_group_{args.widx}'] = worst_group_flags
            output_df['predicted'] = predicted
            prec, rec = precision_recall(wrongness_flags, worst_group_flags)
            logger.write('Worst group prediction: precision:{:.3f}, recall:{:.3f}\n'.format(prec, rec))
            # if epoch in args.save_preds_at:
            output_df = output_df.sort_values('index')
            csv_file_path = os.path.join(args.logs_dir, f'epoch-{epoch}_predictions.csv')
            output_df.to_csv(csv_file_path)
            logger.write('\nSaved predictions to csv file {}\n\n'.format(csv_file_path))
            
        if not is_training:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(models['student'], args))
            csv_logger.flush()
            return loss_computer.log_stats(logger, is_training, target_group_idx=-1)
        
        elif loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(models['student'], args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            loss_computer.reset_stats()

def train(models, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args):
    models['student'] = models['student'].to(device=args.device)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        dataset=dataset['train_data'],
        adj=adjustments,
        normalize_loss=args.use_normalized_loss,
        min_var_weight=args.minimum_variational_weight,
        args=args)

    
    trainable_list = nn.ModuleList([])
    trainable_list.append(models['student'])
    if args.method == 'SimKD': trainable_list.append(models['simkd'])

    if ((args.model_type.startswith("bert") or args.model_type.startswith("distilbert")) and args.use_bert_params): 
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in trainable_list.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay,},
            {"params": [p for n, p in trainable_list.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay":0.0,},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        t_total = len(dataset["train_loader"]) * args.n_epochs
        print(f"\nt_total is {t_total}\n")
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,  t_total=t_total)
    
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, 
                                        trainable_list.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        
        if args.scheduler: # not used finally
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.1,patience=5,threshold=0.0001,min_lr=0,eps=1e-08)
        else:
            scheduler = None

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0
    best_state_dict = copy.deepcopy(models['student'].state_dict())

    best_acc = 0   
    is_last = False
    test_wg_accs, test_avg_accs, test_ub_accs = [], [], []
    
    for epoch in range(args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        
        run_epoch(
            epoch, models, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
            train_dataset=dataset['train_data'] if args.method == 'dedier' else None)

        if args.method == 'dedier' and epoch % args.reweigh_at == 0:
            train_aux(models, dataset['train_loader'], logger, args)
            indxs, wts = reweigh_aux(models, dataset['train_loader'], logger, args)
            dataset['train_data'].update_weights(indxs, wts)
        
        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(dataset=dataset['val_data'], args=args)
        val_accs = run_epoch(
            epoch, models, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False)

        # Test set; don't print to avoid peeking
        logger.write(f'\nTest:\n')
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(dataset=dataset['test_data'], args=args)
            avg_acc, ub_acc, wg_acc = run_epoch(
                epoch, models, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                logger, test_csv_logger, args,
                is_training=False)
            test_avg_accs.append(avg_acc)
            test_ub_accs.append(ub_acc)
            test_wg_accs.append(wg_acc)

        # Inspect learning rates
        # if (epoch+1) % 1 == 0:
        #     for param_group in optimizer.param_groups:
        #         curr_lr = param_group['lr']
        #         logger.write('Current lr: %f\n' % curr_lr)

        # if args.scheduler:
        #     val_loss = val_loss_computer.avg_actual_loss
        #     scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        
        val_ub_acc = val_accs[1] # unbiased accuracy
        curr_val_acc = val_ub_acc
        curr_test_acc = ub_acc

        if curr_val_acc >= best_val_acc or epoch == 0:
            best_val_acc = curr_val_acc
            best_test_acc = curr_test_acc
            is_best = True
            best_state_dict = copy.deepcopy(models['student'].state_dict())
            best_epoch = epoch
        else:
            is_best = False
        if epoch == args.n_epochs-1:
            is_last = True
        state = {'epoch': epoch, 'val_acc': curr_val_acc, 'test_acc': curr_test_acc, 
                'ub_acc': ub_acc, 'avg_acc': avg_acc, 'wg_acc': wg_acc,
                'model': copy.deepcopy(models['student'].state_dict()),
                'best_epoch': best_epoch, 'best_val_acc': best_val_acc, 'best_test_acc': best_test_acc}
        if args.method == 'SimKD':
            state['simkd'] = copy.deepcopy(models['simkd'].state_dict())
        save_checkpoint(state, logs_dir = args.logs_dir, is_best=is_best, save_freq=args.save_step, is_last=is_last)
        
        logger.write(f'Current validation acc: {curr_val_acc:.4f}\n')
        logger.write(f'Current {epoch} test avg acc: {avg_acc:.4f}, unbiased acc: {ub_acc:.4f}, worst acc: {wg_acc:.4f}\n')
        if is_best: logger.write(f'New best!\n')
        logger.write(f'Best epoch {best_epoch} of val acc {best_val_acc:.4f}: avg acc {test_avg_accs[best_epoch]:.4f}, unbiased acc: {test_ub_accs[best_epoch]:.4f}, worst acc: {test_wg_accs[best_epoch]:.4f}\n')        

        logger.write('\n')
    with open(os.path.join(args.logs_dir, f'{100*test_avg_accs[best_epoch]:.2f}_{100*test_wg_accs[best_epoch]:.2f}.result'), 'w'):
        pass
    with open(os.path.join(args.logs_dir, 'train_history.pkl'), 'wb') as file:
        pickle.dump({'test_avg_accs': test_avg_accs,'test_ub_accs': test_ub_accs, 'test_wg_accs': test_wg_accs}, file)
        
    plot_train_progress(test_avg_accs, test_ub_accs, test_wg_accs, os.path.join(args.logs_dir, 'training_curves.png'))
