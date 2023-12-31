import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LossComputer:
    def __init__(self, dataset, adj=None, min_var_weight=0, 
                 normalize_loss=False, args=None):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.min_var_weight = min_var_weight
        self.normalize_loss = normalize_loss
        self.alpha = 1
        
        if args is not None:
            self.device = args.device
            self.alpha = args.kd_alpha
        
        self.n_groups = dataset.n_groups
        self.group_counts = dataset.group_counts().to(device=self.device)
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_str = dataset.group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(device=self.device)
        else:
            self.adj = torch.zeros(self.n_groups).float().to(device=self.device)
        
        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(device=self.device)/self.n_groups

        self.reset_stats()

    def loss_erm(self, yhat, y, group_idx=None, is_training=False, wt=None):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        if wt is not None: 
            per_sample_losses *= wt
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)
        # compute overall loss
        actual_loss = per_sample_losses.mean()
        weights = None
        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def loss_kd(self, yhat, y, teacher_yhat, group_idx=None, is_training=False, wt=None):
        # compute per-sample and per-group losses
        per_sample_ce_losses = self.criterion(yhat, y)
        per_sample_kd_losses = 3*3*nn.KLDivLoss(reduction='none')(F.log_softmax(yhat/3, dim=1),
                                                              F.softmax(teacher_yhat/3, dim=1))
        per_sample_kd_losses = torch.sum(per_sample_kd_losses, dim=1)
        if wt is not None: 
            per_sample_kd_losses *= wt
            per_sample_ce_losses *= wt
        per_sample_losses = self.alpha * per_sample_kd_losses + (1-self.alpha) * per_sample_ce_losses
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)
        # compute overall loss
        actual_loss = per_sample_losses.mean()
        weights = None
        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss
    
    def loss_mse(self, yhat, y, sft, tft, group_idx=None, is_training=False, wt=None):
        # compute per-sample and per-group losses
        per_sample_ce_losses = self.criterion(yhat, y)
        per_sample_kd_losses = torch.mean((sft - tft) ** 2, dim=(1, 2, 3))
        if wt is not None: 
            per_sample_kd_losses *= wt
            per_sample_ce_losses *= wt
        per_sample_losses = self.alpha * per_sample_kd_losses + (1-self.alpha) * per_sample_ce_losses
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat, 1) == y).float(), group_idx)
        # compute overall loss
        actual_loss = per_sample_losses.mean()
        weights = None
        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(device=self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).to(device=self.device)
        self.update_data_counts = torch.zeros(self.n_groups).to(device=self.device)
        self.update_batch_counts = torch.zeros(self.n_groups).to(device=self.device)
        self.avg_group_loss = torch.zeros(self.n_groups).to(device=self.device)
        self.avg_group_acc = torch.zeros(self.n_groups).to(device=self.device)
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts/denom
        curr_weight = group_count/denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + curr_weight*group_loss

        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + curr_weight*group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss

        # counts
        self.processed_data_counts += group_count
        self.update_data_counts += group_count
        self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training, target_group_idx=None):
        if logger is not None:
            logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}, Average sample loss: {self.avg_actual_loss.item():.3f}, Average acc: {self.avg_acc.item():.3f}\n')
            for group_idx in range(self.n_groups):
                logger.write(
                    f'\t{self.group_str(group_idx)}\t'
                    f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                    f'loss = {self.avg_group_loss[group_idx]:.3f}\t'
                    f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
            logger.flush()
        if not is_training and target_group_idx is not None: # i.e. when test/val epoch requests these metrics
            avg_acc = self.avg_acc.item()
            unbiased_acc = np.average([self.avg_group_acc[group_idx].item() for group_idx in range(self.n_groups)])
            if target_group_idx != -1:
                target_group_acc = self.avg_group_acc[target_group_idx].item()
            else:
                target_group_acc = 1
                for group_idx in range(self.n_groups):
                    target_group_acc = min(target_group_acc, self.avg_group_acc[group_idx].item())
            return avg_acc, unbiased_acc, target_group_acc
        
        
