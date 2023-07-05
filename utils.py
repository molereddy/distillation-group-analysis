import sys, os, torch, csv
import numpy as np
from matplotlib import pyplot as plt


def plot_train_progress(test_avg_accs, test_ub_accs, test_wg_accs, save_at):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    
    ax = axes[0, 0]
    ax.plot(range(len(test_avg_accs)), test_avg_accs)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Overall test acc (avg)')
    
    ax = axes[0, 1]
    ax.plot(range(len(test_ub_accs)), test_ub_accs)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Unbiased test acc')
    
    ax = axes[0, 2]
    ax.plot(range(len(test_wg_accs)), test_wg_accs)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Worst-group test acc')
    
    ax = axes[1, 0]
    avg_ub = [avg_acc-ub_acc for avg_acc, ub_acc in zip(test_avg_accs, test_ub_accs)]
    ax.plot(range(len(avg_ub)), avg_ub)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Average-Unbiased acc')
    
    ax = axes[1, 1]
    avg_wg = [avg_acc-wg_acc for avg_acc, wg_acc in zip(test_avg_accs, test_wg_accs)]
    ax.plot(range(len(avg_wg)), avg_wg)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Average-Worst acc')

    ax = axes[1, 2]
    ub_wg = [ub_acc-wg_acc for ub_acc, wg_acc in zip(test_ub_accs, test_wg_accs)]
    ax.plot(range(len(ub_wg)), ub_wg)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Unbiased-Worst acc')

    plt.tight_layout()
    plt.savefig(save_at)
    plt.close()


def save_checkpoint(state, logs_dir, is_best=False, is_last=False, save_freq=20):
    if state['epoch']%save_freq==0:
        torch.save(state, os.path.join(logs_dir, f'{state["epoch"]}_ckpt.pth.tar'))
    if is_best:
        torch.save(state, os.path.join(logs_dir, 'best_ckpt.pth.tar'))
    if is_last:
        torch.save(state, os.path.join(logs_dir, 'last_ckpt.pth.tar'))


class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode='w'):
        columns = ['epoch', 'batch']
        for idx in range(n_groups):
            columns.append(f'avg_loss_group:{idx}')
            columns.append(f'avg_acc_group:{idx}')
            columns.append(f'processed_data_count_group:{idx}')
            columns.append(f'update_data_count_group:{idx}')
            columns.append(f'update_batch_count_group:{idx}')
        columns.append('avg_actual_loss')
        columns.append('avg_per_sample_loss')
        columns.append('avg_acc')
        columns.append('model_norm_sq')
        columns.append('reg_loss')

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode=='w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict['epoch'] = epoch
        stats_dict['batch'] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write('\n')

