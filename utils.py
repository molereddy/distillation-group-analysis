import sys, os, torch, csv, torchvision
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt

def precision_recall(outputs, targets):

    TP = np.sum(outputs & targets)
    FP = np.sum(outputs & ~targets)
    TN = np.sum(~outputs & ~targets)
    FN = np.sum(~outputs & targets)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)

    return precision, recall


def plot_train_progress(test_avg_accs, test_ub_accs, test_wg_accs, save_at):
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    
    ax = axes # [0]
    ax.plot(range(len(test_avg_accs)), test_avg_accs, label='avg acc')
    ax.plot(range(len(test_ub_accs)), test_ub_accs, label='unbiased acc')
    ax.plot(range(len(test_wg_accs)), test_wg_accs, label='worst acc')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Acc')
    ax.legend()
    
    # ax = axes[1]
    # avg_ub = [avg_acc-ub_acc for avg_acc, ub_acc in zip(test_avg_accs, test_ub_accs)]
    # avg_wg = [avg_acc-wg_acc for avg_acc, wg_acc in zip(test_avg_accs, test_wg_accs)]
    # ub_wg = [ub_acc-wg_acc for ub_acc, wg_acc in zip(test_ub_accs, test_wg_accs)]
    # ax.plot(range(len(avg_ub)), avg_ub, label='avg - ub')
    # ax.plot(range(len(avg_wg)), avg_wg, label='avg - wg')
    # ax.plot(range(len(ub_wg)), ub_wg, label='ub - ub')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Difference in accuracies')
    # ax.legend()

    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
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
        # self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        # self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        # self.console.close()
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


def get_model(model, pretrained=True, n_classes=2, dataset='MultiNLI'):
    weights_dict = {'weights': 'DEFAULT'} if pretrained else {}
    if model == "resnet50":
        model = torchvision.models.resnet50(**weights_dict)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model == "resnet18":
        model = torchvision.models.resnet34(**weights_dict)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model.startswith('bert'):
        if dataset == "MultiNLI":
            from pytorch_transformers import BertConfig, BertForSequenceClassification
            config_class = BertConfig
            model_class = BertForSequenceClassification
            config = config_class.from_pretrained("bert-base-uncased",
                                                num_labels=3,
                                                finetuning_task="mnli")
            model = model_class.from_pretrained("bert-base-uncased",
                                                from_tf=False,
                                                config=config)
        elif dataset == "jigsaw":
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                model,
                num_labels=n_classes)
            print(f'n_classes = {n_classes}')
        else: 
            raise NotImplementedError
    elif model.startswith('distilbert'):
        if dataset == "MultiNLI":
            from pytorch_transformers import DistilBertConfig, DistilBertForSequenceClassification
            config_class = DistilBertConfig
            model_class = DistilBertForSequenceClassification
            config = config_class.from_pretrained("distilbert-base-uncased",
                                                num_labels=3,
                                                finetuning_task="mnli")
            model = model_class.from_pretrained("distilbert-base-uncased",
                                                from_tf=False,
                                                config=config)
        elif dataset == "jigsaw":
            from transformers import DistilBertForSequenceClassification
            model = DistilBertForSequenceClassification.from_pretrained(
                model,
                num_labels=n_classes)
            print(f'n_classes = {n_classes}')
        else: 
            raise NotImplementedError
    else:
        raise ValueError(f"{model} Model not recognized.")

    return model


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

