import csv
import random

import torch
import numpy as np
from sklearn.metrics import average_precision_score
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(save_file_path, epoch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    save_states = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(save_states, save_file_path)


class AverageMeter(object):
    """Computes and stores the average and current value."""

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


class Logger(object):
    """CSV logger for training metrics."""

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')
        self.logger.writerow(header)
        self.header = header

    def __del__(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()
        return n_correct_elems / batch_size


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def write_to_batch_logger(batch_logger, epoch, i, data_loader, losses, accuracies, current_lr):
    if batch_logger is not None:
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses,
            'acc': accuracies,
            'lr': current_lr,
        })


def write_to_epoch_logger(epoch_logger, epoch, losses, accuracies, current_lr):
    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses,
            'acc': accuracies,
            'lr': current_lr
        })


def cross_modality_pretrain(conv1_weight, channel):
    """Transform 3-channel pretrained conv1 weight to a target number of channels."""
    S = 0
    for i in range(3):
        S += conv1_weight[:, i, :, :]
    avg = S / 3.
    new_conv1_weight = torch.FloatTensor(64, channel, 7, 7)
    for i in range(channel):
        new_conv1_weight[:, i, :, :] = avg.data
    return new_conv1_weight


def calculate_mAP(outputs, targets):
    """Compute mean Average Precision (mAP) for single-label classification.

    Args:
        outputs: Predicted probabilities, shape [n_samples, n_classes]
        targets: Ground truth labels, shape [n_samples]

    Returns:
        float: mAP value
    """
    n_classes = outputs.shape[1]
    n_samples = targets.shape[0]
    one_hot = np.zeros((n_samples, n_classes))
    for i in range(n_samples):
        if targets[i] < n_classes:
            one_hot[i, int(targets[i])] = 1

    ap_scores = []
    for c in range(n_classes):
        if np.sum(one_hot[:, c]) > 0:
            ap = average_precision_score(one_hot[:, c], outputs[:, c])
            if not np.isnan(ap):
                ap_scores.append(ap)
    return np.mean(ap_scores) if ap_scores else 0.0
