import torch
import torch.nn.functional as F
from utils import AverageMeter, calculate_mAP


def obtain_input(batch, device):
    """Extract input tensors from batch data."""
    visual, audio, labels = batch['clip'], batch['audio'], batch['target']
    return visual.to(device), audio.to(device), labels.to(device)


def accuracy(output, target):
    """Computes the top-1 accuracy."""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size)


def val_epoch(model, data_loader, device, criterion, val_logger=None,
              tb_writer=None, log_tag='val', epoch=None, train_modality='both'):
    """
    Validate model for one epoch.

    Returns:
        Dictionary containing validation metrics (acc, mAP, loss)
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    all_outputs = []
    all_targets = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            visual, audio, target = obtain_input(batch, device)
            batch_size = visual.size(0)

            if train_modality in ('visual', 'audio'):
                output_dict = model(visual, audio, train_modality)
            else:
                output_dict = model(visual, audio)

            fusion_output = output_dict['out']

            all_outputs.append(fusion_output.detach().cpu())
            all_targets.append(target.detach().cpu())

            loss = criterion(fusion_output, target)
            prec1 = accuracy(fusion_output, target)

            losses.update(loss.item(), batch_size)
            top1.update(prec1.item(), batch_size)

    # Calculate mAP
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_probs = F.softmax(all_outputs, dim=1)
    mAP = calculate_mAP(all_probs.numpy(), all_targets.numpy())

    print('\n{} Results: Acc {top1.avg:.3f} Loss {loss.avg:.5f} mAP {mAP:.3f}'.format(
        log_tag, top1=top1, loss=losses, mAP=mAP))

    if val_logger is not None and epoch is not None:
        val_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': top1.avg,
            'mAP': mAP
        })

    if tb_writer is not None and epoch is not None:
        tb_writer.add_scalar(f'{log_tag}/loss', losses.avg, epoch)
        tb_writer.add_scalar(f'{log_tag}/acc', top1.avg, epoch)
        tb_writer.add_scalar(f'{log_tag}/mAP', mAP, epoch)

    return {
        'acc': top1.avg,
        'mAP': mAP,
        'loss': losses.avg
    }
