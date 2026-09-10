import torch
import numpy as np
from datetime import datetime

def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def _progress_points(total, count):
    if total <= 0:
        return set()
    return {min(total, max(1, int(np.ceil(total * i / count)))) for i in range(1, count + 1)}

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def train(model, dataloader, loss_fn_original, loss_fn_aug, optimizer, scheduler=None, model_ema=None, device='cuda', grad_accum_steps=1, verbose=False, epoch=None, total_epochs=None, optimizer_step=0, log_count=5):

    losses = torch.zeros(len(dataloader))
    log_batches = _progress_points(len(dataloader), log_count)

    model.train()

    for batch, (x, y, aug_indices) in enumerate(dataloader):
        x, y, aug_indices = x.to(device), y.to(device), aug_indices.to(device)
        pred = model(x)
        loss_original = loss_fn_original(pred, y)
        loss_aug = loss_fn_aug(pred, y)

        loss = loss_original * ~aug_indices.reshape(-1, 1) + loss_aug * aug_indices.reshape(-1, 1)
        loss = loss.sum()

        loss = loss / grad_accum_steps
        loss.backward()

        if (batch + 1) % grad_accum_steps == 0 or (batch + 1) == len(dataloader):
            optimizer.step()
            optimizer_step += 1

            if scheduler is not None:
                scheduler.step()
                
            model.zero_grad()

            if model_ema is not None:
                model_ema.update(model)

        losses[batch] = loss.detach().cpu()

        if verbose is False and (batch + 1) in log_batches:
            avg_loss = losses[:batch+1].mean().item()
            epoch_text = f'{epoch + 1}/{total_epochs}' if epoch is not None and total_epochs is not None else '-'
            print(
                f'[{timestamp()}] Train epoch={epoch_text} batch={batch+1}/{len(dataloader)} '
                f'optimizer_step={optimizer_step} lr={get_lr(optimizer):.8f} '
                f'loss={loss.item():.4f} avg_loss={avg_loss:.4f}'
            )
    
    return losses, optimizer_step

def evaluate(model, dataloader, device='cuda', verbose=False):
    num_samples = len(dataloader.dataset)
    num_categories = dataloader.dataset.num_categories
    batch_size = dataloader.batch_size

    preds = torch.zeros((num_samples, num_categories))
    targets = torch.zeros((num_samples, num_categories))

    model.eval()
    log_batches = _progress_points(len(dataloader), 3)

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds[batch*batch_size: (batch+1)*batch_size, :] = pred.detach().cpu()
            targets[batch*batch_size: (batch+1)*batch_size, :] = y.detach().cpu()

            if verbose is False and (batch + 1) in log_batches:
                print(f'[{timestamp()}] Validating batch={batch+1}/{len(dataloader)}')

    return preds, targets
