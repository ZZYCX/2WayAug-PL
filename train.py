import torchmetrics
from mlcpl.loss import *
from torch.utils.data import DataLoader
import torchvision
import torch
from mlcpl.helper import *
import os
from pathlib import Path
import torchmetrics
from models import Model
import config
import time
from train_eval_fn import *
from mlcpl.sample_mix import *
from mlcpl.curriculum_labeling import CurriculumLabeling

def main():
    device = config.device
    output_dir = 'output/train'

    train_dataset = config.train_dataset
    valid_dataset = config.valid_dataset

    train_dataset.transform = torchvision.transforms.Compose([
        *config.data_aug,
        torchvision.transforms.Resize(config.image_size),
        torchvision.transforms.ToTensor(),
    ])
    valid_dataset.transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config.image_size),
        torchvision.transforms.ToTensor(),
    ])

    train_dataset_cl = CurriculumLabeling(train_dataset, transform_for_update=torchvision.transforms.Compose([
        torchvision.transforms.Resize(config.image_size),
        torchvision.transforms.ToTensor(),
    ]))

    train_dataset_mix = LogicMix(train_dataset_cl, probability=config.probability, mix_num_samples=config.num_samples)

    num_categories = train_dataset.num_categories

    model = Model(num_categories)

    loss_fn_original = PartialAsymmetricLoss(gamma_neg=config.gamma_n, gamma_pos=config.gamma_p, clip=config.m, reduction=None)
    loss_fn_aug = PartialAsymmetricLoss(gamma_neg=config.omega_n, gamma_pos=config.omega_p, clip=config.n, reduction=None)

    validation_metrics = {
        'mAP@C': torchmetrics.classification.MultilabelAveragePrecision(num_categories, average='macro', validate_args=False),
    }
    monitor_validation_metric_name = 'mAP@C'

    train_dataloader = DataLoader(train_dataset_mix, batch_size=config.batch_size, num_workers=0, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False)
    
    model = model.to(device)
    parameters = add_weight_decay(model, weight_decay=config.weight_decay)

    ema = ModelEma(model, config.ema)

    optimizer = torch.optim.Adam(parameters, lr=config.lr, weight_decay=0)
    steps_per_epoch = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.lr, steps_per_epoch=steps_per_epoch, epochs=config.epochs, pct_start=0.2)

    log_dir = os.path.join(output_dir, 'log')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = MultiLogger(log_dir)

    best_score = 0
    best_at_epoch = -1
    optimizer_step = 0

    for epoch in range(config.epochs):
        print(f'[{timestamp()}] Epoch start: {epoch+1}/{config.epochs}')
        epoch_start_time = time.time()

        # Train Loop
        losses, optimizer_step = train(
            model,
            train_dataloader,
            loss_fn_original,
            loss_fn_aug,
            optimizer,
            scheduler,
            model_ema=ema,
            grad_accum_steps=config.accum_step,
            device=device,
            epoch=epoch,
            total_epochs=config.epochs,
            optimizer_step=optimizer_step,
        )

        logger.add('train_loss', torch.mean(losses).detach().numpy())

        # Valid Loop
        preds, targets = evaluate(ema.module, valid_dataloader, device=device)

        # Calculate metrics and logging
        for name, metric in validation_metrics.items():
            result = metric(preds, targets).detach().numpy()
            logger.add('valid_'+name, result)
        
            if name == monitor_validation_metric_name:
                current_score = result
                print(
                    f'[{timestamp()}] Epoch validation: epoch={epoch+1}/{config.epochs} '
                    f'optimizer_step={optimizer_step} lr={get_lr(optimizer):.8f} '
                    f'validation_mAP={current_score:.4f}'
                )

        if current_score > best_score:
            best_score = current_score
            best_at_epoch = epoch
            print(f'[{timestamp()}] New best {monitor_validation_metric_name}: {best_score:.4f}')
            torch.save(ema.module.state_dict(), os.path.join(output_dir, 'best.pth'))

        if epoch >= (config.E-1): # -1 becasue epoch starts from 0
            train_dataset_cl.update(
                ema.module,
                batch_size=config.batch_size,
                thresholds=config.thresholds,
                device=device,
            )

        epoch_end_time = time.time()
        print(f'[{timestamp()}] Epoch end: {epoch+1}/{config.epochs}. Total time: {(epoch_end_time-epoch_start_time):.2f} sec')
        print()

        if config.early_stopping is not None:
            if epoch - best_at_epoch >= config.early_stopping:
                print(f'[{timestamp()}] Early stopping.')
                break

    logger.flush()

if __name__=='__main__':
    main()
