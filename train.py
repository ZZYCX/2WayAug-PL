import argparse
import random
import numpy as np
import torchmetrics
from mlcpl.loss import *
from torch.utils.data import DataLoader
import torchvision
import torch
from mlcpl.helper import *
import os
import sys
from pathlib import Path
import torchmetrics
from models import Model
import config
import time
from train_eval_fn import *
from mlcpl.sample_mix import *
from mlcpl.curriculum_labeling import CurriculumLabeling


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def seed_everything(seed):
    """Seed training RNGs before model creation or any CUDA work."""
    # Required for deterministic CUDA matrix multiplication (CUDA >= 10.2).
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(dataset, *, shuffle, num_workers=None):
    """Recreate workers each epoch so curriculum-label updates remain visible."""
    if num_workers is None:
        num_workers = int(os.environ.get('DATALOADER_NUM_WORKERS', str(config.num_workers)))
    if num_workers < 0:
        raise ValueError('DATALOADER_NUM_WORKERS must be non-negative')
    # Separate generators keep training shuffle independent of validation.
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset, batch_size=config.batch_size, num_workers=num_workers,
        shuffle=shuffle, persistent_workers=False,
        worker_init_fn=seed_worker, generator=generator,
    )


def main(resume=None):
    seed_everything(config.seed)
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

    train_dataloader = make_dataloader(train_dataset_mix, shuffle=True)
    valid_dataloader = make_dataloader(valid_dataset, shuffle=False)
    print(f'DataLoader num_workers={train_dataloader.num_workers}, seed={config.seed}')
    
    model = model.to(device)
    parameters = add_weight_decay(model, weight_decay=config.weight_decay)

    ema = ModelEma(model, config.ema)

    optimizer = torch.optim.Adam(parameters, lr=config.lr, weight_decay=0)
    steps_per_epoch = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=steps_per_epoch, epochs=config.epochs, pct_start=0.2)

    log_dir = os.path.join(output_dir, 'log')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = MultiLogger(log_dir)

    best_score = 0
    best_at_epoch = -1
    optimizer_step = 0

    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        if (checkpoint.get('stage') != 'pre_pseudo'
                or checkpoint['next_epoch'] != config.E):
            raise ValueError('Resume requires a pre-pseudo checkpoint matching config.E')
        model.load_state_dict(checkpoint['model_state_dict'])
        ema.module.load_state_dict(checkpoint['ema_state_dict'])
        ema.decay = checkpoint['ema_decay']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        optimizer_step = checkpoint['optimizer_step']
        best_score = checkpoint['best_score']
        best_at_epoch = checkpoint['best_at_epoch']
        start_epoch = checkpoint['next_epoch']
        logger.tag_counts = checkpoint['logger_tag_counts']
        if logger.excellog is not None:
            logger.excellog.sheets = checkpoint['logger_sheets']
        random.setstate(checkpoint['python_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if torch.cuda.is_available() and checkpoint['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        train_dataloader.generator.set_state(checkpoint['train_generator_state'])
        valid_dataloader.generator.set_state(checkpoint['valid_generator_state'])
        print(f'[{timestamp()}] Resumed {resume}; generating pseudo-labels before epoch {start_epoch + 1}')
        # This update was deliberately not included in the saved checkpoint.
        train_dataset_cl.update(
            ema.module,
            batch_size=config.batch_size,
            num_workers=train_dataloader.num_workers,
            thresholds=config.thresholds,
            device=device,
        )

    for epoch in range(start_epoch, config.epochs):
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

        if epoch == config.E - 1:
            # Save after validation, before the first pseudo-label update.
            checkpoint_path = os.path.join(output_dir, '10%_pre_pseudo_checkpoint.pth')
            torch.save({
                'stage': 'pre_pseudo',
                'epoch': epoch,  # Zero-based completed epoch (19 for epoch 20).
                'next_epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.module.state_dict(),
                'ema_decay': ema.decay,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_step': optimizer_step,
                'best_score': best_score,
                'best_at_epoch': best_at_epoch,
                'logger_tag_counts': logger.tag_counts,
                'logger_sheets': logger.excellog.sheets if logger.excellog is not None else {},
                'python_rng_state': random.getstate(),
                'numpy_rng_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'train_generator_state': train_dataloader.generator.get_state(),
                'valid_generator_state': valid_dataloader.generator.get_state(),
            }, checkpoint_path)
            print(f'[{timestamp()}] Pre-pseudo checkpoint saved: {checkpoint_path}')

        if epoch >= (config.E-1): # -1 becasue epoch starts from 0
            train_dataset_cl.update(
                ema.module,
                batch_size=config.batch_size,
                num_workers=train_dataloader.num_workers,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None, help='Resume a pre-pseudo training checkpoint')
    args = parser.parse_args()
    Path('output').mkdir(parents=True, exist_ok=True)
    log_path = Path('output') / f'{time.strftime("%Y%m%d_%H%M%S")}.txt'
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_path, 'w', encoding='utf-8') as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f'Training log file: {log_path}')
            main(resume=args.resume)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
