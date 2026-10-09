import os
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from torch.utils.tensorboard import SummaryWriter

from utils import Logger, save_checkpoint, worker_init_fn
from trainer import TrainerFactory
from training_schedule import phase_for_epoch, validate_schedule


def get_dataset(cfg):
    """Create and return train and validation data loaders."""
    train_data = hydra.utils.instantiate(cfg.dataset, mode="train")
    val_data = hydra.utils.instantiate(cfg.dataset, mode="val")

    g = torch.Generator()
    g.manual_seed(cfg.random_seed)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.n_threads, pin_memory=True,
        worker_init_fn=worker_init_fn, generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.n_threads, pin_memory=True,
        worker_init_fn=worker_init_fn, generator=g
    )
    return train_loader, val_loader


def get_loggers(cfg):
    """Initialize and return train, validation, and test loggers."""
    os.makedirs(cfg.result_path, exist_ok=True)

    train_logger = Logger(
        os.path.join(cfg.result_path, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr']
    )
    train_batch_logger = Logger(
        os.path.join(cfg.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr']
    )
    val_logger = Logger(
        os.path.join(cfg.result_path, 'val.log'),
        ['epoch', 'loss', 'acc', 'mAP']
    )
    test_logger = Logger(
        os.path.join(cfg.result_path, 'test.log'),
        ['epoch', 'loss', 'acc']
    )
    return train_logger, train_batch_logger, val_logger, test_logger


def train_model(cfg, model):
    """Main entry point for training."""
    # Load datasets
    train_loader, val_loader = get_dataset(cfg)

    # Inline test dataset creation
    test_data = hydra.utils.instantiate(cfg.dataset, mode="test")
    g = torch.Generator()
    g.manual_seed(cfg.random_seed)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.n_threads, pin_memory=True,
        worker_init_fn=worker_init_fn, generator=g
    )

    # Initialize loggers
    train_logger, train_batch_logger, val_logger, test_logger = get_loggers(cfg)
    tb_writer = SummaryWriter(log_dir=os.path.join(cfg.result_path, 'tensorboard'))

    # Create optimizer and scheduler
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=parameters)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=2, verbose=True
    )

    # Create loss function
    criterion = nn.CrossEntropyLoss().to(cfg.device)

    method_name = cfg.method_name

    print("-" * 60)
    print(f'Learnable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f"Training method: {method_name}")
    print("-" * 60)

    # Instantiate trainer
    trainer = TrainerFactory.create_trainer(
        method_name, model, criterion, optimizer, scheduler, cfg
    )

    # Run the DMIL 3-stage training pipeline
    results = train_decomposition_model(
        cfg, trainer, train_loader, val_loader, test_loader,
        train_logger, val_logger, test_logger, train_batch_logger, tb_writer
    )

    # Print final results
    print("\nTraining completed successfully!")
    if isinstance(results, dict):
        print("\nFinal Results:")
        for k, v in results.items():
            if k not in ('timestamp', 'best_model_path'):
                print(f"{k}: {v}")

    return results


def train_decomposition_model(cfg, trainer, train_loader, val_loader, test_loader,
                              train_logger, val_logger, test_logger, train_batch_logger, tb_writer):
    """
    DMIL 3-stage training pipeline.
    Stage 1: Joint intra-modality VIB training for both modalities.
    Stage 2: Second-level decomposition (redundancy, unique, synergy) and dynamic gating.
    Stage 3: Joint fine-tuning of all parameters end-to-end.
    """
    model = trainer.model
    # Track the best checkpoint for each paper-defined training stage.
    best_acc = {1: float('-inf'), 2: float('-inf'), 3: float('-inf')}
    dataset_name = cfg.dataset.name
    best_model_paths = {
        1: None,
        2: None,
        3: None
    }

    # Stage transition epoch counts
    stage1_epochs = cfg.methods.stage1_epochs
    stage2_epochs = cfg.methods.stage2_epochs
    validate_schedule(stage1_epochs, stage2_epochs, cfg.n_epochs)
    stage1to2 = stage1_epochs + 1
    stage2to3 = stage1_epochs + stage2_epochs + 1

    # Learning-rate settings per stage
    stage1_lr = cfg.methods.stage1_lr
    stage2_lr = cfg.methods.stage2_lr
    stage3_lr = cfg.methods.stage3_lr

    val_freq = cfg.val_freq
    n_epochs = cfg.n_epochs

    parameters = [p for p in model.parameters()]

    # The paper's L1 objective sums both modalities in Stage 1.
    stage = 1
    train_modality = 'both'
    model_core = model.module if hasattr(model, 'module') else model
    model_core.configure_stage(1)

    # Initialize the joint Stage-1 optimizer.
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=parameters)
    for param_group in optimizer.param_groups:
        param_group['lr'] = stage1_lr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2, verbose=True)
    print(f"Stage 1 learning rate: {stage1_lr}")

    print("\n" + "=" * 80)
    print("Stage 1: Training Both Intra-Modality Decompositions")
    print("=" * 80)

    for epoch in range(1, n_epochs + 1):
        scheduled_stage = phase_for_epoch(epoch, stage1_epochs, stage2_epochs)
        if scheduled_stage is None:
            print("Stage 2 is disabled; stopping after Stage 1 as configured.")
            break

        # Stage transition logic
        if epoch == stage1to2 and stage == 1:
            print("\n" + "=" * 80)
            print("SWITCHING FROM STAGE 1 TO STAGE 2")
            print("=" * 80)

            # Stage 1 now has one jointly trained checkpoint for both modalities.
            if best_model_paths[1] and os.path.exists(best_model_paths[1]):
                print(f"Loading Stage 1 model: {best_model_paths[1]}")
                checkpoint_stage1 = torch.load(best_model_paths[1], map_location=cfg.device)
                model_core.load_state_dict(checkpoint_stage1['state_dict'], strict=False)

            stage = 2
            train_modality = 'both'
            model_core.configure_stage(2)

            optimizer = hydra.utils.instantiate(cfg.optimizer, params=parameters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = stage2_lr
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=1, verbose=True)
            print(f"Stage 2 learning rate: {stage2_lr}")

        elif epoch == stage2to3 and stage == 2:
            print("\n" + "=" * 80)
            print("SWITCHING FROM STAGE 2 TO STAGE 3")
            print("=" * 80)

            if best_model_paths[2] and os.path.exists(best_model_paths[2]):
                checkpoint_stage2 = torch.load(best_model_paths[2], map_location=cfg.device)
                model_core.load_state_dict(checkpoint_stage2['state_dict'], strict=False)
                print("Best Stage 2 model loaded successfully!")

            stage = 3
            train_modality = 'both'
            model_core.configure_stage(3)

            optimizer = hydra.utils.instantiate(cfg.optimizer, params=parameters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = stage3_lr
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
            print(f"Stage 3 learning rate: {stage3_lr}")

        if stage != scheduled_stage:
            raise RuntimeError(
                f"Training schedule mismatch at epoch {epoch}: "
                f"expected Stage {scheduled_stage}, got Stage {stage}"
            )

        print(f"CURRENT STAGE: {stage}")

        trainer.train_decomp(
            epoch=epoch,
            data_loader=train_loader,
            model=model,
            criterion=trainer.criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_logger=train_logger,
            batch_logger=train_batch_logger,
            tb_writer=tb_writer,
            opt=cfg,
            train_modality=train_modality
        )

        # Validation
        if epoch % val_freq == 0:
            val_results = trainer.validate(epoch, val_loader, val_logger, tb_writer, train_modality)
            val_loss = val_results.get("loss", None)
            val_acc = val_results.get("acc", None)

            # Save the best checkpoint for the current stage / modality
            if stage == 1:
                if val_acc is not None and val_acc > best_acc[1]:
                    if best_model_paths[1] and os.path.exists(best_model_paths[1]):
                        os.remove(best_model_paths[1])
                    best_acc[1] = val_acc
                    best_model_paths[1] = os.path.join(
                        cfg.result_path, f'stage1_best_epoch_{epoch}_acc_{val_acc:.3f}.pth'
                    )
                    save_checkpoint(best_model_paths[1], epoch, model, optimizer, scheduler)
                    print(f'Stage 1 best model saved with accuracy: {best_acc[1]:.3f}')
            elif stage == 2 or stage == 3:
                if val_acc is not None and val_acc > best_acc[stage]:
                    if best_model_paths[stage] and os.path.exists(best_model_paths[stage]):
                        os.remove(best_model_paths[stage])
                    best_acc[stage] = val_acc
                    best_model_paths[stage] = os.path.join(
                        cfg.result_path, f'stage{stage}_best_epoch_{epoch}_acc_{val_acc:.3f}.pth'
                    )
                    save_checkpoint(best_model_paths[stage], epoch, model, optimizer, scheduler)
                    print(f'Stage {stage} best model saved with accuracy: {best_acc[stage]:.3f}')

            # Update learning rate
            if val_loss is not None:
                scheduler.step(val_loss)

    return {
        'best_model_path': (
            best_model_paths[3] or best_model_paths[2] or best_model_paths[1]
        )
    }
