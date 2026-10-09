"""
Trainer module for DMIL training.

Main components:
1. Base trainer class
2. DMIL-specific trainer (DecompositionTrainer)
3. Trainer factory class
"""
import time
import torch
from utils import AverageMeter, calculate_accuracy, write_to_batch_logger, write_to_epoch_logger


class BaseTrainer:
    """Base trainer providing a common training loop and shared utilities."""

    def __init__(self, model, criterion, optimizer, scheduler, opt=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.opt = opt
        self.device = opt.device
        self.reset_metrics()

    def reset_metrics(self):
        """Reset performance metrics."""
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses_cls = AverageMeter()
        self.accuracies = AverageMeter()

    def train_epoch(self, epoch, data_loader, epoch_logger, batch_logger, tb_writer=None):
        """Train for one epoch."""
        print(f'Training at epoch {epoch}')
        self.model.train()
        self.reset_metrics()
        start_time = time.time()
        current_lr = self.optimizer.param_groups[-1]['lr']
        num_sample = 0

        for i, batch in enumerate(data_loader):
            visual, audio, labels = self.obtain_input(batch)
            batch_size = labels.shape[0]
            num_sample += batch_size
            self.data_time.update(time.time() - start_time)

            outputs, loss = self.forward_backward(visual, audio, labels)

            out = outputs.get("out") if isinstance(outputs, dict) else outputs
            acc = calculate_accuracy(out, labels)
            self.accuracies.update(acc, out.size(0))
            self.losses_cls.update(loss.item(), out.size(0))

            self.optimizer.step()

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

            write_to_batch_logger(batch_logger, epoch, i, data_loader,
                                  self.losses_cls.val, self.accuracies.val, current_lr)

        write_to_epoch_logger(epoch_logger, epoch, self.losses_cls.avg,
                              self.accuracies.avg, current_lr)

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_cls', self.losses_cls.avg, epoch)
            tb_writer.add_scalar('train/acc', self.accuracies.avg, epoch)

        self.on_epoch_end(epoch, tb_writer)
        return self.get_epoch_results()

    def train(self, epoch, data_loader, model, criterion, optimizer, scheduler,
              epoch_logger, batch_logger, tb_writer, opt):
        """Wrapper that temporarily swaps in externally supplied components."""
        orig_model = self.model
        orig_criterion = self.criterion
        orig_optimizer = self.optimizer
        orig_scheduler = self.scheduler
        orig_opt = self.opt

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.opt = opt

        results = self.train_epoch(epoch, data_loader, epoch_logger, batch_logger, tb_writer)

        self.model = orig_model
        self.criterion = orig_criterion
        self.optimizer = orig_optimizer
        self.scheduler = orig_scheduler
        self.opt = orig_opt
        return results

    def obtain_input(self, batch):
        """Extract visual, audio, and label tensors from a batch dict."""
        visual = batch['clip'].to(self.device)
        audio = batch['audio'].to(self.device)
        targets = batch['target'].to(self.device)
        return visual, audio, targets

    def forward_backward(self, visual, audio, labels):
        """Run forward pass, compute loss, and backpropagate."""
        self.optimizer.zero_grad()
        outputs = self.model(visual, audio)
        loss = self.compute_loss(outputs, labels)
        loss.backward()
        return outputs, loss

    def compute_loss(self, outputs, labels):
        out = outputs["out"]
        v_out = outputs["v_out"]
        a_out = outputs["a_out"]
        loss_cls = self.criterion(out, labels)
        loss_v = self.criterion(v_out, labels)
        loss_a = self.criterion(a_out, labels)
        return loss_cls + loss_v + loss_a

    def on_epoch_end(self, epoch, tb_writer=None):
        """End-of-epoch hook; may be overridden by subclasses."""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                pass
            else:
                self.scheduler.step()

    def get_epoch_results(self):
        """Return a dict of training metrics for the completed epoch."""
        return {
            "loss": self.losses_cls.avg,
            "accuracy": self.accuracies.avg
        }

    def validate(self, epoch, val_loader, val_logger=None, tb_writer=None, train_modality='both'):
        """Validate the model on validation dataset."""
        from validation import val_epoch
        return val_epoch(
            model=self.model,
            data_loader=val_loader,
            device=self.device,
            criterion=self.criterion,
            val_logger=val_logger,
            tb_writer=tb_writer,
            log_tag='val',
            epoch=epoch,
            train_modality=train_modality
        )

    def test(self, epoch, test_loader, test_logger=None, tb_writer=None):
        """Evaluate model on the test set."""
        print('Testing at epoch {}'.format(epoch))
        self.model.eval()

        losses = AverageMeter()
        accuracies = AverageMeter()

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                visual, audio, targets = self.obtain_input(batch)

                outputs = self.model(visual, audio)
                out = outputs.get("out") if isinstance(outputs, dict) else outputs

                loss = self.criterion(out, targets)
                acc = calculate_accuracy(out, targets)

                losses.update(loss.item(), targets.size(0))
                accuracies.update(acc, targets.size(0))

        print('Testing Results: Loss {loss.avg:.5f} Acc {acc.avg:.3f}'.format(
            loss=losses, acc=accuracies))

        if tb_writer is not None:
            tb_writer.add_scalar('test/loss', losses.avg, epoch)
            tb_writer.add_scalar('test/acc', accuracies.avg, epoch)

        if test_logger is not None:
            test_logger.log({
                'epoch': epoch,
                'loss': losses.avg,
                'acc': accuracies.avg
            })

        return {
            'loss': losses.avg,
            'acc': accuracies.avg
        }


class DecompositionTrainer(BaseTrainer):
    """DMIL decomposition trainer implementing the 3-stage training pipeline."""

    def __init__(self, model, criterion, optimizer, scheduler, opt=None):
        super().__init__(model, criterion, optimizer, scheduler, opt)

    def reset_metrics(self):
        """Reset all performance meters."""
        super().reset_metrics()
        self.losses_total = AverageMeter()

    def compute_loss(self, outputs, labels, stage=1):
        """Compute the decomposition model loss for the given training stage."""
        if not isinstance(outputs, dict):
            return super().compute_loss(outputs, labels)

        out = outputs.get("out")
        v_out = outputs.get("v_out")
        a_out = outputs.get("a_out")

        loss_cls = self.criterion(out, labels)
        loss_v = self.criterion(v_out, labels) if v_out is not None else 0
        loss_a = self.criterion(a_out, labels) if a_out is not None else 0
        self.losses_cls.update(loss_cls.detach().item(), labels.size(0))

        # KL divergence loss (VIB)
        if "loss_IB" in outputs and isinstance(outputs["loss_IB"], tuple):
            loss_kl_v = torch.sum(outputs["loss_IB"][0])
            loss_kl_a = torch.sum(outputs["loss_IB"][1])
            loss_kl = loss_kl_v + loss_kl_a
        else:
            loss_kl_v = loss_kl_a = loss_kl = torch.tensor(0.0, device=out.device)

        # Input reconstruction loss
        if "loss_rec_input" in outputs and isinstance(outputs["loss_rec_input"], tuple):
            loss_rec_input_v = torch.sum(outputs["loss_rec_input"][0])
            loss_rec_input_a = torch.sum(outputs["loss_rec_input"][1])
            loss_rec_input = loss_rec_input_v + loss_rec_input_a
        else:
            loss_rec_input_v = loss_rec_input_a = loss_rec_input = torch.tensor(0.0, device=out.device)

        # Appendix A.3 reconstruction: recover each M^m from (R, U^m).
        if "loss_rec_tr" in outputs and isinstance(outputs["loss_rec_tr"], tuple):
            loss_rec_tr_v = torch.sum(outputs["loss_rec_tr"][0])
            loss_rec_tr_a = torch.sum(outputs["loss_rec_tr"][1])
            loss_rec_tr = loss_rec_tr_v + loss_rec_tr_a
        else:
            loss_rec_tr_v = loss_rec_tr_a = loss_rec_tr = torch.tensor(0.0, device=out.device)

        # Alignment/conditional-MI upper bound between joint R and each
        # single-modality variational approximation v_phi(R|M^m).
        if "loss_inter" in outputs and isinstance(outputs["loss_inter"], tuple):
            loss_inter_v = torch.sum(outputs["loss_inter"][0])
            loss_inter_a = torch.sum(outputs["loss_inter"][1])
            loss_inter = loss_inter_v + loss_inter_a
        else:
            loss_inter_v = loss_inter_a = loss_inter = torch.tensor(0.0, device=out.device)

        # U compactness: KL(q(U^m|M^m) || N(0, I)). Together with the
        # reconstruction term above this forms the variational R/U split.
        if "loss_uni" in outputs and isinstance(outputs["loss_uni"], tuple):
            loss_uni_v = torch.sum(outputs["loss_uni"][0])
            loss_uni_a = torch.sum(outputs["loss_uni"][1])
            loss_uni = loss_uni_v + loss_uni_a
        else:
            loss_uni_v = loss_uni_a = loss_uni = torch.tensor(0.0, device=out.device)

        # Redundancy IB loss
        loss_red_ib = torch.sum(outputs["loss_red"]) if "loss_red" in outputs else torch.tensor(0.0, device=out.device)

        # Stage-2 classification losses
        loss_synergy = self.criterion(outputs["out_synergy"], labels) if "out_synergy" in outputs else torch.tensor(0.0, device=out.device)
        loss_red_cls = self.criterion(outputs["out_red"], labels) if "out_red" in outputs else torch.tensor(0.0, device=out.device)
        loss_unique = (self.criterion(outputs["out_unique"][0], labels) +
                       self.criterion(outputs["out_unique"][1], labels)) if "out_unique" in outputs else torch.tensor(0.0, device=out.device)

        # Loss weights from config
        beta_cls = self.opt.methods.beta_cls
        lamb = self.opt.methods.lamb

        # Compose stage-specific total loss
        if stage == 1:
            # Paper Eq. L1: sum the target and variational objectives for both
            # modalities instead of training visual/audio in separate phases.
            beta_kl_v = self.opt.methods.stage1_visual_beta_kl
            beta_kl_a = self.opt.methods.stage1_audio_beta_kl
            beta_rec_v = self.opt.methods.stage1_visual_beta_rec
            beta_rec_a = self.opt.methods.stage1_audio_beta_rec
            loss = (
                beta_cls * (loss_v + loss_a)
                + lamb * (
                    beta_kl_v * loss_kl_v
                    + beta_kl_a * loss_kl_a
                    + 0.1 * beta_rec_v * loss_rec_input_v
                    + 0.1 * beta_rec_a * loss_rec_input_a
                )
            )

        elif stage == 2:
            beta_rec_tr = self.opt.methods.stage2_beta_rec_tr
            beta_red = self.opt.methods.stage2_beta_red
            beta_inter = self.opt.methods.stage2_beta_inter
            beta_uni = self.opt.methods.stage2_beta_uni

            loss = (beta_cls * loss_cls +
                    lamb * (beta_red * loss_red_ib +
                            beta_inter * loss_inter +
                            beta_uni * loss_uni +
                            0.1 * beta_rec_tr * loss_rec_tr) +
                    0.5 * (loss_synergy + loss_red_cls + loss_unique))

        elif stage == 3:
            beta_kl_stage1 = self.opt.methods.stage3_beta_kl_stage1
            beta_rec_stage1 = self.opt.methods.stage3_beta_rec_stage1
            beta_cls_stage1 = self.opt.methods.stage3_beta_cls_stage1
            beta_red = self.opt.methods.stage3_beta_red
            beta_inter = self.opt.methods.stage3_beta_inter
            beta_uni = self.opt.methods.stage3_beta_uni
            beta_rec_tr = self.opt.methods.stage3_beta_rec_tr

            loss = (beta_cls * loss_cls +
                    beta_cls_stage1 * (loss_v + loss_a) +
                    lamb * beta_kl_stage1 * loss_kl +
                    0.05 * lamb * beta_rec_stage1 * loss_rec_input +
                    lamb * (beta_red * loss_red_ib +
                            beta_inter * loss_inter +
                            beta_uni * loss_uni +
                            0.05 * beta_rec_tr * loss_rec_tr) +
                    0.8 * (loss_synergy + loss_red_cls + loss_unique))
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.losses_total.update(loss.detach().item(), labels.size(0))

        return loss

    def get_epoch_results(self):
        """Return a dict of training metrics for the completed epoch."""
        results = super().get_epoch_results()
        results["loss_total"] = self.losses_total.avg
        return results

    def train_decomp(self, epoch, data_loader, model, criterion, optimizer, scheduler,
                     epoch_logger, batch_logger, tb_writer, opt, train_modality='both'):
        """DMIL-specific training step implementing per-stage optimization."""
        m = model.module if hasattr(model, 'module') else model
        stage_num = m.stage

        if stage_num == 1:
            print(f'Training Stage 1 (Single Modality VIB) at epoch {epoch}, modality: {train_modality}')
        elif stage_num == 2:
            print(f'Training Stage 2 (Decomposition & Dynamic Routing) at epoch {epoch}')
        elif stage_num == 3:
            print(f'Training Stage 3 (Joint Fine-tuning All Parameters) at epoch {epoch}')

        model.train()
        if hasattr(m, 'configure_stage'):
            # model.train() would otherwise put the frozen Stage-1 BatchNorm
            # layers back into training mode during Stage 2.
            m.configure_stage(stage_num)

        # Save originals
        orig_model = self.model
        orig_criterion = self.criterion
        orig_optimizer = self.optimizer
        orig_scheduler = self.scheduler
        orig_opt = self.opt

        # Swap in supplied values
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.opt = opt

        self.reset_metrics()
        start_time = time.time()
        current_lr = optimizer.param_groups[-1]['lr']

        for i, batch in enumerate(data_loader):
            visual, audio, labels = self.obtain_input(batch)
            if visual.shape[0] == 1:
                continue
            self.data_time.update(time.time() - start_time)

            optimizer.zero_grad()

            if stage_num == 1:
                outputs = model(visual, audio, train_modality=train_modality)
            else:
                outputs = model(visual, audio, train_modality='both')

            self.train_modality = train_modality if stage_num == 1 else 'both'

            loss = self.compute_loss(outputs, labels, stage=stage_num)

            loss.backward()
            optimizer.step()

            out = outputs.get("out") if isinstance(outputs, dict) else outputs
            acc = calculate_accuracy(out, labels)
            self.accuracies.update(acc, labels.size(0))

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

            write_to_batch_logger(batch_logger, epoch, i, data_loader,
                                  self.losses_total.val, self.accuracies.val, current_lr)

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch}][{i+1}/{len(data_loader)}] Stage {stage_num} - '
                      f'Loss: {self.losses_total.avg:.4f} | Acc: {self.accuracies.avg:.3f}')

        # Epoch-level logging
        total_loss = self.losses_total.avg
        write_to_epoch_logger(epoch_logger, epoch, total_loss, self.accuracies.avg, current_lr)

        # Epoch summary
        print(f'\n{"="*80}')
        print(f'Epoch {epoch} Summary (Stage {stage_num}):')
        print(f'{"="*80}')
        print(f'Loss: {self.losses_total.avg:.4f} | Cls Loss: {self.losses_cls.avg:.4f} | Accuracy: {self.accuracies.avg:.3f}')
        print(f'{"="*80}\n')

        # TensorBoard logging
        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_total', total_loss, epoch)
            tb_writer.add_scalar('train/acc', self.accuracies.avg, epoch)
            self.on_epoch_end(epoch, tb_writer)

        # Restore originals
        results = self.get_epoch_results()
        self.model = orig_model
        self.criterion = orig_criterion
        self.optimizer = orig_optimizer
        self.scheduler = orig_scheduler
        self.opt = orig_opt

        return results


class TrainerFactory:
    """Creates the trainer for the specified method."""

    @staticmethod
    def create_trainer(method, model, criterion, optimizer, scheduler, opt=None):
        method_name = opt.method_name
        if method_name.upper() != 'DMIL':
            raise ValueError(f"Unsupported method: {method_name}. Only 'DMIL' is supported.")
        return DecompositionTrainer(model, criterion, optimizer, scheduler, opt)
