"""Training classes for DeepPacket models."""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .utils import (
    AverageMeter,
    binary_accuracy_from_logits,
    multiclass_precision_at_k,
    compute_jacobian_sum,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    """Training arguments dataclass."""
    cuda: bool = False
    nclasses: int = 2
    h_type: str = "input"
    h_sparsity: float = -1.0
    lr: float = 1e-3
    weight_decay: float = 1e-3
    opt: str = "adam"
    print_freq: int = 200
    theta_reg_lambda: float = 1e-2
    theta_reg_type: str = "grad3"
    epochs: int = 1
    model_path: str = "models"
    results_path: str = "out"
    log_path: str = "log"

    limit_files_per_split: int = 0
    max_rows_per_file: Optional[int] = None
    max_batches_per_epoch: int = 0
    eval_batches: int = 0
    
    # Class imbalance handling
    handle_imbalance: bool = False
    weight_method: str = "balanced"
    
    # Undersampling options
    undersample: bool = False
    undersample_ratio: float = 0.1
    undersample_strategy: str = "random"

    flow_split: bool = True
    flow_suffix: str = ".flow.npy"


class ClassificationTrainer:
    """Base trainer for classification tasks."""
    
    def __init__(self, model: nn.Module, args: TrainArgs):
        self.model = model
        self.args = args
        self.device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)

        self.nclasses = int(args.nclasses)
        self.prediction_criterion = nn.BCEWithLogitsLoss() if self.nclasses <= 2 else nn.CrossEntropyLoss()

        opt = args.opt.lower()
        if opt == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif opt == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif opt == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {args.opt}")

        self.loss_history: List[Dict] = []
        self.print_freq = int(args.print_freq)
        self.reset_lstm = bool(getattr(model, "reset_lstm", False))
        self.learning_h = getattr(args, "h_type", "input") != "input"
        self.h_reconst_criterion = nn.MSELoss() if self.learning_h else None
        self.h_sparsity = getattr(args, "h_sparsity", -1.0)

    def train(self, train_loader, val_loader=None, epochs=1, save_path: Optional[str] = None):
        """Train the model for specified number of epochs."""
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader) if val_loader is not None else 0}")
        best_prec1 = 0.0
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs} - Starting training phase")
            self.train_epoch(epoch, train_loader)
            val_prec1 = 0.0
            if val_loader is not None:
                logger.info(f"Epoch {epoch+1}/{epochs} - Starting validation phase")
                val_prec1 = self.validate(val_loader)
            is_best = val_prec1 > best_prec1
            best_prec1 = max(val_prec1, best_prec1)
            if save_path is not None:
                logger.info(f"Epoch {epoch+1}/{epochs} - Saving checkpoint to {save_path}")
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": self.optimizer.state_dict(),
                }
                os.makedirs(save_path, exist_ok=True)
                torch.save(state, os.path.join(save_path, "checkpoint.pth.tar"))
                if is_best:
                    logger.info(f"Epoch {epoch+1}/{epochs} - New best model! Saving best checkpoint.")
                    torch.save(state, os.path.join(save_path, "model_best.pth.tar"))
        logger.info("Training completed")

    def train_batch(self, inputs, targets):
        """Override in subclasses; must return (logits, loss, dict_of_losses)."""
        raise NotImplementedError

    def concept_learning_loss(self, inputs: torch.Tensor, all_losses: dict) -> torch.Tensor:
        """Compute concept learning loss (reconstruction + sparsity)."""
        if not hasattr(self.model, "recons") or self.model.recons is None or self.h_reconst_criterion is None:
            return torch.tensor(0.0, device=self.device)
        # Use inputs directly without detach to avoid creating new tensor
        recons_loss = self.h_reconst_criterion(self.model.recons, inputs)
        all_losses["reconstruction"] = recons_loss.item()
        total = recons_loss
        if self.h_sparsity not in (-1, None) and hasattr(self.model, "h_norm_l1") and self.model.h_norm_l1 is not None:
            sparsity_loss = self.model.h_norm_l1 * float(self.h_sparsity)
            all_losses["h_sparsity"] = sparsity_loss.item()
            total = total + sparsity_loss
        return total

    def train_epoch(self, epoch, train_loader):
        """Train for one epoch."""
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        end = time.time()
        max_batches = int(getattr(self.args, "max_batches_per_epoch", 0) or 0)
        total_batches = max_batches if max_batches > 0 else len(train_loader)
        logger.info(f"  Train epoch {epoch}: Processing up to {total_batches} batches")

        pbar = tqdm(train_loader, desc=f"Train Epoch [{epoch}]", ncols=120)
        for i, (inputs, targets) in enumerate(pbar):
            batch_start_time = time.time()
            logger.debug(f"  Batch {i+1}/{len(train_loader)}: Starting batch load")
            
            data_time.update(time.time() - end)
            logger.debug(f"  Batch {i+1}: Data load time: {time.time() - end:.4f}s, shape: {inputs.shape}")

            # Use non_blocking for async transfer if on GPU, check if already on device
            non_blocking = self.device.type == 'cuda'
            if inputs.device != self.device:
                logger.debug(f"  Batch {i+1}: Moving inputs to {self.device} (non_blocking={non_blocking})")
                inputs = inputs.to(self.device, non_blocking=non_blocking)
            if targets.device != self.device:
                logger.debug(f"  Batch {i+1}: Moving targets to {self.device} (non_blocking={non_blocking})")
                targets = targets.to(self.device, non_blocking=non_blocking)
            
            logger.debug(f"  Batch {i+1}: Starting forward/backward pass")
            forward_start = time.time()
            logits, loss, loss_dict = self.train_batch(inputs, targets)
            forward_time = time.time() - forward_start
            logger.debug(f"  Batch {i+1}: Forward/backward pass completed in {forward_time:.4f}s, loss: {loss.item():.4f}")
            
            loss_dict["iter"] = i + (len(train_loader) * epoch)
            self.loss_history.append(loss_dict)

            if self.nclasses <= 2:
                acc1 = binary_accuracy_from_logits(logits, targets)
                prec1, prec5 = [acc1], [100.0]
            else:
                # Keep tensors on GPU for precision calculation
                precs = multiclass_precision_at_k(logits.detach(), targets, topk=(1, min(5, self.nclasses)))
                prec1 = float(precs[0].item())
                prec5 = [float(precs[1].item())] if self.nclasses >= 5 else [prec1]

            losses_meter.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            batch_time.update(time.time() - end)
            total_batch_time = time.time() - batch_start_time
            logger.debug(f"  Batch {i+1}: Total batch time: {total_batch_time:.4f}s (data: {data_time.val:.4f}s, compute: {forward_time:.4f}s)")
            end = time.time()

            # Update progress bar with current metrics
            pbar.set_postfix({
                'Loss': f'{losses_meter.avg:.4f}',
                'Acc@1': f'{top1.avg:.3f}',
                'Acc@5': f'{top5.avg:.3f}',
                'Time': f'{batch_time.avg:.2f}s'
            })

            if max_batches and (i + 1) >= max_batches:
                logger.info(f"  Reached max_batches limit ({max_batches}), stopping epoch")
                break

    def validate(self, val_loader) -> float:
        """Validate the model."""
        logger.info(f"  Validation: Processing {len(val_loader)} batches")
        batch_time = AverageMeter()
        losses_meter = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        end = time.time()
        eval_batches = int(getattr(self.args, "eval_batches", 0) or 0)
        if eval_batches > 0:
            logger.info(f"  Validation: Limited to {eval_batches} batches")

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Test", ncols=120)
            for i, (inputs, targets) in enumerate(pbar):
                batch_start_time = time.time()
                logger.debug(f"  Val batch {i+1}/{len(val_loader)}: Starting batch load")
                
                # Use non_blocking for async transfer if on GPU, check if already on device
                non_blocking = self.device.type == 'cuda'
                if inputs.device != self.device:
                    logger.debug(f"  Val batch {i+1}: Moving inputs to {self.device}")
                    inputs = inputs.to(self.device, non_blocking=non_blocking)
                if targets.device != self.device:
                    logger.debug(f"  Val batch {i+1}: Moving targets to {self.device}")
                    targets = targets.to(self.device, non_blocking=non_blocking)

                logger.debug(f"  Val batch {i+1}: Starting forward pass")
                forward_start = time.time()
                output = self.model(inputs)
                forward_time = time.time() - forward_start
                logger.debug(f"  Val batch {i+1}: Forward pass completed in {forward_time:.4f}s")
                loss = self.prediction_criterion(
                    output, targets if self.nclasses > 2 else targets.float().unsqueeze(1)
                )

                if self.nclasses <= 2:
                    prec1 = binary_accuracy_from_logits(output, targets)
                    prec5 = [100.0]
                else:
                    # Keep tensors on GPU for precision calculation
                    precs = multiclass_precision_at_k(output.detach(), targets, topk=(1, min(5, self.nclasses)))
                    prec1 = float(precs[0].item())
                    prec5 = [float(precs[1].item())] if self.nclasses >= 5 else [prec1]

                losses_meter.update(loss.item(), inputs.size(0))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

                batch_time.update(time.time() - end)
                total_batch_time = time.time() - batch_start_time
                logger.debug(f"  Val batch {i+1}: Total batch time: {total_batch_time:.4f}s")
                end = time.time()

                # Update progress bar with current metrics
                pbar.set_postfix({
                    'Loss': f'{losses_meter.avg:.4f}',
                    'Acc@1': f'{top1.avg:.3f}',
                    'Acc@5': f'{top5.avg:.3f}',
                    'Time': f'{batch_time.avg:.3f}s'
                })

                if eval_batches and (i + 1) >= eval_batches:
                    logger.info(f"  Reached eval_batches limit ({eval_batches}), stopping validation")
                    break

        return top1.avg

    def evaluate(self, test_loader) -> float:
        """Evaluate the model on test set."""
        logger.info(f"  Evaluation: Processing {len(test_loader)} batches")
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total_seen = 0
        eval_batches = int(getattr(self.args, "eval_batches", 0) or 0)
        if eval_batches > 0:
            logger.info(f"  Evaluation: Limited to {eval_batches} batches")

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluation", ncols=120)
            for i, (data, targets) in enumerate(pbar):
                batch_start_time = time.time()
                logger.debug(f"  Eval batch {i+1}/{len(test_loader)}: Starting batch load")
                
                # Use non_blocking for async transfer if on GPU, check if already on device
                non_blocking = self.device.type == 'cuda'
                if data.device != self.device:
                    logger.debug(f"  Eval batch {i+1}: Moving data to {self.device}")
                    data = data.to(self.device, non_blocking=non_blocking)
                if targets.device != self.device:
                    logger.debug(f"  Eval batch {i+1}: Moving targets to {self.device}")
                    targets = targets.to(self.device, non_blocking=non_blocking)

                logger.debug(f"  Eval batch {i+1}: Starting forward pass")
                forward_start = time.time()
                output = self.model(data)
                forward_time = time.time() - forward_start
                logger.debug(f"  Eval batch {i+1}: Forward pass completed in {forward_time:.4f}s")
                loss = self.prediction_criterion(
                    output, targets if self.nclasses > 2 else targets.float().unsqueeze(1)
                )
                test_loss += loss.item()

                if self.nclasses == 2:
                    probs = torch.sigmoid(output)
                    pred = (probs >= 0.5).long().view(-1)
                else:
                    pred = output.argmax(dim=1).view(-1)

                correct += pred.eq(targets.view(-1)).sum().item()
                total_seen += targets.numel()

                total_batch_time = time.time() - batch_start_time
                logger.debug(f"  Eval batch {i+1}: Total batch time: {total_batch_time:.4f}s")
                
                # Update progress bar with current metrics
                current_acc = 100.0 * correct / max(1, total_seen)
                current_loss = test_loss / max(1, (i + 1))
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
                
                if eval_batches and (i + 1) >= eval_batches:
                    logger.info(f"  Reached eval_batches limit ({eval_batches}), stopping evaluation")
                    break

        test_loss /= max(1, (i + 1))
        acc = 100.0 * correct / max(1, total_seen)
        logger.info(f"  Evaluation complete: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_seen} ({acc:.0f}%)")
        return acc

    def predict(self, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for confusion matrix and other analyses."""
        self.model.eval()
        eval_batches = int(getattr(self.args, "eval_batches", 0) or 0)

        with torch.no_grad():
            # Collect predictions on GPU, transfer to CPU in batches
            pred_list = []
            target_list = []
            for i, (data, targets) in enumerate(test_loader):
                # Use non_blocking for async transfer if on GPU, check if already on device
                non_blocking = self.device.type == 'cuda'
                if data.device != self.device:
                    data = data.to(self.device, non_blocking=non_blocking)
                if targets.device != self.device:
                    targets = targets.to(self.device, non_blocking=non_blocking)

                output = self.model(data)

                if self.nclasses == 2:
                    probs = torch.sigmoid(output)
                    pred = (probs >= 0.5).long().view(-1)
                else:
                    pred = output.argmax(dim=1).view(-1)

                # Keep on GPU, transfer to CPU in batch at end
                pred_list.append(pred)
                target_list.append(targets)
                
                if eval_batches and (i + 1) >= eval_batches:
                    break
            
            # Batch transfer to CPU for numpy conversion
            if pred_list:
                all_preds = torch.cat(pred_list).cpu().numpy()
                all_targets = torch.cat(target_list).cpu().numpy()
            else:
                all_preds = np.array([], dtype=np.int64)
                all_targets = np.array([], dtype=np.int64)

        return all_preds, all_targets


class GradPenaltyTrainer(ClassificationTrainer):
    """Adds SENN-style gradient penalty (type 1/2/3)."""
    
    def __init__(self, model: nn.Module, args: TrainArgs, typ: int = 3):
        super().__init__(model, args)
        self.lambd = float(getattr(args, "theta_reg_lambda", 1e-2))
        self.penalty_type = int(typ)
        self.norm = 2
        self.eps = 1e-8

    def train_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Train on a single batch with gradient penalty."""
        logger.debug("    train_batch: Starting batch processing")
        self.model.train()
        device = self.device
        # Inputs and targets should already be on device from train_epoch
        # Only move if not already on device (defensive check)
        if inputs.device != device:
            logger.debug(f"    train_batch: Moving inputs to {device}")
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
        if targets.device != device:
            logger.debug(f"    train_batch: Moving targets to {device}")
            targets = targets.to(device, non_blocking=device.type == 'cuda')
        # Clone and detach for gradient computation
        logger.debug("    train_batch: Cloning inputs for gradient computation")
        inputs = inputs.clone().detach().requires_grad_(True)
        logger.debug("    train_batch: Zeroing gradients")
        self.optimizer.zero_grad()

        logger.debug("    train_batch: Starting forward pass")
        forward_start = time.time()
        pred = self.model(inputs)
        forward_time = time.time() - forward_start
        logger.debug(f"    train_batch: Forward pass completed in {forward_time:.4f}s")
        if self.nclasses <= 2:
            pred_loss = self.prediction_criterion(pred.view(-1, 1), targets.float().view(-1, 1))
        else:
            pred_loss = self.prediction_criterion(pred, targets.long())
        all_losses = {"prediction": pred_loss.item()}

        # Optional concept loss if learning H
        if self.learning_h:
            h_loss = self.concept_learning_loss(inputs, all_losses)
            loss_base = pred_loss + h_loss
        else:
            loss_base = pred_loss

        # Gradient penalty (type 3 as default SENN)
        # Use smoother scalarization (logsumexp) to stabilize gradients
        if pred.dim() == 2 and pred.size(1) > 1:
            scalar_pred = pred.logsumexp(dim=1)
        else:
            scalar_pred = pred.view(pred.size(0), -1).squeeze(1)

        dF = torch.autograd.grad(
            outputs=scalar_pred, inputs=inputs,
            grad_outputs=torch.ones_like(scalar_pred, device=device),
            create_graph=True, only_inputs=True
        )[0]  # (B, ...)

        grad_penalty = torch.tensor(0.0, device=device)
        thetas_live = getattr(self.model, "thetas", None)
        if self.penalty_type == 3:
            if not self.learning_h:
                if thetas_live is not None:
                    dF_flat = dF.reshape(dF.size(0), -1)
                    theta_flat = thetas_live.reshape(thetas_live.size(0), -1)
                    if theta_flat.size(1) != dF_flat.size(1):
                        if theta_flat.size(1) > dF_flat.size(1):
                            theta_flat = theta_flat[:, :dF_flat.size(1)]
                        else:
                            reps = int(np.ceil(dF_flat.size(1) / theta_flat.size(1)))
                            theta_flat = theta_flat.repeat(1, reps)[:, :dF_flat.size(1)]
                    diff = theta_flat - dF_flat
                    D = float(dF_flat.size(1)) if dF_flat.dim() == 2 else float(max(1, dF.numel() // dF.size(0)))
                    # L2 per-sample, normalized by sqrt(D), then mean across batch
                    grad_penalty = (diff.norm(p=2, dim=1) / (D ** 0.5)).mean()
        elif self.penalty_type == 1:
            if thetas_live is not None:
                dF_flat = dF.reshape(dF.size(0), -1)
                theta_flat = thetas_live.reshape(thetas_live.size(0), -1)
                if theta_flat.size(1) != dF_flat.size(1):
                    if theta_flat.size(1) > dF_flat.size(1):
                        theta_flat = theta_flat[:, :dF_flat.size(1)]
                    else:
                        reps = int(np.ceil(dF_flat.size(1) / theta_flat.size(1)))
                        theta_flat = theta_flat.repeat(1, reps)[:, :dF_flat.size(1)]
                grad_penalty = (dF_flat - theta_flat).norm(p=self.norm)
        elif self.penalty_type == 2:
            # Minimal implementation: ratio ||dTheta/dx|| / sqrt(D)
            if thetas_live is not None:
                JTh = compute_jacobian_sum(inputs, thetas_live.squeeze()).unsqueeze(-1)  # (B, D, 1)
                num = JTh.view(JTh.size(0), -1).norm(p=self.norm)
                denom = (inputs.numel() / inputs.size(0)) ** 0.5 + self.eps
                grad_penalty = num / denom

        all_losses["grad_penalty"] = float(grad_penalty.item())
        total_loss = loss_base + (self.lambd * grad_penalty)
        
        logger.debug("    train_batch: Starting backward pass")
        backward_start = time.time()
        total_loss.backward()
        backward_time = time.time() - backward_start
        logger.debug(f"    train_batch: Backward pass completed in {backward_time:.4f}s")
        
        logger.debug("    train_batch: Optimizer step")
        step_start = time.time()
        self.optimizer.step()
        step_time = time.time() - step_start
        logger.debug(f"    train_batch: Optimizer step completed in {step_time:.4f}s")

        if hasattr(self.model, "clear_runtime_state"):
            self.model.clear_runtime_state()

        logger.debug(f"    train_batch: Batch processing complete (total: {time.time() - forward_start:.4f}s)")
        return pred.detach(), total_loss.detach(), all_losses

