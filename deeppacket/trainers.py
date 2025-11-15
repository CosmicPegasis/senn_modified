"""Training classes for DeepPacket models."""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import (
    AverageMeter,
    binary_accuracy_from_logits,
    multiclass_precision_at_k,
    compute_jacobian_sum,
)


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
        best_prec1 = 0.0
        for epoch in range(epochs):
            self.train_epoch(epoch, train_loader)
            val_prec1 = 0.0
            if val_loader is not None:
                val_prec1 = self.validate(val_loader)
            is_best = val_prec1 > best_prec1
            best_prec1 = max(val_prec1, best_prec1)
            if save_path is not None:
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": self.optimizer.state_dict(),
                }
                os.makedirs(save_path, exist_ok=True)
                torch.save(state, os.path.join(save_path, "checkpoint.pth.tar"))
                if is_best:
                    torch.save(state, os.path.join(save_path, "model_best.pth.tar"))
        print("Training done")

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

        for i, (inputs, targets) in enumerate(train_loader):
            data_time.update(time.time() - end)

            # Use non_blocking for async transfer if on GPU, check if already on device
            non_blocking = self.device.type == 'cuda'
            if inputs.device != self.device:
                inputs = inputs.to(self.device, non_blocking=non_blocking)
            if targets.device != self.device:
                targets = targets.to(self.device, non_blocking=non_blocking)

            logits, loss, loss_dict = self.train_batch(inputs, targets)
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
            end = time.time()

            if i % self.print_freq == 0:
                print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]  "
                      f"Time {batch_time.val:.2f} ({batch_time.avg:.2f})  "
                      f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})  "
                      f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})  "
                      f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})")

            if max_batches and (i + 1) >= max_batches:
                break

    def validate(self, val_loader) -> float:
        """Validate the model."""
        batch_time = AverageMeter()
        losses_meter = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        end = time.time()
        eval_batches = int(getattr(self.args, "eval_batches", 0) or 0)

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                # Use non_blocking for async transfer if on GPU, check if already on device
                non_blocking = self.device.type == 'cuda'
                if inputs.device != self.device:
                    inputs = inputs.to(self.device, non_blocking=non_blocking)
                if targets.device != self.device:
                    targets = targets.to(self.device, non_blocking=non_blocking)

                output = self.model(inputs)
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
                end = time.time()

                if i % self.print_freq == 0:
                    print(f"Test: [{i}/{len(val_loader)}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                          f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})  "
                          f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})  Prec@5 {top5.val:.3f} ({top5.avg:.3f})")

                if eval_batches and (i + 1) >= eval_batches:
                    break

        print(f" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}")
        return top1.avg

    def evaluate(self, test_loader) -> float:
        """Evaluate the model on test set."""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total_seen = 0
        eval_batches = int(getattr(self.args, "eval_batches", 0) or 0)

        with torch.no_grad():
            for i, (data, targets) in enumerate(test_loader):
                # Use non_blocking for async transfer if on GPU, check if already on device
                non_blocking = self.device.type == 'cuda'
                if data.device != self.device:
                    data = data.to(self.device, non_blocking=non_blocking)
                if targets.device != self.device:
                    targets = targets.to(self.device, non_blocking=non_blocking)

                output = self.model(data)
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

                if eval_batches and (i + 1) >= eval_batches:
                    break

        test_loss /= max(1, (i + 1))
        acc = 100.0 * correct / max(1, total_seen)
        print(f"\nEvaluation: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_seen} ({acc:.0f}%)\n")
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
        self.model.train()
        device = self.device
        # Inputs and targets should already be on device from train_epoch
        # Only move if not already on device (defensive check)
        if inputs.device != device:
            inputs = inputs.to(device, non_blocking=device.type == 'cuda')
        if targets.device != device:
            targets = targets.to(device, non_blocking=device.type == 'cuda')
        # Clone and detach for gradient computation
        inputs = inputs.clone().detach().requires_grad_(True)
        self.optimizer.zero_grad()

        pred = self.model(inputs)
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
        total_loss.backward()
        self.optimizer.step()

        if hasattr(self.model, "clear_runtime_state"):
            self.model.clear_runtime_state()

        return pred.detach(), total_loss.detach(), all_losses

