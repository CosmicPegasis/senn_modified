from __future__ import annotations

import os
import time
import glob
import shutil
import bisect
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler

# =========================================================
# Utils
# =========================================================
#
# - [ ] Ensure class balance
# - [ ] Look up Deep Packet implementation
# - [ ] Flow-Based Separation
#
#
# - [ ] Look at the dataset that trustee is using, (we need a baseline model)
# to compare, we want to compare against. NPrintML

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        v = float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

def save_checkpoint(state: dict, is_best: bool, outpath: str) -> None:
    os.makedirs(outpath, exist_ok=True)
    filename = os.path.join(outpath, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outpath, 'model_best.pth.tar'))

def binary_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits.detach())
    preds = (probs >= 0.5).long().view(-1)
    return (preds == targets.view(-1)).float().mean().item() * 100.0

def multiclass_precision_at_k(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().contiguous()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k * (100.0 / batch_size)).detach())
    return res

# =========================================================
# Autograd helpers
# =========================================================

def compute_jacobian_sum(x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
    """Fast L1-style: returns sum of gradients wrt x for all outputs in fx."""
    b, m, c = fx.size(0), fx.size(-2), fx.size(-1)
    grad = torch.ones(b, m, c, device=x.device, dtype=fx.dtype)
    g = torch.autograd.grad(
        outputs=fx, inputs=x, grad_outputs=grad, create_graph=True, only_inputs=True
    )[0]
    return g

def compute_jacobian(x: torch.Tensor, fx: torch.Tensor) -> torch.Tensor:
    """J[:, :, i] = d fx[:, i] / d x (flattened over x dims)."""
    b = x.size(0)
    m = fx.size(-1)
    J = []
    for i in range(m):
        grad_out = torch.zeros(b, m, device=x.device, dtype=fx.dtype)
        grad_out[:, i] = 1
        g = torch.autograd.grad(
            outputs=fx, inputs=x, grad_outputs=grad_out, create_graph=True, only_inputs=True
        )[0]
        J.append(g.view(b, -1, 1))
    return torch.cat(J, dim=2)

def CL_loss(grad: torch.Tensor, n_class: int) -> torch.Tensor:
    """Cross-Lipschitz loss (pairwise gradient differences)."""
    # grad: (C, D) or (B,C,D) -> reduce batch if present
    if grad.dim() == 3:
        grad = grad.mean(dim=0)
    total = 0.0
    for i in range(n_class):
        for j in range(n_class):
            total = total + (grad[i] - grad[j]).norm() ** 2
    return total / float(n_class)

# =========================================================
# Conceptizer / Parametrizer / Aggregator / Model
# =========================================================

class InputConceptizer(nn.Module):
    """Treat raw input features as concepts; optional bias concept at the end."""
    def __init__(self, add_bias: bool = True):
        super().__init__()
        self.add_bias = add_bias
        self.learnable = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept 2D (B,D) or 4D (B,C,H,W). DeepPacket is (B,1,1,1500) -> flatten to (B,1500,1)
        if x.dim() == 4:
            B, C, H, W = x.shape
            out = x.reshape(B, C * H * W, 1)
        elif x.dim() == 2:
            out = x.unsqueeze(-1)
        elif x.dim() == 3:
            B = x.size(0)
            out = x.reshape(B, -1, 1)
        else:
            raise ValueError(f"Unsupported x.ndim={x.dim()}")

        if self.add_bias:
            bias = out.new_ones((out.size(0), 1, out.size(2)))
            out = torch.cat([out, bias], dim=1)
        return out


class LinearParametrizer(nn.Module):
    """
    Simple MLP parametrizer for vector-like inputs (e.g., DeepPacket 1×1500).
    Outputs Theta with shape (B, nconcept, nclass) if concept_dim==1.
    """
    def __init__(self, input_dim: int, nconcept: int, nclass: int, hidden: int = 512, only_positive: bool = False):
        super().__init__()
        self.nconcept = int(nconcept)
        self.nclass = int(nclass)
        self.only_positive = bool(only_positive)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, nconcept * nclass),
        )
        self.pos_act = nn.Softplus() if self.only_positive else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        out = self.net(x)  # (B, nconcept*nclass)
        out = out.view(x.size(0), self.nconcept, self.nclass)
        if self.pos_act is not None:
            out = self.pos_act(out)
        else:
            out = torch.tanh(out)
        return out


class AdditiveScalarAggregator(nn.Module):
    """Logits = Σ_i <h_i, theta_i[:, class]> ; with cdim==1 this is Σ_i h_i * theta_i."""
    def __init__(self, cdim: int = 1, nclasses: int = 1):
        super().__init__()
        self.cdim = int(cdim)
        self.nclasses = int(nclasses)

    def forward(self, H: torch.Tensor, Th: torch.Tensor) -> torch.Tensor:
        # H: (B,k,cdim) or (B,k,1) ; Th: (B,k,nclasses) or (B,k,cdim,nclasses)
        if H.dim() == 4 and H.size(-1) == 1:
            H = H.squeeze(-1)
        if Th.dim() == 3:
            if H.size(-1) != 1:
                raise ValueError("When Th is (B,k,C), H must have cdim==1.")
            logits = torch.einsum("bkd,bkc->bc", H, Th)
        elif Th.dim() == 4:
            if Th.size(2) != H.size(-1):
                raise ValueError("Mismatch between H.cdim and Th[:,:,cdim,:].")
            logits = torch.einsum("bkd,bkdc->bc", H, Th)
        else:
            raise ValueError(f"Unsupported Th shape: {tuple(Th.shape)}")
        return logits


class GSENN(nn.Module):
    """Self-explaining neural net wrapper."""
    def __init__(self, conceptizer: nn.Module, parametrizer: nn.Module, aggregator: nn.Module, debug: bool = False):
        super().__init__()
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
        self.learning_H = bool(getattr(conceptizer, "learnable", False))
        self.reset_lstm = hasattr(conceptizer, "lstm") or hasattr(parametrizer, "lstm")
        self.thetas: Optional[torch.Tensor] = None
        self.concepts: Optional[torch.Tensor] = None
        self.recons: Optional[torch.Tensor] = None
        self.h_norm_l1: Optional[torch.Tensor] = None
        self.debug = debug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learning_H:
            res = self.conceptizer(x)
            if isinstance(res, (tuple, list)) and len(res) >= 2:
                h_x, x_tilde = res[0], res[1]
                self.recons = x_tilde if self.training else x_tilde.detach()
            else:
                h_x = res
                self.recons = None
            self.h_norm_l1 = h_x.norm(p=1)
        else:
            h_x = self.conceptizer(x)

        self.concepts = h_x if self.training else h_x.detach()

        thetas = self.parametrizer(x)
        if thetas.dim() == 2:
            thetas = thetas.unsqueeze(-1)
        self.thetas = thetas if self.training else thetas.detach()

        if h_x.dim() == 4:
            B, k = h_x.shape[0], h_x.shape[1]
            h_x = h_x.view(B, k, -1)

        logits = self.aggregator(h_x, thetas)
        return logits

    def clear_runtime_state(self) -> None:
        self.thetas = None
        self.concepts = None
        self.recons = None
        self.h_norm_l1 = None

# =========================================================
# Trainers
# =========================================================

@dataclass
class TrainArgs:
    cuda: bool = False
    nclasses: int = 2
    h_type: str = "input"
    h_sparsity: float = -1.0
    lr: float = 1e-3
    weight_decay: float = 1e-3
    opt: str = "adam"
    print_freq: int = 50
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

class ClassificationTrainer:
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

    # --------- public API ----------
    def train(self, train_loader, val_loader=None, epochs=1, save_path: Optional[str] = None):
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
                # lightweight saver; plug your own save_checkpoint() if you have one
                import os, torch as _torch
                os.makedirs(save_path, exist_ok=True)
                _torch.save(state, os.path.join(save_path, "checkpoint.pth.tar"))
                if is_best:
                    _torch.save(state, os.path.join(save_path, "model_best.pth.tar"))
        print("Training done")

    def train_batch(self, inputs, targets):
        """Override in subclasses; must return (logits, loss, dict_of_losses)."""
        raise NotImplementedError

    # --------- internals ----------
    def concept_learning_loss(self, inputs: torch.Tensor, all_losses: dict) -> torch.Tensor:
        if not hasattr(self.model, "recons") or self.model.recons is None or self.h_reconst_criterion is None:
            return torch.tensor(0.0, device=self.device)
        recons_loss = self.h_reconst_criterion(self.model.recons, inputs.detach())
        all_losses["reconstruction"] = recons_loss.item()
        total = recons_loss
        if self.h_sparsity not in (-1, None) and hasattr(self.model, "h_norm_l1") and self.model.h_norm_l1 is not None:
            sparsity_loss = self.model.h_norm_l1 * float(self.h_sparsity)
            all_losses["h_sparsity"] = sparsity_loss.item()
            total = total + sparsity_loss
        return total

    def train_epoch(self, epoch, train_loader):
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

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits, loss, loss_dict = self.train_batch(inputs, targets)
            loss_dict["iter"] = i + (len(train_loader) * epoch)
            self.loss_history.append(loss_dict)

            if self.nclasses <= 2:
                acc1 = binary_accuracy_from_logits(logits, targets)
                prec1, prec5 = [acc1], [100.0]
            else:
                precs = multiclass_precision_at_k(logits.detach().cpu(), targets.detach().cpu(), topk=(1, min(5, self.nclasses)))
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
        batch_time = AverageMeter()
        losses_meter = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        end = time.time()
        eval_batches = int(getattr(self.args, "eval_batches", 0) or 0)

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                output = self.model(inputs)
                loss = self.prediction_criterion(
                    output, targets if self.nclasses > 2 else targets.float().unsqueeze(1)
                )

                if self.nclasses <= 2:
                    prec1 = binary_accuracy_from_logits(output, targets)
                    prec5 = [100.0]
                else:
                    precs = multiclass_precision_at_k(output.detach().cpu(), targets.detach().cpu(), topk=(1, min(5, self.nclasses)))
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
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total_seen = 0
        eval_batches = int(getattr(self.args, "eval_batches", 0) or 0)

        with torch.no_grad():
            for i, (data, targets) in enumerate(test_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

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

class GradPenaltyTrainer(ClassificationTrainer):
    """Adds SENN-style gradient penalty (type 1/2/3)."""
    def __init__(self, model: nn.Module, args: TrainArgs, typ: int = 3):
        super().__init__(model, args)
        self.lambd = float(getattr(args, "theta_reg_lambda", 1e-2))
        self.penalty_type = int(typ)
        self.norm = 2
        self.eps = 1e-8

    def train_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.model.train()
        device = self.device
        inputs = inputs.to(device)
        targets = targets.to(device)
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
        if pred.dim() == 2 and pred.size(1) > 1:
            scalar_pred = pred.max(dim=1).values
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
                    grad_penalty = (theta_flat - dF_flat).norm(p=self.norm)
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

# =========================================================
# DeepPacket dataset (NPY, lazy & writable)
# =========================================================

class DeepPacketNPYDataset(Dataset):
    """
    Root/
      class_a/*.npy  # each .npy is (N,1500) or (1500,)
      class_b/*.npy
    Lazy, memmap-backed; returns (B,1,1,1500) tensors.
    """
    def __init__(self, root: str, split_indices: Optional[List[int]] = None, max_rows_per_file: Optional[int] = None, weight_method: str = "balanced"):
        self.root = root
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.files: List[Tuple[str, int]] = []
        for cls in self.classes:
            # Include both chunked and non-chunked .npy files
            for p in glob.glob(os.path.join(root, cls, "*.npy")):
                self.files.append((p, self.class_to_idx[cls]))
        if split_indices is not None:
            self.files = [self.files[i] for i in split_indices]
        if not self.files:
            raise RuntimeError(f"No .npy files found under {root}")

        self.counts: List[int] = []
        for path, _ in self.files:
            arr = np.load(path, mmap_mode="r")
            nrows = 1 if arr.ndim == 1 else int(arr.shape[0])
            if max_rows_per_file is not None:
                nrows = min(nrows, max_rows_per_file)
            self.counts.append(nrows)
        self.offsets = np.cumsum([0] + self.counts)
        self.total = int(self.offsets[-1])
        if self.total == 0:
            raise RuntimeError("All files are empty.")
        
        # Calculate class distribution for imbalance handling
        self.class_counts = self._calculate_class_counts()
        self.class_weights = self._calculate_class_weights(method=weight_method)

    def __len__(self) -> int:
        return self.total

    def _calculate_class_counts(self) -> Dict[int, int]:
        """Calculate the number of samples per class."""
        class_counts = {i: 0 for i in range(len(self.classes))}
        for i, (path, class_idx) in enumerate(self.files):
            class_counts[class_idx] += self.counts[i]
        return class_counts

    def _calculate_class_weights(self, method: str = "balanced") -> Dict[int, float]:
        """
        Calculate class weights for handling class imbalance.
        
        Args:
            method: Method for calculating weights ("balanced", "inverse", "sqrt_inverse")
        """
        total_samples = sum(self.class_counts.values())
        n_classes = len(self.class_counts)
        
        if method == "balanced":
            # sklearn-style balanced weights: n_samples / (n_classes * np.bincount(y))
            weights = {}
            for class_idx, count in self.class_counts.items():
                if count > 0:
                    weights[class_idx] = total_samples / (n_classes * count)
                else:
                    weights[class_idx] = 0.0
        elif method == "inverse":
            # Simple inverse frequency weights
            weights = {}
            for class_idx, count in self.class_counts.items():
                if count > 0:
                    weights[class_idx] = total_samples / count
                else:
                    weights[class_idx] = 0.0
        elif method == "sqrt_inverse":
            # Square root of inverse frequency weights (less aggressive)
            weights = {}
            for class_idx, count in self.class_counts.items():
                if count > 0:
                    weights[class_idx] = np.sqrt(total_samples / count)
                else:
                    weights[class_idx] = 0.0
        else:
            raise ValueError(f"Unknown weight method: {method}")
        
        return weights

    def get_sample_weights(self) -> List[float]:
        """Get sample weights for each sample in the dataset."""
        sample_weights = []
        for i, (path, class_idx) in enumerate(self.files):
            weight = self.class_weights[class_idx]
            # Repeat weight for each sample in this file
            sample_weights.extend([weight] * self.counts[i])
        return sample_weights

    def apply_undersampling(self, ratio: float = 0.1, strategy: str = "random") -> 'DeepPacketNPYDataset':
        """
        Apply mild undersampling to reduce class imbalance.
        
        Args:
            ratio: Ratio of samples to keep from majority classes (0.1 = keep 10% of largest class)
            strategy: Undersampling strategy ("random" or "stratified")
        
        Returns:
            New dataset with undersampled data
        """
        if ratio >= 1.0:
            return self  # No undersampling needed
        
        # Find the maximum class count
        max_count = max(self.class_counts.values())
        if max_count == 0:
            return self  # No data to undersample
        
        # Calculate target count for each class
        target_counts = {}
        target_count = max(1, int(max_count * ratio))
        for class_idx, count in self.class_counts.items():
            if count > 0:
                # Undersample all classes to keep only the specified ratio of the largest class
                # But don't upsample - only reduce if the class is larger than target
                target_counts[class_idx] = min(count, target_count)
            else:
                target_counts[class_idx] = 0
        
        # Create a new dataset that will use sampling indices instead of creating new files
        new_dataset = UndersampledDeepPacketNPYDataset(
            self.root, 
            self.files, 
            self.counts, 
            self.classes, 
            self.class_to_idx,
            target_counts,
            strategy
        )
        
        return new_dataset

    def _locate(self, idx: int) -> Tuple[int, int]:
        f = bisect.bisect_right(self.offsets, idx) - 1
        row = idx - self.offsets[f]
        return f, row

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fidx, ridx = self._locate(idx)
        path, y = self.files[fidx]
        arr = np.load(path, mmap_mode="r")
        vec = arr if arr.ndim == 1 else arr[ridx]
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        # Make it writable & contiguous to avoid PyTorch warning
        if (not getattr(vec, "flags", None) or (not vec.flags.writeable) or (not vec.flags["C_CONTIGUOUS"])):
            vec = np.array(vec, dtype=np.float32, copy=True)
        x = torch.from_numpy(vec).view(1, 1, -1)  # (1,1,1500)
        return x, torch.tensor(y, dtype=torch.long)


class UndersampledDeepPacketNPYDataset(Dataset):
    """
    Fast undersampled version that uses sampling indices instead of creating new files.
    This avoids the expensive disk I/O operations of the original implementation.
    """
    def __init__(self, root: str, files: List[Tuple[str, int]], counts: List[int], 
                 classes: List[str], class_to_idx: Dict[str, int], 
                 target_counts: Dict[int, int], strategy: str = "random"):
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.files = files
        self.counts = counts
        self.strategy = strategy
        
        # Calculate sampling indices for each file
        self.sampling_indices = []
        self.file_offsets = []
        self.total = 0
        
        # Distribute class targets across files
        class_remaining = {class_idx: target_count for class_idx, target_count in target_counts.items()}
        
        for i, (path, class_idx) in enumerate(files):
            current_count = counts[i]
            class_target = target_counts[class_idx]
            
            if class_target > 0 and current_count > 0:
                # Distribute the class target across files proportionally
                total_class_samples = sum(counts[j] for j, (_, c_idx) in enumerate(files) if c_idx == class_idx)
                if total_class_samples > 0:
                    # Calculate proportional target for this file
                    file_target = max(0, min(current_count, int(class_target * current_count / total_class_samples)))
                    # Ensure we don't exceed the remaining class target
                    file_target = min(file_target, class_remaining[class_idx])
                    class_remaining[class_idx] -= file_target
                else:
                    file_target = 0
            else:
                file_target = 0
            
            # Generate sampling indices for this file
            if file_target > 0 and file_target < current_count:
                if strategy == "random":
                    # Random sampling
                    indices = np.random.choice(current_count, file_target, replace=False)
                    indices = sorted(indices)  # Keep sorted for consistency
                elif strategy == "stratified":
                    # Stratified sampling - evenly distributed
                    step = current_count / file_target
                    indices = [int(i * step) for i in range(file_target)]
                else:
                    raise ValueError(f"Unknown undersampling strategy: {strategy}")
            elif file_target > 0:
                # Keep all samples
                indices = list(range(current_count))
            else:
                # No samples to keep
                indices = []
            
            self.sampling_indices.append(indices)
            self.file_offsets.append(self.total)
            self.total += len(indices)
        
        # Calculate class counts for the undersampled dataset
        self.class_counts = self._calculate_class_counts()
        self.class_weights = self._calculate_class_weights()
    
    def __len__(self) -> int:
        return self.total
    
    def _calculate_class_counts(self) -> Dict[int, int]:
        """Calculate the number of samples per class in the undersampled dataset."""
        class_counts = {i: 0 for i in range(len(self.classes))}
        for i, (_, class_idx) in enumerate(self.files):
            class_counts[class_idx] += len(self.sampling_indices[i])
        return class_counts
    
    def _calculate_class_weights(self, method: str = "balanced") -> Dict[int, float]:
        """Calculate class weights for the undersampled dataset."""
        total_samples = sum(self.class_counts.values())
        n_classes = len(self.class_counts)
        
        if method == "balanced":
            weights = {}
            for class_idx, count in self.class_counts.items():
                if count > 0:
                    weights[class_idx] = total_samples / (n_classes * count)
                else:
                    weights[class_idx] = 0.0
        elif method == "inverse":
            weights = {}
            for class_idx, count in self.class_counts.items():
                if count > 0:
                    weights[class_idx] = total_samples / count
                else:
                    weights[class_idx] = 0.0
        elif method == "sqrt_inverse":
            weights = {}
            for class_idx, count in self.class_counts.items():
                if count > 0:
                    weights[class_idx] = np.sqrt(total_samples / count)
                else:
                    weights[class_idx] = 0.0
        else:
            raise ValueError(f"Unknown weight method: {method}")
        
        return weights
    
    def get_sample_weights(self) -> List[float]:
        """Get sample weights for each sample in the undersampled dataset."""
        sample_weights = []
        for i, (_, class_idx) in enumerate(self.files):
            weight = self.class_weights[class_idx]
            # Repeat weight for each sample in this file
            sample_weights.extend([weight] * len(self.sampling_indices[i]))
        return sample_weights
    
    def _locate(self, idx: int) -> Tuple[int, int]:
        """Find which file and which sample within that file corresponds to the global index."""
        f = bisect.bisect_right(self.file_offsets, idx) - 1
        if f >= len(self.file_offsets):
            f = len(self.file_offsets) - 1
        row = idx - self.file_offsets[f]
        return f, row
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the undersampled dataset."""
        fidx, ridx = self._locate(idx)
        path, y = self.files[fidx]
        
        # Get the actual row index from the sampling indices
        actual_row = self.sampling_indices[fidx][ridx]
        
        arr = np.load(path, mmap_mode="r")
        vec = arr if arr.ndim == 1 else arr[actual_row]
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        # Make it writable & contiguous to avoid PyTorch warning
        if (not getattr(vec, "flags", None) or (not vec.flags.writeable) or (not vec.flags["C_CONTIGUOUS"])):
            vec = np.array(vec, dtype=np.float32, copy=True)
        x = torch.from_numpy(vec).view(1, 1, -1)  # (1,1,1500)
        return x, torch.tensor(y, dtype=torch.long)


def split_deeppacket(
    root: str,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 2,
    shuffle: bool = True,
    limit_files_per_split: int = 0,      # <-- NEW
    max_rows_per_file: int = None,       # <-- NEW
    handle_imbalance: bool = False,      # <-- NEW: Enable class imbalance handling
    weight_method: str = "balanced",     # <-- NEW: Method for calculating class weights
    undersample: bool = False,           # <-- NEW: Enable undersampling
    undersample_ratio: float = 0.1,      # <-- NEW: Undersampling ratio
    undersample_strategy: str = "random", # <-- NEW: Undersampling strategy
):
    tmp = DeepPacketNPYDataset(root)
    files = list(tmp.files)
    n_files = len(files)
    indices = np.arange(n_files)
    if shuffle:
        rng = np.random.RandomState(2018)
        rng.shuffle(indices)

    n_test = int(np.floor(test_size * n_files))
    n_val  = int(np.floor(valid_size * (n_files - n_test)))

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    # --- LIMIT FILE COUNT PER SPLIT ---
    def take(arr, n):
        return arr[:min(n, len(arr))] if (n and n > 0) else arr

    test_idx  = take(test_idx,  limit_files_per_split)
    val_idx   = take(val_idx,   limit_files_per_split)
    train_idx = take(train_idx, limit_files_per_split)

    # Pass max_rows_per_file and weight_method down to datasets
    train_ds = DeepPacketNPYDataset(root, split_indices=train_idx.tolist(), max_rows_per_file=max_rows_per_file, weight_method=weight_method)
    val_ds   = DeepPacketNPYDataset(root, split_indices=val_idx.tolist(),   max_rows_per_file=max_rows_per_file, weight_method=weight_method)
    test_ds  = DeepPacketNPYDataset(root, split_indices=test_idx.tolist(),  max_rows_per_file=max_rows_per_file, weight_method=weight_method)
    
    # Apply undersampling to training set if enabled
    if undersample:
        print(f"Applying undersampling (ratio={undersample_ratio}, strategy={undersample_strategy})...")
        train_ds_original = train_ds
        train_ds = train_ds.apply_undersampling(ratio=undersample_ratio, strategy=undersample_strategy)
        print(f"Original training samples: {train_ds_original.total}")
        print(f"Undersampled training samples: {train_ds.total}")

    # Print class distribution information
    print(f"Class distribution in training set:")
    for class_name, class_idx in train_ds.class_to_idx.items():
        count = train_ds.class_counts[class_idx]
        print(f"  {class_name}: {count} samples")
    
    if handle_imbalance:
        print(f"Class weights (method: {weight_method}):")
        for class_name, class_idx in train_ds.class_to_idx.items():
            weight = train_ds.class_weights[class_idx]
            print(f"  {class_name}: {weight:.4f}")

    dl_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    
    # Create data loaders with optional weighted sampling
    if handle_imbalance:
        # Use WeightedRandomSampler for training to handle class imbalance
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **dl_args)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **dl_args)
    
    valid_loader = DataLoader(val_ds,  shuffle=False, **dl_args)
    test_loader  = DataLoader(test_ds, shuffle=False, **dl_args)
    return train_loader, valid_loader, test_loader, train_ds, test_ds
# =========================================================
# Arg/build + main
# =========================================================

def generate_dir_names(dataset: str, args: TrainArgs, make: bool = True) -> Tuple[str, str, str]:
    suffix = f"{args.theta_reg_type}_H{args.h_type}_Reg{args.theta_reg_lambda:.0e}_LR{args.lr}"
    model_path = os.path.join(args.model_path, dataset, suffix)
    log_path = os.path.join(args.log_path, dataset, suffix)
    results_path = os.path.join(args.results_path, dataset, suffix)
    if make:
        for p in [model_path, results_path]:
            os.makedirs(p, exist_ok=True)
    return model_path, log_path, results_path

def build_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="proc_pcaps/", help="DeepPacket root with class folders of .npy files")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--theta_reg_lambda", type=float, default=1e-2)
    parser.add_argument("--theta_reg_type", type=str, default="grad3", choices=["unreg", "none", "grad1", "grad2", "grad3"])
    parser.add_argument("--seed", type=int, default=2018)
    parser.add_argument("--limit_files_per_split", type=int, default=0,
                        help="Max files per split (train/val/test). Set small to speed up.")
    parser.add_argument("--max_rows_per_file", type=int, default=None,
                        help="Cap rows loaded from each .npy file.")
    parser.add_argument("--max_batches_per_epoch", type=int, default=200,
                        help="Train only on this many batches per epoch (0 = no cap).")
    parser.add_argument("--eval_batches", type=int, default=50,
                        help="Validate/test on only this many batches (0 = no cap).")
    parser.add_argument("--handle_imbalance", action="store_true",
                        help="Enable class imbalance handling with weighted sampling.")
    parser.add_argument("--weight_method", type=str, default="balanced", 
                        choices=["balanced", "inverse", "sqrt_inverse"],
                        help="Method for calculating class weights: balanced (sklearn-style), inverse, or sqrt_inverse.")
    parser.add_argument("--undersample", action="store_true",
                        help="Enable mild undersampling to reduce class imbalance.")
    parser.add_argument("--undersample_ratio", type=float, default=0.1,
                        help="Ratio of samples to keep from majority classes (0.1 = keep 10% of largest class).")
    parser.add_argument("--undersample_strategy", type=str, default="random",
                        choices=["random", "stratified"],
                        help="Undersampling strategy: random or stratified.")
    args_ns = parser.parse_args()

    args = TrainArgs(
        cuda=args_ns.cuda,
        nclasses=2,                      # will be overwritten after probing
        lr=args_ns.lr,
        epochs=args_ns.epochs,
        theta_reg_lambda=args_ns.theta_reg_lambda,
        theta_reg_type=args_ns.theta_reg_type,

        # NEW: propagate the caps
        limit_files_per_split=args_ns.limit_files_per_split,
        max_rows_per_file=args_ns.max_rows_per_file,
        max_batches_per_epoch=args_ns.max_batches_per_epoch,
        eval_batches=args_ns.eval_batches,
        
        # Class imbalance handling
        handle_imbalance=args_ns.handle_imbalance,
        weight_method=args_ns.weight_method,
        
        # Undersampling options
        undersample=args_ns.undersample,
        undersample_ratio=args_ns.undersample_ratio,
        undersample_strategy=args_ns.undersample_strategy,
    )

    # also keep these convenience attrs
    args.root = args_ns.root         # type: ignore[attr-defined]
    args.batch_size = args_ns.batch_size  # type: ignore[attr-defined]
    args.seed = args_ns.seed         # type: ignore[attr-defined]
    return args

def print_full_config(args, model=None):
    """Pretty-print full training configuration and model summary."""
    import pprint
    print("\n" + "=" * 70)
    print(" CURRENT CONFIGURATION")
    print("=" * 70)

    # Print all args (dataclass or AttrDict-compatible)
    if hasattr(args, "__dict__"):
        cfg_dict = vars(args)
    else:
        cfg_dict = args.__dict__ if isinstance(args, object) else dict(args)
    for k in sorted(cfg_dict.keys()):
        print(f"{k:25s}: {cfg_dict[k]}")

    # Print device and model info
    if model is not None:
        device = next(model.parameters()).device
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n" + "-" * 70)
        print("MODEL SUMMARY")
        print("-" * 70)
        print(f"Device:        {device}")
        print(f"Total params:  {n_params:,}")
        print(f"Trainable:     {n_trainable:,}")
        print(f"Model type:    {model.__class__.__name__}")
        print("-" * 70)
        print(model)
    print("=" * 70 + "\n")

def main():
    args = build_args()
    set_seed(getattr(args, "seed", 2018))

    # Probe classes
    probe = DeepPacketNPYDataset(args.root)  # type: ignore[attr-defined]
    nclasses = len(probe.classes)
    args.nclasses = nclasses

    # Input dim for DeepPacket vectors
    input_dim = 1500 + 1  # +1 because InputConceptizer appends bias; but parametrizer sees raw x, so keep 1500
    raw_input_dim = 1500

    # Data
    train_loader, valid_loader, test_loader, train_ds, test_ds = split_deeppacket(
        root=args.root,
        valid_size=0.1,
        test_size=0.1,
        batch_size=getattr(args, "batch_size", 128),
        num_workers=2,
        shuffle=True,
        limit_files_per_split=getattr(args, "limit_files_per_split", 0),
        max_rows_per_file=getattr(args, "max_rows_per_file", None),
        handle_imbalance=getattr(args, "handle_imbalance", False),
        weight_method=getattr(args, "weight_method", "balanced"),
        undersample=getattr(args, "undersample", False),
        undersample_ratio=getattr(args, "undersample_ratio", 0.1),
        undersample_strategy=getattr(args, "undersample_strategy", "random"),
    )

    # Model
    conceptizer = InputConceptizer(add_bias=True)  # bias as an extra concept at the end
    nconcepts = raw_input_dim + 1  # each feature + bias
    parametrizer = LinearParametrizer(
        input_dim=raw_input_dim, nconcept=nconcepts, nclass=args.nclasses, hidden=512, only_positive=False
    )
    aggregator = AdditiveScalarAggregator(cdim=1, nclasses=args.nclasses)
    model = GSENN(conceptizer, parametrizer, aggregator, debug=False)

    # Paths
    model_path, _, _ = generate_dir_names("deeppacket", args)

    # Trainer (unregularized or with grad penalty)
    if args.theta_reg_type in ["unreg", "none"]:
        class VanillaTrainer(ClassificationTrainer):
            def train_batch(self, inputs, targets):
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.prediction_criterion(
                    logits, targets if self.nclasses > 2 else targets.float().unsqueeze(1)
                )
                loss.backward()
                self.optimizer.step()
                return logits.detach(), loss.detach(), {"prediction": float(loss.item()), "grad_penalty": 0.0}
        trainer: ClassificationTrainer = VanillaTrainer(model, args)
    else:
        typ = {"grad1": 1, "grad2": 2, "grad3": 3}.get(args.theta_reg_type, 3)
        trainer = GradPenaltyTrainer(model, args, typ=typ)

    # Train & evaluate
    trainer.train(train_loader, valid_loader, epochs=args.epochs, save_path=model_path)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    trainer.model.to(device)
    print_full_config(args, trainer.model)

    print("Computing accuracies…")
    train_acc = trainer.validate(train_loader) if train_loader is not None else None
    val_acc = trainer.validate(valid_loader) if valid_loader is not None else None
    test_acc = trainer.evaluate(test_loader) if test_loader is not None else None

    print("\nFinal accuracies:")
    print(f"  Train Accuracy : {train_acc:.2f}%" if train_acc is not None else "  Train Accuracy : (n/a)")
    print(f"  Val   Accuracy : {val_acc:.2f}%"   if val_acc is not None else "  Val   Accuracy : (n/a)")
    print(f"  Test  Accuracy : {test_acc:.2f}%"  if test_acc is not None else "  Test  Accuracy : (n/a)")

if __name__ == "__main__":
    main()
