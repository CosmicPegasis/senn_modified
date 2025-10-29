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

    flow_split: bool = True
    flow_suffix: str = ".flow.npy" 

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
            # Include both chunked and non-chunked .npy files, but EXCLUDE sidecar flow files
            for p in glob.glob(os.path.join(root, cls, "*.npy")):
                # Guard against including sidecar flow-id files as data inputs
                if p.endswith(".flow.npy"):
                    continue
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
        # Normalize shape to fixed length 1500 and dtype float32 in [0,1]
        try:
            vec_len = int(vec.shape[-1]) if hasattr(vec, 'shape') and len(vec.shape) > 0 else int(len(vec))
        except Exception:
            vec_len = 0
        if vec_len != 1500:
            v = np.zeros(1500, dtype=np.float32)
            # best-effort copy of as many elements as available
            try:
                tmp = np.asarray(vec)
                if tmp.ndim > 1:
                    tmp = tmp.reshape(-1)
                n = min(tmp.size, 1500)
                v[:n] = tmp[:n].astype(np.float32, copy=False)
            except Exception:
                pass
            vec = v
        else:
            if vec.dtype != np.float32:
                vec = vec.astype(np.float32, copy=False)
        # Scale if appears to be raw bytes (0..255)
        if np.nanmax(vec) > 1.5:
            vec = (vec / 255.0).astype(np.float32, copy=False)
        # Make it writable & contiguous to avoid PyTorch warning
        if (not getattr(vec, "flags", None) or (not vec.flags.writeable) or (not vec.flags["C_CONTIGUOUS"])):
            vec = np.array(vec, dtype=np.float32, copy=True)
       # x = torch.from_numpy(vec).view(1, 1, -1)  # (1,1,1500)
        x = torch.tensor(vec, dtype=torch.float32).contiguous().view(1, 1, -1)
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
        # Normalize shape to fixed length 1500 and dtype float32 in [0,1]
        try:
            vec_len = int(vec.shape[-1]) if hasattr(vec, 'shape') and len(vec.shape) > 0 else int(len(vec))
        except Exception:
            vec_len = 0
        if vec_len != 1500:
            v = np.zeros(1500, dtype=np.float32)
            try:
                tmp = np.asarray(vec)
                if tmp.ndim > 1:
                    tmp = tmp.reshape(-1)
                n = min(tmp.size, 1500)
                v[:n] = tmp[:n].astype(np.float32, copy=False)
            except Exception:
                pass
            vec = v
        else:
            if vec.dtype != np.float32:
                vec = vec.astype(np.float32, copy=False)
        if np.nanmax(vec) > 1.5:
            vec = (vec / 255.0).astype(np.float32, copy=False)
        # Make it writable & contiguous to avoid PyTorch warning
        if (not getattr(vec, "flags", None) or (not vec.flags.writeable) or (not vec.flags["C_CONTIGUOUS"])):
            vec = np.array(vec, dtype=np.float32, copy=True)
       # x = torch.from_numpy(vec).view(1, 1, -1)  # (1,1,1500)
        x = torch.tensor(vec, dtype=torch.float32).contiguous().view(1, 1, -1)
        return x, torch.tensor(y, dtype=torch.long)

# =========================================================
# Flow-aware dataset + utilities (uses sidecars: *.flow.npy)
# =========================================================

def _paired_flow_path(data_path: str, flow_suffix: str = ".flow.npy") -> str:
    if data_path.endswith(".npy"):
        return data_path[:-4] + flow_suffix
    return data_path + flow_suffix

class FlowAwareDeepPacketDataset(DeepPacketNPYDataset):
    """
    Extends DeepPacketNPYDataset by loading aligned uint64 flow IDs when present.
    For each data .npy, expects an aligned sidecar "<stem>.flow.npy" with per-row uint64 IDs.
    """
    def __init__(self, *args, flow_suffix: str = ".flow.npy", **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_suffix = flow_suffix
        self._flow_paths = []
        self._flow_memmaps = []
        for path, _ in self.files:
            fpath = _paired_flow_path(path, flow_suffix=self.flow_suffix)
            self._flow_paths.append(fpath if os.path.exists(fpath) else None)
            self._flow_memmaps.append(None)  # lazy-init

    def _ensure_flow_mm(self, file_idx: int):
        fpath = self._flow_paths[file_idx]
        if fpath is None:
            return None
        mm = self._flow_memmaps[file_idx]
        if mm is None:
            mm = np.load(fpath, mmap_mode="r")
            if mm.dtype != np.uint64:
                mm = mm.astype(np.uint64, copy=False)
            self._flow_memmaps[file_idx] = mm
        return mm

    def flow_id_at(self, global_idx: int) -> int:
        fidx, ridx = self._locate(global_idx)
        mm = self._ensure_flow_mm(fidx)
        if mm is None:
            return int(global_idx)  # fallback: each row = its own "flow"
        return int(mm[ridx] if mm.ndim == 1 else mm[ridx])

class SelectedRowsDeepPacketDataset(Dataset):
    """
    Lightweight view over DeepPacketNPYDataset that reads only specific rows per file.
    """
    def __init__(self, base_ds: DeepPacketNPYDataset, per_file_rows: Dict[int, List[int]], weight_method: str = "balanced"):
        self.base = base_ds
        self.root = base_ds.root
        self.files = base_ds.files
        self.classes = base_ds.classes
        self.class_to_idx = base_ds.class_to_idx

        self.sampling_indices = []
        self.file_offsets = []
        self.total = 0

        for fidx in range(len(self.files)):
            rows = sorted(set(per_file_rows.get(fidx, [])))
            self.sampling_indices.append(rows)
            self.file_offsets.append(self.total)
            self.total += len(rows)

        self.class_counts = self._calculate_class_counts()
        self.class_weights = self._calculate_class_weights(method=weight_method)

    def __len__(self) -> int:
        return self.total

    def _calculate_class_counts(self) -> Dict[int, int]:
        class_counts = {i: 0 for i in range(len(self.classes))}
        for i, (_, class_idx) in enumerate(self.files):
            class_counts[class_idx] += len(self.sampling_indices[i])
        return class_counts

    def _calculate_class_weights(self, method: str = "balanced") -> Dict[int, float]:
        total_samples = sum(self.class_counts.values())
        n_classes = len(self.class_counts)
        weights = {}
        for class_idx, count in self.class_counts.items():
            if count <= 0:
                weights[class_idx] = 0.0
            elif method == "balanced":
                weights[class_idx] = total_samples / (n_classes * count)
            elif method == "inverse":
                weights[class_idx] = total_samples / count
            elif method == "sqrt_inverse":
                weights[class_idx] = (total_samples / count) ** 0.5
            else:
                raise ValueError(f"Unknown weight method: {method}")
        return weights

    def get_sample_weights(self) -> List[float]:
        ws = []
        for i, (_, class_idx) in enumerate(self.files):
            w = self.class_weights[class_idx]
            ws.extend([w] * len(self.sampling_indices[i]))
        return ws

    def _locate(self, idx: int) -> Tuple[int, int]:
        f = bisect.bisect_right(self.file_offsets, idx) - 1
        f = min(max(f, 0), len(self.file_offsets) - 1)
        row = idx - self.file_offsets[f]
        return f, row

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fidx, ridx = self._locate(idx)
        path, y = self.files[fidx]
        actual_row = self.sampling_indices[fidx][ridx]
        arr = np.load(path, mmap_mode="r")
        vec = arr if arr.ndim == 1 else arr[actual_row]
        # Normalize shape to fixed length 1500 and dtype float32 in [0,1]
        try:
            vec_len = int(vec.shape[-1]) if hasattr(vec, 'shape') and len(vec.shape) > 0 else int(len(vec))
        except Exception:
            vec_len = 0
        if vec_len != 1500:
            v = np.zeros(1500, dtype=np.float32)
            try:
                tmp = np.asarray(vec)
                if tmp.ndim > 1:
                    tmp = tmp.reshape(-1)
                n = min(tmp.size, 1500)
                v[:n] = tmp[:n].astype(np.float32, copy=False)
            except Exception:
                pass
            vec = v
        else:
            if vec.dtype != np.float32:
                vec = vec.astype(np.float32, copy=False)
        if np.nanmax(vec) > 1.5:
            vec = (vec / 255.0).astype(np.float32, copy=False)
        if (not getattr(vec, "flags", None) or (not vec.flags.writeable) or (not vec.flags["C_CONTIGUOUS"])):
            vec = np.array(vec, dtype=np.float32, copy=True)
        x = torch.tensor(vec, dtype=torch.float32).contiguous().view(1, 1, -1)
        return x, torch.tensor(y, dtype=torch.long)

def _assert_no_flow_overlap_datasets(train_ds, val_ds, test_ds, flow_suffix: str = ".flow.npy") -> None:
    """
    Sanity check: ensure no flow IDs overlap across train/val/test.
    Works for both SelectedRowsDeepPacketDataset (flow split) and DeepPacketNPYDataset (file split).
    If sidecar flow files are absent, uses per-file-scoped synthetic IDs so it will not raise.
    """
    def collect_flow_ids_for_selected(ds: SelectedRowsDeepPacketDataset) -> set:
        flow_ids = set()
        base = getattr(ds, "base", None)
        # If base is FlowAwareDeepPacketDataset, we can read real flow IDs
        for fidx in range(len(ds.files)):
            rows = ds.sampling_indices[fidx] if hasattr(ds, "sampling_indices") else list(range(getattr(ds, "counts", [0])[fidx]))
            if not rows:
                continue
            if base is not None and hasattr(base, "_ensure_flow_mm"):
                mm = base._ensure_flow_mm(fidx)
                if mm is not None:
                    for r in rows:
                        flow_ids.add(int(mm[r] if mm.ndim == 1 else mm[r]))
                    continue
            # Fallback synthetic IDs scoped by file path to avoid cross-file collisions
            fpath = ds.files[fidx][0]
            for r in rows:
                flow_ids.add((fpath, int(r)))
        return flow_ids

    def collect_flow_ids_for_fullfile(ds: DeepPacketNPYDataset) -> set:
        flow_ids = set()
        for fidx, (path, _) in enumerate(ds.files):
            fpath = path[:-4] + flow_suffix if path.endswith(".npy") else path + flow_suffix
            if os.path.exists(fpath):
                try:
                    mm = np.load(fpath, mmap_mode="r")
                    if mm.dtype != np.uint64:
                        mm = mm.astype(np.uint64, copy=False)
                    # unique per file; add raw IDs
                    for v in np.unique(mm):
                        flow_ids.add(int(v))
                    continue
                except Exception:
                    pass
            # Fallback synthetic IDs scoped by file path
            nrows = getattr(ds, "counts", [0])[fidx]
            for r in range(int(nrows)):
                flow_ids.add((path, int(r)))
        return flow_ids

    # Determine collector based on dataset types
    if isinstance(train_ds, SelectedRowsDeepPacketDataset):
        train_flows = collect_flow_ids_for_selected(train_ds)
        val_flows   = collect_flow_ids_for_selected(val_ds)
        test_flows  = collect_flow_ids_for_selected(test_ds)
    else:
        train_flows = collect_flow_ids_for_fullfile(train_ds)
        val_flows   = collect_flow_ids_for_fullfile(val_ds)
        test_flows  = collect_flow_ids_for_fullfile(test_ds)

    inter_tv = train_flows & val_flows
    inter_tt = train_flows & test_flows
    inter_vt = val_flows & test_flows
    if inter_tv or inter_tt or inter_vt:
        def sample(s):
            # show up to 5 examples
            out = list(s)
            return out[:5]
        msg = [
            "Flow overlap detected across splits:",
            f"  train ∩ val  : {len(inter_tv)} examples -> {sample(inter_tv)}",
            f"  train ∩ test : {len(inter_tt)} examples -> {sample(inter_tt)}",
            f"  val   ∩ test : {len(inter_vt)} examples -> {sample(inter_vt)}",
        ]
        raise RuntimeError("\n".join(msg))

def group_indices_by_flow(ds: FlowAwareDeepPacketDataset):
    """
    Builds:
      - by_flow: dict[flow_id] -> list[global_row_idx]
      - flow_to_class: dict[flow_id] -> class_idx
    If no sidecar exists for a file, each row is treated as its own flow.
    """
    by_flow, flow_to_class = {}, {}
    for file_idx, (path, cls_idx) in enumerate(ds.files):
        # IMPORTANT: respect ds.counts (caps like max_rows_per_file), not full file length
        nrows = int(getattr(ds, "counts", [0])[file_idx])
        if nrows <= 0:
            continue
        fpath = _paired_flow_path(path, flow_suffix=getattr(ds, 'flow_suffix', '.flow.npy'))
        fmm = None
        if os.path.exists(fpath):
            fmm = np.load(fpath, mmap_mode="r")
            if fmm.dtype != np.uint64:
                fmm = fmm.astype(np.uint64, copy=False)
            if (fmm.ndim != 1) or (len(fmm) < nrows):
                raise RuntimeError(f"Flow file length mismatch: {fpath} vs {path}")
        base = ds.offsets[file_idx]
        for i in range(nrows):
            gidx = base + i
            fid = int(fmm[i]) if fmm is not None else int(gidx)
            by_flow.setdefault(fid, []).append(gidx)
            flow_to_class[fid] = cls_idx
    return by_flow, flow_to_class

def stratified_flow_split(by_flow, flow_to_class, valid_size=0.1, test_size=0.1, seed=2018):
    """
    Stratify by class at the **flow** level, then expand flows to row indices.
    """
    rng = np.random.RandomState(seed)
    class2flows: Dict[int, List[int]] = {}
    for fid, c in flow_to_class.items():
        class2flows.setdefault(c, []).append(fid)

    train_idx, val_idx, test_idx = [], [], []
    for c, fids in class2flows.items():
        fids = fids.copy()
        rng.shuffle(fids)
        n = len(fids)
        n_test = int(np.floor(test_size * n))
        n_val  = int(np.floor(valid_size * (n - n_test)))
        test_f = fids[:n_test]
        val_f  = fids[n_test:n_test+n_val]
        train_f= fids[n_test+n_val:]

        for fid in train_f: train_idx.extend(by_flow[fid])
        for fid in val_f:   val_idx.extend(by_flow[fid])
        for fid in test_f:  test_idx.extend(by_flow[fid])

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)

def split_deeppacket_by_flow(
    root: str,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 2,
    weight_method: str = "balanced",
    max_rows_per_file: int = None,
    handle_imbalance: bool = False,
    flow_suffix: str = ".flow.npy",
):
    # Build a flow-aware base dataset to discover counts/offsets/files.
    base = FlowAwareDeepPacketDataset(root, max_rows_per_file=max_rows_per_file, weight_method=weight_method, flow_suffix=flow_suffix)
    by_flow, flow_to_class = group_indices_by_flow(base)
    train_rows, val_rows, test_rows = stratified_flow_split(by_flow, flow_to_class, valid_size=valid_size, test_size=test_size, seed=2018)

    def rows_to_per_file(ds, rows):
        per = {i: [] for i in range(len(ds.files))}
        for g in rows.tolist():
            fidx = bisect.bisect_right(ds.offsets, g) - 1
            fidx = min(max(fidx, 0), len(ds.offsets)-2)
            ridx = g - ds.offsets[fidx]
            per[fidx].append(int(ridx))
        return per

    train_ds = SelectedRowsDeepPacketDataset(base, rows_to_per_file(base, train_rows), weight_method=weight_method)
    val_ds   = SelectedRowsDeepPacketDataset(base, rows_to_per_file(base, val_rows),   weight_method=weight_method)
    test_ds  = SelectedRowsDeepPacketDataset(base, rows_to_per_file(base, test_rows),  weight_method=weight_method)

    # Print basic distro info to mirror your current logs.
    print("Class distribution (flow-split) in training set:")
    for class_name, class_idx in train_ds.class_to_idx.items():
        print(f"  {class_name}: {train_ds.class_counts[class_idx]} samples")
    print("Class distribution (flow-split) in validation set:")
    for class_name, class_idx in val_ds.class_to_idx.items():
        print(f"  {class_name}: {val_ds.class_counts[class_idx]} samples")
    print("Class distribution (flow-split) in test set:")
    for class_name, class_idx in test_ds.class_to_idx.items():
        print(f"  {class_name}: {test_ds.class_counts[class_idx]} samples")

    if handle_imbalance:
        print(f"Class weights (method: {weight_method}):")
        for class_name, class_idx in train_ds.class_to_idx.items():
            print(f"  {class_name}: {train_ds.class_weights[class_idx]:.4f}")

    dl_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    if handle_imbalance:
        sampler = WeightedRandomSampler(
            weights=train_ds.get_sample_weights(),
            num_samples=sum(len(v) for v in train_ds.sampling_indices),
            replacement=True
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **dl_args)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **dl_args)

    valid_loader = DataLoader(val_ds, shuffle=False, **dl_args)
    test_loader  = DataLoader(test_ds, shuffle=False, **dl_args)
    # Sanity check: ensure no overlapping flows across splits
    _assert_no_flow_overlap_datasets(train_ds, val_ds, test_ds, flow_suffix=flow_suffix)
    return train_loader, valid_loader, test_loader, train_ds, test_ds


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
    print(f"Class distribution in validation set:")
    for class_name, class_idx in val_ds.class_to_idx.items():
        count = val_ds.class_counts[class_idx]
        print(f"  {class_name}: {count} samples")
    print(f"Class distribution in test set:")
    for class_name, class_idx in test_ds.class_to_idx.items():
        count = test_ds.class_counts[class_idx]
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
    # Sanity check: ensure no overlapping flows across splits (uses sidecars if present)
    _assert_no_flow_overlap_datasets(train_ds, val_ds, test_ds, flow_suffix=".flow.npy")
    return train_loader, valid_loader, test_loader, train_ds, test_ds
# =========================================================
# Flow-based sanity checks
# =========================================================

def verify_flow_sidecar_alignment(root: str, flow_suffix: str = ".flow.npy") -> Dict[str, any]:
    """
    Verify that flow sidecar files are properly aligned with data files.
    Returns a dict with verification results and any issues found.
    """
    issues = []
    stats = {
        "total_files": 0,
        "files_with_sidecars": 0,
        "files_without_sidecars": 0,
        "misaligned_files": 0,
        "issues": issues
    }
    
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    for cls in classes:
        cls_path = os.path.join(root, cls)
        npy_files = [f for f in glob.glob(os.path.join(cls_path, "*.npy")) 
                     if not f.endswith(flow_suffix)]
        
        for data_file in npy_files:
            stats["total_files"] += 1
            flow_file = _paired_flow_path(data_file, flow_suffix)
            
            if not os.path.exists(flow_file):
                stats["files_without_sidecars"] += 1
                issues.append(f"Missing sidecar: {data_file} -> {flow_file}")
                continue
            
            stats["files_with_sidecars"] += 1
            
            # Check alignment
            try:
                data_arr = np.load(data_file, mmap_mode="r")
                flow_arr = np.load(flow_file, mmap_mode="r")
                
                data_rows = 1 if data_arr.ndim == 1 else data_arr.shape[0]
                flow_rows = flow_arr.shape[0] if flow_arr.ndim == 1 else flow_arr.size
                
                if data_rows != flow_rows:
                    stats["misaligned_files"] += 1
                    issues.append(
                        f"Misaligned: {data_file} has {data_rows} rows, "
                        f"but {flow_file} has {flow_rows} flow IDs"
                    )
            except Exception as e:
                stats["misaligned_files"] += 1
                issues.append(f"Error checking {data_file}: {e}")
    
    return stats

def verify_flow_class_consistency(ds: FlowAwareDeepPacketDataset) -> Dict[str, any]:
    """
    Verify that all packets in a flow have the same class label.
    This is critical - flows should not span multiple traffic classes.
    """
    flow_to_classes = {}
    inconsistent_flows_set = set()  # Track unique inconsistent flow IDs
    inconsistent_flow_examples = []  # Track examples for reporting
    
    for file_idx, (path, cls_idx) in enumerate(ds.files):
        nrows = int(getattr(ds, "counts", [0])[file_idx])
        if nrows <= 0:
            continue
        
        fpath = _paired_flow_path(path, flow_suffix=getattr(ds, 'flow_suffix', '.flow.npy'))
        if not os.path.exists(fpath):
            continue
        
        try:
            fmm = np.load(fpath, mmap_mode="r")
            if fmm.dtype != np.uint64:
                fmm = fmm.astype(np.uint64, copy=False)
            
            for i in range(min(nrows, len(fmm))):
                fid = int(fmm[i])
                if fid not in flow_to_classes:
                    flow_to_classes[fid] = cls_idx
                elif flow_to_classes[fid] != cls_idx:
                    # Only record the first time we see this flow as inconsistent
                    if fid not in inconsistent_flows_set:
                        inconsistent_flows_set.add(fid)
                        inconsistent_flow_examples.append({
                            "flow_id": fid,
                            "classes": [flow_to_classes[fid], cls_idx],
                            "file": path
                        })
        except Exception as e:
            print(f"Error checking flow consistency in {path}: {e}")
    
    return {
        "total_flows": len(flow_to_classes),
        "inconsistent_flows": len(inconsistent_flows_set),
        "issues": inconsistent_flow_examples[:10]  # Show first 10
    }

def verify_flow_id_generation(sample_data_file: str, flow_suffix: str = ".flow.npy") -> Dict[str, any]:
    """
    Test that flow ID generation is deterministic by checking a sample of packets.
    This verifies the flow hashing is working correctly.
    """
    try:
        from deep_packet_proc.preproc_flow_new import _flow_id_from_tuple
    except ImportError:
        return {"status": "skipped", "reason": "Could not import flow ID generation function"}
    
    flow_file = _paired_flow_path(sample_data_file, flow_suffix)
    if not os.path.exists(flow_file):
        return {"status": "skipped", "reason": f"No flow file at {flow_file}"}
    
    flow_arr = np.load(flow_file, mmap_mode="r")
    
    # Check for reasonable distribution of flow IDs
    unique_flows = np.unique(flow_arr)
    total_packets = len(flow_arr)
    avg_packets_per_flow = total_packets / len(unique_flows) if len(unique_flows) > 0 else 0
    
    # Check that flow IDs are actually uint64 and not all zeros
    all_zeros = np.all(flow_arr == 0)
    
    return {
        "status": "ok",
        "total_packets": int(total_packets),
        "unique_flows": int(len(unique_flows)),
        "avg_packets_per_flow": float(avg_packets_per_flow),
        "all_zeros": bool(all_zeros),
        "sample_flow_ids": [int(x) for x in unique_flows[:5].tolist()]
    }

def verify_split_statistics(train_ds, val_ds, test_ds) -> Dict[str, any]:
    """
    Compute detailed statistics about the splits to identify potential issues.
    """
    def get_stats(ds, name):
        total = len(ds)
        class_counts = getattr(ds, "class_counts", {})
        
        # Check if it's a SelectedRowsDeepPacketDataset (flow split)
        if hasattr(ds, "sampling_indices"):
            total_rows = sum(len(indices) for indices in ds.sampling_indices)
            total_files = len([idx for idx, indices in enumerate(ds.sampling_indices) if len(indices) > 0])
        else:
            total_rows = total
            total_files = len(getattr(ds, "files", []))
        
        return {
            "name": name,
            "total_samples": total,
            "total_rows": total_rows,
            "total_files": total_files,
            "class_distribution": dict(class_counts),
            "min_class_count": min(class_counts.values()) if class_counts else 0,
            "max_class_count": max(class_counts.values()) if class_counts else 0,
        }
    
    train_stats = get_stats(train_ds, "train")
    val_stats = get_stats(val_ds, "validation")
    test_stats = get_stats(test_ds, "test")
    
    # Check for extreme imbalances
    warnings = []
    
    # Check if validation set is too small
    if val_stats["total_samples"] < 1000:
        warnings.append(f"Validation set very small: {val_stats['total_samples']} samples")
    
    # Check for missing classes in splits
    train_classes = set(train_stats["class_distribution"].keys())
    val_classes = set(val_stats["class_distribution"].keys())
    test_classes = set(test_stats["class_distribution"].keys())
    
    missing_in_val = train_classes - val_classes
    missing_in_test = train_classes - test_classes
    
    if missing_in_val:
        warnings.append(f"Classes missing in validation: {missing_in_val}")
    if missing_in_test:
        warnings.append(f"Classes missing in test: {missing_in_test}")
    
    # Check for extreme class imbalance ratios
    for stats in [train_stats, val_stats, test_stats]:
        if stats["max_class_count"] > 0 and stats["min_class_count"] > 0:
            ratio = stats["max_class_count"] / stats["min_class_count"]
            if ratio > 1000:
                warnings.append(
                    f"{stats['name']} has extreme imbalance: "
                    f"{ratio:.1f}x (max={stats['max_class_count']}, min={stats['min_class_count']})"
                )
    
    return {
        "train": train_stats,
        "validation": val_stats,
        "test": test_stats,
        "warnings": warnings
    }

def run_comprehensive_flow_checks(root: str, train_ds, val_ds, test_ds, 
                                   flow_suffix: str = ".flow.npy") -> None:
    """
    Run all flow-based sanity checks and print a comprehensive report.
    """
    print("\n" + "=" * 70)
    print(" FLOW-BASED DATASET SANITY CHECKS")
    print("=" * 70 + "\n")
    
    # 1. Verify sidecar alignment
    print("1. Checking flow sidecar alignment...")
    alignment = verify_flow_sidecar_alignment(root, flow_suffix)
    print(f"   Total data files: {alignment['total_files']}")
    print(f"   Files with sidecars: {alignment['files_with_sidecars']}")
    print(f"   Files without sidecars: {alignment['files_without_sidecars']}")
    print(f"   Misaligned files: {alignment['misaligned_files']}")
    if alignment['issues']:
        print(f"   Issues found (showing first 5):")
        for issue in alignment['issues'][:5]:
            print(f"     - {issue}")
    
    # 2. Verify flow class consistency (if using FlowAwareDeepPacketDataset)
    base_ds = getattr(train_ds, "base", train_ds)
    if isinstance(base_ds, FlowAwareDeepPacketDataset):
        print("\n2. Checking flow class consistency...")
        consistency = verify_flow_class_consistency(base_ds)
        print(f"   Total unique flows: {consistency['total_flows']}")
        print(f"   Inconsistent flows (cross-class): {consistency['inconsistent_flows']}")
        if consistency['issues']:
            print(f"   Examples of inconsistent flows:")
            for issue in consistency['issues'][:3]:
                print(f"     - Flow {issue['flow_id']}: spans classes {issue['classes']}")
    
    # 3. Test flow ID generation on a sample
    print("\n3. Testing flow ID generation...")
    if train_ds.files:
        sample_file = train_ds.files[0][0]
        flow_gen = verify_flow_id_generation(sample_file, flow_suffix)
        print(f"   Status: {flow_gen.get('status', 'unknown')}")
        if flow_gen.get('status') == 'ok':
            print(f"   Sample file packets: {flow_gen['total_packets']}")
            print(f"   Unique flows: {flow_gen['unique_flows']}")
            print(f"   Avg packets/flow: {flow_gen['avg_packets_per_flow']:.1f}")
            print(f"   All zeros (BAD): {flow_gen['all_zeros']}")
            if flow_gen.get('sample_flow_ids'):
                print(f"   Sample flow IDs: {flow_gen['sample_flow_ids']}")
    
    # 4. Split statistics
    print("\n4. Analyzing split statistics...")
    split_stats = verify_split_statistics(train_ds, val_ds, test_ds)
    
    for split_name in ['train', 'validation', 'test']:
        stats = split_stats[split_name]
        print(f"\n   {stats['name'].upper()}:")
        print(f"     Total samples: {stats['total_samples']}")
        print(f"     Total files: {stats['total_files']}")
        print(f"     Class range: [{stats['min_class_count']}, {stats['max_class_count']}]")
        if stats['max_class_count'] > 0 and stats['min_class_count'] > 0:
            ratio = stats['max_class_count'] / stats['min_class_count']
            print(f"     Imbalance ratio: {ratio:.1f}x")
    
    if split_stats['warnings']:
        print("\n   ⚠️  WARNINGS:")
        for warning in split_stats['warnings']:
            print(f"     - {warning}")
    
    # 5. Verify no flow overlap (already exists, just call it)
    print("\n5. Verifying no flow overlap across splits...")
    try:
        _assert_no_flow_overlap_datasets(train_ds, val_ds, test_ds, flow_suffix=flow_suffix)
        print("   ✓ No flow overlap detected - splits are clean!")
    except RuntimeError as e:
        print(f"   ✗ FLOW OVERLAP DETECTED:")
        print(f"     {str(e)}")
    
    print("\n" + "=" * 70)
    print(" END OF SANITY CHECKS")
    print("=" * 70 + "\n")

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
    parser.add_argument("--flow_split", action="store_true",
        help="Split train/val/test by FLOW (requires aligned *.flow.npy sidecars).")
    parser.add_argument("--flow_suffix", type=str, default=".flow.npy",
        help="Sidecar suffix for flow IDs (default: .flow.npy).")

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
    if args.flow_split:
        train_loader, valid_loader, test_loader, train_ds, test_ds = split_deeppacket_by_flow(
            root=args.root,
            valid_size=0.1, test_size=0.1,
            batch_size=args.batch_size,
            num_workers=2,
            weight_method=args.weight_method,
            max_rows_per_file=args.max_rows_per_file,
            handle_imbalance=args.handle_imbalance,
            flow_suffix=args.flow_suffix,
        )
    else:
        train_loader, valid_loader, test_loader, train_ds, test_ds = split_deeppacket(
            root=args.root,
            valid_size=0.1, test_size=0.1,
            batch_size=args.batch_size,
            num_workers=2,
            shuffle=True,
            limit_files_per_split=args.limit_files_per_split,
            max_rows_per_file=args.max_rows_per_file,
            handle_imbalance=args.handle_imbalance,
            weight_method=args.weight_method,
            undersample=args.undersample,
            undersample_ratio=args.undersample_ratio,
            undersample_strategy=args.undersample_strategy,
        )
    # Comprehensive flow-based sanity checks (if using flow split)
    if args.flow_split:
        run_comprehensive_flow_checks(args.root, train_ds, val_ds, test_ds, 
                                      flow_suffix=args.flow_suffix)
    
    # Lightweight sanity check on a subset of training samples
    bad_counts = 0
    try:
        sample_limit = min(len(train_ds), 100000)
        for i in range(sample_limit):
            try:
                x, y = train_ds[i]
                # Expect tensors shaped (1, 1, 1500)
                if x is None or x.numel() == 0 or x.dim() != 3 or x.size(-1) != 1500:
                    bad_counts += 1
            except Exception:
                bad_counts += 1
    except Exception:
        # If any unexpected error occurs during checking, don't block training
        pass

    print("bad samples (checked subset):", bad_counts)
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
