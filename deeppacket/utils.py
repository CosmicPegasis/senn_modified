"""Utility functions for DeepPacket training and evaluation."""

import os
import shutil
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        """Update meter with new value."""
        v = float(val)
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0


def save_checkpoint(state: dict, is_best: bool, outpath: str) -> None:
    """Save model checkpoint."""
    os.makedirs(outpath, exist_ok=True)
    filename = os.path.join(outpath, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outpath, 'model_best.pth.tar'))


def binary_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute binary classification accuracy from logits."""
    probs = torch.sigmoid(logits.detach())
    preds = (probs >= 0.5).long().view(-1)
    return (preds == targets.view(-1)).float().mean().item() * 100.0


def multiclass_precision_at_k(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute multiclass precision@k."""
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

