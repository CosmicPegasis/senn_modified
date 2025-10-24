import sys, os
import numpy as np
import pickle
import argparse
import operator
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data.dataloader as dataloader
import torchvision.models as tv_models
from typing import Optional

import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Union
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

import multiprocessing

# TODO:
# - [ ] Transformer based conceptizers/parameterizer
#  - [ ] Knowledge/Concept Discovery in ML using
# - [ ] Closed Loop Training

def CL_loss(theta, n_class):
    """ Cross lipshitc loss from https://arxiv.org/pdf/1705.08475.pdf.
        Gradient based.
    """

    total = 0
    for i in range(n_class):
        for j in range(n_class):
            total += (grad[i] - grad[j]).norm()**2

    return total/(n_class)

def compute_jacobian_sum(x, fx):
    """ Much faster than compute_jacobian, but only correct for norm L1 stuff
    since it returns sum of gradients """
    n = x.size(-1)
    b = x.size(0)
    c = fx.size(-1)
    m = fx.size(-2)
    grad = torch.ones(b, m, c)
    if x.is_cuda:
        grad  = grad.cuda()
    g = torch.autograd.grad(outputs=fx, inputs = x, grad_outputs = grad, create_graph=True, only_inputs=True)[0]#, retain_graph = True)[0] -> not sure this should be true or not. Not needed! Defaults to value of create_graph
    return g

def compute_jacobian(x, fx):
    # Ideas from https://discuss.pytorch.org/t/clarification-using-backward-on-non-scalars/1059/2



    b = x.size(0)
    n = x.size(-1)
    # if fx.dim() > 1:
    m = fx.size(-1)
    # else:
    #     #e.g. fx = theta and task is binary classifiction, fx is a vector
    #     m = 1
    #print(fx.size())
    #print(b,n,m)
    J = []
    for i in range(m):
        #print(i)
        grad = torch.zeros(b, m)
        grad[:,i] = 1
        if x.is_cuda:
            grad  = grad.cuda()
        #print(grad.size(), fx.size(), x.size())
        #pdb.set_trace()
        g = torch.autograd.grad(outputs=fx, inputs = x, grad_outputs = grad, create_graph=True, only_inputs=True)[0] #, retain_graph = True)[0]
        J.append(g.view(x.size(0),-1).unsqueeze(-1))
    #print(J[0].size())
    J = torch.cat(J,2)
    return J


#===============================================================================
#==================================   TRAINERS    ==============================
#===============================================================================


def save_checkpoint(state, is_best, outpath):
    # script_dir = dirname(dirname(realpath(__file__)))
    if outpath == None:
        outpath = os.path.join(script_dir, 'checkpoints')

    #outdir = os.path.join(outpath, '{}_LR{}_Lambda{}'.format(state['theta_reg_type'],state['lr'],state['theta_reg_lambda']))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = os.path.join(outpath, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(outpath,'model_best.pth.tar'))


class image_parametrizer(nn.Module):
    """
    Simple CNN-based parametrizer for image inputs.

    Backwards-compatible constructor: ImageParametrizer(din, nconcept, dout, nchannel=1, only_positive=False, pool_output_size=(4,4))
    - din is accepted for compatibility but NOT used to compute spatial dims anymore.
    - Use adaptive pooling to get a fixed final spatial size (pool_output_size).
    Returns: Th of shape (B, nconcept, dout)
    """
    def __init__(self, din, nconcept, dout, nchannel=1, only_positive=False,
                 pool_output_size=(4, 4), dropout_p: float = 0.5):
        super().__init__()
        # preserve some of the original attributes for compatibility
        self.din = din
        self.nconcept = int(nconcept)
        self.dout = int(dout)
        self.nchannel = int(nchannel)
        self.only_positive = bool(only_positive)
        self.pool_output_size = pool_output_size

        # small conv net (same topology as original, but robust)
        self.conv1 = nn.Conv2d(self.nchannel, 10, kernel_size=5, padding=2)  # preserve receptive field but keep shape easy
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)

        # dropout modules
        self.conv2_drop = nn.Dropout2d(p=dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

        # adaptive pooling yields deterministic spatial size regardless of input HxW
        self.adaptive_pool = nn.AdaptiveAvgPool2d(self.pool_output_size)

        # compute fc input dim from pool_output_size and number of channels after conv2
        pooled_H, pooled_W = self.pool_output_size
        fc_in = 20 * pooled_H * pooled_W
        self.fc1 = nn.Linear(fc_in, self.nconcept * self.dout)

        # activation for positivity
        # Softplus gives positive outputs without sharp saturation; switch to torch.sigmoid if you want [0,1]
        if self.only_positive:
            self.pos_activation = nn.Softplus()
        else:
            self.pos_activation = None

        # initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        # conv init: kaiming for ReLU
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)
        # linear init
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: Th of shape (B, nconcept, dout)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected image tensor of shape (B,C,H,W), got {tuple(x.shape)}")

        # conv -> relu -> maxpool / conv drop pattern
        p = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        p = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(p)), kernel_size=2))

        # adaptive pool to deterministic spatial dims
        p = self.adaptive_pool(p)  # (B, 20, pooled_H, pooled_W)

        B = p.size(0)
        p = p.view(B, -1)  # flatten spatial dims and channels
        p = self.fc1(p)    # (B, nconcept * dout)
        p = self.dropout(p)
        out = p.view(B, self.nconcept, self.dout)  # (B, k, dout)

        # apply positivity / sign transform
        if self.only_positive:
            out = self.pos_activation(out)
        else:
            out = torch.tanh(out)

        return out


class vgg_parametrizer(nn.Module):
    """
    Parametrizer wrapper around a VGG-style backbone (or VGG_CIFAR if available).

    Args:
      din: kept for API compatibility (not used for spatial math)
      nconcept: number of concepts (k)
      dout: per-concept output dim
      arch: arch string, e.g. 'vgg11', 'vgg16', or a custom 'vgg_cifar' style handled by VGG_CIFAR
      nchannel: input channels
      only_positive: if True use Softplus to make outputs positive
      pretrained: try to load torchvision pretrained weights when using torchvision backbones
    """
    def __init__(self,
                 din: int,
                 nconcept: int,
                 dout: int,
                 arch: str = 'vgg16',
                 nchannel: int = 3,
                 only_positive: bool = False,
                 pretrained: bool = False):
        super().__init__()
        self.nconcept = int(nconcept)
        self.dout = int(dout)
        self.din = din
        self.arch = arch
        self.nchannel = int(nchannel)
        self.only_positive = bool(only_positive)
        self.pretrained = bool(pretrained)

        num_classes = self.nconcept * self.dout

        # 1) Try to use VGG_CIFAR if available (your repo probably defines this)
        net = None
        try:
            # If user provided a VGG_CIFAR class in the notebook, prefer it (handles CIFAR sizes)
            VGG_CIFAR  # type: ignore[name-defined]
            net = globals().get('VGG_CIFAR')(arch, num_classes=num_classes)
        except Exception:
            # 2) Fallback to torchvision VGGs (may expect 224x224 inputs).
            model_cls = getattr(tv_models, arch, None)
            if model_cls is None:
                # try commonly used vgg names mapping
                # raise helpful error
                raise ValueError(f"Unknown arch '{arch}' and no VGG_CIFAR found. "
                                 "Provide a valid arch (e.g. 'vgg11','vgg16') or define VGG_CIFAR.")
            # attempt constructor across torchvision versions
            try:
                net = model_cls(num_classes=num_classes)
            except TypeError:
                try:
                    net = model_cls(pretrained=self.pretrained, num_classes=num_classes)
                except TypeError:
                    try:
                        net = model_cls(weights=None if not self.pretrained else "DEFAULT", num_classes=num_classes)
                    except Exception as e:
                        raise RuntimeError(f"Failed to instantiate torchvision model '{arch}': {e}")

        self.net = net

        # If backbone expects 3 channels but user passed different nchannel, replace first conv
        if self.nchannel != 3:
            self._replace_first_conv(self.net, in_channels=self.nchannel)

        # dropout module
        self.dropout = nn.Dropout(p=0.5)

        # choose positive activation
        if self.only_positive:
            self.pos_activation = nn.Softplus()
        else:
            self.pos_activation = None

    def _replace_first_conv(self, net: nn.Module, in_channels: int):
        """
        Replace common first conv for VGG-like nets. For torchvision VGG the first conv is usually at net.features[0].
        Copy/replicate weights when possible to avoid losing pretrained info.
        """
        # find candidate conv layer
        replaced = False
        if hasattr(net, 'features') and isinstance(net.features, (nn.Sequential, list)):
            first = net.features[0]
            if isinstance(first, nn.Conv2d):
                old = first
                new = nn.Conv2d(
                    in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=(old.bias is not None)
                )
                with torch.no_grad():
                    old_w = old.weight  # (out_ch, in_ch_old, kh, kw)
                    if old_w.shape[1] == in_channels:
                        new.weight.copy_(old_w)
                        if old.bias is not None:
                            new.bias.copy_(old.bias)
                    else:
                        # strategy: if old had 3 channels, copy first 3 or replicate mean for extra channels
                        if old_w.shape[1] == 3:
                            # fill new weight: copy old for first min(3,in_ch) channels then fill rest with mean
                            oc, ic_old, kh, kw = old_w.shape
                            new_w = torch.zeros((oc, in_channels, kh, kw), dtype=old_w.dtype, device=old_w.device)
                            copy_ch = min(3, in_channels)
                            new_w[:, :copy_ch, :, :].copy_(old_w[:, :copy_ch, :, :])
                            if in_channels > 3:
                                mean_ch = old_w.mean(dim=1, keepdim=True)  # (oc,1,kh,kw)
                                new_w[:, 3:, :, :].copy_(mean_ch.repeat(1, in_channels - 3, 1, 1))
                            new.weight.copy_(new_w)
                            if old.bias is not None:
                                new.bias.copy_(old.bias)
                        else:
                            # fallback: average over old input channels and repeat
                            mean_ch = old_w.mean(dim=1, keepdim=True)
                            new_w = mean_ch.repeat(1, in_channels, 1, 1)
                            new.weight.copy_(new_w)
                            if old.bias is not None:
                                new.bias.copy_(old.bias)

                # set into model
                net.features[0] = new
                replaced = True

        if not replaced:
            # last resort: scan top-level modules for first Conv2d
            for name, module in net.named_children():
                if isinstance(module, nn.Conv2d):
                    old = module
                    new = nn.Conv2d(
                        in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=(old.bias is not None)
                    )
                    with torch.no_grad():
                        # try to copy weights with same heuristics as above
                        old_w = old.weight
                        if old_w.shape[1] == in_channels:
                            new.weight.copy_(old_w)
                            if old.bias is not None:
                                new.bias.copy_(old.bias)
                        else:
                            mean_ch = old_w.mean(dim=1, keepdim=True)
                            new.weight.copy_(mean_ch.repeat(1, in_channels, 1, 1))
                            if old.bias is not None:
                                new.bias.copy_(old.bias)
                    setattr(net, name, new)
                    replaced = True
                    break

        if not replaced:
            import warnings
            warnings.warn(f"Could not automatically replace first Conv2d for model '{self.arch}'. "
                          "You may need to patch the network manually to accept nchannel != 3.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: Th shaped (B, nconcept, dout)
        """
        p = self.net(x)  # expects to return (B, nconcept * dout)
        B = p.size(0)
        p = self.dropout(p)
        out = p.view(B, self.nconcept, self.dout)
        if self.pos_activation is not None:
            out = self.pos_activation(out)
        else:
            out = torch.tanh(out)
        return out



# --- helper meters (kept simple) ----------------
class AverageMeter:
    """Computes and stores the average and current value"""
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

# --- accuracy helpers --------------------------------
def binary_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return percent accuracy (0-100) for binary classification using logits"""
    probs = torch.sigmoid(logits.detach())
    preds = (probs >= 0.5).long()
    return (preds.view(-1) == targets.view(-1)).float().mean().item() * 100.0

def multiclass_precision_at_k(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Return precision@k percentages (each a single-element tensor)"""
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

# ====================================================================
# =======================  ClassificationTrainer  =====================
# ====================================================================
class ClassificationTrainer():
    def __init__(self, model: nn.Module, args):
        """
        Modernized ClassificationTrainer.
        - Expects model to be an nn.Module and moved to device via model.to(device)
        - args must provide: cuda (bool), nclasses (int), objective (str), opt (str), lr, weight_decay, print_freq,
          h_type, h_sparsity, save_dir, theta_reg_lambda (optional)
        """
        self.model = model
        self.args = args
        self.device = torch.device('cuda' if (hasattr(args, 'cuda') and args.cuda and torch.cuda.is_available()) else 'cpu')
        self.model.to(self.device)

        self.nclasses = int(args.nclasses)

        # Prediction criterion: prefer nn modules and assume model outputs logits (not probabilities).
        if self.nclasses <= 2:
            # binary classification: expect logits -> use BCEWithLogitsLoss
            self.prediction_criterion = nn.BCEWithLogitsLoss()
        else:
            # multiclass (C >= 2): expect raw logits -> CrossEntropyLoss
            # CrossEntropyLoss expects targets as long (class indices)
            self.prediction_criterion = nn.CrossEntropyLoss()

        # Concept learning (h) related
        if getattr(args, 'h_type', 'input') != 'input':
            self.learning_h = True
            self.h_reconst_criterion = nn.MSELoss()
            self.h_sparsity = getattr(args, 'h_sparsity', -1)
        else:
            self.learning_h = False
            self.h_reconst_criterion = None
            self.h_sparsity = -1

        # bookkeeping
        self.loss_history = []
        self.print_freq = int(getattr(args, 'print_freq', 100))
        self.reset_lstm = bool(getattr(model, 'reset_lstm', False))

        # Setup optimizer
        optim_betas = (0.9, 0.999)
        opt_name = getattr(args, 'opt', 'adam').lower()
        if opt_name == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=getattr(args, 'lr', 1e-3), betas=optim_betas)
        elif opt_name == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=getattr(args, 'lr', 1e-3))
        elif opt_name == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=getattr(args, 'lr', 1e-2),
                                       weight_decay=getattr(args, 'weight_decay', 0.0), momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer '{opt_name}'")

        # move criterion to device if it has parameters (usually not required)
        # self.prediction_criterion.to(self.device)  # usually unnecessary

    # Public API -------------------------------------------------------
    def train(self, train_loader, val_loader=None, epochs=10, save_path: Optional[str]=None):
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
                    'epoch': epoch + 1,
                    'lr': getattr(self.args, 'lr', None),
                    'theta_reg_lambda': getattr(self.args, 'theta_reg_lambda', None),
                    'theta_reg_type': getattr(self.args, 'theta_reg_type', None),
                    'state_dict': self.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': self.optimizer.state_dict(),
                    'model': self.model
                }
                save_checkpoint(state, is_best, save_path)

        print('Training done')

    def train_batch(self, inputs, targets):
        """Must be implemented by subclasses"""
        raise NotImplementedError('ClassificationTrainer subclasses must implement train_batch(inputs, targets)')

    # Concept learning loss helper
    def concept_learning_loss(self, inputs: torch.Tensor, all_losses: dict):
        """Use model.recons and model.h_norm_l1 if available. inputs are raw input tensors on device."""
        # assume model.recons is present and same shape as inputs
        if not hasattr(self.model, 'recons') or self.model.recons is None:
            return torch.tensor(0.0, device=self.device)

        recons_loss = self.h_reconst_criterion(self.model.recons, inputs.detach())
        all_losses['reconstruction'] = recons_loss.item()
        total = recons_loss
        if self.h_sparsity is not None and self.h_sparsity != -1 and hasattr(self.model, 'h_norm_l1') and self.model.h_norm_l1 is not None:
            sparsity_loss = self.model.h_norm_l1 * float(self.h_sparsity)
            all_losses['h_sparsity'] = sparsity_loss.item()
            total = total + sparsity_loss
        return total

    # Epoch / validation / evaluate -----------------------------------
    def train_epoch(self, epoch, train_loader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_meter = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        end = time.time()

        for i, (inputs, targets) in enumerate(train_loader):
            data_time.update(time.time() - end)

            # move to correct device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # reset LSTM hidden state if needed (ensure on same device)
            if self.reset_lstm and hasattr(self.model.parametrizer, 'init_hidden'):
                # detach hidden state if it exists or re-init
                self.model.parametrizer.hidden = self.model.parametrizer.init_hidden()
                # ensure device correctness
                try:
                    for k, v in self.model.parametrizer.hidden.items():
                        self.model.parametrizer.hidden[k] = v.to(self.device)
                except Exception:
                    try:
                        self.model.parametrizer.hidden = self.model.parametrizer.hidden.to(self.device)
                    except Exception:
                        pass

            outputs, loss, loss_dict = self.train_batch(inputs, targets)
            loss_dict['iter'] = i + (len(train_loader) * epoch)
            self.loss_history.append(loss_dict)

            # compute accuracies (outputs should be raw logits)
            if self.nclasses <= 2:
                acc1 = binary_accuracy_from_logits(outputs, targets)
                prec1, prec5 = [acc1], [100.0]
            else:
                # outputs must be (B, C), targets long
                precs = multiclass_precision_at_k(outputs.detach().cpu(), targets.detach().cpu(), topk=(1, min(5, self.nclasses)))
                prec1, = [float(precs[0].item())]
                if self.nclasses >= 5:
                    prec5 = [float(precs[1].item())]
                else:
                    prec5 = [float(precs[0].item())]

            losses_meter.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]  '
                      f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})  '
                      f'Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})  '
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
                      f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

    def validate(self, val_loader, fold: Optional[str] = None):
        batch_time = AverageMeter()
        losses_meter = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        end = time.time()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = self.model(inputs)
                loss = self.prediction_criterion(output, targets if self.nclasses>2 else targets.float().unsqueeze(1))
                # accuracy
                if self.nclasses <= 2:
                    prec1 = binary_accuracy_from_logits(output, targets)
                    prec5 = [100.0]
                else:
                    precs = multiclass_precision_at_k(output.detach().cpu(), targets.detach().cpu(), topk=(1, min(5, self.nclasses)))
                    prec1 = float(precs[0].item())
                    prec5 = [float(precs[1].item())] if self.nclasses >=5 else [prec1]

                losses_meter.update(loss.item(), inputs.size(0))
                top1.update(prec1, inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    print(f'Test: [{i}/{len(val_loader)}]\t Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          f'Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})\t'
                          f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

        print(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
        return top1.avg

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                if self.reset_lstm and hasattr(self.model.parametrizer, 'init_hidden'):
                    self.model.parametrizer.hidden = self.model.parametrizer.init_hidden()

                output = self.model(data)
                loss = self.prediction_criterion(output, targets if self.nclasses>2 else targets.float().unsqueeze(1))
                test_loss += loss.item()

                if self.nclasses == 2:
                    probs = torch.sigmoid(output)
                    pred = (probs >= 0.5).long().view(-1)
                else:
                    pred = output.argmax(dim=1).view(-1)

                correct += pred.eq(targets.view(-1)).sum().item()

        test_loss /= len(test_loader)
        acc = 100.0 * correct / len(test_loader.dataset)
        print(f'\nEvaluation: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.0f}%)\n')
        return acc

# ====================================================================
# =======================  GradPenaltyTrainer  =========================
# ====================================================================
class GradPenaltyTrainer(ClassificationTrainer):
    """
    Trainer that adds gradient-based penalty to the prediction loss.

    penalty_type:
      1 -> || dF/dx - Theta_projected || (cheapest; assumes mapping from theta->input space or h==x)
      2 -> || dTheta/dx || / || dH/dx ||
      3 -> || dF/dx - J_h(x) @ theta(x) ||  (SENN canonical)
    """
    def __init__(self, model: nn.Module, args, typ: int = 3):
        super().__init__(model, args)
        self.lambd = float(getattr(args, 'theta_reg_lambda', 1e-6))
        self.reconst_criterion = nn.MSELoss()
        self.penalty_type = int(typ)
        self.norm = 2
        self.eps = 1e-8

    def train_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Modernized train_batch:
         - inputs, targets are tensors already on device
         - returns (outputs, loss, all_losses_dict)
        """
        self.model.train()
        device = self.device

        # prepare
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = inputs.clone().detach().requires_grad_(True)  # we require grad w.r.t inputs for jacobians
        self.optimizer.zero_grad()

        # forward prediction (this may populate model.thetas if model does so)
        pred = self.model(inputs)  # raw logits expected
        # compute prediction loss: for binary ensure shape matches BCEWithLogitsLoss input
        if self.nclasses <= 2:
            pred_loss = self.prediction_criterion(pred.view(-1, 1), targets.float().view(-1, 1))
        else:
            pred_loss = self.prediction_criterion(pred, targets.long())

        all_losses = {'prediction': pred_loss.item()}

        # concept learning loss if model is learning H
        if self.learning_h:
            h_loss = self.concept_learning_loss(inputs, all_losses)
            loss_base = pred_loss + h_loss
        else:
            loss_base = pred_loss

        # compute grad penalty according to type
        grad_penalty = torch.tensor(0.0, device=device)

        # Helper: get live thetas (do NOT use detached copies)
        # Prefer computing parametrizer(inputs) directly to avoid relying on possibly stale model.thetas
        thetas_live = None
        try:
            thetas_live = self.model.parametrizer(inputs)
            if thetas_live.dim() == 2:
                thetas_live = thetas_live.unsqueeze(-1)
        except Exception:
            # fallback to model.thetas if parametrizer can't be called standalone
            thetas_live = getattr(self.model, 'thetas', None)

        if self.penalty_type == 1:
            # Type 1: compute per-sample gradient of chosen scalar output and compare to theta (requires mapping)
            # Choose scalar per sample: predicted class logit (multiclass) or scalar logit (binary)
            if pred.dim() == 2 and pred.size(1) > 1:
                # pick predicted class logit per-sample
                pred_vals, _ = torch.max(pred, dim=1)
                scalar_pred = pred_vals
            else:
                scalar_pred = pred.view(pred.size(0), -1).squeeze(1) if pred.dim() > 1 else pred.view(-1)

            grad_outputs = torch.ones_like(scalar_pred, device=device)
            dF = torch.autograd.grad(outputs=scalar_pred, inputs=inputs,
                                     grad_outputs=grad_outputs, create_graph=True, only_inputs=True)[0]  # shape = inputs.shape

            # Flatten both sides
            dF_flat = dF.reshape(dF.size(0), -1)  # (B, D)
            if thetas_live is None:
                # cannot compute theta -> skip penalty
                grad_penalty = torch.tensor(0.0, device=device)
            else:
                theta_flat = thetas_live.reshape(thetas_live.size(0), -1)
                # If shapes mismatch, reduce the larger to smaller by mean (simple heuristic).
                if dF_flat.size(1) != theta_flat.size(1):
                    # project theta -> dF dimension by simple linear up/down sampling (nearest) OR averaging
                    # Here we average/repeat to nearest length (cheap fallback)
                    if theta_flat.size(1) > dF_flat.size(1):
                        theta_flat = theta_flat[:, :dF_flat.size(1)]
                    else:
                        # repeat / tile
                        reps = int(np.ceil(dF_flat.size(1) / theta_flat.size(1)))
                        theta_flat = theta_flat.repeat(1, reps)[:, :dF_flat.size(1)]
                grad_penalty = (dF_flat - theta_flat).norm(p=self.norm)

        elif self.penalty_type == 2:
            # Type 2: ratio || dtheta/dx || / || dh/dx ||
            # J_theta: shape (B, D, nconcept) or (B, D, nconcept, nclass) depending on implementation
            dTh = self.compute_parametrizer_jacobian(inputs)  # expected shape (B, D, nconcept) or similar
            if self.learning_h:
                dH = self.compute_conceptizer_jacobian(inputs)  # (B, D, nconcept)
                num = dTh.view(dTh.size(0), -1).norm(p=self.norm)
                den = dH.view(dH.size(0), -1).norm(p=self.norm) + self.eps
                grad_penalty = num / den
            else:
                # treat dH as identity; denominator is sqrt of input dim as in older code
                denom = (inputs.numel() / inputs.size(0)) ** 0.5
                grad_penalty = dTh.view(dTh.size(0), -1).norm(p=self.norm) / (denom + self.eps)
        else:
            # Type 3: canonical SENN penalty: || dF/dx - J_h(x) @ theta ||
            # Compute dF: per-sample gradient of scalar output (choose predicted class logit)
            if pred.dim() == 2 and pred.size(1) > 1:
                pred_vals, _ = torch.max(pred, dim=1)
                scalar_pred = pred_vals
            else:
                scalar_pred = pred.view(pred.size(0), -1).squeeze(1) if pred.dim() > 1 else pred.view(-1)

            grad_outputs = torch.ones_like(scalar_pred, device=device)
            dF = torch.autograd.grad(outputs=scalar_pred, inputs=inputs,
                                     grad_outputs=grad_outputs, create_graph=True, only_inputs=True)[0]  # (B, ..., input_dims)

            # compute J_h (conceptizer jacobian). This uses provided helper compute_conceptizer_jacobian
            if self.learning_h:
                dH = self.compute_conceptizer_jacobian(inputs)  # expected (B, D, C)
                # thetas_live shape (B, C, cdim) — we need to reduce to (B, C) if cdim==1 or choose correct class dimension
                thetas_use = thetas_live
                if thetas_use is None:
                    # fallback to model.thetas
                    thetas_use = getattr(self.model, 'thetas', None)
                if thetas_use is None:
                    grad_penalty = torch.tensor(0.0, device=device)
                else:
                    # reduce theta last dim if necessary (take mean across concept-dim if >1)
                    if thetas_use.dim() == 3 and thetas_use.size(-1) > 1:
                        theta_reduced = thetas_use.mean(dim=-1)  # (B, C)
                    else:
                        theta_reduced = thetas_use.squeeze(-1)  # (B, C)
                    # compute product J_h @ theta_reduced -> (B, D)
                    theta_vec = theta_reduced.unsqueeze(-1)  # (B, C, 1)
                    # ensure dH shape (B, D, C)
                    prod = torch.bmm(dH, theta_vec).squeeze(-1)  # (B, D)
                    dF_flat = dF.reshape(dF.size(0), -1)
                    grad_penalty = (prod - dF_flat).norm(p=self.norm)
            else:
                # if conceptizer not learned, assume h(x)=x identity; model.thetas must be in input space or shaped accordingly
                thetas_use = thetas_live if thetas_live is not None else getattr(self.model, 'thetas', None)
                if thetas_use is None:
                    grad_penalty = torch.tensor(0.0, device=device)
                else:
                    # pad thetas to match dF channels if needed
                    dF_flat = dF.reshape(dF.size(0), -1)
                    theta_flat = thetas_use.reshape(thetas_use.size(0), -1)
                    if theta_flat.size(1) != dF_flat.size(1):
                        if theta_flat.size(1) > dF_flat.size(1):
                            theta_flat = theta_flat[:, :dF_flat.size(1)]
                        else:
                            reps = int(np.ceil(dF_flat.size(1) / theta_flat.size(1)))
                            theta_flat = theta_flat.repeat(1, reps)[:, :dF_flat.size(1)]
                    grad_penalty = (theta_flat - dF_flat).norm(p=self.norm)

        # record and backpropagate
        all_losses['grad_penalty'] = float(grad_penalty.item()) if isinstance(grad_penalty, torch.Tensor) else float(grad_penalty)

        total_loss = loss_base + (self.lambd * grad_penalty)
        total_loss.backward()
        self.optimizer.step()

        # IMPORTANT: clear any live graphful references held by the model to avoid memory retention
        if hasattr(self.model, 'clear_runtime_state'):
            try:
                self.model.clear_runtime_state()
            except Exception:
                # best-effort: remove common attributes
                for name in ('thetas', 'concepts', 'recons', 'h_norm_l1'):
                    if hasattr(self.model, name):
                        try:
                            setattr(self.model, name, None)
                        except Exception:
                            pass

        return pred.detach(), total_loss.detach(), all_losses

    # Jacobian helpers call wrappers (use the provided compute_jacobian functions)
    def compute_parametrizer_jacobian(self, x: torch.Tensor):
        """Return J_theta: expected shape [B, D, nconcept] or [B, D, nconcept, nclass] depending on compute_jacobian"""
        thetas = getattr(self.model, 'thetas', None)
        if thetas is None:
            # try to compute live thetas (local call)
            thetas = self.model.parametrizer(x)
        nclass = self.nclasses
        if self.norm == 1:
            JTh = compute_jacobian_sum(x, thetas.squeeze()).unsqueeze(-1)
        elif nclass == 1:
            JTh = compute_jacobian(x, thetas[:, :, 0])
        else:
            JTh_list = []
            for i in range(nclass):
                JTh_list.append(compute_jacobian(x, thetas[:, :, i]).unsqueeze(-1))
            JTh = torch.cat(JTh_list, dim=3)
        return JTh

    def compute_conceptizer_jacobian(self, x: torch.Tensor):
        """Return J_h of shape [B, D, C] where C is number of concepts"""
        h = getattr(self.model, 'concepts', None)
        if h is None:
            # compute concepts live
            if self.learning_h:
                out = self.model.conceptizer(x)
                h = out[0] if isinstance(out, (tuple, list)) else out
            else:
                # if conceptizer is fixed, may just be identity over inputs
                raise RuntimeError("Concepts not found and model not learning H")
        # h is expected shape (B, C, ...) -> we squeeze to (B, C) if needed
        h_squeezed = h.squeeze()
        Jh = compute_jacobian(x, h_squeezed)
        return Jh

    def compute_fullmodel_gradient(self, x: torch.Tensor, ypred: torch.Tensor):
        grad = torch.autograd.grad(ypred, x,
                                   grad_outputs=torch.ones_like(ypred, device=self.device),
                                   create_graph=True, only_inputs=True)[0]
        return grad


def generate_dir_names(dataset, args, make = True):
    if args.h_type == 'input':
        suffix = '{}_H{}_Th{}_Reg{:0.0e}_LR{}'.format(
                    args.theta_reg_type,
                    args.h_type,
                    args.theta_arch,
                    args.theta_reg_lambda,
                    args.lr,
                    )
    else:
        suffix = '{}_H{}_Th{}_Cpts{}_Reg{:0.0e}_Sp{}_LR{}'.format(
                    args.theta_reg_type,
                    args.h_type,
                    args.theta_arch,
                    args.nconcepts,
                    args.theta_reg_lambda,
                    args.h_sparsity,
                    args.lr,
                    )

    model_path     = os.path.join(args.model_path, dataset, suffix)
    log_path       = os.path.join(args.log_path, dataset, suffix)
    results_path   = os.path.join(args.results_path, dataset, suffix)

    if make:
        for p in [model_path, results_path]: #, log_path,
            if not os.path.exists(p):
                os.makedirs(p)

    return model_path, log_path, results_path


class input_conceptizer(nn.Module):
    """
    Conceptizer that treats input features (pixels / features) as concepts.

    Accepts:
      - images: (B, C, H, W) -> returns (B, C*H*W [+1], 1) if add_bias
      - vectors: (B, D) -> returns (B, D [+1], 1)

    If add_bias=True, a 1-valued bias concept is appended at dim=1.
    """
    def __init__(self, add_bias: bool = True):
        super().__init__()
        self.add_bias = add_bias
        self.learnable = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # (B, C, H, W)
            B, C, H, W = x.shape
            out = x.reshape(B, C * H * W, 1)   # safe flatten
        elif x.dim() == 3:
            # Keep general 3D: (B, L, F) -> treat L*F as "flattened" features
            B = x.size(0)
            out = x.reshape(B, -1, 1)
        elif x.dim() == 2:  # (B, D)
            out = x.unsqueeze(-1)  # (B, D, 1)
        else:
            raise ValueError(f"Unsupported input tensor with ndim={x.dim()}")

        if self.add_bias:
            # create bias of ones on same device/dtype as out
            bias = out.new_ones((out.size(0), 1, out.size(2)))  # (B,1,1)
            out = torch.cat([out, bias], dim=1)  # append along concept dim

        return out


class additive_scalar_aggregator(nn.Module):
    """
    Linear additive aggregator for SENN-style models.

    Given concepts H and parametrizer outputs Theta (Th), computes
      logits(b, c) = sum_{i=1..k} < h_i, theta_i[:, c] >
    where <.,.> is the inner product over concept-dimension (cdim).

    Supports:
      - H: shape (B, k, cdim)  (or (B, k, 1) which will be squeezed)
      - Th:
          * shape (B, k, nclasses)   (only valid when cdim == 1)
          * shape (B, k, cdim, nclasses)

    By default returns raw logits (unsquashed). Set return_probs=True or return_log_probs=True
    to obtain probabilities or log-probabilities respectively.

    NOTE: returning logits is the recommended default so losses like BCEWithLogitsLoss / CrossEntropyLoss
    can be used directly.
    """
    def __init__(self, cdim: int = 1, nclasses: int = 1):
        super().__init__()
        self.cdim = int(cdim)
        self.nclasses = int(nclasses)
        # treat nclasses==1 as binary scalar-output case (one logit per sample)
        self.binary = (self.nclasses == 1)

    def forward(self,
                H: torch.Tensor,
                Th: torch.Tensor,
                return_logits: bool = True,
                return_probs: bool = False,
                return_log_probs: bool = False) -> torch.Tensor:
        """
        Args:
            H:    (B, k, cdim) or (B, k, 1)
            Th:   (B, k, nclasses)  if cdim==1
                  (B, k, cdim, nclasses) if cdim>1
            return_logits: if True (default) return raw logits
            return_probs: if True return probabilities (softmax or sigmoid)
            return_log_probs: if True return log-softmax (multiclass) or log-sigmoid (binary)
        Returns:
            Tensor of shape (B, nclasses) (or (B,1) for binary)
        """
        # Normalize H shape -> (B, k, cdim)
        if H.dim() == 3:
            B, k, d = H.shape
            H_proc = H
        elif H.dim() == 4 and H.size(-1) == 1:
            # sometimes H arrives as (B, k, cdim, 1) — collapse trailing 1
            H_proc = H.squeeze(-1)
            B, k, d = H_proc.shape
        else:
            # allow (B, k, 1) as common case
            if H.dim() == 3:
                B, k, d = H.shape
                H_proc = H
            elif H.dim() == 2:
                raise ValueError("H must be at least 3-D: (B, k, cdim) -- got shape {}".format(tuple(H.shape)))
            else:
                H_proc = H.view(H.size(0), H.size(1), -1)
                B, k, d = H_proc.shape

        cdim = d

        # Handle Theta shapes and compute logits via einsum for clarity & generality
        if Th.dim() == 3:
            # Th: (B, k, nclasses) -> valid only when cdim == 1
            if cdim != 1:
                raise ValueError("Th has shape (B,k,nclasses) but H has cdim={} > 1. "
                                 "Use Th shape (B,k,cdim,nclasses) when cdim>1.".format(cdim))
            # einsum: sum over k: 'b k d, b k c -> b c' where d==1
            logits = torch.einsum('b k d, b k c -> b c', H_proc, Th)
        elif Th.dim() == 4:
            # Th: (B, k, cdim, nclasses)
            if Th.size(2) != cdim:
                raise ValueError("Mismatch between H.cdim={} and Th.shape[2]={}".format(cdim, Th.size(2)))
            logits = torch.einsum('b k d, b k d c -> b c', H_proc, Th)
        else:
            raise ValueError("Unsupported Th shape {}; expected 3 or 4 dims".format(tuple(Th.shape)))

        # logits shape: (B, nclasses)
        # For binary case where nclasses==1 we keep shape (B,1)
        if return_logits and not (return_probs or return_log_probs):
            return logits

        # convert to probs / log-probs if requested
        if return_log_probs:
            if self.binary:
                # log-sigmoid for binary (log of probability of positive class)
                # keep shape (B,1)
                return torch.log(torch.sigmoid(logits) + 1e-12)
            else:
                return F.log_softmax(logits, dim=1)
        if return_probs:
            if self.binary:
                return torch.sigmoid(logits)
            else:
                return F.softmax(logits, dim=1)

        # default fallback (if user passed contradictory flags) return logits
        return logits



class GSENN(nn.Module):
    """Wrapper for GSENN with optional H-learning.

    Notes:
      - forward(x) returns logits (not probabilities).
      - During training (self.training==True) the returned/assigned attributes
        self.concepts and self.thetas are kept graphful so trainers can compute
        jacobians. After forward() in eval mode those attributes are detached
        to avoid retaining computation graphs.
    """
    def __init__(self, conceptizer: nn.Module, parametrizer: nn.Module, aggregator: nn.Module, debug: bool = False):
        super().__init__()
        self.conceptizer = conceptizer
        self.parametrizer = parametrizer
        self.aggregator = aggregator
        # conceptizer may declare whether it's learnable
        self.learning_H = bool(getattr(conceptizer, 'learnable', False))
        # LSTM detection kept for compatibility with older code
        self.reset_lstm = hasattr(conceptizer, 'lstm') or hasattr(parametrizer, 'lstm')
        # runtime attrs
        self.thetas = None
        self.concepts = None
        self.recons = None
        self.h_norm_l1 = None
        self.debug = bool(debug)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Expects x to be a torch.Tensor on the correct device.
        Returns logits tensor of shape (B, nclasses) (or (B,1) for binary).
        """
        if self.debug:
            print("GSENN.forward input shape:", tuple(x.shape))

        # Compute concepts (and optional reconstruction)
        if self.learning_H:
            # expected: conceptizer(x) -> (h_x, x_tilde)
            h_res = self.conceptizer(x)
            if isinstance(h_res, (tuple, list)) and len(h_res) >= 2:
                h_x, x_tilde = h_res[0], h_res[1]
                self.recons = x_tilde if self.training else x_tilde.detach()
            else:
                # conceptizer claims to be learnable but returned only h(x)
                h_x = h_res
                self.recons = None
            # store L1 norm for sparsity regularizers
            # keep graphful during training
            self.h_norm_l1 = h_x.norm(p=1)
        else:
            # conceptizer returns h_x (likely detached / no autograd required)
            h_x = self.conceptizer(x)

        # store concepts; keep graphful if training, detach otherwise
        self.concepts = h_x if self.training else h_x.detach()

        if self.debug:
            print("Encoded concepts shape:", tuple(h_x.shape))

        # compute theta param scores
        thetas = self.parametrizer(x)
        # ensure theta shape has a concept/class axis: expected (B, k, nclass) or (B,k,cdim,nclass)
        if thetas.dim() == 2:
            # (B, k* d?) ambiguous: but older code unsqueezed dim 2
            thetas = thetas.unsqueeze(-1)

        # store thetas: keep graph during training (for jacobian penalties), else detach
        self.thetas = thetas if self.training else thetas.detach()

        if self.debug:
            print("Theta shape:", tuple(thetas.shape))

        # flatten concepts if they are spatial (e.g., shape (B,k, H, W) -> (B,k, H*W))
        if h_x.dim() == 4:
            B, k = h_x.shape[0], h_x.shape[1]
            h_x = h_x.view(B, k, -1)

        # aggregator expects H (B,k,cdim) and Th shape
        logits = self.aggregator(h_x, thetas)

        if self.debug:
            print("Output (logits) shape:", tuple(logits.shape))

        return logits

    def predict_proba(self, x: Union[np.ndarray, torch.Tensor], to_numpy: bool = False):
        """Return class probabilities for input x.

        Args:
            x: numpy array or torch.Tensor (B,C,H,W or B,D)
            to_numpy: return numpy array if True
        """
        # convert numpy -> tensor
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            x_t = x
        else:
            raise ValueError(f"Unrecognized input type {type(x)}")

        device = next(self.parameters()).device if any(p.numel() for p in self.parameters()) else x_t.device
        x_t = x_t.to(device)

        # compute logits without building graphs
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_t)

            # binary vs multiclass
            if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
                probs = torch.sigmoid(logits)
            else:
                probs = F.softmax(logits, dim=1)

        if to_numpy:
            return probs.cpu().numpy()
        return probs

    def forward_with_params(self, x: torch.Tensor, thetas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute aggregator output using stored or provided thetas (does not call parametrizer)."""
        if self.learning_H:
            h_res = self.conceptizer(x)
            h_x = h_res[0] if isinstance(h_res, (tuple, list)) else h_res
        else:
            h_x = self.conceptizer(x)

        if h_x.dim() == 4:
            B, k = h_x.shape[0], h_x.shape[1]
            h_x = h_x.view(B, k, -1)

        thetas_use = thetas if thetas is not None else self.thetas
        if thetas_use is None:
            raise ValueError("No theta values available. Run forward(x) or pass 'thetas' explicitly.")

        # ensure thetas have correct shape (B,k,nclass) or (B,k,cdim,nclass)
        logits = self.aggregator(h_x, thetas_use)
        return logits

    def explain(self, x: Union[np.ndarray, torch.Tensor], y: Optional[Union[int, list, torch.Tensor, str]] = None,
                skip_bias: bool = True):
        """
        Return per-concept attributions (theta values) for requested class(es).

        Args:
          - x: input batch (B,...)
          - y: None|'all' -> return all classes; 'max' -> predicted argmax class per sample;
               int -> scalar class label applied to all samples; list/tensor length B -> per-sample labels.
          - skip_bias: if True and the conceptizer appended a bias concept (e.g., add_bias), the bias concept is removed.

        Returns:
          - Tensor on CPU with shape:
              * (B, k) for single-class output per sample
              * (B, k, nclasses) for all classes
        """
        # Accept numpy arrays too
        input_is_numpy = isinstance(x, np.ndarray)
        if input_is_numpy:
            x_t = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            x_t = x
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

        device = next(self.parameters()).device if any(p.numel() for p in self.parameters()) else x_t.device
        x_t = x_t.to(device)

        # Run forward to populate thetas; forward will set self.thetas (graphful if training)
        with torch.set_grad_enabled(self.training):
            logits = self.forward(x_t)

        if self.thetas is None:
            raise RuntimeError("Thetas not set. Ensure forward(x) ran successfully.")

        # Work on a CPU detached copy for explanations (avoid large graphs)
        theta = self.thetas.detach().cpu() if isinstance(self.thetas, torch.Tensor) else torch.tensor(self.thetas).cpu()

        # If theta has an extra concept-dim (cdim) reduce it to a scalar relevance per concept by averaging
        # theta shape may be (B,k,nclasses) or (B,k,cdim,nclasses)
        if theta.dim() == 4:
            # reduce over cdim dimension (axis=2)
            theta_reduced = theta.mean(dim=2)  # (B, k, nclasses)
        elif theta.dim() == 3:
            theta_reduced = theta  # (B, k, nclasses)
        else:
            raise ValueError("Unsupported theta shape for explanation: {}".format(tuple(theta.shape)))

        B, k, nclasses = theta_reduced.shape

        # choose classes to return
        if y is None or y == 'all':
            attr = theta_reduced  # (B,k,nclasses)
        elif y == 'max':
            # choose predicted class per sample
            preds = logits.argmax(dim=1).detach().cpu()  # (B,)
            idx = preds.view(-1, 1, 1).expand(-1, k, 1)  # (B,k,1)
            attr = theta_reduced.gather(2, idx).squeeze(-1)  # (B,k)
        else:
            # y may be scalar int, list, numpy array, or tensor
            if isinstance(y, int):
                idx = torch.full((B, 1, 1), y, dtype=torch.long)
                idx = idx.expand(-1, k, -1)  # (B,k,1)
                attr = theta_reduced.gather(2, idx).squeeze(-1)
            else:
                # treat y as per-sample labels
                y_t = torch.as_tensor(y, dtype=torch.long)
                if y_t.dim() == 0:
                    y_t = y_t.unsqueeze(0).expand(B)
                if y_t.numel() != B:
                    raise ValueError("y length must match batch size (B={})".format(B))
                idx = y_t.view(-1, 1, 1).expand(-1, k, 1)
                attr = theta_reduced.gather(2, idx).squeeze(-1)  # (B,k)

        # If attr still has class-dim, it's (B,k,nclasses). If it is (B,k) it's single-class per sample.
        # Remove bias concept if requested and if conceptizer provides `add_bias` attribute
        if skip_bias and getattr(self.conceptizer, 'add_bias', False):
            # assume bias is the last concept
            if attr.dim() == 3:
                attr = attr[:, :-1, :]
            else:
                attr = attr[:, :-1]

        return attr

    # Optional helper to clear stored runtime attributes to free memory
    def clear_runtime_state(self):
        """Clear stored runtime tensors that may hold graph references (useful after training iteration)."""
        # Only clear those used for explanations; do not clear model params
        self.thetas = None
        self.concepts = None
        self.recons = None
        self.h_norm_l1 = None


def load_cifar_data(valid_size=0.1, shuffle=True, resize = None, random_seed=2008, batch_size = 64,
                    num_workers = 1):
    """
        We return train and test for plots and post-training experiments
    """
    transf_seq = [
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ]
    if resize and (resize[0] != 32 or resize[1] != 32):
        transf_seq.insert(0, transforms.Resize(resize))

    transform = transforms.Compose(transf_seq)
    # normalized according to pytorch torchvision guidelines https://chsasank.github.io/vision/models.html
    train = CIFAR10('data/CIFAR', train=True, download=True, transform=transform)
    test  = CIFAR10('data/CIFAR', train=False, download=True, transform=transform)

    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)


    # Create DataLoader
    dataloader_args = dict(batch_size=batch_size,num_workers=num_workers)
    train_loader = dataloader.DataLoader(train, sampler=train_sampler, **dataloader_args)
    valid_loader = dataloader.DataLoader(train, sampler=valid_sampler, **dataloader_args)
    dataloader_args['shuffle'] = False
    test_loader = dataloader.DataLoader(test, **dataloader_args)

    return train_loader, valid_loader, test_loader, train, test


class AttrDict(object):
    """Dict-like object that allows attribute access and dict access."""
    def __init__(self, mapping=None):
        mapping = mapping or {}
        for k, v in mapping.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        setattr(self, key, val)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def keys(self):
        return list(self.__dict__.keys())

    def items(self):
        return list(self.__dict__.items())

    def update(self, other: dict):
        for k, v in other.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"AttrDict({self.__dict__})"


# ==== Deep Packet style dataset & loader ======================================
import os, re, glob
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Accept (1500,) or (N,1500); always return (N,1500) float32 in [0,1]."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr

# ---- Optional: PCAP -> numpy (slow; install scapy if you use this path) -----
def _pcap_to_packets_1500(pcap_path: str, max_len: int = 1500) -> np.ndarray:
    """
    Convert one .pcap to Deep Packet style (N,1500) float32 in [0,1].
    Follows the paper's preprocessing: remove Ethernet header; normalize TCP/UDP header
    length; drop SYN/ACK/FIN without payload and DNS; keep IP header + first 1480 bytes,
    zero-pad; divide by 255. :contentReference[oaicite:1]{index=1}
    """
    try:
        from scapy.all import rdpcap, TCP, UDP, Raw, DNS
    except Exception as e:
        raise RuntimeError("scapy is required for on-the-fly PCAP processing. Install it or "
                           "preconvert to .npy.") from e

    packets = []
    for pkt in rdpcap(pcap_path):
        # Drop DNS
        if pkt.haslayer('DNS') or pkt.haslayer(DNS):
            continue

        # Require transport payload
        if not pkt.haslayer(Raw):
            # also drop pure SYN/ACK/FIN etc. with no payload
            continue

        payload = bytes(pkt[Raw].load)

        # Remove Ethernet header if present (pcap may include it)
        # (Ethernet header is 14 bytes; guard for short)
        if len(payload) >= 14:
            payload_wo_eth = payload[14:]
        else:
            payload_wo_eth = payload

        # Normalize TCP/UDP header lengths to 20 bytes for uniformity (UDP has 8)
        # We keep IP header + 1480 bytes → vector of length 1500 afterwards. :contentReference[oaicite:2]{index=2}
        # Here we simply truncate/pad the transport+payload slice to 1480,
        # assuming 20 bytes accounted for header in the final 1500 pipeline.
        vec = np.zeros(max_len, dtype=np.float32)
        # Put first 1500 bytes (IP hdr + 1480 payload approximation)
        clip = payload_wo_eth[:max_len]
        vec[:len(clip)] = np.frombuffer(clip, dtype=np.uint8).astype(np.float32)

        # Scale to [0,1]
        vec /= 255.0
        packets.append(vec)

    if not packets:
        return np.empty((0, max_len), dtype=np.float32)
    return np.stack(packets, axis=0)


# ==== Lazily-indexed, memmap-backed DeepPacket dataset (no RAM spikes) =======
import os, glob, bisect, numpy as np, torch
from torch.utils.data import Dataset

class DeepPacketNPYDataset(Dataset):
    """
    Root/
      class_a/*.npy   # each .npy is (N,1500) or (1500,)
      class_b/*.npy
    We DO NOT expand to per-packet list at init. We:
      - list files
      - get per-file packet counts using np.load(..., mmap_mode='r') (cheap)
      - build prefix sums to map global index -> (file, row)
    Only the requested row is materialized on __getitem__.
    """
    def __init__(self, root: str, split_indices=None, pcap_ok: bool=False, cache_dir=None,
                 max_rows_per_file: int = None):
        if pcap_ok:
            raise RuntimeError("Use the offline converter to .npy. This dataset is .npy-only for speed.")
        if cache_dir is not None:
            # not needed for .npy path; keep for API compatibility
            pass

        self.root = root
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        # File inventory (by class)
        self.files = []   # [(path, class_idx), ...]
        for cls in self.classes:
            for p in glob.glob(os.path.join(root, cls, "*.npy")):
                self.files.append((p, self.class_to_idx[cls]))

        if split_indices is not None:
            self.files = [self.files[i] for i in split_indices]

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found under {root}")

        # Get per-file row counts WITHOUT loading the arrays (mmap header only)
        self.counts = []
        for path, _ in self.files:
            arr = np.load(path, mmap_mode="r")  # memmap
            # accept (1500,) or (N,1500)
            if arr.ndim == 1:
                nrows = 1
            else:
                nrows = int(arr.shape[0])
            if max_rows_per_file is not None:
                nrows = min(nrows, max_rows_per_file)
            self.counts.append(nrows)
            del arr

        # Prefix sums to locate (file_idx, row_idx) from a global index
        self.offsets = np.cumsum([0] + self.counts)  # length = n_files+1
        self.total = int(self.offsets[-1])
        if self.total == 0:
            raise RuntimeError("All files appear empty after header inspection.")

    def __len__(self):
        return self.total

    def _locate(self, idx: int):
        # returns (file_idx, row_idx_in_file)
        # offsets: [0, c0, c0+c1, ..., total]
        f = bisect.bisect_right(self.offsets, idx) - 1
        row = idx - self.offsets[f]
        return f, row

    def __getitem__(self, idx: int):
        fidx, ridx = self._locate(idx)
        path, y = self.files[fidx]

        arr = np.load(path, mmap_mode="r")
        vec = arr if arr.ndim == 1 else arr[ridx]          # view into memmap

        # ensure float32
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)

        # >>> make writable & contiguous to silence the warning <<<
        if (not vec.flags.writeable) or (not vec.flags['C_CONTIGUOUS']):
            vec = np.array(vec, dtype=np.float32, copy=True)   # forces writable, C-contig

        x = torch.from_numpy(vec).view(1, 1, -1)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

def load_deeppacket_data(root: str,
                         valid_size: float = 0.1,
                         test_size: float = 0.1,
                         batch_size: int = 64,
                         num_workers: int = 2,
                         shuffle: bool = True,
                         pcap_ok: bool = False,
                         cache_dir: Optional[str] = None):

    # Probe without loading data to enumerate files/classes
    tmp = DeepPacketNPYDataset(root, split_indices=None, pcap_ok=pcap_ok, cache_dir=cache_dir)

    # --- split over FILES (lazy dataset uses `files`, not `items`) ---
    file_items = list(tmp.files)                  # [(path, class_idx), ...]
    n_files = len(file_items)
    if n_files == 0:
        raise RuntimeError(f"No .npy files found under '{root}'.")

    indices = np.arange(n_files)
    if shuffle:
        rng = np.random.RandomState(2018)
        rng.shuffle(indices)

    n_test = int(np.floor(test_size * n_files))
    n_val  = int(np.floor(valid_size * (n_files - n_test)))

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train_ds = DeepPacketNPYDataset(root, split_indices=train_idx.tolist(), pcap_ok=pcap_ok, cache_dir=cache_dir)
    val_ds   = DeepPacketNPYDataset(root, split_indices=val_idx.tolist(),   pcap_ok=pcap_ok, cache_dir=cache_dir)
    test_ds  = DeepPacketNPYDataset(root, split_indices=test_idx.tolist(),  pcap_ok=pcap_ok, cache_dir=cache_dir)

    dl_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_args)
    valid_loader = DataLoader(val_ds,  shuffle=False, **dl_args)
    test_loader  = DataLoader(test_ds, shuffle=False, **dl_args)

    return train_loader, valid_loader, test_loader, train_ds, test_ds

def build_args_dict():
    """
    Return an AttrDict containing a comprehensive set of args:
    - All options present in your original get_senn_parser()
    - CIFAR / Lipschitz / Colab-friendly extras
    Edit values directly in Colab after calling build_args_dict().
    """
    defaults = {
        # -------------------------
        # get_senn_parser() defaults
        # -------------------------
        'train': False,
        'test': False,
        'load_model': False,

        # Save Paths
        'model_path': 'models',
        'results_path': 'out',
        'log_path': 'log',
        'summary_path': 'results/summary.csv',

        # Device
        'cuda': False,
        'num_gpus': 1,
        'seed': 2018,

        # Model - Concept Encoder (H)
        'h_type': 'input',            # 'cnn'|'input' etc.
        'concept_dim': 1,
        'nconcepts': 20,
        'h_sparsity': 1e-4,

        # Parametrizer (Theta)
        'nobias': False,
        'positive_theta': False,
        'theta_arch': 'vgg11',     # choices in your repo: ['simple','alexnet','vgg11','vgg16']
        'theta_dim': -1,            # defaults to #classes if left -1 (your main sometimes sets it)
        'theta_reg_type': 'grad1',
        'theta_reg_lambda': 1e-2,

        # Learning
        'opt': 'adam',
        'lr': 0.001,
        'epochs': 1,
        'batch_size': 128,
        'objective': 'cross_entropy',
        'dropout': 0.1,
        'weight_decay': 1e-3,

        # Data
        'dataset': 'pathology',
        'embedding': 'pathology',
        'nclasses': 2,

        # Misc
        'num_workers': 2,
        'print_freq': 10,
        'debug': False,

        # -------------------------
        # Extras / CIFAR / explain
        # -------------------------
        # Keep both 'dataset' and 'datasets' (original had plural list)
        'datasets': ['heart', 'ionosphere', 'breast-cancer','wine','heart',
                     'glass','diabetes','yeast','leukemia','abalone'],

        # Lipschitz/explain settings (from earlier code)
        'lip_calls': 10,
        'lip_eps': 0.01,
        'lip_points': 100,
        'optim_bb': 'gp',           # name change to avoid clashing with optimizer arg; used for BB optimization
        'optim': 'gp',              # kept for compatibility (your code uses args.optim)

        # Colab / CIFAR-friendly defaults
        'nchannel': 3,              # 3 for RGB CIFAR
        'input_dim': None,          # left None; main() currently computes it from H,W (see note below)
        'h_type': 'input',          # set to 'input' by default for image input-concept experiments

        # Behavioral toggles used in other parts of your pipeline
        'positive_theta': False,
        'pretrained_backbone': False,

        # Extra housekeeping
        'results_extra_suffix': '',

        # Keep a placeholder for anything else you may want to add quickly
        # e.g. 'myflag': False
    }

    # Auto-detect and sanitize values
    if defaults['num_workers'] is None or defaults['num_workers'] < 0:
        try:
            defaults['num_workers'] = multiprocessing.cpu_count() or 4
        except Exception:
            defaults['num_workers'] = 4

    # Resolve H,W based on theta_arch heuristic (keeps your previous logic)
    arch = defaults.get('theta_arch', 'simple')
    if arch == 'simple' or ('vgg' in arch):
        H, W = 32, 32
    else:
        H, W = 224, 224

    # If user wants default input_dim computed here, you may uncomment the line below.
    # But to keep backwards compatibility with your existing main() (which itself sets args.input_dim),
    # we leave defaults['input_dim'] = None so main() remains the component that decides input_dim.
    # To set it here instead, uncomment and use:
    # defaults['input_dim'] = defaults['nchannel'] * H * W

    # Make directories if missing (safe no-op if already exist)
    for p in (defaults['model_path'], defaults['results_path'], defaults['log_path']):
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass

    args = AttrDict(defaults)
    # provide H,W for convenience (so user can compute input_dim manually if desired)
    args.H = H
    args.W = W

    # Print parameters in the same format your old parse_args printed them
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args


args = build_args_dict()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# ----- Set SENN args for Deep Packet ---------------------------------
args.nchannel = 1
H, W = 1, 1500
args.input_dim  = H * W * args.nchannel

# Point this to your data root:
#   data/DeepPacket/APP_A/*.npy
#   data/DeepPacket/APP_B/*.npy
# etc. (or .pcap if you set pcap_ok=True)
DEEPPACKET_ROOT = 'proc_pcaps/'

# Discover classes from the folders and set nclasses/theta_dim accordingly
# We’ll peek once using a tiny helper dataset
_probe = DeepPacketNPYDataset(DEEPPACKET_ROOT)
args.nclasses = len(_probe.classes)
args.theta_dim = args.nclasses

model_path, log_path, results_path = generate_dir_names('deeppacket', args)
print("here", flush=True)
train_loader, valid_loader, test_loader, train_tds, test_tds = load_deeppacket_data(
    root=DEEPPACKET_ROOT,
    valid_size=0.1,
    test_size=0.1,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pcap_ok=False,              # set True only if you want on-the-fly .pcap parsing
    cache_dir='data/DeepPacketNPY'  # where to cache converted .pcap -> .npy (optional)
)
print("files ->", len(DeepPacketNPYDataset(DEEPPACKET_ROOT).files), flush=True)
print("train/val/test lens:", len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset))
xb, yb = next(iter(train_loader)); print(xb.shape, yb.shape)  # expect (B, 1, 1, 1500), (B,)
if args.h_type == 'input':
    conceptizer  = input_conceptizer()
    args.nconcepts = args.input_dim + int(not args.nobias)
elif args.h_type == 'cnn':

    # biase. They treat it like any other concept.
    #args.nconcepts +=     int(not args.nobias)
    conceptizer  = image_cnn_conceptizer(args.input_dim, args.nconcepts, args.concept_dim, nchannel = 3) #, sparsity = sparsity_l)
else:
    #args.nconcepts +=     int(not args.nobias)
    conceptizer  = image_fcc_conceptizer(args.input_dim, args.nconcepts, args.concept_dim, nchannel = 3) #, sparsity = sparsity_l)


if args.theta_arch == 'simple':
    parametrizer = image_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, nchannel = 3, only_positive = args.positive_theta)
elif 'vgg' in args.theta_arch:
    parametrizer = vgg_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, arch = args.theta_arch, nchannel = 3, only_positive = args.positive_theta) #torchvision.models.alexnet(num_classes = args.nconcepts*args.theta_dim)
else:
    parametrizer = torchvision_parametrizer(args.input_dim, args.nconcepts, args.theta_dim, arch = args.theta_arch, nchannel = 3, only_positive = args.positive_theta) #torchvision.models.alexnet(num_classes = args.nconcepts*args.theta_dim)

aggregator   = additive_scalar_aggregator(args.concept_dim, args.nclasses)

model        = GSENN(conceptizer, parametrizer, aggregator) #, learn_h = args.train_h)


# if not args.train and args.load_model:
#     checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
#     checkpoint.keys()
#     model = checkpoint['model']
#
#



if args.theta_reg_type in ['unreg','none', None]:
    trainer = VanillaClassTrainer(model, args)
elif args.theta_reg_type == 'grad1':
    trainer = GradPenaltyTrainer(model, args, typ = 1)
elif args.theta_reg_type == 'grad2':
    trainer = GradPenaltyTrainer(model, args, typ = 2)
elif args.theta_reg_type == 'grad3':
    trainer = GradPenaltyTrainer(model, args, typ = 3)
elif args.theta_reg_type == 'crosslip':
    trainer = CLPenaltyTrainer(model, args)
else:
    raise ValueError('Unrecoginzed theta_reg_type')

if args.train or not args.load_model or (not os.path.isfile(os.path.join(model_path,'model_best.pth.tar'))):
    trainer.train(train_loader, valid_loader, epochs = args.epochs, save_path = model_path)
    #trainer.plot_losses(save_path=results_path)
else:
    checkpoint = torch.load(os.path.join(model_path,'model_best.pth.tar'), map_location=lambda storage, loc: storage)
    checkpoint.keys()
    model = checkpoint['model']
    trainer =  VanillaClassTrainer(model, args) # arbtrary trained, only need to compuyte val acc

#trainer.validate(test_loader, fold = 'test')
# Move model to device (harmless if already on device)
device = torch.device('cuda' if (hasattr(args, 'cuda') and args.cuda and torch.cuda.is_available()) else 'cpu')
trainer.model.to(device)

# Compute train accuracy (may be slow on full train set; you can use a subset if desired)
print("Computing accuracies — this will run full epoch evaluations (may take some time).")
train_acc = trainer.validate(train_loader) if train_loader is not None else None

# Validation accuracy
val_acc = trainer.validate(valid_loader) if (valid_loader is not None) else None

# Test accuracy (uses evaluate which prints summary)
test_acc = trainer.evaluate(test_loader) if (test_loader is not None) else None

# Nicely print results
print("\nFinal accuracies:")
if train_acc is not None:
    print(f"  Train Accuracy : {train_acc:.2f}%")
else:
    print("  Train Accuracy : (no train loader)")

if val_acc is not None:
    print(f"  Val   Accuracy : {val_acc:.2f}%")
else:
    print("  Val   Accuracy : (no val loader)")

if test_acc is not None:
    print(f"  Test  Accuracy : {test_acc:.2f}%")
else:
    print("  Test  Accuracy : (no test loader)")

model.eval()

All_Results = {}

### 0. Concept Grid for Visualization
#concept_grid(model, test_loader, top_k = 10, cuda = args.cuda, save_path = results_path + '/concept_grid.pdf')


### 1. Single point lipshiz estimate via black box optim (for fair comparison)
# with other methods in which we have to use BB optimization.
features = None
classes = [str(i) for i in range(10)]
expl = gsenn_wrapper(model,
                    mode      = 'classification',
                    input_type = 'image',
                    multiclass=True,
                    feature_names = features,
                    class_names   = classes,
                    train_data      = train_loader,
                    skip_bias = True,
                    verbose = False)


### Debug single input
# x = next(iter(train_tds))[0]
# attr = expl(x, show_plot = False)
# pdb.set_trace()

# #### Debug multi input
# x = next(iter(test_loader))[0] # Transformed
# x_raw = test_loader.dataset.test_data[:args.batch_size,:,:]
# attr = expl(x, x_raw = x_raw, show_plot = True)
# #pdb.set_trace()

# #### Debug argmaz plot_theta_stability
if args.h_type == 'input':
    x = next(iter(test_tds))[0].numpy()
    y = next(iter(test_tds))[0].numpy()
    x_raw = (test_tds.test_data[0].float()/255).numpy()
    y_raw = revert_to_raw(x)
    att_x = expl(x, show_plot = False)
    att_y = expl(y, show_plot = False)
    lip = 1
    lipschitz_argmax_plot(x_raw, y_raw, att_x,att_y, lip)# save_path=fpath)
    #pdb.set_trace()


### 2. Single example lipschitz estimate with Black Box
do_bb_stability_example = True
if do_bb_stability_example:
    print('**** Performing lipschitz estimation for a single point ****')
    idx = 0
    print('Example index: {}'.format(idx))
    #x = train_tds[idx][0].view(1,28,28).numpy()
    x = next(iter(test_tds))[0].numpy()

    #x_raw = (test_tds.test_data[0].float()/255).numpy()
    x_raw = (test_tds.test_data[0]/255)

    #x_raw = next(iter(train_tds))[0]

    # args.optim     = 'gp'
    # args.lip_eps   = 0.1
    # args.lip_calls = 10
    Results = {}

    lip, argmax = expl.local_lipschitz_estimate(x, bound_type='box_std',
                                            optim=args.optim,
                                            eps=args.lip_eps,
                                            n_calls=4*args.lip_calls,
                                            njobs = 1,
                                            verbose=2)
    #pdb.set_trace()
    Results['lip_argmax'] = (x, argmax, lip)
    # .reshape(inputs.shape[0], inputs.shape[1], -1)
    att = expl(x, None, show_plot=False)#.squeeze()
    # .reshape(inputs.shape[0], inputs.shape[1], -1)
    att_argmax = expl(argmax, None, show_plot=False)#.squeeze()

    #pdb.set_trace()
    Argmax_dict = {'lip': lip, 'argmax': argmax, 'x': x}
    fpath = os.path.join(results_path, 'argmax_lip_gp_senn.pdf')
    if args.h_type == 'input':
        lipschitz_argmax_plot(x_raw, revert_to_raw(argmax), att, att_argmax, lip, save_path=fpath)
    pickle.dump(Argmax_dict, open(
        results_path + '/argmax_lip_gp_senn.pkl', "wb"))


#noise_stability_plots(model, test_tds, cuda = args.cuda, save_path = results_path)
### 3. Local lipschitz estimate over multiple samples with Black BOx Optim
do_bb_stability = True
if do_bb_stability:
    print('**** Performing black-box lipschitz estimation over subset of dataset ****')
    maxpoints = 20
    #valid_loader 0 it's shuffled, so it's like doing random choice
    mini_test = next(iter(valid_loader))[0][:maxpoints].numpy()
    lips = expl.estimate_dataset_lipschitz(mini_test,
                                        n_jobs=-1, bound_type='box_std',
                                        eps=args.lip_eps, optim=args.optim,
                                        n_calls=args.lip_calls, verbose=2)
    Stability_dict = {'lips': lips}
    pickle.dump(Stability_dict, open(results_path + '_stability_blackbox.pkl', "wb"))
    All_Results['stability_blackbox'] = lips


pickle.dump(All_Results, open(results_path + '_combined_metrics.pkl'.format(dataname), "wb"))


# args.epoch_stats = epoch_stats
# save_path = args.results_path
# print("Save train/dev results to", save_path)
# args_dict = vars(args)
# pickle.dump(args_dict, open(save_path,'wb') )
