"""GSENN model components for DeepPacket classification."""

from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def explain(self, x: torch.Tensor, y: Optional[Union[int, str]] = None, skip_bias: bool = True) -> torch.Tensor:
        """
        Return per-concept attributions (theta values) for requested class(es).
        
        Args:
          - x: input batch (B, 1, 1, 1500) or similar
          - y: None|'all' -> return all classes; 'max' -> predicted argmax class per sample;
               int -> scalar class label applied to all samples.
          - skip_bias: if True and the conceptizer appended a bias concept, the bias concept is removed.
        
        Returns:
          - Tensor on CPU with shape:
              * (B, k) for single-class output per sample
              * (B, k, nclasses) for all classes
        """
        device = next(self.parameters()).device if any(p.numel() for p in self.parameters()) else x.device
        x_t = x.to(device)
        
        # Run forward to populate thetas
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_t)
        
        if self.thetas is None:
            raise RuntimeError("Thetas not set. Ensure forward(x) ran successfully.")
        
        # Work on a CPU detached copy for explanations
        theta = self.thetas.detach().cpu()
        
        # Handle different theta shapes: (B, k, nclasses) or (B, k, 1, nclasses)
        if theta.dim() == 4:
            theta = theta.squeeze(-2)  # Remove singleton dim if present
        
        B, k, nclasses = theta.shape[0], theta.shape[1], theta.shape[2]
        
        # Choose classes to return
        if y is None or y == 'all':
            attr = theta  # (B, k, nclasses)
        elif y == 'max':
            # Choose predicted class per sample
            preds = logits.argmax(dim=1).detach().cpu()  # (B,)
            idx = preds.view(-1, 1, 1).expand(-1, k, 1)  # (B, k, 1)
            attr = theta.gather(2, idx).squeeze(-1)  # (B, k)
        else:
            # y is scalar int
            idx = torch.full((B, k, 1), y, dtype=torch.long)
            attr = theta.gather(2, idx).squeeze(-1)  # (B, k)
        
        # Remove bias concept if requested
        if skip_bias and getattr(self.conceptizer, 'add_bias', False):
            # Assume bias is the last concept
            if attr.dim() == 3:
                attr = attr[:, :-1, :]
            else:
                attr = attr[:, :-1]
        
        return attr

    def predict_proba(self, x: Union[np.ndarray, torch.Tensor], to_numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """Return class probabilities for input x.
        
        Args:
            x: numpy array or torch.Tensor. Can be:
               - (B, 1500) - will be reshaped to (B, 1, 1, 1500)
               - (B, 1, 1, 1500) - already in correct format
            to_numpy: if True, return numpy array; otherwise return torch.Tensor
            
        Returns:
            Probabilities tensor/array:
            - Binary: (B, 1) with sigmoid probabilities
            - Multiclass: (B, nclasses) with softmax probabilities
        """
        # Convert numpy -> tensor
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            x_t = x.clone()
        else:
            raise ValueError(f"Unrecognized input type {type(x)}")
        
        # Get device
        device = next(self.parameters()).device if any(p.numel() for p in self.parameters()) else x_t.device
        x_t = x_t.to(device)
        
        # Handle input shape: (B, 1500) -> (B, 1, 1, 1500)
        if x_t.dim() == 2:
            # Assume (B, 1500) format
            x_t = x_t.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 1500)
        elif x_t.dim() == 3:
            # Assume (B, 1, 1500) format
            x_t = x_t.unsqueeze(1)  # (B, 1, 1, 1500)
        # If already 4D, use as-is
        
        # Compute logits without building graphs
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_t)
        
        # Apply activation based on number of classes
        nclasses = self.aggregator.nclasses
        if nclasses <= 2:
            # Binary classification: sigmoid
            # Handle both (B, 1) and (B,) logit shapes
            if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
                probs = torch.sigmoid(logits)
            else:
                # If somehow multiclass format, take first class
                probs = torch.sigmoid(logits[:, 0:1])
        else:
            # Multiclass: softmax
            probs = F.softmax(logits, dim=1)
        
        # Convert to numpy if requested
        if to_numpy:
            return probs.cpu().numpy()
        return probs

    def clear_runtime_state(self) -> None:
        """Clear runtime state variables."""
        self.thetas = None
        self.concepts = None
        self.recons = None
        self.h_norm_l1 = None

