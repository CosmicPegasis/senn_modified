"""Dataset classes for DeepPacket data loading."""

import os
import glob
import bisect
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset


def _paired_flow_path(data_path: str, flow_suffix: str = ".flow.npy") -> str:
    """Get the path to the flow sidecar file for a given data file."""
    if data_path.endswith(".npy"):
        return data_path[:-4] + flow_suffix
    return data_path + flow_suffix


def _normalize_packet_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a packet vector to fixed length 1500 and dtype float32 in [0,1].
    
    Args:
        vec: Input vector (can be any shape/size)
        
    Returns:
        Normalized vector of shape (1500,) with dtype float32
    """
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
    
    return vec


class DeepPacketNPYDataset(Dataset):
    """
    Dataset for loading DeepPacket data from .npy files.
    
    Expected directory structure:
        Root/
          class_a/*.npy  # each .npy is (N,1500) or (1500,)
          class_b/*.npy
    
    Lazy, memmap-backed; returns (B,1,1,1500) tensors.
    """
    
    def __init__(self, root: str, split_indices: Optional[List[int]] = None, 
                 max_rows_per_file: Optional[int] = None, weight_method: str = "balanced"):
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

    def apply_undersampling(self, ratio: float = 0.1, strategy: str = "random") -> 'UndersampledDeepPacketNPYDataset':
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
        """Find which file and row index corresponds to the global index."""
        f = bisect.bisect_right(self.offsets, idx) - 1
        row = idx - self.offsets[f]
        return f, row

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        fidx, ridx = self._locate(idx)
        path, y = self.files[fidx]
        arr = np.load(path, mmap_mode="r")
        vec = arr if arr.ndim == 1 else arr[ridx]
        vec = _normalize_packet_vector(vec)
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
        vec = _normalize_packet_vector(vec)
        x = torch.tensor(vec, dtype=torch.float32).contiguous().view(1, 1, -1)
        return x, torch.tensor(y, dtype=torch.long)


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
        """Lazily load flow memmap for a file."""
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
        """Get flow ID for a given global index."""
        fidx, ridx = self._locate(global_idx)
        mm = self._ensure_flow_mm(fidx)
        if mm is None:
            return int(global_idx)  # fallback: each row = its own "flow"
        return int(mm[ridx] if mm.ndim == 1 else mm[ridx])


class SelectedRowsDeepPacketDataset(Dataset):
    """
    Lightweight view over DeepPacketNPYDataset that reads only specific rows per file.
    """
    
    def __init__(self, base_ds: DeepPacketNPYDataset, per_file_rows: Dict[int, List[int]], 
                 weight_method: str = "balanced"):
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
        """Calculate class counts for selected rows."""
        class_counts = {i: 0 for i in range(len(self.classes))}
        for i, (_, class_idx) in enumerate(self.files):
            class_counts[class_idx] += len(self.sampling_indices[i])
        return class_counts

    def _calculate_class_weights(self, method: str = "balanced") -> Dict[int, float]:
        """Calculate class weights."""
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
        """Get sample weights."""
        ws = []
        for i, (_, class_idx) in enumerate(self.files):
            w = self.class_weights[class_idx]
            ws.extend([w] * len(self.sampling_indices[i]))
        return ws

    def apply_undersampling(self, ratio: float = 0.1, strategy: str = "random") -> 'SelectedRowsDeepPacketDataset':
        """
        Apply undersampling to the already-selected rows.
        
        Args:
            ratio: Ratio of largest class to keep (e.g., 0.1 = keep 10% of largest class)
            strategy: "random" or "stratified"
        
        Returns:
            New SelectedRowsDeepPacketDataset with undersampled indices
        """
        if ratio >= 1.0:
            return self  # No undersampling needed
        
        # Calculate target count for each class
        max_count = max(self.class_counts.values())
        if max_count == 0:
            return self  # No data to undersample
        
        target_count = max(1, int(max_count * ratio))
        target_counts = {}
        for class_idx, count in self.class_counts.items():
            if count > 0:
                target_counts[class_idx] = min(count, target_count)
            else:
                target_counts[class_idx] = 0
        
        # Create new per_file_rows with undersampled indices
        new_per_file_rows = {}
        class_remaining = {class_idx: target_count for class_idx in target_counts}
        
        for fidx, (_, class_idx) in enumerate(self.files):
            current_indices = self.sampling_indices[fidx]
            if len(current_indices) == 0 or class_remaining[class_idx] <= 0:
                new_per_file_rows[fidx] = []
                continue
            
            # Calculate proportional target for this file
            total_class_samples = self.class_counts[class_idx]
            if total_class_samples > 0:
                file_target = max(0, min(len(current_indices), 
                                       int(target_counts[class_idx] * len(current_indices) / total_class_samples)))
                file_target = min(file_target, class_remaining[class_idx])
                class_remaining[class_idx] -= file_target
            else:
                file_target = 0
            
            # Sample from current_indices
            if file_target > 0 and file_target < len(current_indices):
                if strategy == "random":
                    sampled_idx = np.random.choice(len(current_indices), file_target, replace=False)
                    new_per_file_rows[fidx] = [current_indices[i] for i in sorted(sampled_idx)]
                elif strategy == "stratified":
                    step = len(current_indices) / file_target
                    sampled_idx = [int(i * step) for i in range(file_target)]
                    new_per_file_rows[fidx] = [current_indices[i] for i in sampled_idx]
                else:
                    raise ValueError(f"Unknown undersampling strategy: {strategy}")
            elif file_target > 0:
                new_per_file_rows[fidx] = current_indices[:]
            else:
                new_per_file_rows[fidx] = []
        
        # Create new dataset with the undersampled indices
        weight_method = "balanced"  # Default to balanced
        return SelectedRowsDeepPacketDataset(self.base, new_per_file_rows, weight_method=weight_method)

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Find file and row index for global index."""
        f = bisect.bisect_right(self.file_offsets, idx) - 1
        f = min(max(f, 0), len(self.file_offsets) - 1)
        row = idx - self.file_offsets[f]
        return f, row

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        fidx, ridx = self._locate(idx)
        path, y = self.files[fidx]
        actual_row = self.sampling_indices[fidx][ridx]
        arr = np.load(path, mmap_mode="r")
        vec = arr if arr.ndim == 1 else arr[actual_row]
        vec = _normalize_packet_vector(vec)
        x = torch.tensor(vec, dtype=torch.float32).contiguous().view(1, 1, -1)
        return x, torch.tensor(y, dtype=torch.long)

