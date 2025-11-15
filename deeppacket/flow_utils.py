"""Flow-based dataset splitting and verification utilities."""

import os
import glob
import bisect
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import (
    DeepPacketNPYDataset,
    FlowAwareDeepPacketDataset,
    SelectedRowsDeepPacketDataset,
    _paired_flow_path,
)


def _assert_no_flow_overlap_datasets(train_ds, val_ds, test_ds, flow_suffix: str = ".flow.npy") -> None:
    """
    Sanity check: ensure no flow IDs overlap across train/val/test.
    Works for both SelectedRowsDeepPacketDataset (flow split) and DeepPacketNPYDataset (file split).
    If sidecar flow files are absent, uses per-file-scoped synthetic IDs so it will not raise.
    val_ds can be None if there's no validation set (e.g., train/test only split).
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
        val_flows   = collect_flow_ids_for_selected(val_ds) if val_ds is not None else set()
        test_flows  = collect_flow_ids_for_selected(test_ds)
    else:
        train_flows = collect_flow_ids_for_fullfile(train_ds)
        val_flows   = collect_flow_ids_for_fullfile(val_ds) if val_ds is not None else set()
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
        ]
        if val_ds is not None:
            msg.append(f"  train ∩ val  : {len(inter_tv)} examples -> {sample(inter_tv)}")
        msg.append(f"  train ∩ test : {len(inter_tt)} examples -> {sample(inter_tt)}")
        if val_ds is not None:
            msg.append(f"  val   ∩ test : {len(inter_vt)} examples -> {sample(inter_vt)}")
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
        n_val  = int(np.floor(valid_size * n))
        test_f = fids[:n_test]
        val_f  = fids[n_test:n_test+n_val]
        train_f= fids[n_test+n_val:]

        for fid in train_f: train_idx.extend(by_flow[fid])
        for fid in val_f:   val_idx.extend(by_flow[fid])
        for fid in test_f:  test_idx.extend(by_flow[fid])

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def stratified_flow_train_test_split(by_flow, flow_to_class, test_size=0.2, seed=2018):
    """
    Split flow IDs into train/test, stratified by class, then expand to row indices.
    Returns (train_row_indices, test_row_indices) as numpy arrays.
    """
    rng = np.random.RandomState(seed)
    class2flows: Dict[int, List[int]] = {}
    for fid, c in flow_to_class.items():
        class2flows.setdefault(c, []).append(fid)

    train_idx, test_idx = [], []
    for c, fids in class2flows.items():
        fids = fids.copy()
        rng.shuffle(fids)
        n = len(fids)
        n_test = int(np.floor(test_size * n))
        test_f = fids[:n_test]
        train_f = fids[n_test:]
        for fid in train_f:
            train_idx.extend(by_flow[fid])
        for fid in test_f:
            test_idx.extend(by_flow[fid])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def _rows_to_per_file_dict(ds: DeepPacketNPYDataset, rows: np.ndarray) -> Dict[int, List[int]]:
    """
    Convert global row indices to per-file relative row indices dict[file_idx] = [row_idx,...]
    """
    per = {i: [] for i in range(len(ds.files))}
    for g in rows.tolist():
        fidx = bisect.bisect_right(ds.offsets, g) - 1
        fidx = min(max(fidx, 0), len(ds.offsets) - 2)
        ridx = g - ds.offsets[fidx]
        per[fidx].append(int(ridx))
    return per


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
    undersample: bool = False,
    undersample_ratio: float = 0.1,
    undersample_strategy: str = "random",
):
    """Split dataset by flow IDs, ensuring no flow overlap across splits."""
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

    # Apply undersampling to training set if enabled
    if undersample:
        print(f"\nApplying undersampling (ratio={undersample_ratio}, strategy={undersample_strategy})...")
        train_ds_original = train_ds
        train_ds = train_ds.apply_undersampling(ratio=undersample_ratio, strategy=undersample_strategy)
        print(f"Original training samples: {train_ds_original.total}")
        print(f"Undersampled training samples: {train_ds.total}\n")

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

    # Shuffle val/test sets so eval_batches gets representative samples
    valid_loader = DataLoader(val_ds, shuffle=True, **dl_args)
    test_loader  = DataLoader(test_ds, shuffle=True, **dl_args)
    # Sanity check: ensure no overlapping flows across splits
    _assert_no_flow_overlap_datasets(train_ds, val_ds, test_ds, flow_suffix=flow_suffix)
    return train_loader, valid_loader, test_loader, train_ds, val_ds, test_ds


def split_deeppacket(
    root: str,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 2,
    shuffle: bool = True,
    limit_files_per_split: int = 0,
    max_rows_per_file: int = None,
    handle_imbalance: bool = False,
    weight_method: str = "balanced",
    undersample: bool = False,
    undersample_ratio: float = 0.1,
    undersample_strategy: str = "random",
):
    """Split dataset by files (not flow-aware)."""
    tmp = DeepPacketNPYDataset(root)
    files = list(tmp.files)
    n_files = len(files)
    indices = np.arange(n_files)
    if shuffle:
        rng = np.random.RandomState(2018)
        rng.shuffle(indices)

    n_test = int(np.floor(test_size * n_files))
    n_val  = int(np.floor(valid_size * n_files))

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
    
    # Shuffle val/test sets so eval_batches gets representative samples
    valid_loader = DataLoader(val_ds,  shuffle=True, **dl_args)
    test_loader  = DataLoader(test_ds, shuffle=True, **dl_args)
    # Sanity check: ensure no overlapping flows across splits (uses sidecars if present)
    _assert_no_flow_overlap_datasets(train_ds, val_ds, test_ds, flow_suffix=".flow.npy")
    return train_loader, valid_loader, test_loader, train_ds, val_ds, test_ds


def save_flow_train_test_split(
    root: str,
    out_dir: Optional[str] = None,
    test_size: float = 0.2,
    seed: int = 2018,
    flow_suffix: str = ".flow.npy",
) -> str:
    """
    Create a shuffled, flow-based stratified train/test split and permanently save it.
    Saves:
      - JSON manifest with metadata and per-file row indices for train/test
      - NPY arrays of global row indices (train_rows, test_rows)
    Returns path to the saved JSON manifest.
    """
    base = FlowAwareDeepPacketDataset(root, flow_suffix=flow_suffix)
    by_flow, flow_to_class = group_indices_by_flow(base)
    train_rows, test_rows = stratified_flow_train_test_split(by_flow, flow_to_class, test_size=test_size, seed=seed)

    train_per = _rows_to_per_file_dict(base, train_rows)
    test_per = _rows_to_per_file_dict(base, test_rows)

    # Print basic distro info similar to split_deeppacket
    def count_per_class(per_dict):
        counts = {i: 0 for i in range(len(base.classes))}
        for file_idx, rows in per_dict.items():
            _, cls_idx = base.files[file_idx]
            counts[cls_idx] += len(rows)
        return counts
    train_counts = count_per_class(train_per)
    test_counts = count_per_class(test_per)
    total_flows = len(by_flow)
    print("Flow-based train/test split (persistent):")
    print(f"  Total packets: train={len(train_rows)} test={len(test_rows)}")
    print(f"  Total unique flows: {total_flows}")
    print("  Class distribution in training set:")
    for class_name, class_idx in base.class_to_idx.items():
        print(f"    {class_name}: {train_counts.get(class_idx, 0)} samples")
    print("  Class distribution in test set:")
    for class_name, class_idx in base.class_to_idx.items():
        print(f"    {class_name}: {test_counts.get(class_idx, 0)} samples")

    # Determine output directory
    root_abs = os.path.abspath(root)
    splits_dir = out_dir if out_dir is not None else os.path.join(root_abs, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    tag = f"flow_train_test_seed{seed}_ts{int(round(test_size*100))}"
    json_path = os.path.join(splits_dir, f"{tag}.json")
    npy_train_path = os.path.join(splits_dir, f"{tag}.train_rows.npy")
    npy_test_path = os.path.join(splits_dir, f"{tag}.test_rows.npy")

    # Save NPY rows
    np.save(npy_train_path, train_rows)
    np.save(npy_test_path, test_rows)

    # Prepare JSON manifest
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "root": root_abs,
        "type": "flow_train_test_split",
        "flow_suffix": flow_suffix,
        "seed": seed,
        "test_size": float(test_size),
        "shuffled": True,
        "classes": base.classes,
        "files": [path for (path, _) in base.files],
        "train_per_file_rows": {int(k): v for k, v in train_per.items()},
        "test_per_file_rows": {int(k): v for k, v in test_per.items()},
        "artifacts": {
            "train_rows_npy": os.path.abspath(npy_train_path),
            "test_rows_npy": os.path.abspath(npy_test_path),
        },
    }

    with open(json_path, "w") as f:
        json.dump(manifest, f)

    return json_path


def load_flow_train_test_split(
    split_manifest_path: str,
    batch_size: int = 128,
    num_workers: int = 2,
    weight_method: str = "balanced",
    handle_imbalance: bool = False,
    undersample: bool = False,
    undersample_ratio: float = 0.1,
    undersample_strategy: str = "random",
    max_rows_per_file: Optional[int] = None,
    root_override: Optional[str] = None,
):
    """
    Load a previously saved flow-based train/test split manifest and return
    (train_loader, test_loader, train_ds, test_ds).
    
    Args:
        max_rows_per_file: If set, limit the number of rows taken from each file
                          (applies to both train and test). This is useful for
                          running on a subset of the data for faster iteration.
        root_override: If provided, use this root path instead of the one in the manifest.
                      Useful when running on a different system where the original absolute
                      path doesn't exist. If the manifest root doesn't exist and this is
                      not provided, will try to resolve relative to the manifest file.
    """
    with open(split_manifest_path, "r") as f:
        manifest = json.load(f)

    root = manifest["root"]
    
    # Handle path resolution for cross-system compatibility
    if root_override is not None:
        root = root_override
    elif not os.path.exists(root):
        # If the manifest root doesn't exist, try to resolve relative to manifest file
        manifest_dir = os.path.dirname(os.path.abspath(split_manifest_path))
        # The manifest is typically in <root>/splits/, so go up one level
        potential_root = os.path.dirname(manifest_dir)
        if os.path.exists(potential_root):
            root = potential_root
            print(f"Manifest root path '{manifest['root']}' not found. Using '{root}' (resolved from manifest location).")
        else:
            raise FileNotFoundError(
                f"Root path from manifest '{manifest['root']}' does not exist, and could not "
                f"resolve from manifest location. Please provide --root argument to override."
            )
    flow_suffix = manifest.get("flow_suffix", ".flow.npy")
    
    # Create base dataset first to get current file ordering
    base = FlowAwareDeepPacketDataset(root, flow_suffix=flow_suffix, weight_method=weight_method)
    
    # Build mapping from manifest file paths to current file indices
    # Handle both absolute and relative paths, and normalize for comparison
    manifest_files = manifest.get("files", [])
    old_to_new_idx = {}
    
    # Normalize paths for comparison (handle absolute vs relative, different roots)
    # Match by (class, filename) tuple since paths may differ but files are the same
    def normalize_path(p):
        # Get just the filename and class directory name
        filename = os.path.basename(p)
        # Extract class from path - it's the directory containing the file
        dirname = os.path.dirname(p)
        class_name = os.path.basename(dirname)
        # Handle both absolute and relative paths
        if not class_name or class_name not in base.classes:
            # Try to find class in path components
            parts = p.replace(os.sep, '/').split('/')
            for part in reversed(parts[:-1]):  # All parts except filename
                if part in base.classes:
                    class_name = part
                    break
        if class_name in base.classes:
            return (class_name, filename)
        return None
    
    # Build mapping from normalized paths to current indices
    current_file_map = {}
    for new_idx, (path, _) in enumerate(base.files):
        norm = normalize_path(path)
        if norm:
            current_file_map[norm] = new_idx
    
    # Map old indices to new indices based on file paths
    for old_idx, old_path in enumerate(manifest_files):
        norm = normalize_path(old_path)
        if norm and norm in current_file_map:
            new_idx = current_file_map[norm]
            old_to_new_idx[old_idx] = new_idx
    
    # Remap train_per and test_per from old indices to new indices
    train_per_old = {int(k): list(map(int, v)) for k, v in manifest["train_per_file_rows"].items()}
    test_per_old = {int(k): list(map(int, v)) for k, v in manifest["test_per_file_rows"].items()}
    
    train_per = {}
    test_per = {}
    for old_idx, new_idx in old_to_new_idx.items():
        if old_idx in train_per_old:
            train_per[new_idx] = train_per_old[old_idx]
        if old_idx in test_per_old:
            test_per[new_idx] = test_per_old[old_idx]
    
    # Apply max_rows_per_file limit if specified
    if max_rows_per_file is not None and max_rows_per_file > 0:
        train_per = {fidx: rows[:max_rows_per_file] for fidx, rows in train_per.items()}
        test_per = {fidx: rows[:max_rows_per_file] for fidx, rows in test_per.items()}
        print(f"Limited to {max_rows_per_file} rows per file (subsetting for faster iteration)")
    
    files_not_found = len(manifest_files) - len(old_to_new_idx)
    if files_not_found > 0:
        print(f"Warning: {files_not_found} files from manifest were not found in current dataset (may be due to different file ordering or missing files).")
    train_ds = SelectedRowsDeepPacketDataset(base, train_per, weight_method=weight_method)
    original_train_total = len(train_ds)
    original_train_counts = dict(train_ds.class_counts)
    if undersample:
        train_ds = train_ds.apply_undersampling(ratio=undersample_ratio, strategy=undersample_strategy)
    test_ds = SelectedRowsDeepPacketDataset(base, test_per, weight_method=weight_method)

    # Ensure flow-disjointness (should be guaranteed by saved split)
    # No validation set in train/test split, so pass None
    _assert_no_flow_overlap_datasets(train_ds, None, test_ds, flow_suffix=flow_suffix)

    # Print distributions and (if applicable) weights, mirroring original split logging
    print("Class distribution (flow-saved) in training set:")
    for class_name, class_idx in train_ds.class_to_idx.items():
        print(f"  {class_name}: {train_ds.class_counts[class_idx]} samples")
    print("Class distribution (flow-saved) in test set:")
    for class_name, class_idx in test_ds.class_to_idx.items():
        print(f"  {class_name}: {test_ds.class_counts[class_idx]} samples")
    if undersample:
        print(f"Undersampling applied (ratio={undersample_ratio}, strategy={undersample_strategy})")
        print(f"  Original training samples: {original_train_total}")
        print(f"  Undersampled training samples: {len(train_ds)}")
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
    test_loader = DataLoader(test_ds, shuffle=True, **dl_args)
    return train_loader, test_loader, train_ds, test_ds


def verify_flow_sidecar_alignment(root: str, flow_suffix: str = ".flow.npy") -> Dict[str, Any]:
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


def verify_flow_class_consistency(ds: FlowAwareDeepPacketDataset) -> Dict[str, Any]:
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


def verify_flow_id_generation(sample_data_file: str, flow_suffix: str = ".flow.npy") -> Dict[str, Any]:
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


def verify_split_statistics(train_ds, val_ds, test_ds) -> Dict[str, Any]:
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

