"""Flow-based dataset splitting and verification utilities."""

import os
import glob
import bisect
import json
import gc
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

# Set up logger for this module
logger = logging.getLogger(__name__)

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
            # Check if dataset has already-loaded flow arrays (FlowAwareDeepPacketDataset)
            if hasattr(ds, "_ensure_flow_mm"):
                try:
                    mm = ds._ensure_flow_mm(fidx)
                    if mm is not None:
                        # Use already-loaded flow array (no file I/O)
                        for v in np.unique(mm):
                            flow_ids.add(int(v))
                        continue
                except Exception:
                    pass
            # Fallback: try to load from file (only if not already loaded)
            fpath = path[:-4] + flow_suffix if path.endswith(".npy") else path + flow_suffix
            if os.path.exists(fpath):
                try:
                    mm = np.load(fpath, mmap_mode=None)  # Load into RAM, not memmap
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
    Uses already-loaded flow arrays from the dataset (no file I/O).
    """
    by_flow, flow_to_class = {}, {}
    for file_idx, (path, cls_idx) in enumerate(ds.files):
        # IMPORTANT: respect ds.counts (caps like max_rows_per_file), not full file length
        nrows = int(getattr(ds, "counts", [0])[file_idx])
        if nrows <= 0:
            continue
        # Use already-loaded flow arrays from dataset (no file I/O)
        fmm = ds._ensure_flow_mm(file_idx)
        if fmm is not None:
            if (fmm.ndim != 1) or (len(fmm) < nrows):
                raise RuntimeError(f"Flow array length mismatch: file {file_idx} (expected {nrows}, got {len(fmm)})")
        base = ds.offsets[file_idx]
        for i in range(nrows):
            gidx = base + i
            fid = int(fmm[i]) if fmm is not None else int(gidx)
            by_flow.setdefault(fid, []).append(gidx)
            flow_to_class[fid] = cls_idx
    return by_flow, flow_to_class


def stratified_flow_split(by_flow, flow_to_class, valid_size=0.1, test_size=0.1, seed=2018):
    """
    Stratify by class at the **packet** level (not flow level), then expand flows to row indices.
    This ensures train, validation, and test sets have similar packet counts while maintaining
    flow integrity (no flow overlap across splits).
    
    Uses a greedy algorithm to assign flows to splits to achieve target packet counts per class.
    """
    rng = np.random.RandomState(seed)
    class2flows: Dict[int, List[int]] = {}
    # Also track flow sizes (packet counts) for each flow
    flow_sizes: Dict[int, int] = {}
    
    for fid, c in flow_to_class.items():
        class2flows.setdefault(c, []).append(fid)
        flow_sizes[fid] = len(by_flow[fid])
    
    train_idx, val_idx, test_idx = [], [], []
    
    for c, fids in class2flows.items():
        # Calculate total packets for this class
        total_packets = sum(flow_sizes[fid] for fid in fids)
        target_test_packets = int(np.floor(test_size * total_packets))
        target_val_packets = int(np.floor(valid_size * total_packets))
        target_train_packets = total_packets - target_test_packets - target_val_packets
        
        # Shuffle flows to randomize assignment
        fids = fids.copy()
        rng.shuffle(fids)
        
        # Sort flows by size (descending) to help with greedy assignment
        # This helps avoid situations where we need to assign many small flows
        fids_sorted = sorted(fids, key=lambda fid: flow_sizes[fid], reverse=True)
        
        # Greedy assignment: assign flows to splits to get as close as possible to target packet counts
        # We prioritize test and val to ensure they get their target amounts first
        train_f = []
        val_f = []
        test_f = []
        train_packets = 0
        val_packets = 0
        test_packets = 0
        
        for fid in fids_sorted:
            size = flow_sizes[fid]
            
            # Calculate the improvement (reduction in distance from target) for each split
            train_improvement = abs(train_packets - target_train_packets) - abs((train_packets + size) - target_train_packets)
            val_improvement = abs(val_packets - target_val_packets) - abs((val_packets + size) - target_val_packets)
            test_improvement = abs(test_packets - target_test_packets) - abs((test_packets + size) - target_test_packets)
            
            # Prefer splits that are still below target, then choose the one with best improvement
            # Priority: test > val > train (to ensure smaller splits get filled first)
            if test_packets < target_test_packets:
                if val_packets < target_val_packets:
                    # Both test and val need more packets - choose the one with better improvement
                    if test_improvement >= val_improvement:
                        test_f.append(fid)
                        test_packets += size
                    else:
                        val_f.append(fid)
                        val_packets += size
                else:
                    # Only test needs more packets
                    test_f.append(fid)
                    test_packets += size
            elif val_packets < target_val_packets:
                # Only val needs more packets
                val_f.append(fid)
                val_packets += size
            else:
                # Both test and val have reached their targets, assign to best fit
                if test_improvement >= val_improvement and test_improvement >= train_improvement:
                    test_f.append(fid)
                    test_packets += size
                elif val_improvement >= train_improvement:
                    val_f.append(fid)
                    val_packets += size
                else:
                    train_f.append(fid)
                    train_packets += size
        
        # Expand flows to row indices
        for fid in train_f:
            train_idx.extend(by_flow[fid])
        for fid in val_f:
            val_idx.extend(by_flow[fid])
        for fid in test_f:
            test_idx.extend(by_flow[fid])
    
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def stratified_flow_train_test_split(by_flow, flow_to_class, test_size=0.2, seed=2018):
    """
    Split flow IDs into train/test, stratified by class at the **packet** level, then expand to row indices.
    This ensures train and test sets have similar packet counts while maintaining flow integrity.
    Returns (train_row_indices, test_row_indices) as numpy arrays.
    """
    rng = np.random.RandomState(seed)
    class2flows: Dict[int, List[int]] = {}
    # Also track flow sizes (packet counts) for each flow
    flow_sizes: Dict[int, int] = {}
    
    for fid, c in flow_to_class.items():
        class2flows.setdefault(c, []).append(fid)
        flow_sizes[fid] = len(by_flow[fid])

    train_idx, test_idx = [], []
    for c, fids in class2flows.items():
        # Calculate total packets for this class
        total_packets = sum(flow_sizes[fid] for fid in fids)
        target_test_packets = int(np.floor(test_size * total_packets))
        target_train_packets = total_packets - target_test_packets
        
        # Shuffle flows to randomize assignment
        fids = fids.copy()
        rng.shuffle(fids)
        
        # Sort flows by size (descending) to help with greedy assignment
        fids_sorted = sorted(fids, key=lambda fid: flow_sizes[fid], reverse=True)
        
        # Greedy assignment: assign flows to splits to get as close as possible to target packet counts
        train_f = []
        test_f = []
        train_packets = 0
        test_packets = 0
        
        for fid in fids_sorted:
            size = flow_sizes[fid]
            
            # Calculate the improvement (reduction in distance from target) for each split
            train_improvement = abs(train_packets - target_train_packets) - abs((train_packets + size) - target_train_packets)
            test_improvement = abs(test_packets - target_test_packets) - abs((test_packets + size) - target_test_packets)
            
            # Prefer test if it's still below target, otherwise choose best fit
            if test_packets < target_test_packets:
                test_f.append(fid)
                test_packets += size
            elif test_improvement >= train_improvement:
                test_f.append(fid)
                test_packets += size
            else:
                train_f.append(fid)
                train_packets += size
        
        for fid in train_f:
            train_idx.extend(by_flow[fid])
        for fid in test_f:
            test_idx.extend(by_flow[fid])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def rows_to_file_indices(ds, rows):
    """
    Convert global row indices into per-file relative indices.
    Returns a dict mapping file_idx -> list[row_idx].
    """
    per_file = defaultdict(list)
    for gidx in rows.tolist():
        fidx = bisect.bisect_right(ds.offsets, gidx) - 1
        fidx = min(max(fidx, 0), len(ds.offsets) - 2)
        ridx = gidx - ds.offsets[fidx]
        per_file[fidx].append(int(ridx))
    return per_file


def create_flow_split_dataset_files(
    root: str,
    output_root: Optional[str] = None,
    test_size: float = 0.2,
    seed: int = 2018,
    flow_suffix: str = ".flow.npy",
    chunk_size: int = 10000,
    num_workers: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Create permanent flow-split dataset by writing new .npy files.

    This variant loads each source .npy file exactly once (now in parallel),
    keeps all split samples in RAM, and writes one consolidated file per
    class/split at the very end (also in parallel, single open per file).
    The chunk_size argument remains for backward compatibility (unused),
    while num_workers controls the concurrency level for both stages.
    """
    function_start_time = time.time()
    logger.info("="*70)
    logger.info("Starting create_flow_split_dataset_files")
    logger.info(f"  root: {root}")
    logger.info(f"  output_root: {output_root}")
    logger.info(f"  test_size: {test_size}")
    logger.info(f"  seed: {seed}")
    logger.info(f"  flow_suffix: {flow_suffix}")
    logger.info(f"  chunk_size: {chunk_size} (unused)")
    logger.info(f"  num_workers: {num_workers} (parallel readers/writers)")
    logger.info("="*70)

    if output_root is None:
        output_root = os.path.abspath(root) + "_split"
    else:
        output_root = os.path.abspath(output_root)
    logger.info(f"Output directory: {output_root}")

    logger.info("Loading base dataset...")
    load_start = time.time()
    base = FlowAwareDeepPacketDataset(root, max_rows_per_file=None, flow_suffix=flow_suffix)
    load_time = time.time() - load_start
    logger.info(
        f"Loaded base dataset with {base.total} total samples across {len(base.files)} files in {load_time:.2f}s"
    )
    print(f"Loaded base dataset with {base.total} total samples across {len(base.files)} files")

    logger.info("Grouping indices by flow...")
    group_start = time.time()
    by_flow, flow_to_class = group_indices_by_flow(base)
    group_time = time.time() - group_start
    logger.info(f"Grouped {len(by_flow)} flows in {group_time:.2f}s")

    logger.info(f"Splitting flows into train/test (test_size={test_size})...")
    split_start = time.time()
    train_rows, test_rows = stratified_flow_train_test_split(
        by_flow, flow_to_class, test_size=test_size, seed=seed
    )
    split_time = time.time() - split_start
    logger.info(f"Split into {len(train_rows)} train rows and {len(test_rows)} test rows in {split_time:.2f}s")

    logger.info("Grouping rows by file for train/test splits...")
    group_start = time.time()
    train_by_file = rows_to_file_indices(base, train_rows)
    test_by_file = rows_to_file_indices(base, test_rows)
    group_time = time.time() - group_start
    logger.info(
        f"Grouped rows by file in {group_time:.2f}s (train files: {len(train_by_file)}, test files: {len(test_by_file)})"
    )

    del train_rows, test_rows, by_flow, flow_to_class
    gc.collect()

    split_storage = {"train": defaultdict(list), "test": defaultdict(list)}
    split_counts = {"train": defaultdict(int), "test": defaultdict(int)}
    split_locks = {
        "train": defaultdict(threading.Lock),
        "test": defaultdict(threading.Lock),
    }

    worker_count = num_workers if num_workers and num_workers > 0 else min(32, (os.cpu_count() or 1))
    if worker_count < 1:
        worker_count = 1

    logger.info(
        f"Loading each source file once (parallelized across {worker_count} worker(s)) "
        "and caching split contents in memory..."
    )

    def process_file(fidx: int, train_rows_for_file, test_rows_for_file):
        path, class_idx = base.files[fidx]
        class_name = base.classes[class_idx]
        logger.debug(
            f"[reader] Loading file {fidx} ({os.path.basename(path)}) for class {class_name}"
        )
        arr = np.load(path, mmap_mode="r")
        try:
            if train_rows_for_file:
                selected_train = arr[train_rows_for_file]
                with split_locks["train"][class_name]:
                    split_storage["train"][class_name].append(selected_train.copy())
                    split_counts["train"][class_name] += selected_train.shape[0]
                logger.debug(
                    f"[reader] File {fidx}: added {selected_train.shape[0]} train samples"
                )
            if test_rows_for_file:
                selected_test = arr[test_rows_for_file]
                with split_locks["test"][class_name]:
                    split_storage["test"][class_name].append(selected_test.copy())
                    split_counts["test"][class_name] += selected_test.shape[0]
                logger.debug(
                    f"[reader] File {fidx}: added {selected_test.shape[0]} test samples"
                )
        finally:
            del arr

    read_futures = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for fidx in range(len(base.files)):
            train_rows_for_file = train_by_file.get(fidx)
            test_rows_for_file = test_by_file.get(fidx)
            if not train_rows_for_file and not test_rows_for_file:
                continue
            read_futures.append(
                executor.submit(process_file, fidx, train_rows_for_file, test_rows_for_file)
            )
        for future in as_completed(read_futures):
            future.result()

    del train_by_file, test_by_file
    gc.collect()

    train_dir = os.path.join(output_root, "train")
    test_dir = os.path.join(output_root, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    logger.info("="*70)
    logger.info("Writing split files (single write per class, parallelized)")
    logger.info("="*70)

    split_dirs = {"train": train_dir, "test": test_dir}

    def write_class_split(split_name: str, class_name: str, samples_list: List[np.ndarray]):
        base_split_dir = split_dirs[split_name]
        class_dir = os.path.join(base_split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        logger.debug(
            f"[writer] {split_name.upper()} - {class_name}: concatenating {len(samples_list)} arrays"
        )
        data = (
            np.concatenate(samples_list, axis=0)
            if len(samples_list) > 1
            else samples_list[0]
        )
        out_path = os.path.join(class_dir, f"{class_name}.npy")
        np.save(out_path, data)
        logger.info(
            f"  {split_name.upper()} - {class_name}: {data.shape[0]} samples written to {out_path}"
        )
        for arr in samples_list:
            del arr

    write_futures = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for split_name, class_data in split_storage.items():
            logger.info(f"Writing {split_name} split to disk...")
            for class_name, samples_list in list(class_data.items()):
                if not samples_list:
                    logger.debug(
                        f"  {split_name.upper()} - {class_name}: no samples, skipping file write"
                    )
                    continue
                write_futures.append(
                    executor.submit(write_class_split, split_name, class_name, samples_list)
                )
        for future in as_completed(write_futures):
            future.result()

    for class_data in split_storage.values():
        class_data.clear()
    gc.collect()

    train_counts = {cls: split_counts["train"].get(cls, 0) for cls in base.classes}
    test_counts = {cls: split_counts["test"].get(cls, 0) for cls in base.classes}

    logger.info("Saving metadata...")
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_root": os.path.abspath(root),
        "output_root": output_root,
        "type": "flow_split_dataset_files",
        "test_size": float(test_size),
        "seed": seed,
        "flow_suffix": flow_suffix,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "total_train": sum(train_counts.values()),
        "total_test": sum(test_counts.values()),
    }

    metadata_path = os.path.join(output_root, "flow_split_metadata.json")
    logger.info(f"Writing metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved successfully")

    total_time = time.time() - function_start_time
    logger.info("="*70)
    logger.info("create_flow_split_dataset_files COMPLETED")
    logger.info(f"  Train: {train_dir}")
    logger.info(f"  Test: {test_dir}")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Total train samples: {sum(train_counts.values())}")
    logger.info(f"  Total test samples: {sum(test_counts.values())}")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info("="*70)

    print("\nFlow-split dataset created:")
    print(f"  Train: {train_dir}")
    print(f"  Test: {test_dir}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Total train samples: {sum(train_counts.values())}")
    print(f"  Total test samples: {sum(test_counts.values())}")

    return train_dir, test_dir



def has_pre_split_dataset(root: str) -> bool:
    """
    Check if root directory contains pre-split train/ and test/ subdirectories.
    
    Args:
        root: Directory to check
        
    Returns:
        True if both train/ and test/ subdirectories exist and contain class folders
    """
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        return False
    
    # Check that both directories have at least one class folder
    train_classes = [d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))]
    test_classes = [d for d in os.listdir(test_dir) 
                     if os.path.isdir(os.path.join(test_dir, d))]
    
    return len(train_classes) > 0 and len(test_classes) > 0


def load_pre_split_dataset(
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    weight_method: str = "balanced",
    handle_imbalance: bool = False,
    undersample: bool = False,
    undersample_ratio: float = 0.1,
    undersample_strategy: str = "random",
    pin_memory: bool = False,
):
    """
    Load pre-split dataset from root/train/ and root/test/ directories.
    
    Args:
        root: Root directory containing train/ and test/ subdirectories
        batch_size: Batch size for data loaders
        num_workers: Number of data loading workers
        weight_method: Method for calculating class weights
        handle_imbalance: Enable class imbalance handling
        undersample: Enable undersampling
        undersample_ratio: Undersampling ratio
        undersample_strategy: Undersampling strategy
        pin_memory: Enable pin_memory for GPU
        
    Returns:
        Tuple of (train_loader, test_loader, train_ds, test_ds)
    """
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Pre-split dataset not found. Expected {train_dir} and {test_dir} to exist. "
            f"Use create_flow_split_dataset_files() to create them."
        )
    
    # Create datasets - no flow splitting needed, just load the files
    train_ds = DeepPacketNPYDataset(train_dir, weight_method=weight_method)
    test_ds = DeepPacketNPYDataset(test_dir, weight_method=weight_method)
    
    # Apply undersampling to training set if enabled
    if undersample:
        print(f"\nApplying undersampling (ratio={undersample_ratio}, strategy={undersample_strategy})...")
        train_ds_original = train_ds
        train_ds = train_ds.apply_undersampling(ratio=undersample_ratio, strategy=undersample_strategy)
        print(f"Original training samples: {train_ds_original.total}")
        print(f"Undersampled training samples: {train_ds.total}\n")
    
    # Print class distribution
    print("Class distribution in training set:")
    for class_name, class_idx in train_ds.class_to_idx.items():
        count = train_ds.class_counts[class_idx]
        print(f"  {class_name}: {count} samples")
    print("Class distribution in test set:")
    for class_name, class_idx in test_ds.class_to_idx.items():
        count = test_ds.class_counts[class_idx]
        print(f"  {class_name}: {count} samples")
    
    if handle_imbalance:
        print(f"Class weights (method: {weight_method}):")
        for class_name, class_idx in train_ds.class_to_idx.items():
            weight = train_ds.class_weights[class_idx]
            print(f"  {class_name}: {weight:.4f}")
    
    # Optimize DataLoader settings
    dl_args = dict(
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        dl_args['persistent_workers'] = True
        dl_args['prefetch_factor'] = 2
    
    # Create data loaders
    if handle_imbalance:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **dl_args)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **dl_args)
    
    test_loader = DataLoader(test_ds, shuffle=True, **dl_args)
    
    return train_loader, test_loader, train_ds, test_ds


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
    pin_memory: bool = False,
):
    """Split dataset by flow IDs, ensuring no flow overlap across splits."""
    # Build a flow-aware base dataset to discover counts/offsets/files.
    base = FlowAwareDeepPacketDataset(root, max_rows_per_file=max_rows_per_file, weight_method=weight_method, flow_suffix=flow_suffix)
    by_flow, flow_to_class = group_indices_by_flow(base)
    train_rows, val_rows, test_rows = stratified_flow_split(by_flow, flow_to_class, valid_size=valid_size, test_size=test_size, seed=2018)

    # Verify that splits are disjoint at the global row index level
    train_set = set(train_rows)
    val_set = set(val_rows)
    test_set = set(test_rows)
    inter_tv = train_set & val_set
    inter_tt = train_set & test_set
    inter_vt = val_set & test_set
    if inter_tv or inter_tt or inter_vt:
        raise RuntimeError(
            f"Global row index overlap detected in stratified_flow_split:\n"
            f"  train ∩ val: {len(inter_tv)} rows\n"
            f"  train ∩ test: {len(inter_tt)} rows\n"
            f"  val ∩ test: {len(inter_vt)} rows"
        )

    def rows_to_per_file(ds, rows):
        per = {i: [] for i in range(len(ds.files))}
        for g in rows.tolist():
            # Find the file index: offsets[i] <= g < offsets[i+1]
            # bisect_right returns the insertion point, so we subtract 1 to get the file index
            fidx = bisect.bisect_right(ds.offsets, g) - 1
            # Clamp to valid range [0, len(files)-1]
            fidx = min(max(fidx, 0), len(ds.offsets) - 2)
            ridx = g - ds.offsets[fidx]
            per[fidx].append(int(ridx))
        # Deduplicate and sort row indices per file to ensure no duplicates
        # This prevents flow overlap issues when collecting flow IDs
        for fidx in per:
            per[fidx] = sorted(set(per[fidx]))
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

    # Print detailed distribution info including flow-level and packet-level statistics
    print("\n" + "="*70)
    print("FLOW-BASED SPLIT STATISTICS")
    print("="*70)
    
    # Calculate flow-level statistics
    def get_flow_stats(rows, by_flow, flow_to_class):
        """Get flow and packet counts per class for a split."""
        rows_set = set(rows)  # Convert to set for O(1) lookup
        flow_ids_in_split = set()
        row_to_flow = {}  # Map row index to flow ID
        
        # Build reverse mapping: row -> flow_id
        for fid, flow_rows in by_flow.items():
            for row_idx in flow_rows:
                if row_idx in rows_set:
                    flow_ids_in_split.add(fid)
                    row_to_flow[row_idx] = fid
        
        class_flow_counts = {}
        class_packet_counts = {}
        for fid in flow_ids_in_split:
            cls = flow_to_class[fid]
            class_flow_counts[cls] = class_flow_counts.get(cls, 0) + 1
            # Count packets for this flow that are in the split
            flow_packets = sum(1 for r in by_flow[fid] if r in rows_set)
            class_packet_counts[cls] = class_packet_counts.get(cls, 0) + flow_packets
        
        return class_flow_counts, class_packet_counts
    
    train_flow_counts, train_packet_counts = get_flow_stats(train_rows, by_flow, flow_to_class)
    val_flow_counts, val_packet_counts = get_flow_stats(val_rows, by_flow, flow_to_class)
    test_flow_counts, test_packet_counts = get_flow_stats(test_rows, by_flow, flow_to_class)
    
    print("\nFLOW-LEVEL distribution (stratified at flow level):")
    print("  Class | Train Flows | Val Flows | Test Flows")
    print("  " + "-"*50)
    for class_name, class_idx in base.class_to_idx.items():
        train_f = train_flow_counts.get(class_idx, 0)
        val_f = val_flow_counts.get(class_idx, 0)
        test_f = test_flow_counts.get(class_idx, 0)
        print(f"  {class_name:20s} | {train_f:11d} | {val_f:9d} | {test_f:10d}")
    
    print("\nPACKET-LEVEL distribution (actual samples in each split):")
    print("  Class | Train Packets | Val Packets | Test Packets | Train % | Val % | Test %")
    print("  " + "-"*80)
    total_train = sum(train_packet_counts.values())
    total_val = sum(val_packet_counts.values())
    total_test = sum(test_packet_counts.values())
    for class_name, class_idx in base.class_to_idx.items():
        train_p = train_packet_counts.get(class_idx, 0)
        val_p = val_packet_counts.get(class_idx, 0)
        test_p = test_packet_counts.get(class_idx, 0)
        train_pct = 100.0 * train_p / total_train if total_train > 0 else 0
        val_pct = 100.0 * val_p / total_val if total_val > 0 else 0
        test_pct = 100.0 * test_p / total_test if total_test > 0 else 0
        print(f"  {class_name:20s} | {train_p:13d} | {val_p:11d} | {test_p:12d} | {train_pct:6.2f}% | {val_pct:5.2f}% | {test_pct:5.2f}%")
    
    print(f"\n  TOTAL                | {total_train:13d} | {total_val:11d} | {total_test:12d}")
    
    # Note about packet-level vs flow-level distribution
    print("\n" + "="*70)
    print("NOTE: Flow-based splitting is stratified at the PACKET level per class.")
    print("This ensures balanced packet counts between train/val/test splits while")
    print("maintaining flow integrity (no flow overlap across splits).")
    print("="*70 + "\n")
    
    # Also print the simpler version for compatibility
    print("Class distribution (packet-level) in training set:")
    for class_name, class_idx in train_ds.class_to_idx.items():
        print(f"  {class_name}: {train_ds.class_counts[class_idx]} samples")
    print("Class distribution (packet-level) in validation set:")
    for class_name, class_idx in val_ds.class_to_idx.items():
        print(f"  {class_name}: {val_ds.class_counts[class_idx]} samples")
    print("Class distribution (packet-level) in test set:")
    for class_name, class_idx in test_ds.class_to_idx.items():
        print(f"  {class_name}: {test_ds.class_counts[class_idx]} samples")

    if handle_imbalance:
        print(f"Class weights (method: {weight_method}):")
        for class_name, class_idx in train_ds.class_to_idx.items():
            print(f"  {class_name}: {train_ds.class_weights[class_idx]:.4f}")

    # Optimize DataLoader settings: pin_memory for GPU, persistent_workers for efficiency
    # These optimizations significantly improve GPU training performance
    dl_args = dict(
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory,  # Faster CPU->GPU transfer when using GPU
    )
    # Add advanced optimizations (PyTorch 1.7+)
    if num_workers > 0:
        dl_args['persistent_workers'] = True  # Keep workers alive between epochs
        dl_args['prefetch_factor'] = 2  # Prefetch 2 batches per worker
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

    # Print basic packet-level distribution info for compatibility
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
    pin_memory: bool = False,
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

    # Optimize DataLoader settings: pin_memory for GPU, persistent_workers for efficiency
    # These optimizations significantly improve GPU training performance
    dl_args = dict(
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory,  # Faster CPU->GPU transfer when using GPU
    )
    # Add advanced optimizations (PyTorch 1.7+)
    if num_workers > 0:
        dl_args['persistent_workers'] = True  # Keep workers alive between epochs
        dl_args['prefetch_factor'] = 2  # Prefetch 2 batches per worker
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
    Uses already-loaded flow arrays from the dataset (no file I/O).
    """
    flow_to_classes = {}
    inconsistent_flows_set = set()  # Track unique inconsistent flow IDs
    inconsistent_flow_examples = []  # Track examples for reporting
    
    for file_idx, (path, cls_idx) in enumerate(ds.files):
        nrows = int(getattr(ds, "counts", [0])[file_idx])
        if nrows <= 0:
            continue
        
        # Use already-loaded flow arrays from dataset (no file I/O)
        try:
            fmm = ds._ensure_flow_mm(file_idx)
            if fmm is None:
                continue
            
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

