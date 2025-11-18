#!/usr/bin/env python3
"""
Convert flow-split .npy files to parquet format for Hugging Face datasets.

Reads the output from create_flow_split_dataset_files (train/ and test/ directories
with class subdirectories containing .npy files) and converts them to parquet format
that can be loaded with datasets.load_dataset().

Each .npy file is read once, converted to parquet with "feature" (list) and "label" (int)
columns, and written once. Processing is sequential - one class at a time.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_class_npy_to_parquet(
    npy_path: str,
    output_parquet_path: str,
    class_idx: int,
    split_name: str,
    class_name: str,
    compression: str = 'snappy'
) -> Tuple[str, int]:
    """
    Convert a single class .npy file to parquet format.
    
    Opens the .npy file once, reads all data, converts to parquet format,
    and writes the parquet file once. Uses pyarrow directly for faster writes.
    
    Args:
        npy_path: Path to input .npy file
        output_parquet_path: Path where parquet file should be written
        class_idx: Integer class index (label value)
        split_name: Name of split ("train" or "test") for logging
        class_name: Name of class for logging
        compression: Compression codec ('snappy', 'gzip', 'zstd', 'uncompressed')
    
    Returns:
        Tuple of (output_parquet_path, num_samples)
    """
    logger.debug(f"[{split_name}] Converting {class_name} from {npy_path}")
    
    # Read .npy file once
    arr = np.load(npy_path, mmap_mode=None)  # Load into RAM, not memmap
    
    # Handle both 1D and 2D arrays
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    
    num_samples, feature_size = arr.shape
    
    # Optimize array for PyArrow: only convert if necessary to avoid copies
    # PyArrow can handle float32, but float64 is more standard - convert only if needed
    needs_conversion = False
    if arr.dtype != np.float64:
        needs_conversion = True
    if not arr.flags['C_CONTIGUOUS']:
        needs_conversion = True
    
    if needs_conversion:
        # Single conversion to ensure both C-contiguous and float64
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    
    # Create label array in one go (much faster than per-row)
    # Use np.full with dtype to avoid type conversion overhead
    labels = np.full(num_samples, class_idx, dtype=np.int32)
    
    # Create PyArrow arrays directly from numpy arrays - zero-copy where possible
    # Use FixedSizeListArray for much faster conversion (no Python list overhead)
    # Flatten creates a view (no copy) if array is C-contiguous
    flattened_values = arr.flatten()
    
    # Use zero-copy numpy array conversion - PyArrow can directly use numpy memory
    # This avoids an extra copy when creating the PyArrow array
    feature_values_array = pa.array(flattened_values, type=pa.float64(), from_pandas=False)
    feature_array = pa.FixedSizeListArray.from_arrays(feature_values_array, list_size=feature_size)
    label_array = pa.array(labels, type=pa.int32(), from_pandas=False)
    
    # Create table with named columns
    table = pa.table({
        "feature": feature_array,
        "label": label_array
    })
    
    # Write parquet with optimized settings for maximum write speed
    # Optimize row group size: larger groups = faster writes, but more memory
    # Use adaptive sizing based on number of samples for optimal performance
    if num_samples < 10000:
        row_group_size = num_samples  # Single row group for small files
    elif num_samples < 100000:
        row_group_size = 50000  # Medium files: 2 row groups
    else:
        row_group_size = 100000  # Large files: multiple row groups
    
    pq.write_table(
        table,
        output_parquet_path,
        compression=compression,
        use_dictionary=False,  # Disable dictionary encoding for faster writes
        write_statistics=False,  # Disable statistics for faster writes
        row_group_size=row_group_size,  # Adaptive row group sizing
        use_byte_stream_split=False,  # Disable byte stream split for faster writes
    )
    
    # Clean up memory immediately (in order of size to help GC)
    del table, feature_array, feature_values_array, label_array
    del arr, flattened_values, labels
    
    logger.info(
        f"[{split_name}] {class_name}: {num_samples} samples -> {output_parquet_path}"
    )
    
    return output_parquet_path, num_samples


def convert_split_to_parquet(
    split_dir: str,
    output_split_dir: str,
    split_name: str,
    num_workers: Optional[int] = None,
    compression: str = 'snappy'
) -> Dict[str, int]:
    """
    Convert all class .npy files in a split directory to parquet format.
    
    Processes classes sequentially, one at a time. Each class file is read once
    and written once before moving to the next.
    
    Args:
        split_dir: Input directory containing class subdirectories with .npy files
        output_split_dir: Output directory where parquet files will be written
        split_name: Name of split ("train" or "test")
        num_workers: Ignored (kept for API compatibility)
    
    Returns:
        Dictionary mapping class_name -> num_samples
    """
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    # Pre-create output directory once (avoid repeated checks in per-file function)
    os.makedirs(output_split_dir, exist_ok=True)
    
    # Get all class directories - optimize list comprehension
    split_path = Path(split_dir)
    classes = sorted([
        d.name for d in split_path.iterdir()
        if d.is_dir()
    ])
    
    if not classes:
        raise ValueError(f"No class directories found in {split_dir}")
    
    logger.info(f"[{split_name}] Found {len(classes)} classes: {classes}")
    
    # Prepare tasks: (npy_path, output_path, class_idx, class_name)
    tasks = []
    for class_idx, class_name in enumerate(classes):
        npy_file = split_path / class_name / f"{class_name}.npy"
        
        if not npy_file.exists():
            logger.warning(f"[{split_name}] {npy_file} not found, skipping")
            continue
        
        output_parquet = os.path.join(output_split_dir, f"{class_name}.parquet")
        tasks.append((str(npy_file), output_parquet, class_idx, class_name))
    
    if not tasks:
        raise ValueError(f"No .npy files found in {split_dir}")
    
    logger.info(
        f"[{split_name}] Converting {len(tasks)} classes sequentially..."
    )
    
    # Process sequentially, one at a time
    # Disable GC during conversion loop to avoid GC pauses (re-enable after)
    gc.disable()
    class_counts = {}
    try:
        for npy_path, output_path, class_idx, class_name in tqdm(tasks, desc=f"Converting {split_name}"):
            try:
                output_path, num_samples = convert_class_npy_to_parquet(
                    npy_path,
                    output_path,
                    class_idx,
                    split_name,
                    class_name,
                    compression=compression
                )
                class_counts[class_name] = num_samples
            except Exception as e:
                logger.error(f"[{split_name}] Error processing {class_name}: {e}")
                raise
    finally:
        # Re-enable GC and force collection after batch
        gc.enable()
        gc.collect()
    
    return class_counts


def convert_flow_split_to_parquet(
    flow_split_root: str,
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    compression: str = 'snappy'
) -> Tuple[str, str]:
    """
    Convert flow-split .npy files to parquet format.
    
    Converts both train/ and test/ splits. Each .npy file is read once and
    written to parquet once. Processing is sequential - one class at a time.
    Uses pyarrow directly for optimized write performance.
    
    Args:
        flow_split_root: Root directory containing train/ and test/ subdirectories
                        (output from create_flow_split_dataset_files)
        output_dir: Where to save parquet files (default: flow_split_root + "_parquet")
        num_workers: Ignored (kept for API compatibility)
        compression: Compression codec ('snappy', 'gzip', 'zstd', 'uncompressed')
    
    Returns:
        Tuple of (train_parquet_dir, test_parquet_dir)
    """
    start_time = time.time()
    
    flow_split_root = os.path.abspath(flow_split_root)
    logger.info("=" * 70)
    logger.info("Converting flow-split .npy files to parquet format")
    logger.info(f"  Input: {flow_split_root}")
    
    if output_dir is None:
        output_dir = flow_split_root + "_parquet"
    else:
        output_dir = os.path.abspath(output_dir)
    
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 70)
    
    train_dir = os.path.join(flow_split_root, "train")
    test_dir = os.path.join(flow_split_root, "test")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Create output directories
    train_output_dir = os.path.join(output_dir, "train")
    test_output_dir = os.path.join(output_dir, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Convert train split
    logger.info("\nConverting TRAIN split...")
    train_start = time.time()
    train_counts = convert_split_to_parquet(
        train_dir, train_output_dir, "train", num_workers, compression
    )
    train_time = time.time() - train_start
    logger.info(
        f"Train conversion completed in {train_time:.2f}s "
        f"({sum(train_counts.values())} total samples)"
    )
    
    # Convert test split
    logger.info("\nConverting TEST split...")
    test_start = time.time()
    test_counts = convert_split_to_parquet(
        test_dir, test_output_dir, "test", num_workers, compression
    )
    test_time = time.time() - test_start
    logger.info(
        f"Test conversion completed in {test_time:.2f}s "
        f"({sum(test_counts.values())} total samples)"
    )
    
    # Final memory cleanup after all conversions
    gc.collect()
    
    # Save metadata
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": flow_split_root,
        "output_dir": output_dir,
        "type": "flow_split_parquet",
        "train_counts": train_counts,
        "test_counts": test_counts,
        "total_train": sum(train_counts.values()),
        "total_test": sum(test_counts.values()),
    }
    
    # Try to load original metadata if it exists
    original_metadata_path = os.path.join(flow_split_root, "flow_split_metadata.json")
    if os.path.exists(original_metadata_path):
        try:
            with open(original_metadata_path, "r") as f:
                original_metadata = json.load(f)
                metadata["original_metadata"] = {
                    "test_size": original_metadata.get("test_size"),
                    "seed": original_metadata.get("seed"),
                    "flow_suffix": original_metadata.get("flow_suffix"),
                }
        except Exception as e:
            logger.warning(f"Could not load original metadata: {e}")
    
    metadata_path = os.path.join(output_dir, "parquet_metadata.json")
    logger.info(f"\nSaving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Conversion completed successfully")
    logger.info(f"  Train parquet: {train_output_dir}")
    logger.info(f"  Test parquet: {test_output_dir}")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Total train samples: {sum(train_counts.values())}")
    logger.info(f"  Total test samples: {sum(test_counts.values())}")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info("=" * 70)
    
    print("\nParquet conversion completed:")
    print(f"  Train: {train_output_dir}")
    print(f"  Test: {test_output_dir}")
    print(f"  Total train samples: {sum(train_counts.values())}")
    print(f"  Total test samples: {sum(test_counts.values())}")
    
    return train_output_dir, test_output_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert flow-split .npy files to parquet format for Hugging Face datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default settings (snappy compression)
  python convert_flow_split_to_parquet.py /path/to/flow_split_output

  # Specify output directory
  python convert_flow_split_to_parquet.py /path/to/flow_split_output -o /path/to/parquet_output

  # Use uncompressed for fastest writes (larger files)
  python convert_flow_split_to_parquet.py /path/to/flow_split_output -c uncompressed

  # Note: Processing is sequential (one class at a time)
        """
    )
    
    parser.add_argument(
        "flow_split_root",
        type=str,
        help="Root directory containing train/ and test/ subdirectories with .npy files"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory for parquet files (default: <flow_split_root>_parquet)"
    )
    
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Ignored (kept for API compatibility, processing is sequential)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "-c", "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "zstd", "uncompressed"],
        help="Compression codec for parquet files (default: snappy). "
             "Use 'uncompressed' for fastest writes (larger files)."
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        convert_flow_split_to_parquet(
            args.flow_split_root,
            output_dir=args.output,
            num_workers=args.workers,
            compression=args.compression
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

