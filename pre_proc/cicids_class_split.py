#!/usr/bin/env python3
"""
Convert CICIDS-style pcaps + CSV label files into DeepPacket-ready .npy files
organized by traffic class (one directory per class).

Each output directory matches the structure expected by `deep_pkt.py`, i.e.:

    out_root/
      DDos/
        Monday-WorkingHours_DDos.npy
        Tuesday-WorkingHours_DDos.npy
      BENIGN/
        Monday-WorkingHours_BENIGN.npy
        Tuesday-WorkingHours_BENIGN.npy

Each capture file is processed and written immediately as separate .npy files
in the corresponding class directories. The data loader will handle loading
multiple .npy files per class.
"""

from __future__ import annotations

import argparse
import gc
import logging
import multiprocessing
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path so imports work when running as script
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pre_proc.cicids_proc import (  # type: ignore
    _count_pcap_packets,
    _count_pcapng_packets_scapy,
    _extract_class_label_from_csv_filename,
    _filter_drop_dns,
    _find_matching_csvs_for_pcap,
    _flow_id_from_tuple,
    _is_ipv4,
    _is_ipv6,
    _iter_pcap_packets,
    _iter_pcapng_packets_scapy,
    _load_flow_labels_from_csv,
    _prepare_bytes_for_model_ipv4,
    _prepare_bytes_for_model_ipv6,
    _strip_l2,
    _vectorize_bytes,
    _ipv4_fields,
    _ipv6_fields,
)


def _safe_label_name(label: str) -> str:
    cleaned = label.strip()
    if not cleaned:
        cleaned = "UNKNOWN"
    # Allow letters, numbers, dash, underscore, and dot
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    return cleaned or "UNKNOWN"


def _load_label_map_for_capture(cap: Path, csv_dir: Path) -> Dict[int, str]:
    matching_csvs = _find_matching_csvs_for_pcap(cap, csv_dir)
    combined: Dict[int, str] = {}
    for csv_file in matching_csvs:
        # Use labels from CSV file instead of filename-derived labels
        fmap = _load_flow_labels_from_csv(csv_file, label_override=None)
        if fmap:
            combined.update(fmap)
            # Count unique labels for logging
            unique_labels = set(fmap.values())
            logging.info(
                "Loaded %d flow labels for %s from %s (labels: %s)",
                len(fmap),
                cap.name,
                csv_file.name,
                ", ".join(sorted(unique_labels)),
            )
    return combined


def _count_packets_for_capture(cap_path: Path, pcapng_mode: str) -> int:
    """Count total packets in a capture file without loading them into memory."""
    name_lower = cap_path.name.lower()
    if name_lower.endswith(".pcap"):
        return _count_pcap_packets(cap_path)
    if name_lower.endswith(".pcapng"):
        if pcapng_mode == "skip":
            return 0
        return _count_pcapng_packets_scapy(cap_path)
    return 0


def _iter_packets_for_capture(cap_path: Path, pcapng_mode: str) -> Iterable[bytes]:
    name_lower = cap_path.name.lower()
    if name_lower.endswith(".pcap"):
        logging.info("Reading packets from %s as PCAP", cap_path.name)
        return _iter_pcap_packets(cap_path)
    if name_lower.endswith(".pcapng"):
        if pcapng_mode == "skip":
            logging.warning(
                "Skipping %s because it is pcapng and pcapng_mode=skip", cap_path.name
            )
            return ()
        logging.info("Reading packets from %s as PCAPNG (scapy)", cap_path.name)
        return _iter_pcapng_packets_scapy(cap_path)
    raise RuntimeError(f"Unsupported capture format for {cap_path.name}")


def save_capture_buffers(
    buffers: Dict[str, List[np.ndarray]],
    out_root: Path,
    capture_stem: str,
) -> Dict[str, int]:
    """
    Save buffers from a single capture to disk immediately.
    Each class gets its own npy file with a unique name based on the capture.
    
    Args:
        buffers: Dictionary mapping class labels to lists of packet vectors
        out_root: Root directory for output files
        capture_stem: Stem of the capture filename (for unique naming)
    
    Returns:
        Dictionary mapping class labels to number of packets written
    """
    counts: Dict[str, int] = {}
    
    for label, samples in buffers.items():
        if not samples:
            continue
        
        class_dir = out_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Use capture stem to create unique filename
        filename = f"{capture_stem}_{label}.npy"
        path = class_dir / filename
        
        logging.debug("Writing %d packets to %s", len(samples), path)
        
        # Pre-allocate array for faster writing (avoids np.stack overhead)
        num_samples = len(samples)
        sample_shape = samples[0].shape
        dtype = samples[0].dtype
        data = np.empty((num_samples, *sample_shape), dtype=dtype)
        
        # Fill pre-allocated array
        for i, sample in enumerate(samples):
            data[i] = sample
        
        np.save(path, data, allow_pickle=False)
        logging.info("Wrote %d packets to %s", len(samples), path)
        counts[label] = len(samples)
        
        # Clean up memory
        del data, samples
    
    return counts


def process_capture(
    cap: Path,
    csv_dir: Path,
    max_len: int,
    pcapng_mode: str,
    drop_unknown: bool,
    default_label: str,
    show_progress: bool = True,
) -> Tuple[int, int, int, Dict[str, List[np.ndarray]], int, int]:
    local_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)

    label_map = _load_label_map_for_capture(cap, csv_dir)
    if not label_map and drop_unknown:
        raise RuntimeError(f"No labels found for {cap.name}")
    
    # Count total packets first for accurate progress bar
    total_packets = _count_packets_for_capture(cap, pcapng_mode)
    packet_iter = _iter_packets_for_capture(cap, pcapng_mode)

    total_seen = total_kept = total_unlabeled_packets = 0
    unlabeled_flows: set[int] = set()
    all_flows: set[int] = set()
    
    # Only show progress bar if requested (disabled in parallel mode)
    if show_progress:
        packet_iter = tqdm(packet_iter, desc=f"Processing {cap.name}", unit="pkt", total=total_packets)
    
    for raw in packet_iter:
        total_seen += 1
        l3 = _strip_l2(raw)

        if _is_ipv4(l3):
            proto, src, dst, _ihl, ports, _payload, _tcp_flags = _ipv4_fields(l3)
            if proto == -1:
                continue
            if _filter_drop_dns(proto, ports):
                continue
            sport, dport = ports
            fid = _flow_id_from_tuple(proto, src, dst, sport or 0, dport or 0)
            l3_for_model = _prepare_bytes_for_model_ipv4(l3)
        elif _is_ipv6(l3):
            proto, src, dst, ports, _payload = _ipv6_fields(l3)
            if proto == -1:
                continue
            if _filter_drop_dns(proto, ports):
                continue
            sport, dport = ports
            fid = _flow_id_from_tuple(proto, src, dst, sport or 0, dport or 0)
            l3_for_model = _prepare_bytes_for_model_ipv6(l3)
        else:
            continue

        all_flows.add(fid)
        label = label_map.get(fid)
        if label is None:
            if drop_unknown:
                continue
            if fid not in unlabeled_flows:
                unlabeled_flows.add(fid)
            label = default_label
            total_unlabeled_packets += 1

        safe_label = _safe_label_name(label)
        vec = _vectorize_bytes(l3_for_model, max_len)
        local_buffers[safe_label].append(vec)
        total_kept += 1
    return total_seen, total_kept, total_unlabeled_packets, local_buffers, len(all_flows), len(unlabeled_flows)


def process_capture_worker(
    cap_path: str,
    csv_dir: str,
    out_root: str,
    max_len: int,
    pcapng_mode: str,
    drop_unknown: bool,
    default_label: str,
) -> Tuple[str, int, int, int, Dict[str, int], int, int]:
    """
    Worker function that processes a single capture end-to-end:
    reads, processes, and writes the results.
    
    Returns:
        Tuple of (cap_name, seen, kept, unlabeled, class_counts, total_flows, flows_without_labels)
    """
    cap = Path(cap_path)
    csv_dir_path = Path(csv_dir)
    out_root_path = Path(out_root)
    
    try:
        # Process the capture
        seen, kept, unlabeled, buffers, total_flows, flows_without_labels = process_capture(
            cap=cap,
            csv_dir=csv_dir_path,
            max_len=max_len,
            pcapng_mode=pcapng_mode,
            drop_unknown=drop_unknown,
            default_label=default_label,
            show_progress=False,  # Disable per-file progress in parallel mode
        )
        
        # Write buffers immediately
        capture_stem = cap.stem
        class_counts = save_capture_buffers(buffers, out_root_path, capture_stem)
        
        # Clean up
        del buffers
        gc.collect()
        
        return (cap.name, seen, kept, unlabeled, class_counts, total_flows, flows_without_labels)
    except Exception as e:
        logging.error(f"Error processing {cap.name}: {e}", exc_info=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CICIDS pcaps into class-sorted .npy directories"
    )
    parser.add_argument(
        "--in-dir",
        required=True,
        type=Path,
        help="Directory containing pcaps and CSV label files",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Output directory for class-organized .npy files",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1500,
        help="Vector length (default 1500 to match DeepPacket)",
    )
    parser.add_argument(
        "--pcapng-mode",
        choices=["scapy", "skip"],
        default="scapy",
        help="How to handle .pcapng files",
    )
    parser.add_argument(
        "--drop-unknown-labels",
        action="store_true",
        help="Drop packets whose flows lack labels (default: keep using default-label)",
    )
    parser.add_argument(
        "--default-label",
        type=str,
        default="BENIGN",
        help="Label to use for unlabeled packets when not dropping",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    in_dir = args.in_dir.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pcaps = sorted(list(in_dir.glob("*.pcap")) + list(in_dir.glob("*.pcapng")))
    if not pcaps:
        raise RuntimeError(f"No .pcap/.pcapng files found under {in_dir}")

    num_workers = args.num_workers or multiprocessing.cpu_count()
    logging.info("Processing %d pcap files in parallel with %d workers", len(pcaps), num_workers)

    total_seen = total_kept = total_unlabeled = 0
    all_class_counts: Counter[str] = Counter()

    # Prepare arguments for worker function
    worker_args = [
        (
            str(cap),
            str(in_dir),
            str(out_root),
            args.max_len,
            args.pcapng_mode,
            args.drop_unknown_labels,
            args.default_label,
        )
        for cap in pcaps
    ]

    # Process in parallel - each worker handles one capture end-to-end
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.starmap(process_capture_worker, worker_args),
                desc="Processing captures",
                total=len(pcaps),
                unit="file",
            )
        )

    # Aggregate results
    for cap_name, seen, kept, unlabeled, class_counts, total_flows, flows_without_labels in results:
        total_seen += seen
        total_kept += kept
        total_unlabeled += unlabeled
        all_class_counts.update(class_counts)
        
        logging.info(
            "[%s] seen=%d kept=%d (%.1f%%) | flows: total=%d without_labels=%d",
            cap_name,
            seen,
            kept,
            (100.0 * kept / seen) if seen else 0.0,
            total_flows,
            flows_without_labels,
        )

    logging.info("Finished. Total packets seen=%d kept=%d", total_seen, total_kept)
    if total_seen:
        logging.info(
            "Unlabeled packets: %d (%.2f%% of kept packets)",
            total_unlabeled,
            100.0 * total_unlabeled / total_kept if total_kept else 0.0,
        )
    for label, count in sorted(all_class_counts.items()):
        logging.info("  %s: %d packets", label, count)


if __name__ == "__main__":
    main()

