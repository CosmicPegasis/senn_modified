#!/usr/bin/env python3
"""
Convert CICIDS-style pcaps + CSV label files into DeepPacket-ready .npy files
organized by traffic class (one directory per class).

Each output directory matches the structure expected by `deep_pkt.py`, i.e.:

    out_root/
      DDos/
        DDos.npy
      BENIGN/
        BENIGN.npy

Packets are grouped purely by label; capture day/order is ignored.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from pre_proc.cicids_proc import (  # type: ignore
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
        class_label = _extract_class_label_from_csv_filename(csv_file)
        fmap = _load_flow_labels_from_csv(csv_file, label_override=class_label)
        if fmap:
            combined.update(fmap)
            logging.info(
                "Loaded %d flow labels for %s from %s (class=%s)",
                len(fmap),
                cap.name,
                csv_file.name,
                class_label,
            )
    return combined


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


class ClassAccumulator:
    """
    Hold all per-class packet vectors in memory until a final flush.
    """

    def __init__(self) -> None:
        self.buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.counts: Counter[str] = Counter()

    def add(self, label: str, vec: np.ndarray) -> None:
        self.buffers[label].append(vec)
        self.counts[label] += 1

    def merge(self, other: Dict[str, List[np.ndarray]]) -> None:
        for label, samples in other.items():
            if not samples:
                continue
            self.buffers[label].extend(samples)
            self.counts[label] += len(samples)

    def save(self, out_root: Path) -> None:
        for label, samples in self.buffers.items():
            if not samples:
                continue
            class_dir = out_root / label
            class_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{label}.npy"
            path = class_dir / filename
            np.save(path, np.stack(samples, axis=0))
            logging.info("Wrote %d packets to %s", len(samples), path)


def process_capture(
    cap: Path,
    csv_dir: Path,
    max_len: int,
    pcapng_mode: str,
    drop_unknown: bool,
    default_label: str,
) -> Tuple[int, int, int, Dict[str, List[np.ndarray]]]:
    local_buffers: Dict[str, List[np.ndarray]] = defaultdict(list)

    label_map = _load_label_map_for_capture(cap, csv_dir)
    if not label_map and drop_unknown:
        raise RuntimeError(f"No labels found for {cap.name}")
    packet_iter = _iter_packets_for_capture(cap, pcapng_mode)

    total_seen = total_kept = total_unlabeled_packets = 0
    unlabeled_flows: set[int] = set()
    for raw in tqdm(packet_iter, desc=f"Processing {cap.name}", unit="pkt"):
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

        label = label_map.get(fid)
        if label is None:
            if drop_unknown:
                continue
            if fid not in unlabeled_flows:
                unlabeled_flows.add(fid)
                logging.warning(
                    "Missing label for flow %016x in %s; assigning default '%s'",
                    fid,
                    cap.name,
                    default_label,
                )
            label = default_label
            total_unlabeled_packets += 1

        safe_label = _safe_label_name(label)
        vec = _vectorize_bytes(l3_for_model, max_len)
        local_buffers[safe_label].append(vec)
        total_kept += 1
    return total_seen, total_kept, total_unlabeled_packets, local_buffers


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

    accumulator = ClassAccumulator()
    total_seen = total_kept = total_unlabeled = 0

    for cap in pcaps:
        seen, kept, unlabeled, buffers = process_capture(
            cap=cap,
            csv_dir=in_dir,
            max_len=args.max_len,
            pcapng_mode=args.pcapng_mode,
            drop_unknown=args.drop_unknown_labels,
            default_label=args.default_label,
        )
        total_seen += seen
        total_kept += kept
        total_unlabeled += unlabeled
        accumulator.merge(buffers)
        logging.info(
            "[%s] seen=%d kept=%d (%.1f%%)",
            cap.name,
            seen,
            kept,
            (100.0 * kept / seen) if seen else 0.0,
        )

    accumulator.save(out_root)

    logging.info("Finished. Total packets seen=%d kept=%d", total_seen, total_kept)
    if total_seen:
        logging.info(
            "Unlabeled packets: %d (%.2f%% of kept packets)",
            total_unlabeled,
            100.0 * total_unlabeled / total_kept if total_kept else 0.0,
        )
    for label, count in sorted(accumulator.counts.items()):
        logging.info("  %s: %d packets", label, count)


if __name__ == "__main__":
    main()

