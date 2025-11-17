#!/usr/bin/env python3
"""
preproc_cicids2017.py
A DROP-IN preprocessor for CICIDS2017-style folders (pcap files per day).
Behaves like preproc_flow_new.py but:
 - expects a flat folder of pcaps (no class subfolders)
 - optionally reads an accompanying CSV per pcap to attach per-flow labels
 - writes .npy data chunks + .flow.npy + optional .label.npy
"""
from pathlib import Path
import argparse
import logging
import json
import math
import numpy as np
import struct
import hashlib
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

# --- Constants (same defaults as preproc_flow_new) ---
MAX_LEN_DEFAULT = 1500
ETH_LEN = 14

# --- Helper functions (Ethernet strip, IP checks, masking, UDP pad, etc) ---
def _is_ipv4(pkt_l3: bytes) -> bool:
    return len(pkt_l3) >= 1 and (pkt_l3[0] >> 4) == 4

def _is_ipv6(pkt_l3: bytes) -> bool:
    return len(pkt_l3) >= 1 and (pkt_l3[0] >> 4) == 6

def _strip_l2(raw: bytes) -> bytes:
    if len(raw) >= ETH_LEN:
        ether_type = int.from_bytes(raw[12:14], "big")
        if ether_type >= 0x0600:
            return raw[ETH_LEN:]
    return raw

def _mask_ip_addrs_ipv4(pkt_l3: bytes) -> bytes:
    if len(pkt_l3) < 20:
        return pkt_l3
    ihl = (pkt_l3[0] & 0x0F) * 4
    if ihl < 20 or len(pkt_l3) < ihl:
        return pkt_l3
    b = bytearray(pkt_l3)
    for i in range(12, 20):
        b[i] = 0
    return bytes(b)

def _mask_ip_addrs_ipv6(pkt_l3: bytes) -> bytes:
    if len(pkt_l3) < 40:
        return pkt_l3
    b = bytearray(pkt_l3)
    for i in range(8, 40):
        b[i] = 0
    return bytes(b)

# IPv6 ext header set & walker (borrowed approach)
_IPV6_EXT_SET = {0, 43, 44, 50, 51, 60}
def _locate_ipv6_l4(pkt_l3: bytes) -> Tuple[Optional[int], Optional[int]]:
    if len(pkt_l3) < 40:
        return None, None
    nh = pkt_l3[6]
    off = 40
    while nh in _IPV6_EXT_SET:
        if nh == 44:
            if len(pkt_l3) < off + 8:
                return None, None
            nh = pkt_l3[off]
            off += 8
        else:
            if len(pkt_l3) < off + 2:
                return None, None
            next_header = pkt_l3[off]
            hdrlen_units = pkt_l3[off + 1]
            hdr_len = (hdrlen_units + 1) * 8
            if len(pkt_l3) < off + hdr_len:
                return None, None
            nh = next_header
            off += hdr_len
    return nh, off

def _pad_udp_header_ipv4(pkt_l3: bytes) -> bytes:
    if len(pkt_l3) < 20:
        return pkt_l3
    ihl = (pkt_l3[0] & 0x0F) * 4
    if len(pkt_l3) < ihl + 8:
        return pkt_l3
    proto = pkt_l3[9]
    if proto != 17:
        return pkt_l3
    return pkt_l3[:ihl+8] + (b"\x00" * 12) + pkt_l3[ihl+8:]

def _pad_udp_header_ipv6(pkt_l3: bytes) -> bytes:
    nh, off = _locate_ipv6_l4(pkt_l3)
    if nh != 17 or off is None:
        return pkt_l3
    if len(pkt_l3) < off + 8:
        return pkt_l3
    return pkt_l3[:off+8] + (b"\x00" * 12) + pkt_l3[off+8:]

# IPv4/6 parsers (minimal, focused on sport/dport and payload)
def _ipv4_fields(pkt_l3: bytes):
    if len(pkt_l3) < 20:
        return (-1, b"", b"", 0, (0,0), b"", -1)
    ihl = (pkt_l3[0] & 0x0F) * 4
    if ihl < 20 or len(pkt_l3) < ihl:
        return (-1, b"", b"", 0, (0,0), b"", -1)
    total_len = int.from_bytes(pkt_l3[2:4], "big")
    proto = pkt_l3[9]
    src = pkt_l3[12:16]
    dst = pkt_l3[16:20]
    l3_total = min(total_len, len(pkt_l3))
    l4 = pkt_l3[ihl:l3_total]
    if proto == 6:
        if len(l4) < 20:
            return (proto, src, dst, ihl, (0,0), b"", -1)
        sport = int.from_bytes(l4[0:2], "big")
        dport = int.from_bytes(l4[2:4], "big")
        data_offset = (l4[12] >> 4) * 4
        if data_offset < 20 or len(l4) < data_offset:
            return (proto, src, dst, ihl, (sport,dport), b"", -1)
        flags = l4[13]
        payload = l4[data_offset:]
        return (proto, src, dst, ihl, (sport, dport), payload, flags)
    if proto == 17:
        if len(l4) < 8:
            return (proto, src, dst, ihl, (0,0), b"", -1)
        sport = int.from_bytes(l4[0:2], "big")
        dport = int.from_bytes(l4[2:4], "big")
        payload = l4[8:]
        return (proto, src, dst, ihl, (sport, dport), payload, -1)
    return (proto, src, dst, ihl, (0,0), l4, -1)

def _ipv6_fields(pkt_l3: bytes):
    if len(pkt_l3) < 40:
        return (-1, b"", b"", (0,0), b"")
    nh = pkt_l3[6]
    src = pkt_l3[8:24]
    dst = pkt_l3[24:40]
    payload = pkt_l3[40:]
    sport = dport = 0
    if nh == 6 and len(payload) >= 4:
        sport = int.from_bytes(payload[0:2], "big")
        dport = int.from_bytes(payload[2:4], "big")
    elif nh == 17 and len(payload) >= 4:
        sport = int.from_bytes(payload[0:2], "big")
        dport = int.from_bytes(payload[2:4], "big")
    return (nh, src, dst, (sport, dport), payload)

def _filter_drop_dns(proto: int, ports: Tuple[int,int]) -> bool:
    if proto not in (6, 17):
        return False
    sport, dport = ports
    return sport == 53 or dport == 53

def _ipv6_tcp_payload_len(pkt_l3: bytes) -> Optional[int]:
    nh, off = _locate_ipv6_l4(pkt_l3)
    if nh != 6 or off is None:
        return None
    if len(pkt_l3) < off + 20:
        return 0
    data_offset = (pkt_l3[off + 12] >> 4) * 4
    if data_offset < 20 or len(pkt_l3) < off + data_offset:
        return 0
    return len(pkt_l3) - (off + data_offset)

def _prepare_bytes_for_model_ipv4(pkt_l3: bytes) -> bytes:
    if len(pkt_l3) < 20:
        return pkt_l3
    ihl = (pkt_l3[0] & 0x0F) * 4
    m = _mask_ip_addrs_ipv4(pkt_l3)
    if m[9] == 17:
        m = _pad_udp_header_ipv4(m)
    header = m[:ihl]
    payload = m[ihl:ihl + 1480]
    return header + payload

def _prepare_bytes_for_model_ipv6(pkt_l3: bytes) -> bytes:
    if len(pkt_l3) < 40:
        return pkt_l3
    m = _mask_ip_addrs_ipv6(pkt_l3)
    m = _pad_udp_header_ipv6(m)
    header = m[:40]
    payload = m[40:40 + 1480]
    return header + payload

def _vectorize_bytes(l3_bytes: bytes, max_len: int) -> np.ndarray:
    out = np.zeros(max_len, dtype=np.float32)
    sl = l3_bytes[:max_len]
    arr_u8 = np.frombuffer(sl, dtype=np.uint8)
    out[:arr_u8.size] = arr_u8.astype(np.float32)
    out /= 255.0
    return out

def _flow_id_from_tuple(proto: int, src_ip: bytes, dst_ip: bytes, sport: int, dport: int) -> int:
    epA = src_ip + int(sport).to_bytes(2, "big", signed=False)
    epB = dst_ip + int(dport).to_bytes(2, "big", signed=False)
    lo, hi = (epA, epB) if epA <= epB else (epB, epA)
    blob = bytes([proto & 0xFF]) + lo + hi
    digest = hashlib.blake2b(blob, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)

# --- minimal pcap reader and pcapng fallback via scapy if needed ---
def _iter_pcap_packets(path: Path):
    with open(path, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            return
        magic = gh[:4]
        if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            endian = "<"
        elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            endian = ">"
        else:
            return
        pkh_struct = struct.Struct(endian + "IIII")
        while True:
            pkh = f.read(16)
            if len(pkh) < 16:
                break
            _ts_sec, _ts_usec, incl_len, _orig_len = pkh_struct.unpack(pkh)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            yield data

def _iter_pcapng_packets_scapy(path: Path):
    try:
        from scapy.utils import PcapNgReader
    except Exception as e:
        raise RuntimeError("Reading .pcapng requires scapy. Install it or use --pcapng-mode skip.") from e
    rdr = PcapNgReader(str(path))
    for pkt in rdr:
        try:
            yield bytes(pkt)
        except Exception:
            continue

def _count_pcap_packets(path: Path) -> int:
    """Count total number of packets in a pcap file."""
    count = 0
    with open(path, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            return 0
        magic = gh[:4]
        if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            endian = "<"
        elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            endian = ">"
        else:
            return 0
        pkh_struct = struct.Struct(endian + "IIII")
        while True:
            pkh = f.read(16)
            if len(pkh) < 16:
                break
            _ts_sec, _ts_usec, incl_len, _orig_len = pkh_struct.unpack(pkh)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            count += 1
    return count

def _count_pcapng_packets_scapy(path: Path) -> int:
    """Count total number of packets in a pcapng file using scapy."""
    try:
        from scapy.utils import PcapNgReader
    except Exception as e:
        raise RuntimeError("Reading .pcapng requires scapy. Install it or use --pcapng-mode skip.") from e
    count = 0
    rdr = PcapNgReader(str(path))
    for pkt in rdr:
        try:
            _ = bytes(pkt)  # Just check if we can read it
            count += 1
        except Exception:
            continue
    return count


def _human_readable_bytes(num_bytes: float) -> str:
    """Convert byte counts into a human readable string."""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def estimate_dataset_output_size(in_dir: Path, max_len: int, flush_every: int, pcapng_mode: str) -> Dict[str, object]:
    """
    Estimate the total disk footprint of the generated dataset.

    Assumptions:
        * All packets counted will survive filtering (worst-case size).
        * Each chunk produces three .npy files (data/flow/label).
        * Each .npy has ~128 bytes of header/metadata overhead.
    """
    caps = sorted(list(in_dir.glob("*.pcap")) + list(in_dir.glob("*.pcapng")))
    per_file = []
    total_packets = 0
    total_chunks = 0

    if not caps:
        return {
            "captures": 0,
            "total_packets": 0,
            "total_chunks": 0,
            "bytes": {
                "data": 0,
                "flows": 0,
                "labels": 0,
                "overhead": 0,
                "total": 0,
            },
            "per_file": per_file,
        }

    for cap in caps:
        note = None
        try:
            with cap.open("rb") as fh:
                magic = fh.read(4)
        except FileNotFoundError:
            logging.warning("File missing while estimating size: %s", cap)
            per_file.append({"file": cap.name, "packets": 0, "note": "missing"})
            continue

        is_pcapng = magic == b"\x0a\x0d\x0d\x0a"
        if is_pcapng and pcapng_mode == "skip":
            logging.info("Skipping %s (pcapng + --pcapng-mode skip)", cap.name)
            per_file.append({"file": cap.name, "packets": 0, "note": "pcapng skipped"})
            continue

        logging.info("Counting packets in %s ...", cap.name)
        if is_pcapng:
            packet_count = _count_pcapng_packets_scapy(cap)
            note = "pcapng"
        else:
            packet_count = _count_pcap_packets(cap)
            note = "pcap"
            if packet_count == 0:
                logging.warning("Magic bytes %s not recognized as PCAP/PCAPNG for %s", magic.hex(), cap.name)

        per_file.append({"file": cap.name, "packets": packet_count, "note": note})
        total_packets += packet_count
        if packet_count:
            total_chunks += math.ceil(packet_count / max(1, flush_every))

    bytes_data = total_packets * (max_len * 4)  # float32
    bytes_flows = total_packets * 8  # uint64
    bytes_labels = total_packets * 2  # uint16
    # 3 files per chunk, each ~128 bytes of header/metadata
    bytes_overhead = total_chunks * 3 * 128
    total_bytes = bytes_data + bytes_flows + bytes_labels + bytes_overhead

    return {
        "captures": len(caps),
        "total_packets": total_packets,
        "total_chunks": total_chunks,
        "bytes": {
            "data": bytes_data,
            "flows": bytes_flows,
            "labels": bytes_labels,
            "overhead": bytes_overhead,
            "total": total_bytes,
        },
        "per_file": per_file,
    }

# --- optional CICIDS CSV loader (tolerant mapping of column names) ---
COMMON_COLS = [
    ("src_ip","src","source","src ip","source ip"),
    ("dst_ip","dst","destination","dst ip","destination ip"),
    ("sport","source port","src port","sport"),
    ("dport","dest port","dst port","dport","destination port"),
    ("protocol","proto","protocol"),
    ("label","Label","attack","category","traffic label")
]

def _normalize_col_name(c: str) -> str:
    return c.strip().lower().replace(" ", "").replace("_","")

def _find_columns(headers: List[str]) -> Dict[str,int]:
    norm = { _normalize_col_name(h): i for i,h in enumerate(headers) }
    picked = {}
    for canonical, *aliases in COMMON_COLS:
        for cand in (canonical,) + tuple(aliases):
            nc = _normalize_col_name(cand)
            if nc in norm:
                picked[canonical] = norm[nc]
                break
    return picked

def _extract_class_label_from_csv_filename(csv_path: Path) -> str:
    """
    Extract class label from CSV filename.
    Examples:
    - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv -> DDos
    - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv -> PortScan
    - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv -> Infilteration
    - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv -> WebAttacks
    - Friday-WorkingHours-Morning.pcap_ISCX.csv -> Morning (or BENIGN if no attack type)
    """
    # Remove .pcap_ISCX.csv or .csv suffix
    stem = csv_path.stem
    if stem.endswith('.pcap_ISCX'):
        stem = stem[:-10]  # Remove .pcap_ISCX
    
    # Split by hyphens
    parts = stem.split('-')
    
    # Pattern: Day-WorkingHours-[Time]-[AttackType] or Day-WorkingHours-[Time]
    # We want the last part as the class label
    if len(parts) >= 4:
        # Has attack type: Day-WorkingHours-Afternoon-DDos -> DDos
        last_part = parts[-1]
        # Preserve original capitalization (DDos, PortScan, etc.)
        return last_part
    elif len(parts) == 3:
        # Day-WorkingHours-Morning -> Morning
        # This might be BENIGN traffic, but use "Morning" as label per user request
        last_part = parts[-1]
        return last_part
    elif len(parts) == 2:
        # Day-WorkingHours -> likely BENIGN
        return "BENIGN"
    else:
        # Fallback: use the stem as-is
        return stem

def _load_flow_labels_from_csv(csv_path: Path, label_override: Optional[str] = None) -> Dict[int,str]:
    """
    Try to read CSV and build mapping flow_id -> label string
    Accepts common column names; if columns missing, returns empty map.
    
    Args:
        csv_path: Path to CSV file
        label_override: If provided, use this label for all flows instead of the Label column
    """
    mapping = {}
    try:
        import csv
        with csv_path.open("r", newline='', encoding='utf-8', errors='ignore') as fh:
            rdr = csv.reader(fh)
            headers = next(rdr)
            colpos = _find_columns(headers)
            # need at least src,dst,sport,dport,protocol
            # label column is optional if label_override is provided
            needed = {"src_ip","dst_ip","sport","dport","protocol"}
            if not needed.issubset(set(colpos.keys())):
                missing = needed - set(colpos.keys())
                logging.warning(f"Missing required columns in {csv_path.name}: {missing}. Found: {list(colpos.keys())}")
                return {}
            # If no label override, we need the label column
            if label_override is None and "label" not in colpos:
                logging.warning(f"Missing 'label' column in {csv_path.name} and no label_override provided")
                return {}
            
            rows_processed = 0
            rows_failed = 0
            for row in rdr:
                try:
                    srcs = row[colpos["src_ip"]].strip()
                    dsts = row[colpos["dst_ip"]].strip()
                    sport = int(row[colpos["sport"]])
                    dport = int(row[colpos["dport"]])
                    proto_str = row[colpos["protocol"]].strip()
                    
                    # Use override label if provided, otherwise use column label
                    if label_override is not None:
                        label = label_override
                    else:
                        label = row[colpos["label"]].strip()
                    
                    # parse proto: try numeric or name
                    proto = 0
                    if proto_str.isdigit():
                        proto = int(proto_str)
                    else:
                        if proto_str.lower().startswith("tcp"):
                            proto = 6
                        elif proto_str.lower().startswith("udp"):
                            proto = 17
                        else:
                            try:
                                proto = int(proto_str)
                            except Exception:
                                proto = 0
                    # convert ips to bytes (IPv4 assumed for CICIDS)
                    def ip4_to_bytes(s):
                        parts = s.split(".")
                        if len(parts) != 4:
                            return b"\x00\x00\x00\x00"
                        return bytes(int(p) & 0xFF for p in parts)
                    srcb = ip4_to_bytes(srcs)
                    dstb = ip4_to_bytes(dsts)
                    fid = _flow_id_from_tuple(proto, srcb, dstb, sport or 0, dport or 0)
                    mapping[fid] = label
                    rows_processed += 1
                except Exception as e:
                    rows_failed += 1
                    if rows_failed <= 5:  # Only log first few errors
                        logging.debug(f"Failed to process row in {csv_path.name}: {e}")
                    continue
            
            if rows_processed == 0:
                logging.warning(f"No rows successfully processed from {csv_path.name}")
            else:
                logging.debug(f"Loaded {len(mapping)} flow labels from {csv_path.name} ({rows_processed} rows processed, {rows_failed} failed)")
    except Exception as e:
        logging.error(f"Error loading CSV {csv_path.name}: {e}", exc_info=True)
        return {}
    return mapping

# --- core convert_capture_streaming (modified to output optional labels) ---
def convert_capture_streaming(cap_path: Path,
                              max_len: int,
                              chunk: int,
                              pcapng_mode: str = "scapy",
                              out_dir: Optional[Path] = None,
                              base_name: Optional[str] = None,
                              label_map_for_file: Optional[Dict[int,str]] = None):
    name_lower = cap_path.name.lower()
    iterator = None
    if name_lower.endswith(".pcapng"):
        if pcapng_mode == "skip":
            return [], {"seen":0,"kept":0,"dns":0,"tcp_no_payload":0,"malformed":0,"non_ip":0,"file":str(cap_path)}
        logging.info(f"Processing {cap_path.name} as PCAPNG (streaming, no pre-count)...")
        iterator = _iter_pcapng_packets_scapy(cap_path)
    elif name_lower.endswith(".pcap"):
        logging.info(f"Processing {cap_path.name} as PCAP (streaming, no pre-count)...")
        iterator = _iter_pcap_packets(cap_path)
    else:
        logging.warning(f"Unsupported capture extension for {cap_path.name}; expected .pcap or .pcapng")
        return [], {"seen":0,"kept":0,"dns":0,"tcp_no_payload":0,"malformed":0,"non_ip":0,"file":str(cap_path)}

    vecs: List[np.ndarray] = []
    flows: List[int] = []
    labels_for_packets: List[int] = []
    out_paths: List[Path] = []
    chunk_idx = 0

    total_seen = total_kept = 0
    dropped_dns = dropped_no_payload = dropped_malformed = dropped_non_ip = 0

    # Create progress bar for this pcap file with total count
    pbar = tqdm(iterator, desc=f"Processing {cap_path.name}", unit="pkt", total=None,
                ncols=120, leave=True, mininterval=0.5, maxinterval=2.0)
    
    for raw in pbar:
        total_seen += 1
        l3 = _strip_l2(raw)

        if _is_ipv4(l3):
            proto, src, dst, ihl, ports, payload, tcp_flags = _ipv4_fields(l3)
            if proto == -1:
                dropped_malformed += 1
                continue
            if _filter_drop_dns(proto, ports):
                dropped_dns += 1
                continue
            # Keep TCP packets even if they have no payload
            sport, dport = ports
            fid = _flow_id_from_tuple(proto, src, dst, sport or 0, dport or 0)
            l3_for_model = _prepare_bytes_for_model_ipv4(l3)

        elif _is_ipv6(l3):
            proto, src, dst, ports, payload = _ipv6_fields(l3)
            if proto == -1:
                dropped_malformed += 1
                continue
            if _filter_drop_dns(proto, ports):
                dropped_dns += 1
                continue
            # Keep TCP packets even if they have no payload
            sport, dport = ports
            fid = _flow_id_from_tuple(proto, src, dst, sport or 0, dport or 0)
            l3_for_model = _prepare_bytes_for_model_ipv6(l3)
        else:
            dropped_non_ip += 1
            continue

        vec = _vectorize_bytes(l3_for_model, max_len)
        vecs.append(vec)
        flows.append(fid)
        # map label for this packet's flow id (always generate labels, use 0 if no map)
        if label_map_for_file:
            labels_for_packets.append(label_map_for_file.get(fid, 0))  # 0 = unknown
        else:
            labels_for_packets.append(0)  # 0 = unknown when no label map
        total_kept += 1
        
        # Update progress bar stats periodically (after counts are updated)
        if total_seen % 1000 == 0:
            total_dropped = dropped_dns + dropped_no_payload + dropped_malformed + dropped_non_ip
            pbar.set_postfix({
                'kept': f'{total_kept:,}',
                'dropped': f'{total_dropped:,}'
            })

        if out_dir is not None and len(vecs) >= chunk:
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = base_name or cap_path.stem
            base = f"{stem}.chunk{chunk_idx:03d}"
            data_path = out_dir / f"{base}.npy"
            flow_path = out_dir / f"{base}.flow.npy"
            label_path = out_dir / f"{base}.label.npy"
            np.save(data_path, np.stack(vecs, axis=0))
            np.save(flow_path, np.asarray(flows, dtype=np.uint64))
            # Always save label files (with 0 for unknown when no label map)
            lbls = np.asarray(labels_for_packets, dtype=np.uint16)
            np.save(label_path, lbls)
            out_paths.append(data_path)
            vecs.clear(); flows.clear(); labels_for_packets.clear(); chunk_idx += 1

    # Final update to show complete stats
    total_dropped = dropped_dns + dropped_no_payload + dropped_malformed + dropped_non_ip
    pbar.set_postfix({
        'kept': f'{total_kept:,}',
        'dropped': f'{total_dropped:,}'
    })
    pbar.close()

    if total_seen == 0:
        logging.warning(f"No packets found in {cap_path.name}")

    # final flush
    if out_dir is not None and vecs:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = base_name or cap_path.stem
        base = f"{stem}.chunk{chunk_idx:03d}"
        data_path = out_dir / f"{base}.npy"
        flow_path = out_dir / f"{base}.flow.npy"
        label_path = out_dir / f"{base}.label.npy"
        np.save(data_path, np.stack(vecs, axis=0))
        np.save(flow_path, np.asarray(flows, dtype=np.uint64))
        # Always save label files (with 0 for unknown when no label map)
        lbls = np.asarray(labels_for_packets, dtype=np.uint16)
        np.save(label_path, lbls)
        out_paths.append(data_path)

    metrics = {
        "seen": total_seen,
        "kept": total_kept,
        "dns": dropped_dns,
        "tcp_no_payload": dropped_no_payload,
        "malformed": dropped_malformed,
        "non_ip": dropped_non_ip,
        "file": str(cap_path),
    }
    
    # Print formatted summary after processing each pcap
    total_dropped = dropped_dns + dropped_no_payload + dropped_malformed + dropped_non_ip
    drop_rate = (total_dropped / total_seen * 100) if total_seen > 0 else 0.0
    
    logging.info("")
    logging.info("=" * 70)
    logging.info(f"Summary for {cap_path.name}")
    logging.info("=" * 70)
    logging.info(f"  Total packets seen:     {total_seen:>12,}")
    logging.info(f"  Packets kept:           {total_kept:>12,} ({100-drop_rate:.1f}%)")
    logging.info(f"  Packets dropped:        {total_dropped:>12,} ({drop_rate:.1f}%)")
    logging.info("")
    logging.info("  Drop breakdown:")
    if total_seen > 0:
        logging.info(f"    - DNS packets:        {dropped_dns:>12,} ({dropped_dns/total_seen*100:.1f}%)")
        logging.info(f"    - TCP no payload:     {dropped_no_payload:>12,} ({dropped_no_payload/total_seen*100:.1f}%)")
        logging.info(f"    - Malformed:          {dropped_malformed:>12,} ({dropped_malformed/total_seen*100:.1f}%)")
        logging.info(f"    - Non-IP:             {dropped_non_ip:>12,} ({dropped_non_ip/total_seen*100:.1f}%)")
    else:
        logging.info("    - DNS packets:        0")
        logging.info("    - TCP no payload:     0")
        logging.info("    - Malformed:          0")
        logging.info("    - Non-IP:             0")
    logging.info("=" * 70)
    logging.info("")
    
    return out_paths, metrics

# --- worker + directory scan (flat folder) ---
def _process_one_capture(cap: Path, dst_dir: Path, max_len: int, flush_every: int, pcapng_mode: str, label_map_for_file: Optional[Dict[int,str]]):
    try:
        chunk_paths, metrics = convert_capture_streaming(
            cap_path=cap,
            max_len=max_len,
            chunk=flush_every,
            pcapng_mode=pcapng_mode,
            out_dir=dst_dir,
            base_name=cap.stem,
            label_map_for_file=label_map_for_file
        )
        return cap, chunk_paths, metrics
    except Exception:
        logging.exception("Failed to process %s", cap)
        return cap, [], {"seen":0,"kept":0,"dns":0,"tcp_no_payload":0,"malformed":0,"non_ip":0,"file":str(cap)}

def _find_matching_csvs_for_pcap(pcap_path: Path, csv_dir: Path) -> List[Path]:
    """
    Find all CSV files that match a PCAP file's base name.
    Example: Friday-WorkingHours.pcap matches:
    - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    - Friday-WorkingHours-Morning.pcap_ISCX.csv
    
    But does NOT match:
    - Friday-WorkingHours-Afternoon-DDos-extra.pcap_ISCX.csv (if such exists)
    - OtherDay-WorkingHours-*.csv
    """
    pcap_stem = pcap_path.stem  # e.g., "Friday-WorkingHours"
    matching_csvs = []
    
    # Look for CSV files that start with the PCAP stem followed by a hyphen or end
    # Pattern: {pcap_stem}-*.csv or {pcap_stem}-*.pcap_ISCX.csv
    for csv_file in csv_dir.glob("*.csv"):
        csv_stem = csv_file.stem
        # Handle .pcap_ISCX.csv files
        if csv_stem.endswith('.pcap_ISCX'):
            csv_base = csv_stem[:-10]  # Remove .pcap_ISCX
        else:
            csv_base = csv_stem
        
        # Check if CSV base name starts with PCAP stem followed by hyphen or equals it
        # e.g., "Friday-WorkingHours-Afternoon-DDos" starts with "Friday-WorkingHours-"
        # or "Friday-WorkingHours" equals "Friday-WorkingHours"
        if csv_base == pcap_stem or csv_base.startswith(pcap_stem + '-'):
            matching_csvs.append(csv_file)
    
    return sorted(matching_csvs)

def process_folder(in_dir: Path, out_root: Path, max_len: int, flush_every: int, workers: int, pcapng_mode: str, keep_unknown_labels: bool):
    # collect pcaps
    caps = sorted(list(in_dir.glob("*.pcap")) + list(in_dir.glob("*.pcapng")))
    logging.info("Found %d capture files in %s", len(caps), in_dir)
    
    # Check for labels at the start
    logging.info("=" * 60)
    logging.info("Checking for label files...")
    csv_found_count = 0
    labels_loaded_count = 0
    labels_not_found_count = 0
    
    for cap in caps:
        matching_csvs = _find_matching_csvs_for_pcap(cap, in_dir)
        if matching_csvs:
            csv_found_count += len(matching_csvs)
            # Try to load labels from all matching CSVs
            combined_fmap = {}
            for csv_file in matching_csvs:
                class_label = _extract_class_label_from_csv_filename(csv_file)
                fmap = _load_flow_labels_from_csv(csv_file, label_override=class_label)
                if fmap:
                    # Merge into combined map (later CSVs may overwrite earlier ones for same flow_id)
                    combined_fmap.update(fmap)
                    logging.info("  ✓ Labels loaded: %s -> %s (class: %s, %d flows)", 
                                cap.name, csv_file.name, class_label, len(fmap))
            
            if combined_fmap:
                labels_loaded_count += 1
                logging.info("  ✓ Total labels for %s: %d flows from %d CSV(s)", 
                            cap.name, len(combined_fmap), len(matching_csvs))
            else:
                labels_not_found_count += 1
                logging.warning("  ✗ No labels loaded for: %s (from %d CSV file(s))", 
                              cap.name, len(matching_csvs))
        else:
            logging.info("  - No matching CSV files for: %s (will process without labels)", cap.name)
    
    logging.info("=" * 60)
    logging.info("Label check summary:")
    logging.info("  Total pcap files: %d", len(caps))
    logging.info("  CSV files found: %d", csv_found_count)
    logging.info("  Labels successfully loaded: %d", labels_loaded_count)
    logging.info("  CSV files with no valid labels: %d", labels_not_found_count)
    logging.info("  Files without CSV: %d", len(caps) - labels_loaded_count)
    logging.info("=" * 60)
    
    # Throw error if labels are missing
    if labels_loaded_count < len(caps):
        missing_files = []
        for cap in caps:
            matching_csvs = _find_matching_csvs_for_pcap(cap, in_dir)
            if not matching_csvs:
                missing_files.append(f"  - {cap.name} (no matching CSV files)")
            else:
                combined_fmap = {}
                for csv_file in matching_csvs:
                    class_label = _extract_class_label_from_csv_filename(csv_file)
                    fmap = _load_flow_labels_from_csv(csv_file, label_override=class_label)
                    if fmap:
                        combined_fmap.update(fmap)
                if not combined_fmap:
                    missing_files.append(f"  - {cap.name} (CSV files exist but no valid labels)")
        
        error_msg = f"ERROR: Labels are required but missing for {len(caps) - labels_loaded_count} file(s):\n" + "\n".join(missing_files)
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    
    tasks = []
    label_maps_global = {}  # combined label maps to write a labels_map.json
    for cap in caps:
        # Find all matching CSV files and combine their labels
        matching_csvs = _find_matching_csvs_for_pcap(cap, in_dir)
        if matching_csvs:
            combined_fmap = {}
            for csv_file in matching_csvs:
                class_label = _extract_class_label_from_csv_filename(csv_file)
                fmap = _load_flow_labels_from_csv(csv_file, label_override=class_label)
                if fmap:
                    # Merge into combined map (later CSVs may overwrite earlier ones for same flow_id)
                    combined_fmap.update(fmap)
            
            if combined_fmap:
                # map string label -> small integer later; store strings per file temporarily
                tasks.append((cap, out_root, combined_fmap))
                # collect unique label strings
                for v in combined_fmap.values():
                    label_maps_global.setdefault(v, None)
            else:
                # This should not happen due to error check above, but keep for safety
                raise RuntimeError(f"Labels not found for {cap.name} despite CSV files existing")
        else:
            # This should not happen due to error check above, but keep for safety
            raise RuntimeError(f"No matching CSV files found for {cap.name}")

    # assign numeric ids for label strings (1..N); 0 reserved for unknown
    label_to_id = {}
    if label_maps_global:
        idx = 1
        for label_str in sorted(label_maps_global.keys()):
            label_to_id[label_str] = idx
            idx += 1

    # convert per-file maps from flowid->label_str to flowid->label_id
    tasks_with_label_ids = []
    for cap, out_root_dir, fmap in tasks:
        if fmap:
            fmap_id = { fid: label_to_id.get(lbl, 0) for fid,lbl in fmap.items() }
            tasks_with_label_ids.append((cap, out_root_dir, fmap_id))
        else:
            tasks_with_label_ids.append((cap, out_root_dir, None))

    agg = {"seen":0, "kept":0, "dns":0, "tcp_no_payload":0, "malformed":0, "non_ip":0, "zero_kept_files":0}
    if workers <= 1:
        # Sequential processing
        for cap, dst, fmap in tasks_with_label_ids:
            src, chunks, m = _process_one_capture(cap, dst, max_len, flush_every, pcapng_mode, fmap)
            if m.get("kept",0) == 0:
                agg["zero_kept_files"] += 1
            for k in ("seen","kept","dns","tcp_no_payload","malformed","non_ip"):
                agg[k] += int(m.get(k,0))
            logging.info("[OK] %s -> %d chunk(s)", src.name, len(chunks))
    else:
        # Parallel processing across different days/files
        nproc = min(max(1, workers), cpu_count())
        logging.info("Processing %d files in parallel using %d workers", len(tasks_with_label_ids), nproc)
        with Pool(processes=nproc) as pool:
            # Use starmap to pass all arguments correctly
            results = pool.starmap(_process_one_capture, 
                                   [(c, d, max_len, flush_every, pcapng_mode, f) 
                                    for (c, d, f) in tasks_with_label_ids])
            for src, chunks, m in results:
                if m.get("kept",0) == 0:
                    agg["zero_kept_files"] += 1
                for k in ("seen","kept","dns","tcp_no_payload","malformed","non_ip"):
                    agg[k] += int(m.get(k,0))
                logging.info("[OK] %s -> %d chunk(s)", src.name, len(chunks))

    logging.info("Summary: files=%d seen=%d kept=%d dropped: dns=%d tcp_no_payload=%d malformed=%d non_ip=%d zero_kept_files=%d",
                 len(caps), agg["seen"], agg["kept"], agg["dns"], agg["tcp_no_payload"], agg["malformed"], agg["non_ip"], agg["zero_kept_files"])

    # if label map present, write labels_map.json mapping numeric id->string
    if label_to_id:
        out_root.mkdir(parents=True, exist_ok=True)
        labels_map_inv = {str(v): k for k, v in label_to_id.items()}
        with (out_root / "labels_map.json").open("w", encoding="utf-8") as fh:
            json.dump(labels_map_inv, fh, indent=2)

# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="Preprocess CICIDS2017 pcaps → DeepPacket tensors + flows + optional labels")
    ap.add_argument("--in-dir", required=True, help="Directory with CICIDS2017 pcaps (flat folder)")
    ap.add_argument("--out-root", required=True, help="Output directory for .npy chunks and sidecars")
    ap.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT, help="Vector length (default 1500)")
    ap.add_argument("--flush-every", type=int, default=50000, help="Flush every N packets into a chunk")
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker processes (files in parallel)")
    ap.add_argument("--pcapng-mode", choices=["scapy","skip"], default="scapy", help="How to handle .pcapng")
    ap.add_argument("--no-verify", action="store_true", help="Skip post-processing verification step")
    ap.add_argument("--keep-unknown-labels", action="store_true", help="Keep unknown labels (0) rather than error")
    ap.add_argument("--estimate-output-size", action="store_true", help="Estimate total dataset output size and exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    in_dir = Path(args.in_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if args.estimate_output_size:
        estimate = estimate_dataset_output_size(in_dir, args.max_len, args.flush_every, args.pcapng_mode)
        captures = estimate["captures"]
        total_packets = estimate["total_packets"]
        total_chunks = estimate["total_chunks"]
        bytes_breakdown = estimate["bytes"]

        if captures == 0:
            print(f"No .pcap/.pcapng captures found under {in_dir}")
            return

        print("")
        print(f"Estimated dataset output size for {captures} capture file(s)")
        print("Assumes every counted packet is kept (worst-case size).")
        print(f"  Total packets:        {total_packets:,}")
        print(f"  Estimated chunks:     {total_chunks:,} (flush_every={args.flush_every})")
        print(f"  Data tensors:         {_human_readable_bytes(bytes_breakdown['data'])}")
        print(f"  Flow sidecars:        {_human_readable_bytes(bytes_breakdown['flows'])}")
        print(f"  Label sidecars:       {_human_readable_bytes(bytes_breakdown['labels'])}")
        print(f"  NPY headers (approx): {_human_readable_bytes(bytes_breakdown['overhead'])}")
        print(f"  -------------------------------")
        print(f"  Total estimated size: {_human_readable_bytes(bytes_breakdown['total'])}")
        print("")
        print("Per-file packet counts:")
        for entry in estimate["per_file"]:
            note = f" ({entry['note']})" if entry["note"] else ""
            print(f"  - {entry['file']}: {entry['packets']:,} packets{note}")
        print("")
        print("Re-run without --estimate-output-size to generate the dataset.")
        return

    process_folder(in_dir, out_root, max_len=args.max_len, flush_every=args.flush_every,
                   workers=args.workers, pcapng_mode=args.pcapng_mode, keep_unknown_labels=args.keep_unknown_labels)

    if not args.no_verify:
        # Simple verification: check shapes & sidecars for each chunk
        print("\nRunning verification pass...\n")
        # from pathlib import Path
        total_files = 0
        total_packets = 0
        problematic = 0
        for arr in sorted(out_root.glob("*.npy")):
            if arr.name.endswith(".flow.npy") or arr.name.endswith(".label.npy") or arr.name == "labels_map.json":
                continue
            total_files += 1
            try:
                data = np.load(arr, mmap_mode="r")
                flow_path = arr.with_name(arr.stem + ".flow.npy")
                if not flow_path.exists():
                    print(f"Missing flow sidecar for {arr.name}")
                    problematic += 1
                    continue
                flows = np.load(flow_path, mmap_mode="r")
                if data.shape[0] != flows.shape[0]:
                    print(f"Misaligned counts: {arr.name} ({data.shape[0]} vs {flows.shape[0]})")
                    problematic += 1
                    continue
                # label exists?
                label_path = arr.with_name(arr.stem + ".label.npy")
                if label_path.exists():
                    labels = np.load(label_path, mmap_mode="r")
                    if labels.shape[0] != data.shape[0]:
                        print(f"Label misaligned: {arr.name}")
                        problematic += 1
                        continue
                total_packets += data.shape[0]
            except Exception as e:
                print(f"Error reading {arr.name}: {e}")
                problematic += 1
        print(f"\nVerification: files={total_files} packets={total_packets} problems={problematic}\n")
        if problematic == 0:
            print("✓ Verification passed")
        else:
            print("✗ Issues found - inspect logs/output")

if __name__ == "__main__":
    main()
