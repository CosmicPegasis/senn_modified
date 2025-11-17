#!/usr/bin/env python3
# preprocess_pcaps_with_flows.py — streaming PCAP/PCAPNG → DeepPacket tensors + flow sidecars
# - Streams packets (no full decode for .pcap)
# - Lightweight header parsing (Ethernet→IPv4/IPv6→TCP/UDP)
# - Drops DNS (port 53) + TCP no-payload handshakes
# - Produces 1500-length float32 vectors in [0,1]
# - ALSO writes aligned .flow.npy containing deterministic 64-bit flow IDs
# - Flushes in chunks to avoid high RAM; optional multiprocessing
#
# Notes on flow IDs:
# - Direction-agnostic: (min(endpoint), max(endpoint)) where endpoint=(ip,port)
# - Endpoint IP bytes are 4B for IPv4 and 16B for IPv6; ports are 2B (0 if unknown)
# - Protocol is included; for non-TCP/UDP we keep port=0
# - Stable across runs via blake2b(digest_size=8) → uint64

import os, sys, argparse, struct, numpy as np, hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from functools import partial
from multiprocessing import Pool, cpu_count

MAX_LEN_DEFAULT = 1500
ETH_LEN = 14

# ------------------------ low-level parsers (no Scapy) ------------------------

def _is_ipv4(pkt_l3: bytes) -> bool:
    return len(pkt_l3) >= 1 and (pkt_l3[0] >> 4) == 4

def _is_ipv6(pkt_l3: bytes) -> bool:
    return len(pkt_l3) >= 1 and (pkt_l3[0] >> 4) == 6

def _strip_l2(raw: bytes) -> bytes:
    if len(raw) >= ETH_LEN:
        ether_type = int.from_bytes(raw[12:14], "big")
        if ether_type >= 0x0600:  # Ethernet II EtherType
            return raw[ETH_LEN:]
    return raw

def _ipv4_fields(pkt_l3: bytes) -> Tuple[int, bytes, bytes, int, Tuple[int,int], bytes, int]:
    """
    Return (proto, src_ip4_bytes, dst_ip4_bytes, ihl_bytes,
            (sport,dport or (0,0) if N/A), payload_bytes, tcp_flags or -1).
    If malformed, returns proto=-1.
    """
    if len(pkt_l3) < 20:
        return (-1, b"", b"", 0, (0,0), b"", -1)
    ver_ihl = pkt_l3[0]
    ihl = (ver_ihl & 0x0F) * 4
    if ihl < 20 or len(pkt_l3) < ihl:
        return (-1, b"", b"", 0, (0,0), b"", -1)
    total_len = int.from_bytes(pkt_l3[2:4], "big")
    proto = pkt_l3[9]
    src = pkt_l3[12:16]
    dst = pkt_l3[16:20]
    l3_total = min(total_len, len(pkt_l3))
    l4 = pkt_l3[ihl:l3_total]

    if proto == 6:  # TCP
        if len(l4) < 20:
            return (proto, src, dst, ihl, (0,0), b"", -1)
        sport = int.from_bytes(l4[0:2], "big")
        dport = int.from_bytes(l4[2:4], "big")
        data_offset = (l4[12] >> 4) * 4
        if data_offset < 20 or len(l4) < data_offset:
            return (proto, src, dst, ihl, (sport,dport), b"", -1)
        flags = l4[13]
        payload = l4[data_offset:]
        return (proto, src, dst, ihl, (sport,dport), payload, flags)

    if proto == 17:  # UDP
        if len(l4) < 8:
            return (proto, src, dst, ihl, (0,0), b"", -1)
        sport = int.from_bytes(l4[0:2], "big")
        dport = int.from_bytes(l4[2:4], "big")
        payload = l4[8:]
        return (proto, src, dst, ihl, (sport,dport), payload, -1)

    # other L4
    return (proto, src, dst, ihl, (0,0), l4, -1)

def _ipv6_fields(pkt_l3: bytes) -> Tuple[int, bytes, bytes, Tuple[int,int], bytes]:
    """
    Minimal IPv6: Return (next_header (proto), src_ip16, dst_ip16, (sport,dport or (0,0)), payload_bytes)
    We do not walk extension headers; ports will be (0,0) unless L4 starts immediately (rare in presence of ext hdrs).
    """
    if len(pkt_l3) < 40:
        return (-1, b"", b"", (0,0), b"")
    nh = pkt_l3[6]  # next header (approx proto)
    src = pkt_l3[8:24]
    dst = pkt_l3[24:40]
    payload = pkt_l3[40:]
    sport = dport = 0
    # If next header directly TCP/UDP, we can parse ports quickly
    if nh == 6 and len(payload) >= 4:  # TCP
        sport = int.from_bytes(payload[0:2], "big")
        dport = int.from_bytes(payload[2:4], "big")
    elif nh == 17 and len(payload) >= 4:  # UDP
        sport = int.from_bytes(payload[0:2], "big")
        dport = int.from_bytes(payload[2:4], "big")
    return (nh, src, dst, (sport,dport), payload)

def _filter_drop_dns(proto: int, ports: Tuple[int,int]) -> bool:
    if proto not in (6, 17) or ports is None:
        return False
    sport, dport = ports
    return sport == 53 or dport == 53

def _filter_drop_tcp_no_payload(proto: int, payload: bytes, tcp_flags: int) -> bool:
    if proto != 6:
        return False
    return len(payload) == 0

def _vectorize_bytes(l3_start: bytes, max_len: int) -> np.ndarray:
    out = np.zeros(max_len, dtype=np.float32)
    sl = l3_start[:max_len]
    view = memoryview(sl)
    arr_u8 = np.frombuffer(view, dtype=np.uint8)
    out[:arr_u8.size] = arr_u8.astype(np.float32)
    out /= 255.0
    return out

def _flow_id_from_tuple(proto: int, src_ip: bytes, dst_ip: bytes, sport: int, dport: int) -> int:
    """
    Build a direction-agnostic 5-tuple and hash to uint64 (stable).
    IP bytes can be 4 (IPv4) or 16 (IPv6). Ports should be 0..65535 (0 if unknown).
    """
    # Normalize endpoints as bytes: ip || port(2B)
    epA = src_ip + int(sport).to_bytes(2, "big", signed=False)
    epB = dst_ip + int(dport).to_bytes(2, "big", signed=False)
    lo, hi = (epA, epB) if epA <= epB else (epB, epA)
    blob = bytes([proto & 0xFF]) + lo + hi
    digest = hashlib.blake2b(blob, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)

# ------------------------ PCAP readers ---------------------------------------

def _iter_pcap_packets(path: Path):
    with open(path, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            return
        magic = gh[:4]
        if magic == b'\xd4\xc3\xb2\xa1':
            endian = "<"
        elif magic == b'\xa1\xb2\xc3\xd4':
            endian = ">"
        elif magic == b'\x4d\x3c\xb2\xa1':
            endian = "<"
        elif magic == b'\xa1\xb2\x3c\x4d':
            endian = ">"
        else:
            return
        pkh_struct = struct.Struct(endian + "IIII")
        while True:
            pkh = f.read(16)
            if len(pkh) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = pkh_struct.unpack(pkh)
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
            raw = bytes(pkt)
        except Exception:
            continue
        yield raw

# ------------------------ conversion core ------------------------------------

def convert_capture_streaming(cap_path: Path,
                             max_len: int,
                             chunk: int,
                             pcapng_mode: str = "scapy",
                             out_dir: Optional[Path] = None,
                             base_name: Optional[str] = None) -> Tuple[List[Path], dict]:
    """
    Stream a capture, flush chunks to disk: returns list of (data .npy) paths.
    Also writes aligned .flow.npy with uint64 flow IDs.
    """
    assert chunk > 0
    ext = cap_path.suffix.lower()
    if ext == ".pcap":
        it = _iter_pcap_packets(cap_path)
    elif ext == ".pcapng":
        if pcapng_mode == "skip":
            return []
        it = _iter_pcapng_packets_scapy(cap_path)
    else:
        return []

    vecs: List[np.ndarray] = []
    flows: List[int] = []
    out_paths: List[Path] = []
    chunk_idx = 0

    # Metrics
    total_seen = 0
    total_kept = 0
    dropped_dns = 0
    dropped_no_payload = 0
    dropped_malformed = 0
    dropped_non_ip = 0

    pbar = it
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
            if _filter_drop_tcp_no_payload(proto, payload, tcp_flags):
                dropped_no_payload += 1
                continue
            sport, dport = ports
            fid = _flow_id_from_tuple(proto, src, dst, sport or 0, dport or 0)
elif _is_ipv6(l3):
            proto, src, dst, ports, payload = _ipv6_fields(l3)
            if proto == -1:
                dropped_malformed += 1
                continue
            if _filter_drop_dns(proto, ports):
                dropped_dns += 1
                continue
            # For TCP over IPv6 we don't compute flags here; still fine to keep payload length check off.
            sport, dport = ports
            fid = _flow_id_from_tuple(proto, src, dst, sport or 0, dport or 0)
else:
            dropped_non_ip += 1
            continue

        vec = _vectorize_bytes(l3, max_len)
vecs.append(vec)
        flows.append(fid)
        total_kept += 1

        if len(vecs) >= chunk:
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = base_name or cap_path.stem
                base = f"{stem}.chunk{chunk_idx:03d}"
                out_path_data = out_dir / f"{base}.npy"
                out_path_flow = out_dir / f"{base}.flow.npy"
np.save(out_path_data, np.stack(vecs, axis=0))
                np.save(out_path_flow, np.asarray(flows, dtype=np.uint64))
                out_paths.append(out_path_data)
                vecs.clear(); flows.clear()
                chunk_idx += 1

    if vecs and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = base_name or cap_path.stem
        base = f"{stem}.chunk{chunk_idx:03d}"
        out_path_data = out_dir / f"{base}.npy"
        out_path_flow = out_dir / f"{base}.flow.npy"
np.save(out_path_data, np.stack(vecs, axis=0))
        np.save(out_path_flow, np.asarray(flows, dtype=np.uint64))
out_paths.append(out_path_data)

    metrics = {
        "seen": total_seen,
        "kept": total_kept,
        "dns": dropped_dns,
        "tcp_no_payload": dropped_no_payload,
        "malformed": dropped_malformed,
        "non_ip": dropped_non_ip,
        "file": str(cap_path),
    }

    logging.info(
        "[%s] packets: seen=%d kept=%d dropped_dns=%d dropped_tcp_no_payload=%d dropped_malformed=%d dropped_non_ip=%d",
        cap_path.name, total_seen, total_kept, dropped_dns, dropped_no_payload, dropped_malformed, dropped_non_ip,
    )
if total_kept == 0:
        logging.warning("[%s] kept 0 packets → no output chunks written", cap_path.name)

    return out_paths, metrics

# ------------------------ folder processing ----------------------------------



def _process_one_capture(cap: Path, dst_dir: Path, max_len: int, flush_every: int, pcapng_mode: str) -> Tuple[Path, List[Path], dict]:
    try:
        chunk_paths, metrics = convert_capture_streaming(
            cap_path=cap,
            max_len=max_len,
            chunk=flush_every,
            pcapng_mode=pcapng_mode,
            out_dir=dst_dir,
            base_name=cap.stem
        )
        return cap, chunk_paths, metrics
    except Exception as e:
        logging.exception("Failed to process %s", cap)
        return cap, [], {"seen": 0, "kept": 0, "dns": 0, "tcp_no_payload": 0, "malformed": 0, "non_ip": 0, "file": str(cap)}

def process_tree(in_root: Path,
                 out_root: Path,
                 max_len: int,
                 flush_every: int,
                 workers: int,
                 pcapng_mode: str):
    classes = [p for p in in_root.iterdir() if p.is_dir()]
tasks = []
    for cls_dir in sorted(classes):
        dst_dir = out_root / cls_dir.name
        files = sorted(list(cls_dir.glob("*.pcap")) + list(cls_dir.glob("*.pcapng")))
for cap in files:
            tasks.append((cap, dst_dir))

    worker_fn = partial(_process_one_capture,
                        max_len=max_len,
                        flush_every=flush_every,
                        pcapng_mode=pcapng_mode)

    total_tasks = len(tasks)
    logging.info("Processing %d capture files...", total_tasks)

    # Global aggregates
    agg = {"seen": 0, "kept": 0, "dns": 0, "tcp_no_payload": 0, "malformed": 0, "non_ip": 0, "zero_kept_files": 0}
    if workers <= 1:
        for cap, dst in tasks:
            src, chunks, m = worker_fn(cap, dst)
            if m["kept"] == 0:
                agg["zero_kept_files"] += 1
            for k in ("seen", "kept", "dns", "tcp_no_payload", "malformed", "non_ip"):
                agg[k] += int(m.get(k, 0))
            logging.info("[OK] %s -> %d chunk(s)", src, len(chunks))
    else:
        nproc = min(max(1, workers), cpu_count())
        with Pool(processes=nproc) as pool:
            for src, chunks, m in pool.starmap(worker_fn, tasks):
                if m["kept"] == 0:
                    agg["zero_kept_files"] += 1
                for k in ("seen", "kept", "dns", "tcp_no_payload", "malformed", "non_ip"):
                    agg[k] += int(m.get(k, 0))
                logging.info("[OK] %s -> %d chunk(s)", src, len(chunks))

    # Summary
    logging.info(
        "Summary: files=%d seen=%d kept=%d dropped: dns=%d tcp_no_payload=%d malformed=%d non_ip=%d zero_kept_files=%d",
        total_tasks, agg["seen"], agg["kept"], agg["dns"], agg["tcp_no_payload"], agg["malformed"], agg["non_ip"], agg["zero_kept_files"]
    )

# ------------------------ CLI -------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stream PCAP/PCAPNG → DeepPacket tensors + flow sidecars (optimized).")
    ap.add_argument("--in-root", required=True, help="Input root with class subfolders")
    ap.add_argument("--out-root", required=True, help="Output root (mirrors class folders)")
    ap.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT, help="Vector length (default 1500)")
    ap.add_argument("--flush-every", type=int, default=50000, help="Flush to disk every N packets")
    ap.add_argument("--workers", type=int, default=1, help="Parallel processes (files in parallel)")
    ap.add_argument("--pcapng-mode", choices=["scapy", "skip"], default="scapy",
                    help="How to read .pcapng (scapy fallback is slower).")
    args = ap.parse_args()

    in_root = Path(args.in_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    process_tree(in_root, out_root,
                 max_len=args.max_len,
                 flush_every=args.flush_every,
                 workers=args.workers,
                 pcapng_mode=args.pcapng_mode)

if __name__ == "__main__":
    main()
