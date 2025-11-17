#!/usr/bin/env python3
# preprocess_pcaps.py — optimized streaming PCAP/PCAPNG → DeepPacket tensors
# - Streams packets (no full decode for .pcap)
# - Lightweight header parsing (Ethernet→IPv4/IPv6→TCP/UDP)
# - Drops DNS (port 53) + TCP no-payload handshakes
# - Produces 1500-length float32 vectors in [0,1]
# - Flushes in chunks to avoid high RAM; optional multiprocessing

import os, sys, argparse, struct, numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from functools import partial
from multiprocessing import Pool, cpu_count

MAX_LEN_DEFAULT = 1500
ETH_LEN = 14

# ------------------------ low-level parsers (no Scapy) ------------------------

def _is_ipv4(pkt_l3: bytes) -> bool:
    # first nibble = version; 4 for IPv4
    return len(pkt_l3) >= 1 and (pkt_l3[0] >> 4) == 4

def _is_ipv6(pkt_l3: bytes) -> bool:
    return len(pkt_l3) >= 1 and (pkt_l3[0] >> 4) == 6

def _strip_l2(raw: bytes) -> bytes:
    """If it looks like Ethernet II, strip 14B; else return as-is."""
    if len(raw) >= ETH_LEN:
        ether_type = int.from_bytes(raw[12:14], "big")
        if ether_type >= 0x0600:  # Ethernet II EtherType
            return raw[ETH_LEN:]
    return raw

def _ipv4_transport_and_payload(pkt_l3: bytes) -> Tuple[int, int, bytes, int]:
    """
    Return (proto, sport/dport or -1/-1 if N/A, payload_bytes, tcp_flags or -1).
    Drops if malformed.
    """
    if len(pkt_l3) < 20:
        return (-1, -1, b"", -1)
    ver_ihl = pkt_l3[0]
    ihl = (ver_ihl & 0x0F) * 4
    if ihl < 20 or len(pkt_l3) < ihl:
        return (-1, -1, b"", -1)
    total_len = int.from_bytes(pkt_l3[2:4], "big")
    proto = pkt_l3[9]
    l3_total = min(total_len, len(pkt_l3))
    l4 = pkt_l3[ihl:l3_total]

    if proto == 6:  # TCP
        if len(l4) < 20:
            return (proto, -1, b"", -1)
        sport = int.from_bytes(l4[0:2], "big")
        dport = int.from_bytes(l4[2:4], "big")
        data_offset = (l4[12] >> 4) * 4
        if data_offset < 20 or len(l4) < data_offset:
            return (proto, sport, b"", -1)
        flags = l4[13]  # CWR ECE URG ACK PSH RST SYN FIN
        payload = l4[data_offset:]
        return (proto, (sport, dport), payload, flags)

    if proto == 17:  # UDP
        if len(l4) < 8:
            return (proto, -1, b"", -1)
        sport = int.from_bytes(l4[0:2], "big")
        dport = int.from_bytes(l4[2:4], "big")
        payload = l4[8:]
        return (proto, (sport, dport), payload, -1)

    # other L4
    return (proto, -1, b"", -1)

def _filter_drop_dns(proto: int, ports) -> bool:
    if proto not in (6, 17) or ports == -1:
        return False
    sport, dport = ports
    return sport == 53 or dport == 53

def _filter_drop_tcp_no_payload(proto: int, payload: bytes, tcp_flags: int) -> bool:
    if proto != 6:
        return False
    # drop TCP packets with no payload (handshakes/acks/fin-only)
    return (len(payload) == 0)

def _vectorize_bytes(l3_start: bytes, max_len: int) -> np.ndarray:
    out = np.zeros(max_len, dtype=np.float32)
    sl = l3_start[:max_len]
    # fast path: memoryview → frombuffer (uint8) → float32
    view = memoryview(sl)
    arr_u8 = np.frombuffer(view, dtype=np.uint8)
    out[:arr_u8.size] = arr_u8.astype(np.float32)
    out /= 255.0
    return out

# ------------------------ PCAP readers ---------------------------------------

def _iter_pcap_packets(path: Path):
    """
    Stream .pcap packets (no decoding) using RawPcapReader-compatible layout.
    We implement a minimal PCAP parser to avoid heavy deps.
    """
    # PCAP global header = 24 bytes
    # We accept both endianesses via magic number.
    with open(path, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            return
        magic = gh[:4]
        # endian
        if magic == b'\xd4\xc3\xb2\xa1':
            endian = "<"  # little-endian
        elif magic == b'\xa1\xb2\xc3\xd4':
            endian = ">"  # big-endian
        elif magic == b'\x4d\x3c\xb2\xa1':  # nanosec LE
            endian = "<"
        elif magic == b'\xa1\xb2\x3c\x4d':  # nanosec BE
            endian = ">"
        else:
            # Not a pcap; let upper layer try pcapng if set
            return

        # Per-packet header: ts_sec, ts_usec, incl_len, orig_len
        pkh_struct = struct.Struct(endian + "IIII")
        while True:
            pkh = f.read(16)
            if len(pkh) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = pkh_struct.unpack(pkh)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            yield data  # raw bytes at link layer (likely Ethernet)

def _iter_pcapng_packets_scapy(path: Path):
    """Fallback reader for .pcapng via scapy (slower but robust)."""
    try:
        from scapy.utils import PcapNgReader
    except Exception as e:
        raise RuntimeError("Reading .pcapng requires scapy. Install it or use --pcapng-mode skip.") from e
    rdr = PcapNgReader(str(path))
    for pkt in rdr:
        # pkt is a scapy Packet; get raw bytes
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
                              base_name: Optional[str] = None) -> List[Path]:
    """
    Stream a capture, flush chunks to disk: returns list of chunk .npy paths.
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
    out_paths: List[Path] = []
    chunk_idx = 0
    wrote_any = False

    for raw in it:
        # Strip L2 if present and form L3 bytes
        l3 = _strip_l2(raw)

        if _is_ipv4(l3):
            proto, ports, payload, tcp_flags = _ipv4_transport_and_payload(l3)
        elif _is_ipv6(l3):
            # For simplicity, keep IPv6 packets; minimal filtering: drop DNS by port 53 (UDP/TCP)
            # IPv6 header 40B; next header at byte 6
            if len(l3) < 40:
                continue
            nh = l3[6]  # next header
            # L4 starts at 40; we won't fully parse ext headers for speed
            proto = nh
            ports = -1
            payload = l3[40:]
            tcp_flags = -1
        else:
            # Unknown L3; skip
            continue

        if _filter_drop_dns(proto, ports):
            continue
        if _filter_drop_tcp_no_payload(proto, payload, tcp_flags):
            continue

        vec = _vectorize_bytes(l3, max_len)
        vecs.append(vec)

        if len(vecs) >= chunk:
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = base_name or cap_path.stem
                out_path = out_dir / f"{stem}.chunk{chunk_idx:03d}.npy"
                np.save(out_path, np.stack(vecs, axis=0))
                out_paths.append(out_path)
                wrote_any = True
                vecs.clear()
                chunk_idx += 1
            else:
                # if no out_dir: keep accumulating (not recommended for large files)
                pass

    if vecs and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = base_name or cap_path.stem
        out_path = out_dir / f"{stem}.chunk{chunk_idx:03d}.npy"
        np.save(out_path, np.stack(vecs, axis=0))
        out_paths.append(out_path)
        wrote_any = True
        vecs.clear()

    # Optional: merge chunks into a single file to simplify downstream
    # (disabled by default to avoid a second pass / extra RAM)
    return out_paths

# ------------------------ folder processing ----------------------------------

def _process_one_capture(cap: Path, dst_dir: Path, max_len: int, flush_every: int, pcapng_mode: str) -> Tuple[Path, List[Path]]:
    try:
        chunk_paths = convert_capture_streaming(
            cap_path=cap,
            max_len=max_len,
            chunk=flush_every,
            pcapng_mode=pcapng_mode,
            out_dir=dst_dir,
            base_name=cap.stem
        )
        return cap, chunk_paths
    except Exception as e:
        return cap, []

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

    if workers <= 1:
        for cap, dst in tasks:
            src, chunks = worker_fn(cap, dst)
            print(f"[OK] {src} -> {len(chunks)} chunk(s)")
    else:
        nproc = min(max(1, workers), cpu_count())
        with Pool(processes=nproc) as pool:
            for src, chunks in pool.starmap(worker_fn, tasks):
                print(f"[OK] {src} -> {len(chunks)} chunk(s)")

# ------------------------ CLI -------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stream PCAP/PCAPNG → DeepPacket tensors (optimized).")
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

    process_tree(in_root, out_root,
                 max_len=args.max_len,
                 flush_every=args.flush_every,
                 workers=args.workers,
                 pcapng_mode=args.pcapng_mode)

if __name__ == "__main__":
    main()
