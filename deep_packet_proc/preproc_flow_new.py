#!/usr/bin/env python3
# preprocess_pcaps_with_flows.py — streaming PCAP/PCAPNG → DeepPacket tensors + flow sidecars
# -------------------------------------------------------------------------
# Complies with Deep Packet preprocessing pipeline:
#   ✓ Strip Ethernet header
#   ✓ Drop DNS (TCP/UDP port 53) for IPv4/IPv6
#   ✓ Drop TCP handshake / zero-payload packets (IPv4 + IPv6)
#   ✓ Mask IP addresses (src/dst → zeros)
#   ✓ Pad UDP headers to 20 bytes
#   ✓ Keep (IP header + first 1480 bytes payload) → 1500-byte vector
#   ✓ Normalize to [0,1]
# Adds: deterministic 64-bit direction-agnostic flow IDs and detailed logging.
# -------------------------------------------------------------------------

import argparse
import hashlib
import logging
import numpy as np
import struct
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple, Optional

MAX_LEN_DEFAULT = 1500
ETH_LEN = 14

# ------------------------ basic helpers ------------------------

def _is_ipv4(pkt_l3: bytes) -> bool:
    return len(pkt_l3) >= 1 and (pkt_l3[0] >> 4) == 4

def _is_ipv6(pkt_l3: bytes) -> bool:
    return len(pkt_l3) >= 1 and (pkt_l3[0] >> 4) == 6

def _strip_l2(raw: bytes) -> bytes:
    """Remove Ethernet header (no VLAN/QinQ handling)."""
    if len(raw) >= ETH_LEN:
        ether_type = int.from_bytes(raw[12:14], "big")
        if ether_type >= 0x0600:  # Ethernet II EtherType
            return raw[ETH_LEN:]
    return raw

# ------------------------ IP address masking ------------------------

def _mask_ip_addrs_ipv4(pkt_l3: bytes) -> bytes:
    if len(pkt_l3) < 20:
        return pkt_l3
    ihl = (pkt_l3[0] & 0x0F) * 4
    if ihl < 20 or len(pkt_l3) < ihl:
        return pkt_l3
    b = bytearray(pkt_l3)
    for i in range(12, 20):  # src(12:16), dst(16:20)
        b[i] = 0
    return bytes(b)

def _mask_ip_addrs_ipv6(pkt_l3: bytes) -> bytes:
    if len(pkt_l3) < 40:
        return pkt_l3
    b = bytearray(pkt_l3)
    for i in range(8, 40):  # src(8:24), dst(24:40)
        b[i] = 0
    return bytes(b)

# ------------------------ UDP header padding ------------------------

# Common IPv6 extension headers (subset)
_IPV6_EXT_SET = {0, 43, 44, 50, 51, 60}  # Hop-by-Hop, Routing, Fragment, ESP, AH, Dest-Opts

def _locate_ipv6_l4(pkt_l3: bytes) -> Tuple[Optional[int], Optional[int]]:
    """Return (next_header, l4_offset) walking common IPv6 extension headers."""
    if len(pkt_l3) < 40:
        return None, None
    nh = pkt_l3[6]
    off = 40
    while nh in _IPV6_EXT_SET:
        if nh == 44:  # Fragment header, fixed 8 bytes
            if len(pkt_l3) < off + 8:
                return None, None
            nh = pkt_l3[off]  # next header at first byte of fragment header
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
    """Insert 12 zero bytes after the UDP header to make it 20 bytes (like TCP)."""
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
    """Insert 12 zero bytes after the UDP header to make it 20 bytes (like TCP)."""
    nh, off = _locate_ipv6_l4(pkt_l3)
    if nh != 17 or off is None:
        return pkt_l3
    if len(pkt_l3) < off + 8:
        return pkt_l3
    return pkt_l3[:off+8] + (b"\x00" * 12) + pkt_l3[off+8:]

# ------------------------ parsers ------------------------

def _ipv4_fields(pkt_l3: bytes):
    """
    Return (proto, src_ip4_bytes, dst_ip4_bytes, ihl_bytes,
            (sport,dport or (0,0)), payload_bytes, tcp_flags or -1).
    If malformed, returns proto == -1.
    """
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

    if proto == 6:  # TCP
        if len(l4) < 20:
            return (proto, src, dst, ihl, (0,0), b"", -1)
        sport = int.from_bytes(l4[0:2], "big")
        dport = int.from_bytes(l4[2:4], "big")
        data_offset = (l4[12] >> 4) * 4
        if data_offset < 20 or len(l4) < data_offset:
            return (proto, src, dst, ihl, (sport, dport), b"", -1)
        flags = l4[13]
        payload = l4[data_offset:]
        return (proto, src, dst, ihl, (sport, dport), payload, flags)

    if proto == 17:  # UDP
        if len(l4) < 8:
            return (proto, src, dst, ihl, (0,0), b"", -1)
        sport = int.from_bytes(l4[0:2], "big")
        dport = int.from_bytes(l4[2:4], "big")
        payload = l4[8:]
        return (proto, src, dst, ihl, (sport, dport), payload, -1)

    # other L4
    return (proto, src, dst, ihl, (0,0), l4, -1)

def _ipv6_fields(pkt_l3: bytes):
    """
    Minimal IPv6 parse (no full ext header walk here):
    Return (next_header (proto), src_ip16, dst_ip16, (sport,dport or (0,0)), payload_bytes)
    """
    if len(pkt_l3) < 40:
        return (-1, b"", b"", (0,0), b"")
    nh = pkt_l3[6]
    src = pkt_l3[8:24]
    dst = pkt_l3[24:40]
    payload = pkt_l3[40:]
    sport = dport = 0
    if nh == 6 and len(payload) >= 4:  # TCP (no ext hdrs)
        sport = int.from_bytes(payload[0:2], "big")
        dport = int.from_bytes(payload[2:4], "big")
    elif nh == 17 and len(payload) >= 4:  # UDP (no ext hdrs)
        sport = int.from_bytes(payload[0:2], "big")
        dport = int.from_bytes(payload[2:4], "big")
    return (nh, src, dst, (sport, dport), payload)

# ------------------------ filters ------------------------

def _filter_drop_dns(proto: int, ports: Tuple[int,int]) -> bool:
    """Drop DNS packets (either TCP or UDP, port 53)."""
    if proto not in (6, 17):
        return False
    sport, dport = ports
    return sport == 53 or dport == 53

def _ipv6_tcp_payload_len(pkt_l3: bytes) -> Optional[int]:
    """Compute TCP payload length for IPv6 by locating TCP header after ext headers."""
    nh, off = _locate_ipv6_l4(pkt_l3)
    if nh != 6 or off is None:
        return None
    if len(pkt_l3) < off + 20:
        return 0  # treat as malformed/zero-payload
    data_offset = (pkt_l3[off + 12] >> 4) * 4
    if data_offset < 20 or len(pkt_l3) < off + data_offset:
        return 0
    return len(pkt_l3) - (off + data_offset)

# ------------------------ DeepPacket 1500B construction ------------------------

def _prepare_bytes_for_model_ipv4(pkt_l3: bytes) -> bytes:
    """Mask IPs, pad UDP header, then return (IP header + first 1480 payload) bytes."""
    if len(pkt_l3) < 20:
        return pkt_l3
    ihl = (pkt_l3[0] & 0x0F) * 4
    m = _mask_ip_addrs_ipv4(pkt_l3)
    if m[9] == 17:  # UDP
        m = _pad_udp_header_ipv4(m)
    header = m[:ihl]
    payload = m[ihl:ihl + 1480]
    return header + payload

def _prepare_bytes_for_model_ipv6(pkt_l3: bytes) -> bytes:
    """Mask IPs, pad UDP header (after ext headers), then (IPv6 hdr + first 1480 payload)."""
    if len(pkt_l3) < 40:
        return pkt_l3
    m = _mask_ip_addrs_ipv6(pkt_l3)
    m = _pad_udp_header_ipv6(m)
    header = m[:40]
    payload = m[40:40 + 1480]
    return header + payload

# ------------------------ vectorization ------------------------

def _vectorize_bytes(l3_bytes: bytes, max_len: int) -> np.ndarray:
    out = np.zeros(max_len, dtype=np.float32)
    sl = l3_bytes[:max_len]
    arr_u8 = np.frombuffer(sl, dtype=np.uint8)
    out[:arr_u8.size] = arr_u8.astype(np.float32)
    out /= 255.0
    return out

# ------------------------ flow IDs ------------------------

def _flow_id_from_tuple(proto: int, src_ip: bytes, dst_ip: bytes, sport: int, dport: int) -> int:
    """
    Direction-agnostic, deterministic 64-bit flow ID:
    hash(proto || min(endpoint), max(endpoint)), endpoint = ip||port(2B).
    """
    epA = src_ip + int(sport).to_bytes(2, "big", signed=False)
    epB = dst_ip + int(dport).to_bytes(2, "big", signed=False)
    lo, hi = (epA, epB) if epA <= epB else (epB, epA)
    blob = bytes([proto & 0xFF]) + lo + hi
    digest = hashlib.blake2b(blob, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)

# ------------------------ PCAP readers ------------------------

def _iter_pcap_packets(path: Path):
    """Minimal PCAP reader (no timestamps used)."""
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
    """PCAPNG reader via scapy (fallback)."""
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

# ------------------------ conversion core ------------------------

def convert_capture_streaming(cap_path: Path,
                              max_len: int,
                              chunk: int,
                              pcapng_mode: str = "scapy",
                              out_dir: Optional[Path] = None,
                              base_name: Optional[str] = None):
    """
    Stream a capture, flush chunks to disk: returns (list of data .npy paths, metrics dict).
    Also writes aligned .flow.npy with uint64 flow IDs.
    """
    assert chunk > 0
    ext = cap_path.suffix.lower()
    if ext == ".pcap":
        it = _iter_pcap_packets(cap_path)
    elif ext == ".pcapng":
        if pcapng_mode == "skip":
            return [], {"seen":0,"kept":0,"dns":0,"tcp_no_payload":0,"malformed":0,"non_ip":0,"file":str(cap_path)}
        it = _iter_pcapng_packets_scapy(cap_path)
    else:
        return [], {"seen":0,"kept":0,"dns":0,"tcp_no_payload":0,"malformed":0,"non_ip":0,"file":str(cap_path)}

    vecs: List[np.ndarray] = []
    flows: List[int] = []
    out_paths: List[Path] = []
    chunk_idx = 0

    # Metrics
    total_seen = total_kept = 0
    dropped_dns = dropped_no_payload = dropped_malformed = dropped_non_ip = 0

    for raw in it:
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
            # Drop TCP zero-payload (handshakes/acks without data)
            if proto == 6 and len(payload) == 0:
                dropped_no_payload += 1
                continue
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
            # IPv6 TCP zero-payload (requires ext hdr walk)
            plen = _ipv6_tcp_payload_len(l3)
            if plen == 0 and proto == 6:
                dropped_no_payload += 1
                continue
            sport, dport = ports
            fid = _flow_id_from_tuple(proto, src, dst, sport or 0, dport or 0)
            l3_for_model = _prepare_bytes_for_model_ipv6(l3)

        else:
            dropped_non_ip += 1
            continue

        # Vectorize and accumulate
        vec = _vectorize_bytes(l3_for_model, max_len)
        vecs.append(vec)
        flows.append(fid)
        total_kept += 1

        # Flush chunk
        if out_dir is not None and len(vecs) >= chunk:
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = base_name or cap_path.stem
            base = f"{stem}.chunk{chunk_idx:03d}"
            data_path = out_dir / f"{base}.npy"
            flow_path = out_dir / f"{base}.flow.npy"
            np.save(data_path, np.stack(vecs, axis=0))
            np.save(flow_path, np.asarray(flows, dtype=np.uint64))
            out_paths.append(data_path)
            vecs.clear(); flows.clear(); chunk_idx += 1

    # Final flush
    if out_dir is not None and vecs:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = base_name or cap_path.stem
        base = f"{stem}.chunk{chunk_idx:03d}"
        data_path = out_dir / f"{base}.npy"
        flow_path = out_dir / f"{base}.flow.npy"
        np.save(data_path, np.stack(vecs, axis=0))
        np.save(flow_path, np.asarray(flows, dtype=np.uint64))
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

    logging.info(
        "[%s] packets: seen=%d kept=%d dropped_dns=%d dropped_tcp_no_payload=%d dropped_malformed=%d dropped_non_ip=%d",
        cap_path.name, total_seen, total_kept, dropped_dns, dropped_no_payload, dropped_malformed, dropped_non_ip,
    )
    if total_kept == 0:
        logging.warning("[%s] kept 0 packets → no output chunks written", cap_path.name)

    return out_paths, metrics

# ------------------------ folder processing ------------------------

def _process_one_capture(cap: Path, dst_dir: Path, max_len: int, flush_every: int, pcapng_mode: str):
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
    except Exception:
        logging.exception("Failed to process %s", cap)
        return cap, [], {"seen":0, "kept":0, "dns":0, "tcp_no_payload":0, "malformed":0, "non_ip":0, "file":str(cap)}

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

    total_tasks = len(tasks)
    logging.info("Processing %d capture files...", total_tasks)

    # Worker function
    worker_fn = partial(_process_one_capture,
                        max_len=max_len,
                        flush_every=flush_every,
                        pcapng_mode=pcapng_mode)

    # Aggregates
    agg = {"seen":0, "kept":0, "dns":0, "tcp_no_payload":0, "malformed":0, "non_ip":0, "zero_kept_files":0}

    if workers <= 1:
        for cap, dst in tasks:
            src, chunks, m = worker_fn(cap, dst)
            if m.get("kept", 0) == 0:
                agg["zero_kept_files"] += 1
            for k in ("seen","kept","dns","tcp_no_payload","malformed","non_ip"):
                agg[k] += int(m.get(k, 0))
            logging.info("[OK] %s -> %d chunk(s)", src, len(chunks))
    else:
        nproc = min(max(1, workers), cpu_count())
        with Pool(processes=nproc) as pool:
            for src, chunks, m in pool.starmap(worker_fn, tasks):
                if m.get("kept", 0) == 0:
                    agg["zero_kept_files"] += 1
                for k in ("seen","kept","dns","tcp_no_payload","malformed","non_ip"):
                    agg[k] += int(m.get(k, 0))
                logging.info("[OK] %s -> %d chunk(s)", src, len(chunks))

    # Summary
    logging.info(
        "Summary: files=%d seen=%d kept=%d dropped: dns=%d tcp_no_payload=%d malformed=%d non_ip=%d zero_kept_files=%d",
        total_tasks, agg["seen"], agg["kept"], agg["dns"], agg["tcp_no_payload"], agg["malformed"], agg["non_ip"], agg["zero_kept_files"]
    )

# ------------------------ verification ------------------------

def verify_output_integrity(out_root: Path) -> None:
    """
    Verify integrity of processed output: check alignment, flow IDs, etc.
    Run this after processing to ensure everything is correct.
    """
    print("\n" + "=" * 70)
    print(" VERIFYING OUTPUT INTEGRITY")
    print("=" * 70 + "\n")
    
    classes = [p for p in out_root.iterdir() if p.is_dir()]
    total_files = 0
    total_packets = 0
    total_flows = 0
    total_misaligned = 0
    flow_id_issues = []
    
    for cls_dir in sorted(classes):
        print(f"Checking class: {cls_dir.name}")
        data_files = sorted(cls_dir.glob("*.npy"))
        # Filter out .flow.npy files
        data_files = [f for f in data_files if not f.name.endswith(".flow.npy")]
        
        class_packets = 0
        class_flows = set()
        
        for data_file in data_files:
            total_files += 1
            flow_file = data_file.parent / (data_file.stem + ".flow.npy")
            
            if not flow_file.exists():
                print(f"  ⚠️  Missing flow file: {flow_file.name}")
                continue
            
            try:
                data_arr = np.load(data_file, mmap_mode="r")
                flow_arr = np.load(flow_file, mmap_mode="r")
                
                # Check dimensions
                data_rows = data_arr.shape[0] if data_arr.ndim > 1 else 1
                flow_rows = flow_arr.shape[0] if flow_arr.ndim > 0 else 1
                
                if data_rows != flow_rows:
                    print(f"  ✗ Misaligned: {data_file.name} ({data_rows} vs {flow_rows})")
                    total_misaligned += 1
                    continue
                
                # Check flow IDs
                unique_flows = np.unique(flow_arr)
                if len(unique_flows) == 0:
                    flow_id_issues.append(f"{cls_dir.name}/{data_file.name}: no flows")
                elif np.all(flow_arr == 0):
                    flow_id_issues.append(f"{cls_dir.name}/{data_file.name}: all zeros")
                
                class_packets += data_rows
                class_flows.update(unique_flows.tolist())
                
            except Exception as e:
                print(f"  ✗ Error reading {data_file.name}: {e}")
                continue
        
        total_packets += class_packets
        total_flows += len(class_flows)
        avg_pkt_per_flow = class_packets / len(class_flows) if class_flows else 0
        
        print(f"  Files: {len(data_files)}")
        print(f"  Packets: {class_packets}")
        print(f"  Unique flows: {len(class_flows)}")
        print(f"  Avg packets/flow: {avg_pkt_per_flow:.1f}\n")
    
    print("-" * 70)
    print(f"TOTAL FILES: {total_files}")
    print(f"TOTAL PACKETS: {total_packets}")
    print(f"TOTAL UNIQUE FLOWS: {total_flows}")
    print(f"MISALIGNED FILES: {total_misaligned}")
    
    if flow_id_issues:
        print(f"\n⚠️  Flow ID Issues ({len(flow_id_issues)}):")
        for issue in flow_id_issues[:10]:
            print(f"  - {issue}")
        if len(flow_id_issues) > 10:
            print(f"  ... and {len(flow_id_issues) - 10} more")
    
    if total_misaligned == 0 and not flow_id_issues:
        print("\n✓ All checks passed!")
    else:
        print("\n✗ Issues found - review above")
    
    print("=" * 70 + "\n")

# ------------------------ CLI ------------------------

def main():
    ap = argparse.ArgumentParser(description="Stream PCAP/PCAPNG → DeepPacket tensors + flow sidecars (optimized, compliant).")
    ap.add_argument("--in-root", help="Input root with class subfolders")
    ap.add_argument("--out-root", required=True, help="Output root (mirrors class folders)")
    ap.add_argument("--max-len", type=int, default=MAX_LEN_DEFAULT, help="Vector length (default 1500)")
    ap.add_argument("--flush-every", type=int, default=50000, help="Flush to disk every N packets")
    ap.add_argument("--workers", type=int, default=1, help="Parallel processes (files in parallel)")
    ap.add_argument("--pcapng-mode", choices=["scapy", "skip"], default="scapy",
                    help="How to read .pcapng (scapy fallback is slower).")
    ap.add_argument("--verify-only", action="store_true",
                    help="Only verify output integrity (skip processing)")
    args = ap.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.verify_only:
        # Just run verification on existing output
        verify_output_integrity(out_root)
    else:
        # Process captures
        if not args.in_root:
            ap.error("--in-root is required when not using --verify-only")
        
        in_root = Path(args.in_root).expanduser().resolve()
        process_tree(in_root, out_root,
                     max_len=args.max_len,
                     flush_every=args.flush_every,
                     workers=args.workers,
                     pcapng_mode=args.pcapng_mode)
        
        # Auto-verify after processing
        print("\nProcessing complete. Running verification...\n")
        verify_output_integrity(out_root)

if __name__ == "__main__":
    main()
