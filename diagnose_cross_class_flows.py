#!/usr/bin/env python3
"""
Diagnose cross-class flows to understand the dataset contamination issue.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def analyze_cross_class_flows(root: str, flow_suffix: str = ".flow.npy"):
    """
    Find all flows that appear in multiple classes and analyze them.
    """
    root_path = Path(root)
    
    # Map: flow_id -> list of (class_name, file_path, packet_count)
    flow_to_locations = defaultdict(list)
    
    classes = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    print("Scanning dataset...")
    for cls_dir in classes:
        data_files = sorted(cls_dir.glob("*.npy"))
        # Filter out .flow.npy sidecar files
        data_files = [f for f in data_files if not f.name.endswith(flow_suffix)]
        
        for data_file in data_files:
            flow_file = data_file.parent / (data_file.stem + flow_suffix)
            
            if not flow_file.exists():
                continue
            
            try:
                flow_arr = np.load(flow_file, mmap_mode="r")
                unique_flows, counts = np.unique(flow_arr, return_counts=True)
                
                for fid, count in zip(unique_flows, counts):
                    flow_to_locations[int(fid)].append((
                        cls_dir.name,
                        data_file.name,
                        int(count)
                    ))
            except Exception as e:
                print(f"Error reading {flow_file}: {e}")
                continue
    
    # Find cross-class flows
    cross_class_flows = {
        fid: locs for fid, locs in flow_to_locations.items()
        if len(set(loc[0] for loc in locs)) > 1
    }
    
    print(f"\n{'='*70}")
    print(f"CROSS-CLASS FLOW ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Total unique flows: {len(flow_to_locations)}")
    print(f"Cross-class flows: {len(cross_class_flows)} ({100*len(cross_class_flows)/len(flow_to_locations):.1f}%)")
    
    # Count packets affected
    cross_class_packets = sum(
        sum(loc[2] for loc in locs)
        for locs in cross_class_flows.values()
    )
    total_packets = sum(
        sum(loc[2] for loc in locs)
        for locs in flow_to_locations.values()
    )
    
    print(f"Packets in cross-class flows: {cross_class_packets}/{total_packets} ({100*cross_class_packets/total_packets:.1f}%)\n")
    
    # Analyze class pair contamination
    class_pairs = defaultdict(int)
    for fid, locs in cross_class_flows.items():
        classes_involved = sorted(set(loc[0] for loc in locs))
        for i, c1 in enumerate(classes_involved):
            for c2 in classes_involved[i+1:]:
                class_pairs[(c1, c2)] += 1
    
    print(f"Top class pair contaminations:")
    for (c1, c2), count in sorted(class_pairs.items(), key=lambda x: -x[1])[:10]:
        print(f"  {c1} ↔ {c2}: {count} flows")
    
    # Show examples of most contaminated flows
    print(f"\nMost contaminated flows (top 10):")
    sorted_flows = sorted(
        cross_class_flows.items(),
        key=lambda x: sum(loc[2] for loc in x[1]),
        reverse=True
    )
    
    for i, (fid, locs) in enumerate(sorted_flows[:10], 1):
        classes_involved = set(loc[0] for loc in locs)
        total_pkts = sum(loc[2] for loc in locs)
        print(f"\n  {i}. Flow ID: {fid}")
        print(f"     Total packets: {total_pkts}")
        print(f"     Classes: {', '.join(sorted(classes_involved))}")
        print(f"     Distribution:")
        for cls, fname, count in sorted(locs, key=lambda x: -x[2]):
            print(f"       - {cls}/{fname}: {count} packets")
    
    # Statistics by class
    print(f"\n{'='*70}")
    print(f"CONTAMINATION BY CLASS")
    print(f"{'='*70}\n")
    
    class_stats = defaultdict(lambda: {'total_flows': set(), 'cross_class_flows': set(), 'total_packets': 0, 'cross_class_packets': 0})
    
    for fid, locs in flow_to_locations.items():
        classes_involved = set(loc[0] for loc in locs)
        is_cross_class = len(classes_involved) > 1
        
        for cls, fname, count in locs:
            class_stats[cls]['total_flows'].add(fid)  # Track unique flow IDs
            class_stats[cls]['total_packets'] += count
            if is_cross_class:
                class_stats[cls]['cross_class_flows'].add(fid)  # Track unique flow IDs
                class_stats[cls]['cross_class_packets'] += count
    
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        total_flows = len(stats['total_flows'])
        cross_flows = len(stats['cross_class_flows'])
        flow_pct = 100 * cross_flows / total_flows if total_flows > 0 else 0
        pkt_pct = 100 * stats['cross_class_packets'] / stats['total_packets'] if stats['total_packets'] > 0 else 0
        
        print(f"{cls}:")
        print(f"  Total flows: {total_flows}")
        print(f"  Cross-class flows: {cross_flows} ({flow_pct:.1f}%)")
        print(f"  Total packets: {stats['total_packets']}")
        print(f"  Cross-class packets: {stats['cross_class_packets']} ({pkt_pct:.1f}%)")
        print()
    
    print(f"{'='*70}\n")
    
    return cross_class_flows, flow_to_locations


def main():
    parser = argparse.ArgumentParser(description="Diagnose cross-class flow contamination")
    parser.add_argument("--root", default="./pcaps_flow", help="Dataset root directory")
    parser.add_argument("--flow-suffix", default=".flow.npy", help="Flow file suffix")
    args = parser.parse_args()
    
    analyze_cross_class_flows(args.root, args.flow_suffix)


if __name__ == "__main__":
    main()

