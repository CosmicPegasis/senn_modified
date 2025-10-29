#!/usr/bin/env python3
"""
Standalone script to verify flow-based dataset integrity.
Run this after preprocessing to ensure flow splitting will work correctly.

Usage:
    python verify_flow_dataset.py --root ./proc_pcaps_by_flow
    python verify_flow_dataset.py --root ./proc_pcaps_by_flow --detailed
"""

import argparse
import sys
import os
import glob
import numpy as np
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from deep_pkt, but provide fallback implementations
try:
    from deep_pkt import (
        verify_flow_sidecar_alignment,
        verify_flow_class_consistency,
        verify_flow_id_generation,
        FlowAwareDeepPacketDataset,
        _paired_flow_path
    )
    HAS_TORCH = True
except ImportError as e:
    print(e);
    HAS_TORCH = False
    print("Warning: Could not import torch/deep_pkt. Using lightweight verification mode.")
    print("For full verification, ensure torch is installed and run in proper environment.\n")
    
    # Fallback implementations without torch dependency
    def _paired_flow_path(data_path: str, flow_suffix: str = ".flow.npy") -> str:
        if data_path.endswith(".npy"):
            return data_path[:-4] + flow_suffix
        return data_path + flow_suffix
    
    def verify_flow_sidecar_alignment(root: str, flow_suffix: str = ".flow.npy") -> Dict:
        """Lightweight version without torch dependency"""
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
    
    def verify_flow_id_generation(sample_data_file: str, flow_suffix: str = ".flow.npy") -> Dict:
        """Lightweight version without torch dependency"""
        flow_file = _paired_flow_path(sample_data_file, flow_suffix)
        if not os.path.exists(flow_file):
            return {"status": "skipped", "reason": f"No flow file at {flow_file}"}
        
        try:
            flow_arr = np.load(flow_file, mmap_mode="r")
            unique_flows = np.unique(flow_arr)
            total_packets = len(flow_arr)
            avg_packets_per_flow = total_packets / len(unique_flows) if len(unique_flows) > 0 else 0
            all_zeros = np.all(flow_arr == 0)
            
            return {
                "status": "ok",
                "total_packets": int(total_packets),
                "unique_flows": int(len(unique_flows)),
                "avg_packets_per_flow": float(avg_packets_per_flow),
                "all_zeros": bool(all_zeros),
                "sample_flow_ids": [int(x) for x in unique_flows[:5].tolist()]
            }
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    FlowAwareDeepPacketDataset = None
    verify_flow_class_consistency = None


def main():
    parser = argparse.ArgumentParser(
        description="Verify flow-based dataset integrity before training"
    )
    parser.add_argument("--root", required=True, help="Path to processed dataset root")
    parser.add_argument("--flow-suffix", default=".flow.npy", 
                       help="Flow sidecar suffix (default: .flow.npy)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed per-class statistics")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" FLOW DATASET VERIFICATION")
    print("=" * 70)
    print(f"\nDataset root: {args.root}")
    print(f"Flow suffix: {args.flow_suffix}\n")

    # 1. Check sidecar alignment
    print("=" * 70)
    print("1. SIDECAR ALIGNMENT CHECK")
    print("=" * 70)
    alignment = verify_flow_sidecar_alignment(args.root, args.flow_suffix)
    
    print(f"Total data files:        {alignment['total_files']}")
    print(f"Files with sidecars:     {alignment['files_with_sidecars']}")
    print(f"Files without sidecars:  {alignment['files_without_sidecars']}")
    print(f"Misaligned files:        {alignment['misaligned_files']}")
    
    if alignment['issues']:
        print(f"\n⚠️  Issues detected ({len(alignment['issues'])} total):")
        for i, issue in enumerate(alignment['issues'][:10], 1):
            print(f"  {i}. {issue}")
        if len(alignment['issues']) > 10:
            print(f"  ... and {len(alignment['issues']) - 10} more issues")
    else:
        print("\n✓ All sidecars properly aligned!")

    # 2. Check flow class consistency
    print("\n" + "=" * 70)
    print("2. FLOW CLASS CONSISTENCY CHECK")
    print("=" * 70)
    
    if not HAS_TORCH or FlowAwareDeepPacketDataset is None:
        print("   ⚠️  Skipped - requires torch (run in training environment for full check)")
        
        # 3. Sample flow ID generation (lightweight version)
        print("\n" + "=" * 70)
        print("3. FLOW ID GENERATION CHECK")
        print("=" * 70)
        
        # Find a sample file
        classes = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))])
        sample_file = None
        if classes:
            cls_path = os.path.join(args.root, classes[0])
            npy_files = [f for f in glob.glob(os.path.join(cls_path, "*.npy")) 
                        if not f.endswith(args.flow_suffix)]
            if npy_files:
                sample_file = npy_files[0]
        
        if sample_file:
            flow_gen = verify_flow_id_generation(sample_file, args.flow_suffix)
            print(f"Status:                  {flow_gen.get('status', 'unknown')}")
            if flow_gen.get('status') == 'ok':
                print(f"Sample file:             {os.path.basename(sample_file)}")
                print(f"Total packets:           {flow_gen['total_packets']}")
                print(f"Unique flows:            {flow_gen['unique_flows']}")
                print(f"Avg packets/flow:        {flow_gen['avg_packets_per_flow']:.1f}")
                print(f"All zeros (BAD):         {flow_gen['all_zeros']}")
                
                if flow_gen['all_zeros']:
                    print("\n⚠️  WARNING: Flow IDs are all zeros - flow generation may be broken!")
                else:
                    print(f"\nSample flow IDs:         {flow_gen['sample_flow_ids']}")
                    print("✓ Flow IDs appear to be properly generated!")
            else:
                print(f"Reason: {flow_gen.get('reason', 'unknown')}")
        
        # 4. Detailed class statistics (lightweight version)
        if args.detailed:
            print("\n" + "=" * 70)
            print("4. DETAILED CLASS STATISTICS (Lightweight)")
            print("=" * 70)
            print("   ⚠️  Run with torch for full detailed stats")
            
            from collections import defaultdict
            
            class_stats = defaultdict(lambda: {"files": 0, "packets": 0, "flows": set()})
            classes = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))])
            
            for cls_name in classes:
                cls_path = os.path.join(args.root, cls_name)
                npy_files = [f for f in glob.glob(os.path.join(cls_path, "*.npy")) 
                            if not f.endswith(args.flow_suffix)]
                
                for data_file in npy_files:
                    class_stats[cls_name]["files"] += 1
                    try:
                        data_arr = np.load(data_file, mmap_mode="r")
                        nrows = 1 if data_arr.ndim == 1 else data_arr.shape[0]
                        class_stats[cls_name]["packets"] += nrows
                        
                        flow_file = _paired_flow_path(data_file, args.flow_suffix)
                        if os.path.exists(flow_file):
                            flow_arr = np.load(flow_file, mmap_mode="r")
                            unique = np.unique(flow_arr)
                            class_stats[cls_name]["flows"].update(unique.tolist())
                    except:
                        pass
            
            for cls_name in sorted(class_stats.keys()):
                stats = class_stats[cls_name]
                n_flows = len(stats["flows"])
                avg_pkt = stats["packets"] / n_flows if n_flows > 0 else 0
                
                print(f"\n{cls_name}:")
                print(f"  Files:              {stats['files']}")
                print(f"  Total packets:      {stats['packets']}")
                print(f"  Unique flows:       {n_flows}")
                print(f"  Avg packets/flow:   {avg_pkt:.1f}")
    
    else:
        # Full verification with torch
        try:
            base_ds = FlowAwareDeepPacketDataset(
                args.root, 
                max_rows_per_file=None,
                flow_suffix=args.flow_suffix
            )
            consistency = verify_flow_class_consistency(base_ds)
            
            print(f"Total unique flows:      {consistency['total_flows']}")
            print(f"Inconsistent flows:      {consistency['inconsistent_flows']}")
            
            if consistency['issues']:
                print(f"\n⚠️  Flows spanning multiple classes ({len(consistency['issues'])} detected):")
                for i, issue in enumerate(consistency['issues'][:5], 1):
                    print(f"  {i}. Flow ID {issue['flow_id']}: classes {issue['classes']}")
                    print(f"     File: {issue['file']}")
                if len(consistency['issues']) > 5:
                    print(f"  ... and {len(consistency['issues']) - 5} more")
                print("\n⚠️  WARNING: Cross-class flows will cause data leakage!")
            else:
                print("\n✓ All flows are class-consistent!")
            
            # 3. Sample flow ID generation
            print("\n" + "=" * 70)
            print("3. FLOW ID GENERATION CHECK")
            print("=" * 70)
            if base_ds.files:
                sample_file = base_ds.files[0][0]
                flow_gen = verify_flow_id_generation(sample_file, args.flow_suffix)
                
                print(f"Status:                  {flow_gen.get('status', 'unknown')}")
                if flow_gen.get('status') == 'ok':
                    print(f"Sample file:             {os.path.basename(sample_file)}")
                    print(f"Total packets:           {flow_gen['total_packets']}")
                    print(f"Unique flows:            {flow_gen['unique_flows']}")
                    print(f"Avg packets/flow:        {flow_gen['avg_packets_per_flow']:.1f}")
                    print(f"All zeros (BAD):         {flow_gen['all_zeros']}")
                    
                    if flow_gen['all_zeros']:
                        print("\n⚠️  WARNING: Flow IDs are all zeros - flow generation may be broken!")
                    else:
                        print(f"\nSample flow IDs:         {flow_gen['sample_flow_ids']}")
                        print("✓ Flow IDs appear to be properly generated!")
                else:
                    print(f"Reason: {flow_gen.get('reason', 'unknown')}")
            
            # 4. Detailed class statistics (optional)
            if args.detailed:
                print("\n" + "=" * 70)
                print("4. DETAILED CLASS STATISTICS")
                print("=" * 70)
                
                from collections import defaultdict
                
                class_stats = defaultdict(lambda: {"files": 0, "packets": 0, "flows": set()})
                
                for cls_name, cls_idx in base_ds.class_to_idx.items():
                    for file_idx, (path, file_cls_idx) in enumerate(base_ds.files):
                        if file_cls_idx != cls_idx:
                            continue
                        
                        class_stats[cls_name]["files"] += 1
                        nrows = base_ds.counts[file_idx]
                        class_stats[cls_name]["packets"] += nrows
                        
                        # Get flow IDs
                        flow_file = _paired_flow_path(path, args.flow_suffix)
                        if os.path.exists(flow_file):
                            try:
                                flow_arr = np.load(flow_file, mmap_mode="r")
                                unique = np.unique(flow_arr[:nrows])
                                class_stats[cls_name]["flows"].update(unique.tolist())
                            except:
                                pass
                
                for cls_name in sorted(class_stats.keys()):
                    stats = class_stats[cls_name]
                    n_flows = len(stats["flows"])
                    avg_pkt = stats["packets"] / n_flows if n_flows > 0 else 0
                    
                    print(f"\n{cls_name}:")
                    print(f"  Files:              {stats['files']}")
                    print(f"  Total packets:      {stats['packets']}")
                    print(f"  Unique flows:       {n_flows}")
                    print(f"  Avg packets/flow:   {avg_pkt:.1f}")
        
        except Exception as e:
            print(f"\n✗ Error during verification: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Summary
    print("\n" + "=" * 70)
    print(" VERIFICATION SUMMARY")
    print("=" * 70)
    
    issues_found = False
    if alignment['misaligned_files'] > 0:
        print("✗ Misaligned sidecar files detected")
        issues_found = True
    if alignment['files_without_sidecars'] > 0:
        print("⚠️  Some files missing flow sidecars")
        issues_found = True
    
    # Check consistency and flow_gen if they were computed
    if not HAS_TORCH:
        if flow_gen and flow_gen.get('all_zeros'):
            print("✗ Flow IDs are all zeros")
            issues_found = True
    else:
        if 'consistency' in locals() and consistency.get('inconsistent_flows', 0) > 0:
            print("✗ Cross-class flows detected (DATA LEAKAGE RISK)")
            issues_found = True
        if 'flow_gen' in locals() and flow_gen.get('all_zeros'):
            print("✗ Flow IDs are all zeros")
            issues_found = True
    
    if not issues_found:
        print("✓ All checks passed - dataset ready for flow-based splitting!")
        return 0
    else:
        print("\n⚠️  Issues detected - please review above and fix before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())

