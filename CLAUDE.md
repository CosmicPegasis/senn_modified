# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start: What You Need for DeepPacket

**Minimal files required:**
- ✅ `deep_pkt.py` - Main training script
- ✅ `deeppacket/` - Self-contained module (NO external SENN dependency)
- ✅ Your data in `pcaps_flow/` directory

**What you DON'T need:**
- ❌ `SENN/` directory - Original framework (not a dependency for DeepPacket)
- ❌ `robust_interpret/` - Not needed for `--use_gpu_explanations`
- ❌ `scripts/` - Only for original SENN examples
- ❌ `modified.py` - Deprecated legacy script

**Dependencies:**
- Core: `torch`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `matplotlib`, `seaborn`
- NOT needed: `robust_interpret`, `shap`, `lime` (GPU explanations are built-in)

## Overview

This repository contains a Self-Explaining Neural Network (SENN) implementation adapted for network traffic classification (DeepPacket). The codebase has two primary components:

1. **Original SENN**: Framework for self-explaining neural networks on various datasets (MNIST, CIFAR, COMPAS, etc.) - **NOT REQUIRED for DeepPacket**
2. **DeepPacket Extension**: Network intrusion detection using SENN on raw packet data from PCAP files - **FULLY SELF-CONTAINED** in `deeppacket/` module

**Important:** The `deeppacket/` module is completely self-contained with all SENN components refactored into `deeppacket/models.py`. The `SENN/` directory is only needed for running original SENN examples (MNIST, CIFAR, COMPAS) and is **NOT a dependency** for DeepPacket training.

## Project Structure

```
deeppacket/                # ✅ REQUIRED: Self-contained DeepPacket module
  models.py                # GSENN model for packet classification (all SENN components included)
  trainers.py              # ClassificationTrainer, GradPenaltyTrainer
  datasets.py              # DeepPacketNPYDataset with flow-aware splitting
  flow_utils.py            # Flow-based train/test splitting utilities
  gpu_explanations.py      # GPU-accelerated explanation generation
  utils.py                 # Utility functions

deep_pkt.py                # ✅ REQUIRED: Main entry point for DeepPacket training

pre_proc/                  # Optional: PCAP preprocessing utilities
  preproc_flow.py          # Streaming PCAP → .npy tensors + .flow.npy sidecars
  cicids_proc.py           # CIC-IDS-2017 specific preprocessing

verify_flow_dataset.py     # Optional: Dataset integrity verification tool

# OPTIONAL - NOT NEEDED FOR DEEPPACKET:
SENN/                      # ❌ NOT REQUIRED: Original SENN framework (for MNIST/CIFAR/COMPAS only)
  models.py                # Base SENN architecture
  conceptizers.py          # Concept extraction layers (h(x))
  parametrizers.py         # Parameter generation layers (theta(x))
  aggregators.py           # Aggregation functions
  trainers.py              # Training utilities and regularizers

scripts/                   # ❌ NOT REQUIRED: Training scripts for original SENN datasets only
  main_mnist.py
  main_cifar.py
  main_compas.py

modified.py                # ❌ DEPRECATED: Legacy monolithic script (use deep_pkt.py instead)
```

## Core Architecture

### SENN Components (implemented in `deeppacket/models.py`)

All SENN components are self-contained in the `deeppacket` module. SENN models follow a three-stage architecture:

1. **Conceptizer** h(x): Maps inputs → interpretable concepts
2. **Parametrizer** θ(x): Generates class-specific parameters from inputs
3. **Aggregator**: Combines concepts and parameters → predictions

**DeepPacket Implementation** (all in `deeppacket/models.py`):
- **Conceptizer**: `InputConceptizer` treats raw packet bytes as concepts
- **Parametrizer**: `LinearParametrizer` is an MLP generating θ(x) ∈ ℝ^(nconcept × nclass)
- **Aggregator**: `AdditiveScalarAggregator` computes logits = Σᵢ hᵢ·θᵢ
- **GSENN**: Complete model combining all three components

**No external dependencies on SENN/ directory** - all components are refactored and included in `deeppacket/models.py`.

### Flow-Based Dataset Splitting

Network traffic must be split by **flow** (not packet) to prevent data leakage. A flow = bidirectional 5-tuple (src_ip, dst_ip, src_port, dst_port, protocol).

- **Preprocessing** (`preproc_flow.py`): Generates `.npy` packet tensors + `.flow.npy` sidecars with 64-bit flow IDs
- **Dataset splitting** (`deeppacket/flow_utils.py`): `split_deeppacket_by_flow()` ensures all packets from same flow go to same split
- **Verification** (`verify_flow_dataset.py`): Checks alignment, flow consistency

## Common Commands

### Training DeepPacket

Basic training (with flow-based splitting):
```bash
python deep_pkt.py --root ./proc_pcaps_by_flow --train --epochs 10
```

Key arguments:
- `--root`: Directory with preprocessed `.npy` files organized by class
- `--train`: Train from scratch (omit to load existing model)
- `--epochs`: Number of training epochs
- `--nconcepts`: Number of interpretable concepts (default: 10)
- `--theta_reg_lambda`: Regularization strength for θ stability (default: 1e-2)
- `--theta_reg_type`: Regularization type (grad1/grad2/grad3, default: grad3)
- `--cuda`: Use GPU if available
- `--save_model`: Save trained model

### GPU-Optimized GSENN Explanations (NEW)

**Fast, memory-efficient explanation generation for all datasets:**

```bash
# Generate GSENN explanations for train+val+test sets using GPU batching
python deep_pkt.py --root ./proc_pcaps_by_flow --use_gpu_explanations --cuda

# With custom batch size (larger = faster but more memory)
python deep_pkt.py --root ./proc_pcaps_by_flow --use_gpu_explanations \
  --explanation_batch_size 512 --cuda

# Limit samples to save memory
python deep_pkt.py --root ./proc_pcaps_by_flow --use_gpu_explanations \
  --max_explanation_samples 10000 --cuda
```

**Key features:**
- **Pre-loads all data** into memory once (no repeated file I/O)
- **GPU batched processing** (10-100x faster than legacy method)
- **Progress bars** with tqdm for real-time monitoring
- **All datasets**: Generates explanations for train, validation, and test sets
- **Memory efficient**: Processes in configurable batches, frees GPU memory regularly
- **Automatic visualization**: Saves top-K feature importance plots per class
- **Saves results**: Raw and aggregated explanations saved as `.npz` files

Arguments:
- `--use_gpu_explanations`: Enable GPU-optimized explanation generation
- `--explanation_batch_size`: Batch size for GPU processing (default: 256)
- `--max_explanation_samples`: Max samples per dataset (None = all)

**Output files** (saved to `models/` directory):
- `gsenn_gpu_{train|val|test}_raw.npz`: Raw attributions, predictions, targets
- `gsenn_gpu_{train|val|test}_aggregated.npz`: Mean attributions per class
- `gsenn_gpu_{train|val|test}_class{N}_{classname}_top20.png`: Visualizations

**Legacy explanation method** (slow, limited samples, SHAP/LIME support):
```bash
python deep_pkt.py --root ./proc_pcaps_by_flow --use_gsenn --use_shap --use_lime --cuda
```

Training with options optimized for HPC:
```bash
python deep_pkt.py --root ./data --train --cuda --epochs 20 \
  --limit_files_per_split 100 --max_rows_per_file 10000 \
  --print_freq 100 --disable_metrics --disable_explanations
```

Memory management flags:
- `--limit_files_per_split`: Max files per train/test split (0 = unlimited)
- `--max_rows_per_file`: Max packets per file to load
- `--max_batches_per_epoch`: Limit batches per epoch (for large datasets)
- `--disable_metrics`: Skip confusion matrix/classification report
- `--disable_explanations`: Skip SHAP/LIME analysis (saves time)

### Preprocessing PCAPs

Convert PCAP files to DeepPacket format with flow sidecars:
```bash
python pre_proc/preproc_flow.py -i ./pcaps -o ./proc_pcaps_by_flow
```

For CIC-IDS-2017 dataset:
```bash
python pre_proc/cicids_proc.py --input ./CIC-IDS-2017 --output ./cicids_processed
```

### Dataset Verification

Check flow-based dataset integrity:
```bash
python verify_flow_dataset.py --root ./proc_pcaps_by_flow --detailed
```

### Training Original SENN (MNIST example)

```bash
python scripts/main_mnist.py --train
```

### Transferring Files to HPC

**Minimal transfer (code only):**
```bash
# Transfer only what's needed for DeepPacket
scp -r deep_pkt.py deeppacket cs1221290@hpc.iitd.ac.in:~/scratch/SENN/
```

**Full transfer including data:**
```bash
# If you need to upload data as well
scp -r deep_pkt.py deeppacket pcaps_flow cs1221290@hpc.iitd.ac.in:~/scratch/SENN/
```

**What NOT to transfer:**
- ❌ `SENN/` directory (not needed for DeepPacket)
- ❌ `scripts/` directory (only for original SENN examples)
- ❌ `modified.py` (deprecated)
- ❌ `robust_interpret/` (not needed for GPU explanations)
- ❌ `models/`, `log/`, `__pycache__/` (generated files)

## Key Implementation Details

### Flow ID Generation

Flow IDs are deterministic 64-bit hashes (blake2b) of:
- Direction-agnostic endpoint tuple: `(min(endpoint), max(endpoint))`
- Endpoint = `(ip_bytes, port_bytes, protocol)`
- IPv4: 4-byte IP, IPv6: 16-byte IP
- Port bytes: 2 bytes (0 if non-TCP/UDP)

This ensures packets from both directions of a connection get the same flow ID.

### Gradient Penalty Regularization

The `GradPenaltyTrainer` in `deeppacket/trainers.py` implements three regularization types:
- `grad1`: L1 norm of ∂θ/∂x (encourages sparse gradients)
- `grad2`: L2 norm of ∂θ/∂x (encourages small gradients)
- `grad3`: L1 norm summed over classes (computational optimization)

Controlled by `--theta_reg_type` and `--theta_reg_lambda`.

### Memory Management

DeepPacket datasets can be massive (GB-TB scale). The codebase uses:
- **Memory-mapped arrays** via `np.load(..., mmap_mode='r')`
- **Lazy loading**: Only loads data chunks as needed
- **File limits**: `--limit_files_per_split`, `--max_rows_per_file`
- **Batch limits**: `--max_batches_per_epoch` for partial epoch training
- **Garbage collection**: Explicit `gc.collect()` after large operations

### Class Imbalance Handling

Options for imbalanced datasets:
- `--handle_imbalance`: Enable weighted loss
- `--weight_method`: "balanced" (sklearn) or "inverse" (1/count)
- `--undersample`: Random undersampling of majority class
- `--undersample_ratio`: Target ratio (e.g., 0.1 = 10% of original size)

## Dependencies

### Core Dependencies (Required)

For DeepPacket training with GPU explanations:
```bash
pip install torch numpy scipy scikit-learn matplotlib seaborn tqdm
```

**Minimal requirements:**
- PyTorch (any recent version, 1.x or 2.x)
- numpy, scipy
- scikit-learn (for metrics)
- tqdm (progress bars)
- matplotlib, seaborn (for visualization, optional on HPC)

### Optional Dependencies (NOT Required for GPU Explanations)

**Only needed for legacy explanation methods:**
- `robust_interpret` (GSENN wrappers - NOT needed, GPU explanations are built-in)
- `shap` (SHAP explainer - NOT needed for GPU explanations)
- `lime` (LIME explainer - NOT needed for GPU explanations)

**Note:** The `--use_gpu_explanations` flag uses built-in GPU-accelerated GSENN explanations and does NOT depend on `robust_interpret`, `shap`, or `lime`.

## Dataset Formats

### Input: Preprocessed .npy files

Directory structure:
```
root/
  class_0/*.npy          # Each .npy: shape (N, 1500) or (1500,)
  class_0/*.flow.npy     # Flow IDs: shape (N,) dtype uint64
  class_1/*.npy
  class_1/*.flow.npy
```

Each packet vector is 1500 bytes, normalized to float32 [0,1].

### Flow Sidecar Format

`.flow.npy` files contain uint64 flow IDs, one per packet row. Must align exactly with corresponding `.npy` file row count.

## Testing and Validation

No formal test suite exists. Validation workflow:
1. Preprocess small PCAP subset
2. Run `verify_flow_dataset.py` to check integrity
3. Train with `--limit_files_per_split 10` to test pipeline
4. Check logs for flow statistics and cross-class flow warnings

## HPC Considerations

When running on HPC clusters:
- Use `--disable_metrics` and `--disable_explanations` to reduce dependencies
- Set `--print_freq` high to reduce I/O
- Use `--max_batches_per_epoch` for time-limited jobs
- Monitor memory with `--limit_files_per_split`
- The codebase gracefully handles missing matplotlib/seaborn/sklearn.metrics

### IIT Delhi HPC Workflow

**Step 1: SSH into HPC**
```bash
ssh cs1221290@hpc.iitd.ac.in
```

**Step 2: Request compute node with GPU**
```bash
qsub -I -P netgen.spons -l select=1:ncpus=8:mem=100G:ngpus=1:centos=skylake -l walltime=6:00:00
```

Parameters:
- `-I`: Interactive mode
- `-P netgen.spons`: Project name
- `select=1:ncpus=8:mem=100G:ngpus=1:centos=skylake`: 1 node, 8 CPUs, 100GB RAM, 1 GPU, Skylake architecture
- `walltime=6:00:00`: 6 hour time limit

**Step 3: Load Apptainer module**
```bash
module load $(cat avail.txt | grep apptainer)
```

**Step 4: Launch container with GPU support**
```bash
apptainer shell --bind /scratch:/scratch --nv myimage.sif
```

Flags:
- `--bind /scratch:/scratch`: Mount scratch directory
- `--nv`: Enable NVIDIA GPU support
- `myimage.sif`: Container image with PyTorch environment

**Step 5: Navigate to project and run training**
```bash
cd ~/scratch/SENN
python deep_pkt.py --root ~/scratch/SENN/pcaps_flow --train --cuda --epochs 20 --use_gpu_explanations
```

**Minimal Required Files on HPC:**
```
~/scratch/SENN/
├── deep_pkt.py           # ✅ Main training script
├── deeppacket/           # ✅ Self-contained core module (models, trainers, datasets, gpu_explanations)
│   ├── __init__.py
│   ├── models.py         # All SENN components included here
│   ├── trainers.py
│   ├── datasets.py
│   ├── flow_utils.py
│   ├── gpu_explanations.py
│   └── utils.py
├── pcaps_flow/           # ✅ Your preprocessed data
│   ├── Chat/             # .npy and .flow.npy files
│   ├── Email/
│   ├── FileTransfer/
│   ├── Streaming/
│   ├── VoIP/
│   ├── VPN-Chat/
│   ├── VPN-Email/
│   ├── VPN-FileTransfer/
│   ├── VPN-P2P/
│   ├── VPN-Streaming/
│   └── VPN-VoIP/
├── models/               # (auto-generated) Saved models and explanation outputs
└── log/                  # (auto-generated) Training logs

# NOT NEEDED:
# ❌ SENN/                # Original framework (not a dependency)
# ❌ scripts/             # Original SENN examples only
# ❌ modified.py          # Legacy script
```

## Git Status Notes

Modified files:
- `modified.py`: Legacy monolithic script (consider using `deep_pkt.py` instead)

Untracked:
- `pcaps_flow/`: Likely preprocessed data (should be in .gitignore)
- `transfer.py`: Parallel rsync utility for HPC uploads
- `upload_to_hpc.sh`: Wrapper for parallel rsync with progress monitoring
