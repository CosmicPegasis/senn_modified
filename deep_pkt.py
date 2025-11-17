"""Main script for DeepPacket training and evaluation.

This script has been refactored to use modular components from the deeppacket package.
All model components, trainers, datasets, and utilities are now in separate modules.
"""

from __future__ import annotations

import os
import gc
import argparse
import logging
from typing import Optional, Tuple, List
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import warnings

# Configure logging with timestamps and detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optional imports for plotting and metrics (may not be available on HPC clusters)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plotting will be disabled.", UserWarning)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("seaborn not available. Some plots may be disabled.", UserWarning)

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN_METRICS = True
except ImportError:
    HAS_SKLEARN_METRICS = False
    warnings.warn("sklearn.metrics not available. Confusion matrix and classification report will be skipped.", UserWarning)

# Import from refactored modules
from deeppacket import (
    set_seed,
    TrainArgs,
    InputConceptizer,
    LinearParametrizer,
    AdditiveScalarAggregator,
    GSENN,
    ClassificationTrainer,
    GradPenaltyTrainer,
    DeepPacketNPYDataset,
    split_deeppacket_by_flow,
    save_flow_train_test_split,
    load_flow_train_test_split,
    create_flow_split_dataset_files,
    has_pre_split_dataset,
    load_pre_split_dataset,
    run_comprehensive_flow_checks,
    get_dataset_classes,
    GPUExplanationGenerator,
)

# Optional imports for feature explanations
try:
    from robust_interpret.explainers import gsenn_wrapper
    HAS_ROBUST_INTERPRET = True
except ImportError:
    HAS_ROBUST_INTERPRET = False
    print("Warning: robust_interpret not available. Feature explanations will be skipped.")

# Ignore ConvergenceWarning (if sklearn is available)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
        module="sklearn.linear_model._least_angle"
    )
except ImportError:
    pass  # sklearn not available, skip warning filter

# =========================================================
# Argument parsing and configuration
# =========================================================

def generate_dir_names(dataset: str, args: TrainArgs, make: bool = True) -> Tuple[str, str, str]:
    """Generate directory names for model, log, and results paths."""
    suffix = f"{args.theta_reg_type}_H{args.h_type}_Reg{args.theta_reg_lambda:.0e}_LR{args.lr}"
    model_path = os.path.join(args.model_path, dataset, suffix)
    log_path = os.path.join(args.log_path, dataset, suffix)
    results_path = os.path.join(args.results_path, dataset, suffix)
    if make:
        for p in [model_path, results_path]:
            os.makedirs(p, exist_ok=True)
    return model_path, log_path, results_path

def build_args() -> TrainArgs:
    """Build training arguments from command line."""
    parser = argparse.ArgumentParser(
        description="DeepPacket training and evaluation script with GSENN model support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Basic training with default settings:\n"
            "  python deep_pkt.py --root proc_pcaps/ --epochs 10 --cuda\n\n"
            "  # Training with flow-based split:\n"
            "  python deep_pkt.py --root proc_pcaps/ --epochs 20 --cuda\n\n"
            "  # Training with class imbalance handling:\n"
            "  python deep_pkt.py --root proc_pcaps/ --handle_imbalance --undersample --cuda\n\n"
            "  # GPU-optimized GSENN explanations (fast, all datasets):\n"
            "  python deep_pkt.py --root proc_pcaps/ --use_gpu_explanations --cuda\n\n"
            "  # GPU explanations with custom batch size:\n"
            "  python deep_pkt.py --root proc_pcaps/ --use_gpu_explanations --explanation_batch_size 512 --cuda\n\n"
            "  # Legacy GSENN explanations on CPU:\n"
            "  python deep_pkt.py --root proc_pcaps/ --use_gsenn --cuda"
        )
    )
    parser.add_argument("--root", type=str, default="proc_pcaps/", 
                        help="DeepPacket root directory with class folders containing .npy files")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training and evaluation (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer (default: 1e-3)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA (GPU) if available (default: False, uses CPU)")
    parser.add_argument("--theta_reg_lambda", type=float, default=1e-2,
                        help="Regularization strength for theta (concept parameters) (default: 1e-2)")
    parser.add_argument("--theta_reg_type", type=str, default="grad3", 
                        choices=["unreg", "none", "grad1", "grad2", "grad3"],
                        help="Type of gradient penalty regularization: unreg/none (no regularization), grad1/grad2/grad3 (gradient penalty order) (default: grad3)")
    parser.add_argument("--seed", type=int, default=2018,
                        help="Random seed for reproducibility (default: 2018)")
    parser.add_argument("--limit_files_per_split", type=int, default=0,
                        help="Max files per split (train/val/test). Set small to speed up.")
    parser.add_argument("--max_rows_per_file", type=int, default=None,
                        help="Cap rows loaded from each .npy file. Set to 0 or None for no limit.")
    parser.add_argument("--max_batches_per_epoch", type=int, default=200,
                        help="Train only on this many batches per epoch (0 = no cap).")
    parser.add_argument("--eval_batches", type=int, default=50,
                        help="Validate/test on only this many batches (0 = no cap).")
    parser.add_argument("--handle_imbalance", action="store_true",
                        help="Enable class imbalance handling with weighted sampling.")
    parser.add_argument("--weight_method", type=str, default="balanced", 
                        choices=["balanced", "inverse", "sqrt_inverse"],
                        help="Method for calculating class weights: balanced (sklearn-style), inverse, or sqrt_inverse.")
    parser.add_argument("--undersample", action="store_true",
                        help="Enable mild undersampling to reduce class imbalance.")
    parser.add_argument("--undersample_ratio", type=float, default=0.1,
                        help="Ratio of samples to keep from majority classes (0.1 = keep 10%% of largest class).")
    parser.add_argument("--undersample_strategy", type=str, default="random",
                        choices=["random", "stratified"],
                        help="Undersampling strategy: random or stratified.")
    parser.add_argument("--flow_suffix", type=str, default=".flow.npy",
        help="Sidecar suffix for flow IDs (default: .flow.npy).")
    # Saved flow split options
    parser.add_argument("--use_flow_split_manifest", type=str, default=None,
        help="Path to a saved flow-based train/test split manifest JSON. If set, uses it.")
    parser.add_argument("--save_flow_split", action="store_true",
        help="Create and save a new flow-based train/test split before training, then use it.")
    parser.add_argument("--flow_test_size", type=float, default=0.2,
        help="Test size for creating a new flow-based split when --save_flow_split is set.")
    # Pre-split dataset options (default behavior)
    parser.add_argument("--create_pre_split", action="store_true",
        help="Create permanent flow-split dataset files (train/ and test/ directories) if they don't exist.")
    parser.add_argument("--pre_split_output", type=str, default=None,
        help="Output directory for pre-split dataset. If None, uses root + '_split' or root itself.")
    parser.add_argument("--skip_pre_split_check", action="store_true",
        help="Skip checking for pre-split dataset and always regenerate splits on the fly.")
    parser.add_argument("--use_gsenn", action="store_true", default=False,
        help="Enable GSENN explanations. Disabled by default.")
    parser.add_argument("--use_gpu_explanations", action="store_true", default=False,
        help="Use GPU-optimized batch explanation generation (faster, uses more memory). "
             "Automatically enables GSENN and generates explanations for train+val+test sets.")
    parser.add_argument("--explanation_batch_size", type=int, default=256,
        help="Batch size for GPU explanation generation (default: 256)")
    parser.add_argument("--max_explanation_samples", type=int, default=None,
        help="Max samples per dataset for explanations (None = all, useful to limit memory)")
    parser.add_argument("--num_workers", type=int, default=None,
        help="Number of data loading workers (default: auto-detect based on CPU count, min 2)")
    parser.add_argument("--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level: DEBUG (verbose), INFO (default), WARNING, ERROR")

    args_ns = parser.parse_args()
    
    # Configure logging level based on argument
    log_level = getattr(logging, args_ns.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger.setLevel(log_level)
    # Also set level for deeppacket modules
    logging.getLogger('deeppacket').setLevel(log_level)

    # Normalize max_rows_per_file: 0 means no limit (same as None)
    max_rows_per_file = args_ns.max_rows_per_file
    if max_rows_per_file == 0:
        max_rows_per_file = None

    args = TrainArgs(
        cuda=args_ns.cuda,
        nclasses=2,                      # will be overwritten after probing
        lr=args_ns.lr,
        epochs=args_ns.epochs,
        theta_reg_lambda=args_ns.theta_reg_lambda,
        theta_reg_type=args_ns.theta_reg_type,

        # NEW: propagate the caps
        limit_files_per_split=args_ns.limit_files_per_split,
        max_rows_per_file=max_rows_per_file,
        max_batches_per_epoch=args_ns.max_batches_per_epoch,
        eval_batches=args_ns.eval_batches,
        
        # Class imbalance handling
        handle_imbalance=args_ns.handle_imbalance,
        weight_method=args_ns.weight_method,
        
        # Undersampling options
        undersample=args_ns.undersample,
        undersample_ratio=args_ns.undersample_ratio,
        undersample_strategy=args_ns.undersample_strategy,
    )

    # also keep these convenience attrs
    args.root = args_ns.root         # type: ignore[attr-defined]
    args.batch_size = args_ns.batch_size  # type: ignore[attr-defined]
    args.seed = args_ns.seed         # type: ignore[attr-defined]
    args.flow_suffix = args_ns.flow_suffix  # type: ignore[attr-defined]
    args.use_flow_split_manifest = args_ns.use_flow_split_manifest  # type: ignore[attr-defined]
    args.save_flow_split = args_ns.save_flow_split  # type: ignore[attr-defined]
    args.flow_test_size = args_ns.flow_test_size  # type: ignore[attr-defined]
    args.create_pre_split = args_ns.create_pre_split  # type: ignore[attr-defined]
    args.pre_split_output = args_ns.pre_split_output  # type: ignore[attr-defined]
    args.skip_pre_split_check = args_ns.skip_pre_split_check  # type: ignore[attr-defined]
    args.use_gsenn = args_ns.use_gsenn  # type: ignore[attr-defined]
    args.use_gpu_explanations = args_ns.use_gpu_explanations  # type: ignore[attr-defined]
    args.explanation_batch_size = args_ns.explanation_batch_size  # type: ignore[attr-defined]
    args.max_explanation_samples = args_ns.max_explanation_samples  # type: ignore[attr-defined]

    # Auto-detect num_workers if not specified
    if args_ns.num_workers is None:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Use min of 4 workers or half of CPU count, but at least 2
        args.num_workers = max(2, min(4, cpu_count // 2))  # type: ignore[attr-defined]
    else:
        args.num_workers = args_ns.num_workers  # type: ignore[attr-defined]
    
    return args

def print_full_config(args, model=None):
    """Pretty-print full training configuration and model summary."""
    print("\n" + "=" * 70)
    print(" CURRENT CONFIGURATION")
    print("=" * 70)

    # Print all args (dataclass or AttrDict-compatible)
    if hasattr(args, "__dict__"):
        cfg_dict = vars(args)
    else:
        cfg_dict = args.__dict__ if isinstance(args, object) else dict(args)
    for k in sorted(cfg_dict.keys()):
        print(f"{k:25s}: {cfg_dict[k]}")

    # Print device and model info
    if model is not None:
        device = next(model.parameters()).device
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n" + "-" * 70)
        print("MODEL SUMMARY")
        print("-" * 70)
        print(f"Device:        {device}")
        print(f"Total params:  {n_params:,}")
        print(f"Trainable:     {n_trainable:,}")
        print(f"Model type:    {model.__class__.__name__}")
        print("-" * 70)
        print(model)
    print("=" * 70 + "\n")

# =========================================================
# Main training function
# =========================================================

class VanillaTrainer(ClassificationTrainer):
    """Vanilla trainer without gradient penalty."""
    
    def train_batch(self, inputs, targets):
        """Train on a single batch without gradient penalty."""
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = self.prediction_criterion(
            logits, targets if self.nclasses > 2 else targets.float().unsqueeze(1)
        )
        loss.backward()
        self.optimizer.step()
        return logits.detach(), loss.detach(), {"prediction": float(loss.item()), "grad_penalty": 0.0}


def main():
    logger.info("="*70)
    logger.info("Starting DeepPacket training script")
    logger.info("="*70)
    
    args = build_args()
    set_seed(getattr(args, "seed", 2018))
    logger.info(f"Random seed set to: {getattr(args, 'seed', 2018)}")

    # Probe classes (lightweight - just scan directories, don't load data)
    logger.info(f"Probing dataset classes from: {args.root}")
    classes = get_dataset_classes(args.root)  # type: ignore[attr-defined]
    nclasses = len(classes)
    args.nclasses = nclasses
    logger.info(f"Found {nclasses} classes: {classes}")

    # Input dim for DeepPacket vectors
    input_dim = 1500 + 1  # +1 because InputConceptizer appends bias; but parametrizer sees raw x, so keep 1500
    raw_input_dim = 1500

    # Data
    logger.info("="*70)
    logger.info("Loading datasets and creating data loaders")
    logger.info(f"  Root: {args.root}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Num workers: {args.num_workers}")
    logger.info(f"  CUDA enabled: {args.cuda}")
    logger.info("="*70)
    
    # Check for pre-split dataset first (default behavior)
    use_pre_split = False
    dataset_root = args.root  # type: ignore[attr-defined]
    
    if not args.skip_pre_split_check:  # type: ignore[attr-defined]
        if has_pre_split_dataset(args.root):
            logger.info("Found pre-split dataset (train/ and test/ directories). Using pre-split dataset.")
            use_pre_split = True
            dataset_root = args.root
        elif args.create_pre_split:  # type: ignore[attr-defined]
            # Create pre-split dataset
            logger.info("Creating permanent flow-split dataset files...")
            output_root = args.pre_split_output  # type: ignore[attr-defined]
            if output_root is None:
                # Check if root already has train/test, if not use root + "_split"
                if not has_pre_split_dataset(args.root):
                    output_root = os.path.abspath(args.root) + "_split"
                else:
                    output_root = args.root
            
            try:
                train_dir, test_dir = create_flow_split_dataset_files(
                    root=args.root,
                    output_root=output_root,
                    test_size=getattr(args, "flow_test_size", 0.2),
                    seed=getattr(args, "seed", 2018),
                    flow_suffix=getattr(args, "flow_suffix", ".flow.npy"),
                    num_workers=args.num_workers,  # type: ignore[attr-defined]
                )
                logger.info(f"Created pre-split dataset at: {output_root}")
                # Verify it was created successfully
                if has_pre_split_dataset(output_root):
                    use_pre_split = True
                    dataset_root = output_root
                    logger.info(f"Pre-split dataset verified. Will use: {dataset_root}")
                else:
                    logger.warning(f"Pre-split dataset creation completed but verification failed. Falling back to legacy split.")
            except Exception as e:
                logger.error(f"Failed to create pre-split dataset: {e}")
                logger.warning("Falling back to legacy split method.")
                import traceback
                traceback.print_exc()
                raise
    
    if use_pre_split:
        # Load pre-split dataset (no mapping needed, just load files)
        logger.info("Loading pre-split dataset...")
        train_loader, test_loader, train_ds, test_ds = load_pre_split_dataset(
            root=dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,  # type: ignore[arg-type]
            weight_method=args.weight_method,
            handle_imbalance=args.handle_imbalance,
            undersample=args.undersample,
            undersample_ratio=args.undersample_ratio,
            undersample_strategy=args.undersample_strategy,
            pin_memory=args.cuda,  # Enable pin_memory when using GPU
        )
        logger.info(f"Train dataset size: {len(train_ds)}")
        logger.info(f"Test dataset size: {len(test_ds)}")
        valid_loader = None
    elif args.use_flow_split_manifest or args.save_flow_split:
        # Use persistent flow-based train/test split
        if args.save_flow_split:
            logger.info("Creating and saving flow-based train/test split...")
            manifest_path = save_flow_train_test_split(
                root=args.root,
                out_dir=None,
                test_size=getattr(args, "flow_test_size", 0.2),
                seed=getattr(args, "seed", 2018),
                flow_suffix=getattr(args, "flow_suffix", ".flow.npy"),
            )
            logger.info(f"Saved flow split manifest: {manifest_path}")
            args.use_flow_split_manifest = manifest_path  # type: ignore[attr-defined]

        logger.info("Loading flow-based train/test split from manifest...")
        train_loader, test_loader, train_ds, test_ds = load_flow_train_test_split(
            split_manifest_path=args.use_flow_split_manifest,  # type: ignore[arg-type]
            batch_size=args.batch_size,
            num_workers=args.num_workers,  # type: ignore[arg-type]
            weight_method=args.weight_method,
            handle_imbalance=args.handle_imbalance,
            undersample=args.undersample,
            undersample_ratio=args.undersample_ratio,
            undersample_strategy=args.undersample_strategy,
            max_rows_per_file=args.max_rows_per_file,
            root_override=args.root,  # Allow overriding manifest root for cross-system compatibility
            pin_memory=args.cuda,  # Enable pin_memory when using GPU
        )
        logger.info(f"Train dataset size: {len(train_ds)}")
        logger.info(f"Test dataset size: {len(test_ds)}")
        valid_loader = None
    else:
        logger.info("Creating flow-based split (train/val/test)...")
        train_loader, valid_loader, test_loader, train_ds, val_ds, test_ds = split_deeppacket_by_flow(
            root=args.root,
            valid_size=0.1, test_size=0.1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,  # type: ignore[arg-type]
            weight_method=args.weight_method,
            max_rows_per_file=args.max_rows_per_file,
            handle_imbalance=args.handle_imbalance,
            flow_suffix=args.flow_suffix,
            undersample=args.undersample,
            undersample_ratio=args.undersample_ratio,
            undersample_strategy=args.undersample_strategy,
            pin_memory=args.cuda,  # Enable pin_memory when using GPU
        )
        logger.info(f"Train dataset size: {len(train_ds)}")
        logger.info(f"Val dataset size: {len(val_ds)}")
        logger.info(f"Test dataset size: {len(test_ds)}")
    # Comprehensive flow-based sanity checks (if using runtime flow split)
    # Skip for pre-split datasets, saved manifest path, and pre-split datasets
    # (no val split available, and we assume manifest/pre-split is authoritative)
    if (not use_pre_split) and (not args.use_flow_split_manifest):
        run_comprehensive_flow_checks(args.root, train_ds, val_ds, test_ds, 
                                      flow_suffix=args.flow_suffix)
    
    # Lightweight sanity check on a subset of training samples
    bad_counts = 0
    try:
        sample_limit = min(len(train_ds), 100000)
        for i in range(sample_limit):
            try:
                x, y = train_ds[i]
                # Expect tensors shaped (1, 1, 1500)
                if x is None or x.numel() == 0 or x.dim() != 3 or x.size(-1) != 1500:
                    bad_counts += 1
            except Exception:
                bad_counts += 1
    except Exception:
        # If any unexpected error occurs during checking, don't block training
        pass

    print("bad samples (checked subset):", bad_counts)
    # Model
    conceptizer = InputConceptizer(add_bias=True)  # bias as an extra concept at the end
    nconcepts = raw_input_dim + 1  # each feature + bias
    parametrizer = LinearParametrizer(
        input_dim=raw_input_dim, nconcept=nconcepts, nclass=args.nclasses, hidden=512, only_positive=False
    )
    aggregator = AdditiveScalarAggregator(cdim=1, nclasses=args.nclasses)
    model = GSENN(conceptizer, parametrizer, aggregator, debug=False)

    # Paths
    model_path, _, _ = generate_dir_names("deeppacket", args)

    # Trainer (unregularized or with grad penalty)
    if args.theta_reg_type in ["unreg", "none"]:
        trainer: ClassificationTrainer = VanillaTrainer(model, args)
    else:
        typ = {"grad1": 1, "grad2": 2, "grad3": 3}.get(args.theta_reg_type, 3)
        trainer = GradPenaltyTrainer(model, args, typ=typ)

    # Try to load existing checkpoint
    checkpoint_path = os.path.join(model_path, "model_best.pth.tar")
    if os.path.exists(checkpoint_path):
        logger.info("="*70)
        logger.info(f"Found existing checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            trainer.model.load_state_dict(checkpoint['state_dict'])
            logger.info(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"✓ Best validation accuracy: {checkpoint.get('best_prec1', 'unknown'):.2f}%")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Continuing with fresh model...")
        logger.info("="*70)
    else:
        logger.info("="*70)
        logger.info("No checkpoint found at: " + checkpoint_path)
        logger.info("Will start with fresh model weights")
        logger.info("="*70)

    # Train & evaluate
    if args.epochs > 0:
        logger.info("="*70)
        logger.info("Starting training")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Model path: {model_path}")
        logger.info("="*70)
        trainer.train(train_loader, valid_loader, epochs=args.epochs, save_path=model_path)
    else:
        logger.info("="*70)
        logger.info("Skipping training (epochs=0). Using loaded model for evaluation/explanations.")
        logger.info("="*70)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    logger.info(f"Moving model to device: {device}")
    trainer.model.to(device)
    print_full_config(args, trainer.model)

    # Explanation generation and evaluation
    use_gpu_explanations = getattr(args, 'use_gpu_explanations', False)

    if use_gpu_explanations:
        # Skip accuracy calculations when generating GPU explanations
        print("\nSkipping accuracy calculations and confusion matrix (--use_gpu_explanations enabled)")
        print("\n" + "=" * 70)
        print(" GPU-OPTIMIZED GSENN EXPLANATION GENERATION")
        print("=" * 70)
        print("Using GPU-accelerated batch processing for all datasets (train+val+test)")

        # Initialize GPU explanation generator
        explanation_batch_size = getattr(args, 'explanation_batch_size', 256)
        max_explanation_samples = getattr(args, 'max_explanation_samples', None)

        gpu_explainer = GPUExplanationGenerator(
            model=trainer.model,
            device=device,
            batch_size=explanation_batch_size,
            skip_bias=True,
        )

        # Generate explanations for all datasets
        all_explanations = gpu_explainer.generate_all_explanations(
            train_loader=train_loader if train_loader is not None else None,
            val_loader=valid_loader if valid_loader is not None else None,
            test_loader=test_loader if test_loader is not None else None,
            max_train_samples=max_explanation_samples,
            max_val_samples=max_explanation_samples,
            max_test_samples=max_explanation_samples,
        )

        # Aggregate explanations by class for each dataset
        print("\n" + "=" * 70)
        print("Aggregating feature importance per class...")
        print("=" * 70)

        aggregated_by_dataset = {}
        for dataset_name, explanations in all_explanations.items():
            print(f"\nAggregating {dataset_name} explanations...")
            aggregated = gpu_explainer.aggregate_explanations_by_class(
                explanations, class_names=classes
            )
            aggregated_by_dataset[dataset_name] = aggregated

            # Print summary
            for cls_idx, stats in aggregated.items():
                n_samples = stats['count']
                print(f"  Class {cls_idx} ({classes[cls_idx]}): {n_samples} samples")

        # Visualize and save results
        if HAS_MATPLOTLIB:
            print("\n" + "=" * 70)
            print("Visualizing global feature importance...")
            print("=" * 70)

            for dataset_name, aggregated in aggregated_by_dataset.items():
                for cls_idx, class_stats in aggregated.items():
                    mean_attributions = class_stats['mean']
                    importance = class_stats['mean_abs']
                    top_k = min(20, len(importance))
                    if top_k == 0:
                        continue
                    top_indices = np.argsort(importance)[-top_k:][::-1]
                    top_values = mean_attributions[top_indices]

                    plt.figure(figsize=(10, 6))
                    colors = ['red' if v > 0 else 'blue' for v in top_values]
                    plt.barh(range(top_k), top_values, color=colors, alpha=0.8)
                    plt.yticks(range(top_k), [f"byte_{i}" for i in top_indices])
                    plt.xlabel('Mean Attribution')
                    plt.title(f'Top {top_k} Important Bytes - {dataset_name} - {classes[cls_idx]}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()

                    # Save plot
                    plot_path = os.path.join(
                        model_path,
                        f'gsenn_gpu_{dataset_name}_class{cls_idx}_{classes[cls_idx]}_top{top_k}.png'
                    )
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved: {plot_path}")

        # Build plaintext summary similar to robust_interpret output
        if aggregated_by_dataset:
            print("\n" + "=" * 70)
            print(" GPU EXPLANATION SUMMARY")
            print("=" * 70)

            summary_lines: List[str] = []
            summary_lines.append("=" * 70)
            summary_lines.append(" GPU EXPLANATION SUMMARY")
            datasets_list = ", ".join(aggregated_by_dataset.keys())
            summary_lines.append(f" Datasets: {datasets_list}")
            summary_lines.append("=" * 70)
            summary_lines.append("")

            top_k = 20

            for dataset_name, aggregated in aggregated_by_dataset.items():
                summary_lines.append(f"\nDataset: {dataset_name}")
                summary_lines.append("-" * 70)
                print(f"\nDataset: {dataset_name}")
                print("-" * 70)

                for cls_idx, class_stats in aggregated.items():
                    class_name = classes[cls_idx]
                    count = class_stats['count']
                    summary_lines.append(f"\n{class_name} (class {cls_idx})")
                    summary_lines.append(f"  Samples: {count}")
                    print(f"\n{class_name} (class {cls_idx})")
                    print(f"  Samples: {count}")

                    if count == 0:
                        summary_lines.append("  No samples available for this class.")
                        print("  No samples available for this class.")
                        continue

                    mean_vals = class_stats['mean']
                    std_vals = class_stats['std']
                    mean_abs_vals = class_stats['mean_abs']

                    k = min(top_k, len(mean_vals))
                    top_indices = np.argsort(mean_abs_vals)[-k:][::-1]

                    header = "    Byte | Mean | Std | Mean |abs||"
                    divider = "    " + "-" * 50
                    summary_lines.append(f"\n  Top {k} important bytes:")
                    summary_lines.append(header)
                    summary_lines.append(divider)
                    print(f"\n  Top {k} important bytes:")
                    print(header)
                    print(divider)
                    for idx in top_indices:
                        line = (
                            f"    byte_{idx:4d} | "
                            f"{mean_vals[idx]:7.4f} | "
                            f"{std_vals[idx]:6.4f} | "
                            f"{mean_abs_vals[idx]:8.4f}"
                        )
                        summary_lines.append(line)
                        print(line)

            summary_lines.append("\n" + "=" * 70)
            summary_lines.append(" GPU EXPLANATIONS COMPLETE")
            summary_lines.append("=" * 70)

            summary_path = os.path.join(model_path, "gsenn_gpu_feature_importance_summary.txt")
            with open(summary_path, "w") as f:
                f.write("\n".join(summary_lines))
            print(f"\nSummary saved to: {summary_path}")

        # Clean up large arrays to free memory
        del all_explanations

        print("\n" + "=" * 70)
        print("GPU-optimized explanation generation complete!")
        print("=" * 70 + "\n")

    # Legacy explanation generation (old slow path) - includes accuracy calculations
    elif HAS_ROBUST_INTERPRET:
        # First compute accuracies
        logger.info("="*70)
        logger.info("Computing accuracies on train/val/test sets")
        logger.info("="*70)

        print("Computing accuracies…")
        if train_loader is not None:
            logger.info("Evaluating on training set...")
            train_acc = trainer.validate(train_loader)
        else:
            train_acc = None
            logger.info("Skipping training set evaluation (no train_loader)")

        if valid_loader is not None:
            logger.info("Evaluating on validation set...")
            val_acc = trainer.validate(valid_loader)
        else:
            val_acc = None
            logger.info("Skipping validation set evaluation (no valid_loader)")

        if test_loader is not None:
            logger.info("Evaluating on test set...")
            test_acc = trainer.evaluate(test_loader)
        else:
            test_acc = None
            logger.info("Skipping test set evaluation (no test_loader)")

        print("\nFinal accuracies:")
        print(f"  Train Accuracy : {train_acc:.2f}%" if train_acc is not None else "  Train Accuracy : (n/a)")
        print(f"  Val   Accuracy : {val_acc:.2f}%"   if val_acc is not None else "  Val   Accuracy : (n/a)")
        print(f"  Test  Accuracy : {test_acc:.2f}%"  if test_acc is not None else "  Test  Accuracy : (n/a)")

        # Generate and plot confusion matrix on test set
        if test_loader is not None:
            y_pred, y_true = trainer.predict(test_loader)

            # Get all class names and labels (including those with zero predictions)
            class_names = classes
            all_labels = list(range(len(class_names)))

            # Compute and print confusion matrix and classification report if sklearn.metrics is available
            if HAS_SKLEARN_METRICS:
                print("\nGenerating confusion matrix on test set...")
                cm = confusion_matrix(y_true, y_pred, labels=all_labels)

                # Print classification report
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred, labels=all_labels,
                                           target_names=class_names, zero_division=0))

                # Plot confusion matrix if matplotlib and seaborn are available
                if HAS_MATPLOTLIB and HAS_SEABORN:
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=class_names, yticklabels=class_names,
                                cbar_kws={'label': 'Count'})
                    plt.title('Confusion Matrix - Test Set', fontsize=16, pad=20)
                    plt.ylabel('True Label', fontsize=12)
                    plt.xlabel('Predicted Label', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()

                    # Save confusion matrix
                    cm_path = os.path.join(model_path, 'confusion_matrix.png')
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    print(f"Confusion matrix saved to: {cm_path}")
                elif HAS_MATPLOTLIB:
                    # Fallback: use matplotlib only (no seaborn)
                    plt.figure(figsize=(12, 10))
                    plt.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.title('Confusion Matrix - Test Set', fontsize=16, pad=20)
                    plt.ylabel('True Label', fontsize=12)
                    plt.xlabel('Predicted Label', fontsize=12)
                    plt.colorbar(label='Count')
                    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
                    plt.yticks(range(len(class_names)), class_names)
                    # Add text annotations
                    for i in range(len(class_names)):
                        for j in range(len(class_names)):
                            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
                    plt.tight_layout()

                    cm_path = os.path.join(model_path, 'confusion_matrix.png')
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    print(f"Confusion matrix saved to: {cm_path}")
                else:
                    print("Warning: matplotlib not available. Skipping confusion matrix plot.")
                    print("Confusion matrix values (raw):")
                    print(cm)
            else:
                print("\nWarning: sklearn.metrics not available. Skipping confusion matrix and classification report.")

        # Now generate legacy explanations
        print("\n" + "=" * 70)
        print(" GENERATING GLOBAL FEATURE EXPLANATIONS")
        print("=" * 70)
        use_gsenn_flag = getattr(args, 'use_gsenn', False)
        enabled_methods = []
        if use_gsenn_flag:
            enabled_methods.append("GSENN")
        
        if not use_gsenn_flag:
            print("WARNING: GSENN explanations are disabled by default. Use --use_gsenn to enable them.")
            warnings.warn(
                "GSENN explanations are disabled by default. Use --use_gsenn to enable them.",
                UserWarning,
                stacklevel=2
            )
        
        if not enabled_methods:
            print("WARNING: No explanation methods enabled. Skipping feature explanations.")
            print("         Use --use_gsenn to enable GSENN explanations.")
            print("=" * 70 + "\n")
        else:
            print("INFO: Processing samples in small batches to control memory usage.")
            print(f"      Enabled methods: {', '.join(enabled_methods)}")
            print("      Total samples per class: GSENN=100")
            print("=" * 70 + "\n")
        
        # Create feature names for packet bytes (0-1499)
        feature_names = [f"byte_{i}" for i in range(1500)]
        class_names = classes
        
        # Initialize explainers
        explainers = {}
        
        # GSENN explainer (if enabled)
        use_gsenn_flag = getattr(args, 'use_gsenn', False)
        if not use_gsenn_flag:
            print("Skipping GSENN explanations (disabled by default, use --use_gsenn to enable)")
        if use_gsenn_flag:
            print("Creating GSENN explainer...")
            explainers['GSENN'] = gsenn_wrapper(
                trainer.model,
                mode='classification',
                input_type='feature',
                multiclass=(args.nclasses > 2),
                feature_names=feature_names,
                class_names=class_names,
                train_data=train_loader,
                skip_bias=True,
                verbose=False
            )
        def collect_explanations_for_dataset(data_loader, dataset_name, explainer_name, explainer, 
                                             max_samples_per_class=None, batch_process_size=10,
                                             max_batches=None):
            """
            Collect explanations for all samples in a dataset, grouped by class.
            Processes in small batches and aggregates incrementally to save memory.
            
            Args:
                max_batches: Maximum number of batches to process (None = no limit, respects eval_batches if set)
            """
            trainer.model.eval()
            explanations_by_class = {cls_idx: [] for cls_idx in range(len(class_names))}
            samples_processed = {cls_idx: 0 for cls_idx in range(len(class_names))}
            
            print(f"\nCollecting {explainer_name} explanations from {dataset_name} set...")
            print(f"  Target: {max_samples_per_class} samples per class")
            print(f"  Processing in batches of {batch_process_size} samples")
            if max_batches is not None:
                print(f"  Maximum batches to process: {max_batches}")
            batch_count = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(data_loader):
                    # Respect max_batches limit
                    if max_batches is not None and batch_idx >= max_batches:
                        print(f"    Reached max_batches limit ({max_batches}), stopping")
                        break
                    
                    inputs = inputs.to(device)
                    targets_np = targets.cpu().numpy()
                    
                    # Convert to numpy for explanation (B, 1, 1, 1500) -> (B, 1500)
                    x_batch_np = inputs.squeeze().cpu().numpy()
                    
                    # Check if we need more samples for each class (check early, before processing)
                    need_more = False
                    for cls_idx in range(len(class_names)):
                        if max_samples_per_class is None or samples_processed[cls_idx] < max_samples_per_class:
                            need_more = True
                            break
                    
                    if not need_more:
                        print(f"    Collected enough samples for all classes, stopping at batch {batch_idx + 1}")
                        break
                    
                    # Process samples in small sub-batches to control memory
                    batch_indices = list(range(len(targets_np)))
                    for sub_batch_start in range(0, len(batch_indices), batch_process_size):
                        sub_batch_end = min(sub_batch_start + batch_process_size, len(batch_indices))
                        sub_indices = batch_indices[sub_batch_start:sub_batch_end]
                        
                        if len(sub_indices) == 0:
                            continue
                        
                        # Check if we still need samples for any class in this sub-batch
                        sub_batch_needed = False
                        for i in sub_indices:
                            cls_idx = int(targets_np[i])
                            if max_samples_per_class is None or samples_processed[cls_idx] < max_samples_per_class:
                                sub_batch_needed = True
                                break
                        
                        if not sub_batch_needed:
                            # Double-check: if we have enough for all classes, exit the entire batch loop
                            all_classes_satisfied = True
                            for cls_idx in range(len(class_names)):
                                if max_samples_per_class is not None and samples_processed[cls_idx] < max_samples_per_class:
                                    all_classes_satisfied = False
                                    break
                            if all_classes_satisfied:
                                # Set flag to exit outer batch loop
                                need_more = False
                                break  # Exit sub-batch loop
                            continue  # Skip this sub-batch but continue
                        
                        # Check again before processing (might have been set by previous check)
                        if not need_more:
                            break
                        
                        # Extract sub-batch
                        x_sub_batch = x_batch_np[sub_indices]
                        targets_sub = targets_np[sub_indices]
                        
                        # Generate explanations for this sub-batch
                        try:
                            attributions = explainer(x_sub_batch, y=targets_sub, show_plot=False)
                            
                            if isinstance(attributions, np.ndarray):
                                if attributions.size == 0:
                                    continue
                                if attributions.ndim == 1:
                                    attributions = attributions.reshape(1, -1)
                            elif isinstance(attributions, torch.Tensor):
                                attributions = attributions.detach().cpu().numpy()
                                if attributions.ndim == 1:
                                    attributions = attributions.reshape(1, -1)
                            else:
                                attributions = np.array(attributions)
                                if attributions.ndim == 1:
                                    attributions = attributions.reshape(1, -1)
                            
                            # Store attributions grouped by true class (for GSENN)
                            for i in range(len(targets_sub)):
                                cls_idx = int(targets_sub[i])
                                if max_samples_per_class is None or samples_processed[cls_idx] < max_samples_per_class:
                                    # Get attributions for this sample
                                    if attributions.ndim == 2:
                                        attr_sample = attributions[i]
                                    else:
                                        attr_sample = attributions
                                    
                                    # Ensure it's a 1D array of length 1500
                                    if attr_sample.ndim > 1:
                                        attr_sample = attr_sample.flatten()
                                    if len(attr_sample) > 1500:
                                        attr_sample = attr_sample[:1500]
                                    elif len(attr_sample) < 1500:
                                        # Pad if needed (shouldn't happen, but be safe)
                                        padded = np.zeros(1500)
                                        padded[:len(attr_sample)] = attr_sample
                                        attr_sample = padded
                                    
                                    explanations_by_class[cls_idx].append(attr_sample)
                                    samples_processed[cls_idx] += 1
                                    
                                    # Check if we've reached the limit for all classes
                                    if max_samples_per_class is not None:
                                        all_done = True
                                        for c_idx in range(len(class_names)):
                                            if samples_processed[c_idx] < max_samples_per_class:
                                                all_done = False
                                                break
                                        if all_done:
                                            need_more = False
                                            break  # Exit the inner loop
                            
                            # Check if we should exit outer loops
                            if not need_more:
                                break
                            
                            # Clear memory after each sub-batch to control memory usage
                            del attributions, x_sub_batch, targets_sub
                            gc.collect()
                        
                        except Exception as e:
                            print(f"    Warning: Could not generate {explainer_name} explanations for sub-batch: {e}")
                            import traceback
                            traceback.print_exc()
                            # Clear memory even on error
                            try:
                                del x_sub_batch, targets_sub, attributions
                            except:
                                pass
                            gc.collect()
                            continue
                        
                        # Check if we should exit sub-batch loop (all classes satisfied)
                        if not need_more:
                            break
                    
                    # Check if we should exit outer batch loop (all classes satisfied)
                    if not need_more:
                        break
                    
                    batch_count += 1
                    
                    # Print progress every 100 batches (more frequent for slower methods)
                    if (batch_idx + 1) % 100 == 0:
                        total_collected = sum(samples_processed.values())
                        print(f"    Processed {batch_idx + 1} batches, collected {total_collected} explanations")
            
            # Print summary
            print(f"\n  Collected {explainer_name} explanations per class:")
            for cls_idx in range(len(class_names)):
                count = len(explanations_by_class[cls_idx])
                print(f"    {class_names[cls_idx]}: {count} samples")
            
            return explanations_by_class
        
        # Only proceed if we have at least one explainer enabled
        if not explainers:
            print("No explainers enabled. Skipping explanation generation.")
        else:
            # Collect explanations from train and test sets for all explainers
            # Keep original sample counts, but process in small batches to save memory
            max_samples_per_class = {'GSENN': 100}
            # Batch sizes for incremental processing (process this many at a time, then aggregate)
            batch_process_sizes = {'GSENN': 100}  # Process in small batches
            
            all_train_explanations = {}
            all_test_explanations = {}
            
            for explainer_name, explainer in explainers.items():
                max_samples = max_samples_per_class.get(explainer_name, 200)
                batch_size = batch_process_sizes.get(explainer_name, 10)
                
                # Calculate max_batches based on max_samples_per_class and batch size
                # Rough estimate: (max_samples * n_classes) / batch_size, with some buffer
                # Also respect eval_batches if set
                eval_batches_limit = getattr(args, 'eval_batches', 0) or 0
                if eval_batches_limit > 0:
                    # Use eval_batches as a hard limit
                    max_batches_limit = eval_batches_limit
                else:
                    # Estimate: max_samples per class * n_classes / batch_size, with 2x buffer
                    estimated_batches = int((max_samples * len(class_names)) / args.batch_size * 2)
                    max_batches_limit = max(100, estimated_batches)  # At least 100 batches, but cap at reasonable limit
                
                print(f"\n{'='*70}")
                print(f"Processing {explainer_name} explanations (max {max_samples} samples per class)")
                print(f"  Processing in batches of {batch_size} to control memory")
                print(f"  Max batches limit: {max_batches_limit}")
                print(f"{'='*70}")
                
                # Clear memory before starting
                gc.collect()
                
                all_train_explanations[explainer_name] = collect_explanations_for_dataset(
                    train_loader, "train", explainer_name, explainer, 
                    max_samples_per_class=max_samples,
                    batch_process_size=batch_size,
                    max_batches=max_batches_limit
                )
                
                # Clear memory between train and test
                gc.collect()
                
                all_test_explanations[explainer_name] = collect_explanations_for_dataset(
                    test_loader, "test", explainer_name, explainer,
                    max_samples_per_class=max_samples,
                    batch_process_size=batch_size,
                    max_batches=max_batches_limit
                )
                
                # Clear memory after each explainer
                gc.collect()
                print(f"  Memory cleared after {explainer_name} processing")
            
            def aggregate_explanations_per_class(explanations_by_class):
                """Aggregate explanations per class (mean, std, median)."""
                aggregated = {}
                for cls_idx, attr_list in explanations_by_class.items():
                    if len(attr_list) == 0:
                        aggregated[cls_idx] = {
                            'mean': np.zeros(1500),
                            'std': np.zeros(1500),
                            'median': np.zeros(1500),
                            'mean_abs': np.zeros(1500),
                            'count': 0
                        }
                        continue
                    
                    # Stack all attributions for this class
                    attr_matrix = np.array(attr_list)  # (n_samples, 1500)
                    
                    aggregated[cls_idx] = {
                        'mean': np.mean(attr_matrix, axis=0),
                        'std': np.std(attr_matrix, axis=0),
                        'median': np.median(attr_matrix, axis=0),
                        'mean_abs': np.mean(np.abs(attr_matrix), axis=0),
                        'count': len(attr_list)
                    }
                return aggregated
            
            # Aggregate explanations for all explainers
            print("\n" + "=" * 70)
            print("Aggregating feature importance per class for all explainers...")
            print("=" * 70)
            
            all_train_aggregated = {}
            all_test_aggregated = {}
            
            for explainer_name in explainers.keys():
                print(f"\nAggregating {explainer_name} explanations...")
                all_train_aggregated[explainer_name] = aggregate_explanations_per_class(
                    all_train_explanations[explainer_name]
                )
                all_test_aggregated[explainer_name] = aggregate_explanations_per_class(
                    all_test_explanations[explainer_name]
                )
            
            # Visualize global feature importance per class for all explainers
            if HAS_MATPLOTLIB:
                print("\n" + "=" * 70)
                print("Generating global feature importance visualizations...")
                print("=" * 70)
                
                n_classes = len(class_names)
                n_explainers = len(explainers)
                
                # Create plots for each explainer
                for explainer_name in explainers.keys():
                    train_aggregated = all_train_aggregated[explainer_name]
                    test_aggregated = all_test_aggregated[explainer_name]
                    
                    # Plot 1: Train set
                    fig, axes = plt.subplots(n_classes, 1, figsize=(16, 4 * n_classes))
                    if n_classes == 1:
                        axes = [axes]
                    
                    for cls_idx, class_name in enumerate(class_names):
                        ax = axes[cls_idx]
                        mean_abs = train_aggregated[cls_idx]['mean_abs']
                        
                        # Get top K features
                        top_k = min(50, len(mean_abs))
                        top_indices = np.argsort(mean_abs)[-top_k:][::-1]
                        top_values = train_aggregated[cls_idx]['mean'][top_indices]
                        
                        # Plot with color coding (positive/negative)
                        colors = ['red' if v > 0 else 'blue' for v in top_values]
                        ax.barh(range(len(top_indices)), top_values, color=colors, alpha=0.7)
                        ax.set_yticks(range(len(top_indices)))
                        ax.set_yticklabels([f"byte_{top_indices[i]}" for i in range(len(top_indices))], fontsize=8)
                        ax.set_xlabel(f'Mean Feature Importance ({explainer_name})', fontsize=10)
                        ax.set_title(
                            f"Train Set - {class_name} ({explainer_name}) (n={train_aggregated[cls_idx]['count']} samples, "
                            f"top {top_k} features by |mean|)",
                            fontsize=11, fontweight='bold'
                        )
                        ax.grid(axis='x', alpha=0.3)
                        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                    
                    plt.tight_layout()
                    train_expl_path = os.path.join(model_path, f'global_feature_importance_train_{explainer_name}.png')
                    plt.savefig(train_expl_path, dpi=300, bbox_inches='tight')
                    print(f"Train set global feature importance ({explainer_name}) saved to: {train_expl_path}")
                    # plt.show()
                    
                    # Plot 2: Test set
                    fig, axes = plt.subplots(n_classes, 1, figsize=(16, 4 * n_classes))
                    if n_classes == 1:
                        axes = [axes]
                    
                    for cls_idx, class_name in enumerate(class_names):
                        ax = axes[cls_idx]
                        mean_abs = test_aggregated[cls_idx]['mean_abs']
                        
                        # Get top K features
                        top_k = min(50, len(mean_abs))
                        top_indices = np.argsort(mean_abs)[-top_k:][::-1]
                        top_values = test_aggregated[cls_idx]['mean'][top_indices]
                        
                        # Plot with color coding (positive/negative)
                        colors = ['red' if v > 0 else 'blue' for v in top_values]
                        ax.barh(range(len(top_indices)), top_values, color=colors, alpha=0.7)
                        ax.set_yticks(range(len(top_indices)))
                        ax.set_yticklabels([f"byte_{top_indices[i]}" for i in range(len(top_indices))], fontsize=8)
                        ax.set_xlabel(f'Mean Feature Importance ({explainer_name})', fontsize=10)
                        ax.set_title(
                            f"Test Set - {class_name} ({explainer_name}) (n={test_aggregated[cls_idx]['count']} samples, "
                            f"top {top_k} features by |mean|)",
                            fontsize=11, fontweight='bold'
                        )
                        ax.grid(axis='x', alpha=0.3)
                        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
                    
                    plt.tight_layout()
                    test_expl_path = os.path.join(model_path, f'global_feature_importance_test_{explainer_name}.png')
                    plt.savefig(test_expl_path, dpi=300, bbox_inches='tight')
                    print(f"Test set global feature importance ({explainer_name}) saved to: {test_expl_path}")
                    # plt.show()
            else:
                print("\nWarning: matplotlib not available. Skipping global feature importance visualizations.")
                print("Feature importance statistics are still available in the summary file.")
            
            # Print summary statistics per class and write to file for all explainers
            summary_lines = []
            summary_lines.append("=" * 70)
            summary_lines.append(" GLOBAL FEATURE IMPORTANCE SUMMARY")
            summary_lines.append(f" Methods: {', '.join(explainers.keys())}")
            summary_lines.append("=" * 70)
            summary_lines.append("")
            
            print("\n" + "=" * 70)
            print(" GLOBAL FEATURE IMPORTANCE SUMMARY")
            print(f" Methods: {', '.join(explainers.keys())}")
            print("=" * 70 + "\n")
            
            # Generate summary for each explainer
            for explainer_name in explainers.keys():
                train_aggregated = all_train_aggregated[explainer_name]
                test_aggregated = all_test_aggregated[explainer_name]
                
                summary_lines.append(f"\n{'='*70}")
                summary_lines.append(f" {explainer_name} EXPLANATIONS")
                summary_lines.append(f"{'='*70}")
                print(f"\n{'='*70}")
                print(f" {explainer_name} EXPLANATIONS")
                print(f"{'='*70}")
                
                for cls_idx, class_name in enumerate(class_names):
                    summary_lines.append(f"\n{class_name}:")
                    summary_lines.append(f"  Train samples: {train_aggregated[cls_idx]['count']}")
                    summary_lines.append(f"  Test samples: {test_aggregated[cls_idx]['count']}")
                    
                    print(f"\n{class_name}:")
                    print(f"  Train samples: {train_aggregated[cls_idx]['count']}")
                    print(f"  Test samples: {test_aggregated[cls_idx]['count']}")
                    
                    # Get top features by mean absolute importance
                    train_mean_abs = train_aggregated[cls_idx]['mean_abs']
                    test_mean_abs = test_aggregated[cls_idx]['mean_abs']
                    
                    top_k = 20
                    train_top_indices = np.argsort(train_mean_abs)[-top_k:][::-1]
                    test_top_indices = np.argsort(test_mean_abs)[-top_k:][::-1]
                    
                    summary_lines.append(f"\n  Top {top_k} important bytes (Train Set):")
                    summary_lines.append("    Byte | Mean | Std | Mean |abs||")
                    summary_lines.append("    " + "-" * 50)
                    print(f"\n  Top {top_k} important bytes (Train Set):")
                    print("    Byte | Mean | Std | Mean |abs||")
                    print("    " + "-" * 50)
                    for idx in train_top_indices:
                        mean_val = train_aggregated[cls_idx]['mean'][idx]
                        std_val = train_aggregated[cls_idx]['std'][idx]
                        mean_abs_val = train_mean_abs[idx]
                        line = f"    byte_{idx:4d} | {mean_val:7.4f} | {std_val:6.4f} | {mean_abs_val:8.4f}"
                        summary_lines.append(line)
                        print(line)
                    
                    summary_lines.append(f"\n  Top {top_k} important bytes (Test Set):")
                    summary_lines.append("    Byte | Mean | Std | Mean |abs||")
                    summary_lines.append("    " + "-" * 50)
                    print(f"\n  Top {top_k} important bytes (Test Set):")
                    print("    Byte | Mean | Std | Mean |abs||")
                    print("    " + "-" * 50)
                    for idx in test_top_indices:
                        mean_val = test_aggregated[cls_idx]['mean'][idx]
                        std_val = test_aggregated[cls_idx]['std'][idx]
                        mean_abs_val = test_mean_abs[idx]
                        line = f"    byte_{idx:4d} | {mean_val:7.4f} | {std_val:6.4f} | {mean_abs_val:8.4f}"
                        summary_lines.append(line)
                        print(line)
                    
                    # Compare train vs test top features
                    train_top_set = set(train_top_indices[:10])
                    test_top_set = set(test_top_indices[:10])
                    overlap = train_top_set & test_top_set
                    overlap_line = f"\n  Top 10 feature overlap (Train ∩ Test): {len(overlap)}/10"
                    summary_lines.append(overlap_line)
                    print(overlap_line)
                    if overlap:
                        overlap_bytes = f"    Overlapping bytes: {sorted(overlap)}"
                        summary_lines.append(overlap_bytes)
                        print(overlap_bytes)
                
                # Compare explainers: find common top features across methods
                if len(explainers) > 1:
                    summary_lines.append(f"\n  Cross-Method Comparison:")
                    print(f"\n  Cross-Method Comparison:")
                    for cls_idx, class_name in enumerate(class_names):
                        all_top_features = {}
                        for exp_name in explainers.keys():
                            agg = all_test_aggregated[exp_name]
                            mean_abs = agg[cls_idx]['mean_abs']
                            top_10 = set(np.argsort(mean_abs)[-10:][::-1])
                            all_top_features[exp_name] = top_10
                        
                        # Find intersection of top features across all methods
                        common_features = set.intersection(*all_top_features.values()) if all_top_features else set()
                        summary_lines.append(f"    {class_name} - Common top 10 features across all methods: {len(common_features)}")
                        print(f"    {class_name} - Common top 10 features across all methods: {len(common_features)}")
                        if common_features:
                            summary_lines.append(f"      Common bytes: {sorted(common_features)}")
                            print(f"      Common bytes: {sorted(common_features)}")
            
            summary_lines.append("\n" + "=" * 70)
            summary_lines.append(" GLOBAL FEATURE EXPLANATIONS COMPLETE")
            summary_lines.append("=" * 70)
            
            # Write summary to file
            summary_path = os.path.join(model_path, 'global_feature_importance_summary.txt')
            with open(summary_path, 'w') as f:
                f.write('\n'.join(summary_lines))
            print(f"\nGlobal feature importance summary saved to: {summary_path}")
            
            print("\n" + "=" * 70)
            print(" GLOBAL FEATURE EXPLANATIONS COMPLETE")
            print("=" * 70 + "\n")

    # Final else - neither GPU nor legacy explanations available
    else:
        # Compute accuracies even without explanations
        logger.info("="*70)
        logger.info("Computing accuracies on train/val/test sets")
        logger.info("="*70)

        print("Computing accuracies…")
        if train_loader is not None:
            logger.info("Evaluating on training set...")
            train_acc = trainer.validate(train_loader)
        else:
            train_acc = None
            logger.info("Skipping training set evaluation (no train_loader)")

        if valid_loader is not None:
            logger.info("Evaluating on validation set...")
            val_acc = trainer.validate(valid_loader)
        else:
            val_acc = None
            logger.info("Skipping validation set evaluation (no valid_loader)")

        if test_loader is not None:
            logger.info("Evaluating on test set...")
            test_acc = trainer.evaluate(test_loader)
        else:
            test_acc = None
            logger.info("Skipping test set evaluation (no test_loader)")

        print("\nFinal accuracies:")
        print(f"  Train Accuracy : {train_acc:.2f}%" if train_acc is not None else "  Train Accuracy : (n/a)")
        print(f"  Val   Accuracy : {val_acc:.2f}%"   if val_acc is not None else "  Val   Accuracy : (n/a)")
        print(f"  Test  Accuracy : {test_acc:.2f}%"  if test_acc is not None else "  Test  Accuracy : (n/a)")

        # Generate and plot confusion matrix on test set
        if test_loader is not None:
            y_pred, y_true = trainer.predict(test_loader)

            # Get all class names and labels (including those with zero predictions)
            class_names = classes
            all_labels = list(range(len(class_names)))

            # Compute and print confusion matrix and classification report if sklearn.metrics is available
            if HAS_SKLEARN_METRICS:
                print("\nGenerating confusion matrix on test set...")
                cm = confusion_matrix(y_true, y_pred, labels=all_labels)

                # Print classification report
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred, labels=all_labels,
                                           target_names=class_names, zero_division=0))

                # Plot confusion matrix if matplotlib and seaborn are available
                if HAS_MATPLOTLIB and HAS_SEABORN:
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                xticklabels=class_names, yticklabels=class_names,
                                cbar_kws={'label': 'Count'})
                    plt.title('Confusion Matrix - Test Set', fontsize=16, pad=20)
                    plt.ylabel('True Label', fontsize=12)
                    plt.xlabel('Predicted Label', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()

                    # Save confusion matrix
                    cm_path = os.path.join(model_path, 'confusion_matrix.png')
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    print(f"Confusion matrix saved to: {cm_path}")
                elif HAS_MATPLOTLIB:
                    # Fallback: use matplotlib only (no seaborn)
                    plt.figure(figsize=(12, 10))
                    plt.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.title('Confusion Matrix - Test Set', fontsize=16, pad=20)
                    plt.ylabel('True Label', fontsize=12)
                    plt.xlabel('Predicted Label', fontsize=12)
                    plt.colorbar(label='Count')
                    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
                    plt.yticks(range(len(class_names)), class_names)
                    # Add text annotations
                    for i in range(len(class_names)):
                        for j in range(len(class_names)):
                            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
                    plt.tight_layout()

                    cm_path = os.path.join(model_path, 'confusion_matrix.png')
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    print(f"Confusion matrix saved to: {cm_path}")
                else:
                    print("Warning: matplotlib not available. Skipping confusion matrix plot.")
                    print("Confusion matrix values (raw):")
                    print(cm)
            else:
                print("\nWarning: sklearn.metrics not available. Skipping confusion matrix and classification report.")

        print("\nSkipping feature explanations (robust_interpret not available)")


if __name__ == "__main__":
    main()
