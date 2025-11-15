"""Main script for DeepPacket training and evaluation.

This script has been refactored to use modular components from the deeppacket package.
All model components, trainers, datasets, and utilities are now in separate modules.
"""

from __future__ import annotations

import os
import gc
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

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
    split_deeppacket,
    save_flow_train_test_split,
    load_flow_train_test_split,
    run_comprehensive_flow_checks,
)

# Optional imports for feature explanations
try:
    from robust_interpret.explainers import gsenn_wrapper, shap_wrapper, lime_wrapper
    HAS_ROBUST_INTERPRET = True
except ImportError:
    HAS_ROBUST_INTERPRET = False
    print("Warning: robust_interpret not available. Feature explanations will be skipped.")

# Check if SHAP and LIME are available
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not available. SHAP explanations will be skipped.")
try:
    import lime
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("Warning: LIME not available. LIME explanations will be skipped.")

# Ignore ConvergenceWarning
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module="sklearn.linear_model._least_angle"
)

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
            "  python deep_pkt.py --root proc_pcaps/ --flow_split --epochs 20 --cuda\n\n"
            "  # Training with class imbalance handling:\n"
            "  python deep_pkt.py --root proc_pcaps/ --handle_imbalance --undersample --cuda\n\n"
            "  # Training with feature explanations:\n"
            "  python deep_pkt.py --root proc_pcaps/ --use_shap --use_lime --cuda"
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
                        help="Cap rows loaded from each .npy file.")
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
    parser.add_argument("--flow_split", action="store_true",
        help="Split train/val/test by FLOW (requires aligned *.flow.npy sidecars).")
    parser.add_argument("--flow_suffix", type=str, default=".flow.npy",
        help="Sidecar suffix for flow IDs (default: .flow.npy).")
    # Saved flow split options
    parser.add_argument("--use_flow_split_manifest", type=str, default=None,
        help="Path to a saved flow-based train/test split manifest JSON. If set, uses it.")
    parser.add_argument("--save_flow_split", action="store_true",
        help="Create and save a new flow-based train/test split before training, then use it.")
    parser.add_argument("--flow_test_size", type=float, default=0.2,
        help="Test size for creating a new flow-based split when --save_flow_split is set.")
    parser.add_argument("--use_gsenn", action="store_true", default=False,
        help="Enable GSENN explanations. Disabled by default.")
    parser.add_argument("--use_shap", action="store_true", default=False,
        help="Enable SHAP explanations (requires shap package). Disabled by default.")
    parser.add_argument("--use_lime", action="store_true", default=False,
        help="Enable LIME explanations (requires lime package). Disabled by default.")
    parser.add_argument("--num_workers", type=int, default=None,
        help="Number of data loading workers (default: auto-detect based on CPU count, min 2)")

    args_ns = parser.parse_args()

    args = TrainArgs(
        cuda=args_ns.cuda,
        nclasses=2,                      # will be overwritten after probing
        lr=args_ns.lr,
        epochs=args_ns.epochs,
        theta_reg_lambda=args_ns.theta_reg_lambda,
        theta_reg_type=args_ns.theta_reg_type,

        # NEW: propagate the caps
        limit_files_per_split=args_ns.limit_files_per_split,
        max_rows_per_file=args_ns.max_rows_per_file,
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
    args.flow_split = args_ns.flow_split  # type: ignore[attr-defined]
    args.flow_suffix = args_ns.flow_suffix  # type: ignore[attr-defined]
    args.use_flow_split_manifest = args_ns.use_flow_split_manifest  # type: ignore[attr-defined]
    args.save_flow_split = args_ns.save_flow_split  # type: ignore[attr-defined]
    args.flow_test_size = args_ns.flow_test_size  # type: ignore[attr-defined]
    args.use_gsenn = args_ns.use_gsenn  # type: ignore[attr-defined]
    args.use_shap = args_ns.use_shap  # type: ignore[attr-defined]
    args.use_lime = args_ns.use_lime  # type: ignore[attr-defined]
    
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
    args = build_args()
    set_seed(getattr(args, "seed", 2018))

    # Probe classes
    probe = DeepPacketNPYDataset(args.root)  # type: ignore[attr-defined]
    nclasses = len(probe.classes)
    args.nclasses = nclasses

    # Input dim for DeepPacket vectors
    input_dim = 1500 + 1  # +1 because InputConceptizer appends bias; but parametrizer sees raw x, so keep 1500
    raw_input_dim = 1500

    # Data
    if args.use_flow_split_manifest or args.save_flow_split:
        # Use persistent flow-based train/test split
        if args.save_flow_split:
            manifest_path = save_flow_train_test_split(
                root=args.root,
                out_dir=None,
                test_size=getattr(args, "flow_test_size", 0.2),
                seed=getattr(args, "seed", 2018),
                flow_suffix=getattr(args, "flow_suffix", ".flow.npy"),
            )
            print(f"Saved flow split manifest: {manifest_path}")
            args.use_flow_split_manifest = manifest_path  # type: ignore[attr-defined]

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
        valid_loader = None
    else:
        if args.flow_split:
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
        else:
            train_loader, valid_loader, test_loader, train_ds, val_ds, test_ds = split_deeppacket(
                root=args.root,
                valid_size=0.1, test_size=0.1,
                batch_size=args.batch_size,
                num_workers=args.num_workers,  # type: ignore[arg-type]
                shuffle=True,
                limit_files_per_split=args.limit_files_per_split,
                max_rows_per_file=args.max_rows_per_file,
                handle_imbalance=args.handle_imbalance,
                weight_method=args.weight_method,
                undersample=args.undersample,
                undersample_ratio=args.undersample_ratio,
                undersample_strategy=args.undersample_strategy,
                pin_memory=args.cuda,  # Enable pin_memory when using GPU
            )
    # Comprehensive flow-based sanity checks (if using runtime flow split)
    # Skip for saved manifest path (no val split available, and we assume manifest is authoritative)
    if getattr(args, "flow_split", False) and (not args.use_flow_split_manifest):
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

    # Train & evaluate
    trainer.train(train_loader, valid_loader, epochs=args.epochs, save_path=model_path)
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    trainer.model.to(device)
    print_full_config(args, trainer.model)

    print("Computing accuracies…")
    train_acc = trainer.validate(train_loader) if train_loader is not None else None
    val_acc = trainer.validate(valid_loader) if valid_loader is not None else None
    test_acc = trainer.evaluate(test_loader) if test_loader is not None else None

    print("\nFinal accuracies:")
    print(f"  Train Accuracy : {train_acc:.2f}%" if train_acc is not None else "  Train Accuracy : (n/a)")
    print(f"  Val   Accuracy : {val_acc:.2f}%"   if val_acc is not None else "  Val   Accuracy : (n/a)")
    print(f"  Test  Accuracy : {test_acc:.2f}%"  if test_acc is not None else "  Test  Accuracy : (n/a)")

    # Generate and plot confusion matrix on test set
    if test_loader is not None:
        print("\nGenerating confusion matrix on test set...")
        y_pred, y_true = trainer.predict(test_loader)
        
        # Get all class names and labels (including those with zero predictions)
        class_names = probe.classes
        all_labels = list(range(len(class_names)))
        
        # Compute confusion matrix with all labels
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=all_labels, 
                                   target_names=class_names, zero_division=0))
        
        # Plot confusion matrix
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
        
        # Show plot
        # plt.show()
        
        # Generate global feature explanations (aggregated per class)
        if HAS_ROBUST_INTERPRET:
            print("\n" + "=" * 70)
            print(" GENERATING GLOBAL FEATURE EXPLANATIONS")
            print("=" * 70)
            use_gsenn_flag = getattr(args, 'use_gsenn', False)
            use_shap_flag = getattr(args, 'use_shap', False)
            use_lime_flag = getattr(args, 'use_lime', False)
            enabled_methods = []
            if use_gsenn_flag:
                enabled_methods.append("GSENN")
            if HAS_SHAP and use_shap_flag:
                enabled_methods.append("SHAP")
            if HAS_LIME and use_lime_flag:
                enabled_methods.append("LIME")
            
            # Warn if GSENN is disabled
            if not use_gsenn_flag:
                print("WARNING: GSENN explanations are disabled by default. Use --use_gsenn to enable them.")
                warnings.warn(
                    "GSENN explanations are disabled by default. Use --use_gsenn to enable them.",
                    UserWarning,
                    stacklevel=2
                )
            
            if not enabled_methods:
                print("WARNING: No explanation methods enabled. Skipping feature explanations.")
                print("         Use --use_gsenn, --use_shap, or --use_lime to enable explanations.")
                print("=" * 70 + "\n")
            else:
                print("INFO: Processing samples in small batches to control memory usage.")
                print(f"      Enabled methods: {', '.join(enabled_methods)}")
                print("      Total samples per class: GSENN=100, SHAP=50, LIME=50")
                print("=" * 70 + "\n")
            
            # Create feature names for packet bytes (0-1499)
            feature_names = [f"byte_{i}" for i in range(1500)]
            class_names = probe.classes
            
            # Prepare training data for SHAP/LIME (need numpy arrays) - LIMIT SIZE
            print("Preparing training data for explainers (limiting to 5000 samples for memory)...")
            train_data_list = []
            train_labels_list = []
            max_train_samples = 5000
            sample_count = 0
            for inputs, targets in train_loader:
                if sample_count >= max_train_samples:
                    break
                x_np = inputs.squeeze().cpu().numpy()  # (B, 1500)
                y_np = targets.cpu().numpy()
                # Only take what we need
                remaining = max_train_samples - sample_count
                if len(x_np) > remaining:
                    x_np = x_np[:remaining]
                    y_np = y_np[:remaining]
                train_data_list.append(x_np)
                train_labels_list.append(y_np)
                sample_count += len(x_np)
            X_train = np.concatenate(train_data_list, axis=0) if train_data_list else np.array([])
            y_train = np.concatenate(train_labels_list, axis=0) if train_labels_list else np.array([])
            print(f"  Prepared {len(X_train)} training samples (limited for memory)")
            # Clear intermediate lists
            del train_data_list, train_labels_list
            gc.collect()
            
            # Create model wrapper function for SHAP/LIME
            def model_predict_proba(x):
                """Wrapper for model.predict_proba that SHAP/LIME can use."""
                if isinstance(x, np.ndarray):
                    x_tensor = torch.from_numpy(x).float()
                else:
                    x_tensor = x.float()
                
                # Reshape to (B, 1, 1, 1500) if needed
                if x_tensor.dim() == 2:
                    x_tensor = x_tensor.unsqueeze(1).unsqueeze(1)
                
                trainer.model.eval()
                with torch.no_grad():
                    probs = trainer.model.predict_proba(x_tensor, to_numpy=True)
                return probs
            
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
            
            # SHAP explainer (if available and enabled) - MEMORY OPTIMIZED
            use_shap_flag = getattr(args, 'use_shap', False)
            has_shap_local = HAS_SHAP and use_shap_flag
            if not has_shap_local:
                if not use_shap_flag:
                    print("Skipping SHAP explanations (disabled by default, use --use_shap to enable)")
                elif not HAS_SHAP:
                    print("Skipping SHAP explanations (shap package not available)")
            if has_shap_local:
                print("Creating SHAP explainer (MEMORY-OPTIMIZED: small background set)...")
                try:
                    # Use a VERY small subset of training data for SHAP background to save memory
                    n_background = min(10, len(X_train))  # Reduced from 100 to 10
                    background_indices = np.random.choice(len(X_train), n_background, replace=False)
                    X_background = X_train[background_indices].copy()
                    y_background = y_train[background_indices].copy()
                    
                    explainers['SHAP'] = shap_wrapper(
                        model_predict_proba,
                        shap_type='kernel',
                        link='identity',
                        mode='classification',
                        multiclass=(args.nclasses > 2),
                        feature_names=feature_names,
                        class_names=class_names,
                        train_data=(X_background, y_background),
                        nsamples=50,  # Reduced from 100 to 50 for memory
                        verbose=False
                    )
                    print("  SHAP explainer created successfully (memory-optimized)")
                    # Clear background data from memory
                    del X_background, y_background, background_indices
                    gc.collect()
                except Exception as e:
                    print(f"  Warning: Could not create SHAP explainer: {e}")
                    print(f"  SHAP is memory-intensive. Consider skipping it or reducing dataset size.")
                    has_shap_local = False
                    if 'SHAP' in explainers:
                        del explainers['SHAP']
                    gc.collect()
            
            # LIME explainer (if available and enabled) - MEMORY OPTIMIZED
            use_lime_flag = getattr(args, 'use_lime', False)
            has_lime_local = HAS_LIME and use_lime_flag
            if not has_lime_local:
                if not use_lime_flag:
                    print("Skipping LIME explanations (disabled by default, use --use_lime to enable)")
                elif not HAS_LIME:
                    print("Skipping LIME explanations (lime package not available)")
            if has_lime_local:
                print("Creating LIME explainer (MEMORY-OPTIMIZED: limited training data)...")
                try:
                    # Use a smaller subset for LIME training data to save memory
                    n_lime_train = min(1000, len(X_train))  # Limit to 1000 samples
                    lime_indices = np.random.choice(len(X_train), n_lime_train, replace=False)
                    X_lime_train = X_train[lime_indices].copy()
                    y_lime_train = y_train[lime_indices].copy()
                    
                    explainers['LIME'] = lime_wrapper(
                        model_predict_proba,
                        lime_type='tabular',
                        mode='classification',
                        multiclass=(args.nclasses > 2),
                        feature_names=feature_names,
                        class_names=class_names,
                        train_data=(X_lime_train, y_lime_train),
                        num_samples=500,  # Reduced from 1000 to 500 for memory
                        num_features=1500,  # All features
                        verbose=False
                    )
                    print("  LIME explainer created successfully (memory-optimized)")
                    # Clear LIME training data from memory
                    del X_lime_train, y_lime_train, lime_indices
                    gc.collect()
                except Exception as e:
                    print(f"  Warning: Could not create LIME explainer: {e}")
                    print(f"  LIME is memory-intensive. Consider skipping it or reducing dataset size.")
                    has_lime_local = False
                    if 'LIME' in explainers:
                        del explainers['LIME']
                    gc.collect()
            
            # Clear training data arrays now that explainers are created (they have their own copies)
            try:
                del X_train, y_train
                gc.collect()
                print("Cleared training data arrays from memory")
            except NameError:
                pass  # Already deleted or never created
            
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
                            
                            # Get predicted classes for SHAP/LIME (use predicted class for explanations)
                            predicted_classes_sub = None
                            if explainer_name in ['SHAP', 'LIME']:
                                # Get predictions for this sub-batch
                                x_sub_batch_tensor = torch.from_numpy(x_sub_batch).float().to(device)
                                # Reshape to (B, 1, 1, 1500) if needed
                                if x_sub_batch_tensor.dim() == 2:
                                    x_sub_batch_tensor = x_sub_batch_tensor.unsqueeze(1).unsqueeze(1)
                                
                                trainer.model.eval()
                                with torch.no_grad():
                                    logits = trainer.model(x_sub_batch_tensor)
                                    # Use argmax for all cases (works for both binary nclasses=2 and multiclass)
                                    preds = logits.argmax(dim=1).view(-1)
                                predicted_classes_sub = preds.cpu().numpy()
                                del x_sub_batch_tensor, logits, preds
                            
                            # Generate explanations for this sub-batch
                            try:
                                # Get attributions for samples in sub-batch
                                # GSENN uses true class labels; SHAP/LIME use predicted class labels
                                if explainer_name == 'GSENN':
                                    attributions = explainer(x_sub_batch, y=targets_sub, show_plot=False)
                                elif explainer_name == 'SHAP':
                                    # SHAP: Process samples in small sub-batch, store directly
                                    # Use PREDICTED class for explanations (not true class)
                                    attributions = []
                                    for i in range(len(targets_sub)):
                                        true_cls_idx = int(targets_sub[i])
                                        pred_cls_idx = int(predicted_classes_sub[i]) if predicted_classes_sub is not None else true_cls_idx
                                        
                                        # Skip if we already have enough samples for the true class
                                        if max_samples_per_class is not None and samples_processed[true_cls_idx] >= max_samples_per_class:
                                            continue
                                        
                                        # Generate explanation for the PREDICTED class only
                                        try:
                                            attr = explainer(x_sub_batch[i:i+1], y=pred_cls_idx, show_plot=False)
                                        except (IndexError, KeyError) as e:
                                            # If that fails, try with None (let explainer use its default)
                                            try:
                                                attr = explainer(x_sub_batch[i:i+1], y=None, show_plot=False)
                                            except Exception as e2:
                                                # If both fail, skip this sample
                                                print(f"    Warning: Could not generate SHAP explanation for sample {i} (true_class={true_cls_idx}, pred_class={pred_cls_idx}): {e2}")
                                                continue
                                        
                                        if attr.ndim == 1:
                                            attr_clean = attr
                                        else:
                                            attr_clean = attr[0]
                                        # Ensure it's a 1D array of length 1500
                                        if attr_clean.ndim > 1:
                                            attr_clean = attr_clean.flatten()
                                        if len(attr_clean) > 1500:
                                            attr_clean = attr_clean[:1500]
                                        elif len(attr_clean) < 1500:
                                            padded = np.zeros(1500)
                                            padded[:len(attr_clean)] = attr_clean
                                            attr_clean = padded
                                        # Store directly in explanations_by_class by true class (for analysis)
                                        # But explanation is for predicted class
                                        explanations_by_class[true_cls_idx].append(attr_clean.copy())
                                        samples_processed[true_cls_idx] += 1
                                        
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
                                    # Set to empty since we've already stored everything
                                    attributions = []
                                    # Check if we should exit outer loops
                                    if not need_more:
                                        break
                                elif explainer_name == 'LIME':
                                    # LIME: Process samples in small sub-batch, store directly
                                    # Use PREDICTED class for explanations (not true class)
                                    attributions = []
                                    for i in range(len(targets_sub)):
                                        true_cls_idx = int(targets_sub[i])
                                        pred_cls_idx = int(predicted_classes_sub[i]) if predicted_classes_sub is not None else true_cls_idx
                                        
                                        # Skip if we already have enough samples for the true class
                                        if max_samples_per_class is not None and samples_processed[true_cls_idx] >= max_samples_per_class:
                                            continue
                                        
                                        # Generate explanation for the PREDICTED class only
                                        try:
                                            attr = explainer(x_sub_batch[i:i+1], y=pred_cls_idx, show_plot=False)
                                        except (IndexError, KeyError) as e:
                                            # If that fails, try with None (let explainer use its default)
                                            try:
                                                attr = explainer(x_sub_batch[i:i+1], y=None, show_plot=False)
                                            except Exception as e2:
                                                # If both fail, skip this sample
                                                print(f"    Warning: Could not generate LIME explanation for sample {i} (true_class={true_cls_idx}, pred_class={pred_cls_idx}): {e2}")
                                                continue
                                        
                                        if attr.ndim == 1:
                                            attr_clean = attr
                                        else:
                                            attr_clean = attr[0]
                                        # Ensure it's a 1D array of length 1500
                                        if attr_clean.ndim > 1:
                                            attr_clean = attr_clean.flatten()
                                        if len(attr_clean) > 1500:
                                            attr_clean = attr_clean[:1500]
                                        elif len(attr_clean) < 1500:
                                            padded = np.zeros(1500)
                                            padded[:len(attr_clean)] = attr_clean
                                            attr_clean = padded
                                        # Store directly in explanations_by_class by true class (for analysis)
                                        # But explanation is for predicted class
                                        explanations_by_class[true_cls_idx].append(attr_clean.copy())
                                        samples_processed[true_cls_idx] += 1
                                        
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
                                    # Set to empty since we've already stored everything
                                    attributions = []
                                    # Check if we should exit outer loops
                                    if not need_more:
                                        break
                                else:
                                    attributions = explainer(x_sub_batch, y=targets_sub, show_plot=False)
                            
                                # Handle different attribution shapes (only for non-SHAP/LIME explainers)
                                # SHAP and LIME already stored their attributions directly above
                                if explainer_name not in ['SHAP', 'LIME']:
                                    if isinstance(attributions, np.ndarray):
                                        if attributions.size == 0:
                                            # Skip empty arrays
                                            continue
                                        if attributions.ndim == 1:
                                            attributions = attributions.reshape(1, -1)
                                    elif isinstance(attributions, torch.Tensor):
                                        attributions = attributions.detach().cpu().numpy()
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
                                    # SHAP and LIME already stored directly above, nothing more to do here
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
                max_samples_per_class = {'GSENN': 100, 'SHAP': 50, 'LIME': 50}
                # Batch sizes for incremental processing (process this many at a time, then aggregate)
                batch_process_sizes = {'GSENN': 100, 'SHAP': 25, 'LIME': 25}  # Process in small batches
                
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
        else:
            print("\nSkipping feature explanations (robust_interpret not available)")


if __name__ == "__main__":
    main()
