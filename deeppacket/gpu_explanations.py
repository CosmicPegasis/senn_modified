"""GPU-optimized GSENN explanation generation.

This module provides memory-efficient, GPU-accelerated explanation generation
for GSENN models, with the following optimizations:
- Pre-loads all data into GPU memory (avoiding repeated file I/O)
- Batched processing for GPU efficiency
- Progress bars for monitoring
- Generates explanations for train+val+test sets
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)


class GPUExplanationGenerator:
    """GPU-optimized explanation generator for GSENN models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 256,
        skip_bias: bool = True,
    ):
        """
        Initialize GPU explanation generator.

        Args:
            model: Trained GSENN model
            device: Device to use (cuda or cpu)
            batch_size: Batch size for explanation generation
            skip_bias: Skip bias concept in attributions
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.skip_bias = skip_bias

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Initialized GPUExplanationGenerator:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Skip bias: {self.skip_bias}")

    def _preload_dataset(
        self,
        data_loader: DataLoader,
        dataset_name: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-load entire dataset into GPU memory.

        Args:
            data_loader: DataLoader to load from
            dataset_name: Name for logging (e.g., 'train', 'val', 'test')
            max_samples: Maximum samples to load (None = all)

        Returns:
            Tuple of (inputs, targets) as tensors on device
        """
        logger.info(f"Pre-loading {dataset_name} dataset into memory...")

        all_inputs = []
        all_targets = []
        total_samples = 0

        # Use tqdm for progress bar
        with tqdm(
            data_loader,
            desc=f"Loading {dataset_name} data",
            unit="batch",
            ncols=100
        ) as pbar:
            for inputs, targets in pbar:
                # Add to lists
                all_inputs.append(inputs.cpu())  # Keep on CPU initially to save GPU memory
                all_targets.append(targets.cpu())

                total_samples += len(inputs)
                pbar.set_postfix({"samples": total_samples})

                # Check if we've reached max_samples
                if max_samples is not None and total_samples >= max_samples:
                    logger.info(f"Reached max_samples limit ({max_samples})")
                    break

        # Concatenate all batches
        logger.info(f"Concatenating {len(all_inputs)} batches...")
        inputs_tensor = torch.cat(all_inputs, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        # Truncate if needed
        if max_samples is not None and len(inputs_tensor) > max_samples:
            inputs_tensor = inputs_tensor[:max_samples]
            targets_tensor = targets_tensor[:max_samples]

        logger.info(f"Loaded {len(inputs_tensor)} samples from {dataset_name} set")
        logger.info(f"  Input shape: {inputs_tensor.shape}")
        logger.info(f"  Memory: {inputs_tensor.element_size() * inputs_tensor.nelement() / 1024**3:.2f} GB")

        return inputs_tensor, targets_tensor

    def _generate_batch_explanations(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate explanations for a batch of inputs.

        Args:
            inputs: Input tensor (B, 1, 1, 1500) on device
            targets: Target tensor (B,) on device

        Returns:
            Tuple of (attributions, predictions, targets) as numpy arrays
            attributions: (B, n_features) - GSENN theta values for predicted class
            predictions: (B,) - predicted class indices
            targets: (B,) - true class indices
        """
        with torch.no_grad():
            # Forward pass to get predictions and thetas
            logits = self.model(inputs)

            # Get predicted classes
            if logits.dim() == 2:
                # Multi-class: (B, nclasses)
                predictions = logits.argmax(dim=1)
            else:
                # Binary: (B,)
                predictions = (logits.squeeze() > 0).long()

            # Get theta attributions
            # self.model.thetas has shape (B, nconcepts, nclasses)
            attrib_mat = self.model.thetas.cpu()

            nx, natt, nclass = attrib_mat.shape

            # Extract attributions for predicted class
            # Use gather to select attrib_mat[i, :, predictions[i]] for each sample i
            pred_indices = predictions.cpu().view(-1, 1, 1).expand(-1, natt, 1)  # (B, natt, 1)
            attributions = attrib_mat.gather(2, pred_indices).squeeze(-1)  # (B, natt)

            # Skip bias concept if requested
            if self.skip_bias and getattr(self.model.conceptizer, "add_bias", None):
                attributions = attributions[:, :-1]

            # Convert to numpy
            attributions_np = attributions.numpy()
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()

        return attributions_np, predictions_np, targets_np

    def generate_explanations(
        self,
        inputs_tensor: torch.Tensor,
        targets_tensor: torch.Tensor,
        dataset_name: str,
    ) -> Dict[str, np.ndarray]:
        """
        Generate explanations for all samples in a pre-loaded dataset.

        Args:
            inputs_tensor: Pre-loaded inputs (N, 1, 1, 1500) on CPU
            targets_tensor: Pre-loaded targets (N,) on CPU
            dataset_name: Name for logging (e.g., 'train', 'val', 'test')

        Returns:
            Dictionary containing:
                - 'attributions': (N, n_features) explanations
                - 'predictions': (N,) predicted classes
                - 'targets': (N,) true classes
        """
        logger.info(f"\nGenerating explanations for {dataset_name} set...")
        logger.info(f"  Total samples: {len(inputs_tensor)}")
        logger.info(f"  Batch size: {self.batch_size}")

        n_samples = len(inputs_tensor)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        # Lists to accumulate results
        all_attributions = []
        all_predictions = []
        all_targets = []

        # Process in batches with progress bar
        with tqdm(
            total=n_samples,
            desc=f"Generating {dataset_name} explanations",
            unit="sample",
            ncols=100
        ) as pbar:
            for batch_idx in range(n_batches):
                # Get batch slice
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)

                # Move batch to device
                inputs_batch = inputs_tensor[start_idx:end_idx].to(self.device)
                targets_batch = targets_tensor[start_idx:end_idx].to(self.device)

                # Generate explanations for batch
                attributions, predictions, targets = self._generate_batch_explanations(
                    inputs_batch, targets_batch
                )

                # Store results
                all_attributions.append(attributions)
                all_predictions.append(predictions)
                all_targets.append(targets)

                # Update progress
                pbar.update(end_idx - start_idx)

                # Free GPU memory
                del inputs_batch, targets_batch
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Concatenate all results
        logger.info(f"Concatenating results from {len(all_attributions)} batches...")
        results = {
            'attributions': np.concatenate(all_attributions, axis=0),
            'predictions': np.concatenate(all_predictions, axis=0),
            'targets': np.concatenate(all_targets, axis=0),
        }

        logger.info(f"Generated {len(results['attributions'])} explanations")
        logger.info(f"  Attribution shape: {results['attributions'].shape}")

        return results

    def generate_all_explanations(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate explanations for train/val/test sets.

        Args:
            train_loader: Training data loader (None to skip)
            val_loader: Validation data loader (None to skip)
            test_loader: Test data loader (None to skip)
            max_train_samples: Max samples from train set (None = all)
            max_val_samples: Max samples from val set (None = all)
            max_test_samples: Max samples from test set (None = all)

        Returns:
            Dictionary mapping dataset name to explanations:
            {
                'train': {'attributions': ..., 'predictions': ..., 'targets': ...},
                'val': {...},
                'test': {...}
            }
        """
        logger.info("=" * 70)
        logger.info("GPU-OPTIMIZED GSENN EXPLANATION GENERATION")
        logger.info("=" * 70)

        results = {}

        # Process each dataset
        datasets = [
            ('train', train_loader, max_train_samples),
            ('val', val_loader, max_val_samples),
            ('test', test_loader, max_test_samples),
        ]

        for dataset_name, data_loader, max_samples in datasets:
            if data_loader is None:
                logger.info(f"\nSkipping {dataset_name} set (no loader provided)")
                continue

            try:
                # Pre-load dataset
                inputs_tensor, targets_tensor = self._preload_dataset(
                    data_loader, dataset_name, max_samples
                )

                # Generate explanations
                dataset_results = self.generate_explanations(
                    inputs_tensor, targets_tensor, dataset_name
                )

                results[dataset_name] = dataset_results

                # Free memory
                del inputs_tensor, targets_tensor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error processing {dataset_name} set: {e}")
                logger.exception(e)
                continue

        logger.info("\n" + "=" * 70)
        logger.info("EXPLANATION GENERATION COMPLETE")
        logger.info("=" * 70)

        return results

    def aggregate_explanations_by_class(
        self,
        explanations: Dict[str, np.ndarray],
        class_names: List[str],
    ) -> Dict[int, np.ndarray]:
        """
        Aggregate explanations by true class.

        Args:
            explanations: Dictionary from generate_explanations/generate_all_explanations
            class_names: List of class names

        Returns:
            Dictionary mapping class_idx -> mean_attributions (n_features,)
        """
        attributions = explanations['attributions']
        targets = explanations['targets']

        aggregated = {}

        for class_idx in range(len(class_names)):
            # Get all samples for this class
            class_mask = targets == class_idx
            class_attributions = attributions[class_mask]

            if len(class_attributions) > 0:
                # Compute mean attributions
                mean_attr = np.mean(class_attributions, axis=0)
                aggregated[class_idx] = mean_attr

                logger.info(f"Class {class_idx} ({class_names[class_idx]}): "
                          f"{len(class_attributions)} samples")
            else:
                logger.warning(f"Class {class_idx} ({class_names[class_idx]}): "
                             f"No samples found")
                aggregated[class_idx] = np.zeros(attributions.shape[1])

        return aggregated
