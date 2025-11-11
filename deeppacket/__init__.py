"""DeepPacket: Self-Explaining Neural Networks for Packet Classification"""

from .utils import (
    set_seed,
    AverageMeter,
    save_checkpoint,
    binary_accuracy_from_logits,
    multiclass_precision_at_k,
    compute_jacobian_sum,
    compute_jacobian,
    CL_loss,
)

from .models import (
    InputConceptizer,
    LinearParametrizer,
    AdditiveScalarAggregator,
    GSENN,
)

from .trainers import (
    TrainArgs,
    ClassificationTrainer,
    GradPenaltyTrainer,
)

from .datasets import (
    DeepPacketNPYDataset,
    UndersampledDeepPacketNPYDataset,
    FlowAwareDeepPacketDataset,
    SelectedRowsDeepPacketDataset,
)

from .flow_utils import (
    split_deeppacket_by_flow,
    split_deeppacket,
    save_flow_train_test_split,
    load_flow_train_test_split,
    run_comprehensive_flow_checks,
    _assert_no_flow_overlap_datasets,
)

__all__ = [
    # Utils
    "set_seed",
    "AverageMeter",
    "save_checkpoint",
    "binary_accuracy_from_logits",
    "multiclass_precision_at_k",
    "compute_jacobian_sum",
    "compute_jacobian",
    "CL_loss",
    # Models
    "InputConceptizer",
    "LinearParametrizer",
    "AdditiveScalarAggregator",
    "GSENN",
    # Trainers
    "TrainArgs",
    "ClassificationTrainer",
    "GradPenaltyTrainer",
    # Datasets
    "DeepPacketNPYDataset",
    "UndersampledDeepPacketNPYDataset",
    "FlowAwareDeepPacketDataset",
    "SelectedRowsDeepPacketDataset",
    # Flow utils
    "split_deeppacket_by_flow",
    "split_deeppacket",
    "save_flow_train_test_split",
    "load_flow_train_test_split",
    "run_comprehensive_flow_checks",
    "_assert_no_flow_overlap_datasets",
]

