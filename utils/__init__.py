"""Public API for the ``utils`` package.

Importing this package also triggers the registration side effects in each
submodule so that their decorators populate the global registries before any
``build_*`` factory is called.
"""

from .datasets import (
    DATASET_REGISTRY,
    BaseArrayDataset,
    build_dataloader,
    build_dataset,
    register_dataset,
)
from .logger import TrainingLogger
from .losses import LOSS_REGISTRY, BaseLoss, build_loss, register_loss
from .metrics import (
    METRIC_REGISTRY,
    BaseMetric,
    build_metrics,
    compute_metrics,
    register_metric,
)
from .train_utils import (
    apply_denoise_experiment_name_from_model,
    barrier_if_distributed,
    build_loaders,
    build_optimizer,
    build_scheduler,
    count_parameters,
    default_config_relpath_for_train_script,
    destroy_distributed,
    evaluate,
    find_latest_checkpoint,
    init_distributed,
    load_checkpoint,
    load_config,
    maybe_wrap_ddp,
    resolve_denoise_metrics,
    resolve_repo_root,
    save_checkpoint,
    sampler_set_epoch,
    set_seed,
    setup_experiment_dir,
    setup_experiment_dir_distributed,
    train_one_epoch,
    training_device,
    unwrap_ddp,
)
from .visualization import (
    plot_loss_curve,
    plot_sample,
    plot_single_metric_curve,
    visualize_random_sample,
)

__all__ = [
    # datasets
    "BaseArrayDataset",
    "DATASET_REGISTRY",
    "build_dataloader",
    "build_dataset",
    "register_dataset",
    # losses
    "BaseLoss",
    "LOSS_REGISTRY",
    "build_loss",
    "register_loss",
    # metrics
    "BaseMetric",
    "METRIC_REGISTRY",
    "build_metrics",
    "compute_metrics",
    "register_metric",
    # visualization
    "plot_loss_curve",
    "plot_single_metric_curve",
    "plot_sample",
    "visualize_random_sample",
    # logger
    "TrainingLogger",
    # training utilities
    "apply_denoise_experiment_name_from_model",
    "barrier_if_distributed",
    "build_loaders",
    "build_optimizer",
    "count_parameters",
    "build_scheduler",
    "default_config_relpath_for_train_script",
    "destroy_distributed",
    "evaluate",
    "find_latest_checkpoint",
    "init_distributed",
    "load_checkpoint",
    "load_config",
    "maybe_wrap_ddp",
    "resolve_denoise_metrics",
    "resolve_repo_root",
    "save_checkpoint",
    "sampler_set_epoch",
    "set_seed",
    "setup_experiment_dir",
    "setup_experiment_dir_distributed",
    "train_one_epoch",
    "training_device",
    "unwrap_ddp",
]
