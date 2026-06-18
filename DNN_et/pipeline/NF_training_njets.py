import argparse
import json
import logging
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from classes import load_data, load_variables
from groupings import GROUPING_NAMES


SEED = 42
PATIENCE = 30

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataframe_complete.feather"
MASKS_PATH = PROJECT_ROOT / "configs" / "masks.yaml"
VARIABLES_PATH = PROJECT_ROOT / "configs" / "training_variables_nf.yaml"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config_NF.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "Training_resuluts_NF"

REDUCED_WEIGHT_GROUPINGS = GROUPING_NAMES
PROCESSES = ("wjets", "qcd")
REGIONS = ("AR-like", "SR-like")

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    t.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclass(frozen=True)
class NFConfig:
    bsize_train: int
    bsize_val: int
    grad_clip: float
    n_epochs: int
    use_amp: bool
    n_layers: int
    hidden_dims: int
    s_scale: float
    use_cut_preprocessing: bool
    cut_preprocessing_index: tuple[int, ...]
    cut_preprocessing_thresholds: tuple[float, ...]
    cut_preprocessing_epsilon: float
    use_tail_preprocessing: bool
    tail_preprocessing_index: tuple[int, ...]
    tail_preprocessing_type: tuple[str, ...]
    tail_preprocessing_center: tuple[float, ...]
    tail_preprocessing_scale: tuple[float, ...]
    tail_preprocessing_epsilon: float
    lr: float
    weight_decay: float
    eps: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_threshold: float
    scheduler_cooldown: int
    scheduler_min_lr: float
    scheduler_eps: float

    def to_nested_dict(self) -> dict:
        return {
            "training": {
                "bsize_train": self.bsize_train,
                "bsize_val": self.bsize_val,
                "grad_clip": self.grad_clip,
                "n_epochs": self.n_epochs,
                "use_amp": self.use_amp,
            },
            "model": {
                "n_layers": self.n_layers,
                "hidden_dims": self.hidden_dims,
                "s_scale": self.s_scale,
                "use_cut_preprocessing": self.use_cut_preprocessing,
                "cut_preprocessing_index": list(
                    self.cut_preprocessing_index
                ),
                "cut_preprocessing_thresholds": list(
                    self.cut_preprocessing_thresholds
                ),
                "cut_preprocessing_epsilon": (
                    self.cut_preprocessing_epsilon
                ),
                "use_tail_preprocessing": self.use_tail_preprocessing,
                "tail_preprocessing_index": list(
                    self.tail_preprocessing_index
                ),
                "tail_preprocessing_type": list(
                    self.tail_preprocessing_type
                ),
                "tail_preprocessing_center": list(
                    self.tail_preprocessing_center
                ),
                "tail_preprocessing_scale": list(
                    self.tail_preprocessing_scale
                ),
                "tail_preprocessing_epsilon": (
                    self.tail_preprocessing_epsilon
                ),
            },
            "optimizer": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "eps": self.eps,
            },
            "scheduler": {
                "factor": self.scheduler_factor,
                "patience": self.scheduler_patience,
                "threshold": self.scheduler_threshold,
                "cooldown": self.scheduler_cooldown,
                "min_lr": self.scheduler_min_lr,
                "eps": self.scheduler_eps,
            },
        }

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "NFConfig":
        with open(path, "r") as stream:
            raw = yaml.safe_load(stream)

        training = raw["training"]
        model = raw["model"]
        optimizer = raw["optimizer"]
        scheduler = raw["scheduler"]

        return cls(
            bsize_train=int(training["bsize_train"]),
            bsize_val=int(training["bsize_val"]),
            grad_clip=float(training["grad_clip"]),
            n_epochs=int(training["n_epochs"]),
            use_amp=bool(training["use_amp"]),
            n_layers=int(model["n_layers"]),
            hidden_dims=int(model["hidden_dims"]),
            s_scale=float(model["s_scale"]),
            use_cut_preprocessing=bool(
                model.get("use_cut_preprocessing", True)
            ),
            cut_preprocessing_index=tuple(
                int(value)
                for value in model.get("cut_preprocessing_index", (0, 1))
            ),
            cut_preprocessing_thresholds=tuple(
                float(value)
                for value in model.get(
                    "cut_preprocessing_thresholds",
                    (33.0, 30.0),
                )
            ),
            cut_preprocessing_epsilon=float(
                model.get("cut_preprocessing_epsilon", 1e-6)
            ),
            use_tail_preprocessing=bool(
                model.get("use_tail_preprocessing", False)
            ),
            tail_preprocessing_index=_as_tuple(
                model.get("tail_preprocessing_index", 2),
                int,
            ),
            tail_preprocessing_type=_as_tuple(
                model.get("tail_preprocessing_type", "asinh"),
                str,
            ),
            tail_preprocessing_center=_as_tuple(
                model.get("tail_preprocessing_center", 0.0),
                float,
            ),
            tail_preprocessing_scale=_as_tuple(
                model.get("tail_preprocessing_scale", 1.0),
                float,
            ),
            tail_preprocessing_epsilon=float(
                model.get("tail_preprocessing_epsilon", 1e-6)
            ),
            lr=float(optimizer["lr"]),
            weight_decay=float(optimizer["weight_decay"]),
            eps=float(optimizer["eps"]),
            scheduler_factor=float(scheduler["factor"]),
            scheduler_patience=int(scheduler["patience"]),
            scheduler_threshold=float(scheduler["threshold"]),
            scheduler_cooldown=int(scheduler["cooldown"]),
            scheduler_min_lr=float(scheduler["min_lr"]),
            scheduler_eps=float(scheduler["eps"]),
        )


def _as_tuple(value, cast):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(cast(item) for item in value)
    return (cast(value),)


def _broadcast(values: tuple, length: int, name: str) -> tuple:
    if len(values) == 1:
        return values * length
    if len(values) != length:
        raise ValueError(
            f"{name} must contain one value or {length} values."
        )
    return values


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: t.Tensor) -> t.Tensor:
        return self.net(inputs)


class ConditionalAffineCoupling(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        mask: t.Tensor,
        hidden_dim: int,
        s_scale: float,
    ):
        super().__init__()
        self.register_buffer("mask", mask)
        self.s_scale = s_scale
        self.st_net = MLP(
            input_dim=dim + cond_dim,
            output_dim=2 * dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, x: t.Tensor, condition: t.Tensor):
        masked = x * self.mask
        scale, shift = t.chunk(
            self.st_net(t.cat([masked, condition], dim=-1)),
            2,
            dim=-1,
        )
        scale = t.tanh(scale) * self.s_scale
        output = masked + (1.0 - self.mask) * (
            x * t.exp(scale) + shift
        )
        log_det = ((1.0 - self.mask) * scale).sum(dim=-1)
        return output, log_det

    def inverse(self, output: t.Tensor, condition: t.Tensor):
        masked = output * self.mask
        scale, shift = t.chunk(
            self.st_net(t.cat([masked, condition], dim=-1)),
            2,
            dim=-1,
        )
        scale = t.tanh(scale) * self.s_scale
        return masked + (1.0 - self.mask) * (
            (output - shift) * t.exp(-scale)
        )


class ConditionalRealNVP(nn.Module):
    def __init__(self, dim: int, config: NFConfig):
        super().__init__()
        self.dim = dim
        self.cond_dim = 1
        self.config = config

        base_mask = t.tensor(
            [index % 2 for index in range(dim)],
            dtype=t.float32,
        )
        self.couplings = nn.ModuleList(
            ConditionalAffineCoupling(
                dim=dim,
                cond_dim=self.cond_dim,
                mask=base_mask if layer % 2 == 0 else 1.0 - base_mask,
                hidden_dim=config.hidden_dims,
                s_scale=config.s_scale,
            )
            for layer in range(config.n_layers)
        )

        self.register_buffer("base_mean", t.zeros(dim))
        self.register_buffer("base_log_std", t.zeros(dim))
        self.register_buffer("_scaler_shift", t.zeros(dim))
        self.register_buffer("_scaler_scale", t.ones(dim))
        self.register_buffer(
            "_cut_preprocess_indices",
            (
                t.tensor(config.cut_preprocessing_index, dtype=t.long)
                if config.use_cut_preprocessing
                else t.empty(0, dtype=t.long)
            ),
        )
        self.register_buffer(
            "_cut_preprocess_thresholds",
            (
                t.tensor(
                    config.cut_preprocessing_thresholds,
                    dtype=t.float32,
                )
                if config.use_cut_preprocessing
                else t.empty(0, dtype=t.float32)
            ),
        )
        self.register_buffer(
            "_tail_preprocessing_indices",
            (
                t.tensor(config.tail_preprocessing_index, dtype=t.long)
                if config.use_tail_preprocessing
                else t.empty(0, dtype=t.long)
            ),
        )

        tail_length = len(config.tail_preprocessing_index)
        self.tail_types = _broadcast(
            config.tail_preprocessing_type,
            tail_length,
            "tail_preprocessing_type",
        )
        self.register_buffer(
            "_tail_preprocessing_centers",
            (
                t.tensor(
                    _broadcast(
                        config.tail_preprocessing_center,
                        tail_length,
                        "tail_preprocessing_center",
                    ),
                    dtype=t.float32,
                )
                if config.use_tail_preprocessing
                else t.empty(0, dtype=t.float32)
            ),
        )
        self.register_buffer(
            "_tail_preprocessing_scales",
            (
                t.tensor(
                    _broadcast(
                        config.tail_preprocessing_scale,
                        tail_length,
                        "tail_preprocessing_scale",
                    ),
                    dtype=t.float32,
                )
                if config.use_tail_preprocessing
                else t.empty(0, dtype=t.float32)
            ),
        )

    def apply_preprocessing(self, values: t.Tensor):
        transformed = values.clone()
        log_det = t.zeros(
            len(values),
            dtype=values.dtype,
            device=values.device,
        )
        valid = t.ones(len(values), dtype=t.bool, device=values.device)

        if self.config.use_cut_preprocessing:
            indices = self._cut_preprocess_indices.to(values.device)
            thresholds = self._cut_preprocess_thresholds.to(
                values.device,
                values.dtype,
            )
            shifted = (
                transformed[:, indices]
                - thresholds
                + self.config.cut_preprocessing_epsilon
            )
            valid &= (shifted > 0).all(dim=-1)
            safe_shifted = shifted.clamp_min(
                self.config.cut_preprocessing_epsilon
            )
            transformed[:, indices] = t.log(safe_shifted)
            log_det -= t.log(safe_shifted).sum(dim=-1)

        if self.config.use_tail_preprocessing:
            indices = self._tail_preprocessing_indices.to(values.device)
            centers = self._tail_preprocessing_centers.to(
                values.device,
                values.dtype,
            )
            scales = self._tail_preprocessing_scales.to(
                values.device,
                values.dtype,
            )
            for position, index in enumerate(indices.tolist()):
                scaled = (
                    transformed[:, index] - centers[position]
                ) / scales[position]
                if self.tail_types[position] == "asinh":
                    transformed[:, index] = t.asinh(scaled)
                    log_det -= (
                        t.log(scales[position])
                        + 0.5 * t.log1p(scaled.square())
                    )
                elif self.tail_types[position] == "log1p":
                    current_valid = 1.0 + scaled > 0.0
                    valid &= current_valid
                    safe_scaled = t.where(
                        current_valid,
                        scaled,
                        scaled.new_full((), -1.0 + 1e-6),
                    )
                    transformed[:, index] = t.log1p(safe_scaled)
                    log_det -= (
                        t.log(scales[position])
                        + t.log1p(safe_scaled)
                    )
                else:
                    raise ValueError(
                        f"Unsupported tail preprocessing: "
                        f"{self.tail_types[position]}"
                    )

        return transformed, log_det, valid

    def initialize_scaler(self, values: t.Tensor) -> float:
        preprocessed, _, valid = self.apply_preprocessing(values)
        if not valid.any():
            raise RuntimeError("No valid events remain after preprocessing.")
        valid_values = preprocessed[valid]
        self._scaler_shift.copy_(valid_values.mean(dim=0))
        self._scaler_scale.copy_(
            valid_values.std(dim=0, unbiased=False).clamp_min(1e-12)
        )
        return valid.float().mean().item()

    def flow_forward(self, values: t.Tensor, condition: t.Tensor):
        log_det = t.zeros(
            len(values),
            dtype=values.dtype,
            device=values.device,
        )
        latent = values
        for coupling in self.couplings:
            latent, layer_log_det = coupling(latent, condition)
            log_det += layer_log_det
        return latent, log_det

    def forward(self, inputs: t.Tensor):
        condition = inputs[:, :self.cond_dim]
        values = inputs[:, self.cond_dim:]
        preprocessed, preprocess_log_det, valid = self.apply_preprocessing(
            values
        )
        scaled = (
            preprocessed - self._scaler_shift.to(values.device)
        ) / self._scaler_scale.to(values.device)
        latent, flow_log_det = self.flow_forward(scaled, condition)

        std = t.exp(self.base_log_std)
        base_log_prob = (
            -0.5
            * (((latent - self.base_mean) / std).square()).sum(dim=-1)
            - 0.5 * self.dim * math.log(2.0 * math.pi)
            - self.base_log_std.sum()
        )
        scale_log_det = -t.log(
            self._scaler_scale.to(values.device)
        ).sum()
        log_prob = (
            base_log_prob
            + flow_log_det
            + preprocess_log_det
            + scale_log_det
        )
        return t.where(valid, log_prob, t.full_like(log_prob, -t.inf))


def _region_frame(analysis_df, process: str, region: str) -> pd.DataFrame:
    return getattr(
        analysis_df.data,
        f"{region.replace('-', '_')}_{process}",
    ).events.copy()


def _prepare_frame(
    analysis_df,
    process: str,
    region: str,
    grouping: str,
) -> tuple[pd.DataFrame, str]:
    source_weight = f"reduced_weight_{process}_{grouping}_nominal"
    analysis_df.ensure_column(source_weight)
    frame = _region_frame(analysis_df, process, region)
    frame = frame.copy()
    frame["training_weight"] = analysis_df.full.events.loc[
        frame.index,
        source_weight,
    ]
    return frame, source_weight


def _make_loader(
    frame: pd.DataFrame,
    variables: list[str],
    batch_size: int,
    shuffle: bool,
) -> tuple[TensorDataset, DataLoader]:
    finite_columns = ["njets", "training_weight", *variables]
    finite = np.isfinite(
        frame[finite_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)
    frame = frame.loc[finite]
    if frame.empty:
        raise ValueError("No finite events are available for NF training.")

    inputs = np.column_stack(
        [
            frame["njets"].to_numpy(dtype=np.float32),
            frame[variables].to_numpy(dtype=np.float32),
        ]
    )
    weights = frame["training_weight"].to_numpy(dtype=np.float32)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or abs(weight_sum) < 1e-12:
        raise ValueError("NF training weights have a zero or invalid sum.")
    weights = weights / weight_sum

    dataset = TensorDataset(
        t.as_tensor(inputs),
        t.as_tensor(weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=t.cuda.is_available(),
    )
    return dataset, loader


def _evaluate(model, loader, device) -> float:
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    with t.no_grad():
        for inputs, weights in loader:
            inputs = inputs.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            log_prob = model(inputs)
            finite = t.isfinite(log_prob) & t.isfinite(weights)
            loss_sum += (-(log_prob[finite]) * weights[finite]).sum().item()
            weight_sum += weights[finite].sum().item()
    if abs(weight_sum) < 1e-12:
        raise RuntimeError("Validation weights sum to zero.")
    return loss_sum / weight_sum


def _save_checkpoint(
    output_dir: Path,
    checkpoint: dict,
    history: list[dict],
    config: NFConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    t.save(checkpoint, output_dir / "model_checkpoint.pth")
    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv",
        index=False,
    )
    with open(output_dir / "config.yaml", "w") as stream:
        yaml.safe_dump(config.to_nested_dict(), stream, sort_keys=False)
    (output_dir / "training_metadata.json").write_text(
        json.dumps(
            {
                "schema": "conditional_nf_two_fold_v1",
                "config": str(CONFIG_PATH),
                "variables": checkpoint["variables"],
                "condition": "njets",
                "best_val_nll": checkpoint["best_val_nll"],
            },
            indent=2,
        )
    )


def _train_fold(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    variables: list[str],
    config: NFConfig,
    device: t.device,
    output_dir: Path,
    metadata: dict,
) -> None:
    train_dataset, train_loader = _make_loader(
        train_frame,
        variables,
        config.bsize_train,
        shuffle=True,
    )
    _, val_loader = _make_loader(
        val_frame,
        variables,
        config.bsize_val,
        shuffle=False,
    )

    model = ConditionalRealNVP(len(variables), config).to(device)
    train_inputs = train_dataset.tensors[0][:, 1:].to(device)
    valid_fraction = model.initialize_scaler(train_inputs)
    logger.info(
        "%s scaler valid fraction: %.4f",
        metadata["label"],
        valid_fraction,
    )

    optimizer = t.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.scheduler_threshold,
        threshold_mode="rel",
        cooldown=config.scheduler_cooldown,
        min_lr=config.scheduler_min_lr,
        eps=config.scheduler_eps,
    )
    use_amp = device.type == "cuda" and config.use_amp
    scaler = t.amp.GradScaler("cuda", enabled=use_amp)

    best_val_nll = float("inf")
    epochs_without_improvement = 0
    best_checkpoint = None
    history = []

    for epoch in range(1, config.n_epochs + 1):
        started = time.time()
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0

        for inputs, weights in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with t.amp.autocast("cuda", enabled=use_amp):
                log_prob = model(inputs)
                finite = t.isfinite(log_prob) & t.isfinite(weights)
                loss = (-(log_prob[finite]) * weights[finite]).sum()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_weight_sum += weights[finite].sum().item()

        train_nll = train_loss_sum / max(abs(train_weight_sum), 1e-12)
        val_nll = _evaluate(model, val_loader, device)
        scheduler.step(val_nll)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "lr": current_lr,
                "time_s": time.time() - started,
            }
        )

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            epochs_without_improvement = 0
            best_checkpoint = {
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "variables": variables,
                "condition": "njets",
                "schema": "conditional_nf_two_fold_v1",
                "best_val_nll": best_val_nll,
                **metadata,
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            logger.info("Early stopping %s at epoch %d", metadata["label"], epoch)
            break

    if best_checkpoint is None:
        raise RuntimeError(f"No checkpoint was created for {metadata['label']}.")
    _save_checkpoint(output_dir, best_checkpoint, history, config)


def train_conditional_flows(
    data_path: Union[str, Path] = DATA_PATH,
    masks_path: Union[str, Path] = MASKS_PATH,
    variables_path: Union[str, Path] = VARIABLES_PATH,
    config_path: Union[str, Path] = CONFIG_PATH,
    output_root: Union[str, Path] = OUTPUT_ROOT,
    test_size: float = 0.25,
    random_state: int = SEED,
) -> Path:
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    _set_seed(random_state)
    config = NFConfig.from_yaml(config_path)
    variables = load_variables(variables_path)
    analysis_df = load_data(data_path, masks_path)
    output_root = Path(output_root)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    for grouping in REDUCED_WEIGHT_GROUPINGS:
        for process in PROCESSES:
            for region in REGIONS:
                frame, source_weight = _prepare_frame(
                    analysis_df,
                    process,
                    region,
                    grouping,
                )
                even = frame[frame["event"] % 2 == 0]
                odd = frame[frame["event"] % 2 == 1]

                fold_inputs = {
                    "fold_even": odd,
                    "fold_odd": even,
                }
                for fold, fold_frame in fold_inputs.items():
                    train_frame, val_frame = train_test_split(
                        fold_frame,
                        test_size=test_size,
                        random_state=random_state,
                    )
                    label = f"{grouping}/{process}/{region}/{fold}"
                    logger.info(
                        "Training %s with %d train and %d validation events",
                        label,
                        len(train_frame),
                        len(val_frame),
                    )
                    _train_fold(
                        train_frame=train_frame,
                        val_frame=val_frame,
                        variables=variables,
                        config=config,
                        device=device,
                        output_dir=(
                            output_root
                            / grouping
                            / process
                            / region
                            / fold
                        ),
                        metadata={
                            "label": label,
                            "grouping": grouping,
                            "process": process,
                            "region": region,
                            "fold": fold,
                            "source_weight": source_weight,
                            "trained_on_parity": (
                                "odd" if fold == "fold_even" else "even"
                            ),
                        },
                    )

    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--masks", type=Path, default=MASKS_PATH)
    parser.add_argument("--variables", type=Path, default=VARIABLES_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train_conditional_flows(
        data_path=args.data,
        masks_path=args.masks,
        variables_path=args.variables,
        config_path=args.config,
        output_root=args.output_root,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
