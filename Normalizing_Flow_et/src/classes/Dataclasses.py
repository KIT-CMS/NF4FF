import yaml
import torch as t
import pandas as pd
import numpy as np


import CODE.HELPER as helper

from dataclasses import KW_ONLY, dataclass
from typing import (Any, Callable, Dict, Iterable, List, Protocol, Tuple,
                    Union, runtime_checkable)

TailIndexValue = Union[int, List[int], Tuple[int, ...]]
TailStringValue = Union[str, List[str], Tuple[str, ...]]
TailFloatValue = Union[float, List[float], Tuple[float, ...]]
CutIndexValue = Union[int, List[int], Tuple[int, ...]]
CutFloatValue = Union[float, List[float], Tuple[float, ...]]

@dataclass
class ModelConfig:
    n_layers: int
    hidden_dims: int
    s_scale: float
    use_cut_preprocessing: bool = True
    cut_preprocessing_index: CutIndexValue = (0, 1)
    cut_preprocessing_thresholds: CutFloatValue = (33.0, 30.0)
    cut_preprocessing_epsilon: float = 1e-6
    use_tail_preprocessing: bool = False
    tail_preprocessing_index: TailIndexValue = 2
    tail_preprocessing_type: TailStringValue = "asinh"
    tail_preprocessing_center: TailFloatValue = 0.0
    tail_preprocessing_scale: TailFloatValue = 1.0
    tail_preprocessing_epsilon: float = 1e-6

@dataclass
class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[t.Tensor, pd.DataFrame, np.ndarray]
    os: Union[t.Tensor, pd.DataFrame, np.ndarray]

@dataclass
class _collection:
    values: Any
    weights: Any
    histograms: Any
    
    @property
    def unrolled(self) -> tuple[Any, ...]:
        return (self.values, self.weights, self.histograms)

@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    _: KW_ONLY
    X: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    Njets: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    process: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None

@dataclass
class Config:
    # training
    bsize_train: int
    bsize_val: int
    bsize_test: int
    grad_clip: float
    n_epochs: int
    use_amp: bool
    s_scale_max: float

    # model
    n_layers: int
    hidden_dims: int
    s_scale: float
    dropout: float

    # optimizer
    lr: float
    weight_decay: float
    eps: float

    # scheduler
    scheduler_step_size: int
    scheduler_gamma: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_threshold: float
    scheduler_cooldown: int
    scheduler_min_lr: float
    scheduler_eps: float

    # classifier training (optional)
    label_smoothing: float = 0.0

    # preprocessing (optional)
    use_cut_preprocessing: bool = True
    cut_preprocessing_index: CutIndexValue = (0, 1)
    cut_preprocessing_thresholds: CutFloatValue = (33.0, 30.0)
    cut_preprocessing_epsilon: float = 1e-6
    use_tail_preprocessing: bool = False
    tail_preprocessing_index: TailIndexValue = 2
    tail_preprocessing_type: TailStringValue = "asinh"
    tail_preprocessing_center: TailFloatValue = 0.0
    tail_preprocessing_scale: TailFloatValue = 1.0
    tail_preprocessing_epsilon: float = 1e-6

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "Config":
        """Construct from the original nested YAML structure."""
        training = cfg["training"]
        model = cfg["model"]
        optimizer = cfg["optimizer"]
        scheduler = cfg["scheduler"]
        cut_preprocessing_thresholds = model.get("cut_preprocessing_thresholds", [33.0, 30.0])

        return Config(
            # training
            bsize_train=training["bsize_train"],
            bsize_val=training["bsize_val"],
            bsize_test=training["bsize_test"],
            grad_clip=training["grad_clip"],
            n_epochs=training["n_epochs"],
            use_amp=training["use_amp"],
            s_scale_max=training["s_scale_max"],

            # model
            n_layers=model["n_layers"],
            hidden_dims=model["hidden_dims"],
            s_scale=model["s_scale"],
            dropout=model["dropout"],
            use_cut_preprocessing=model.get("use_cut_preprocessing", True),
            cut_preprocessing_index=model.get("cut_preprocessing_index", [0, 1]),
            cut_preprocessing_thresholds=cut_preprocessing_thresholds,
            cut_preprocessing_epsilon=model.get("cut_preprocessing_epsilon", 1e-6),
            use_tail_preprocessing=model.get("use_tail_preprocessing", False),
            tail_preprocessing_index=model.get("tail_preprocessing_index", 2),
            tail_preprocessing_type=model.get("tail_preprocessing_type", "asinh"),
            tail_preprocessing_center=model.get("tail_preprocessing_center", 0.0),
            tail_preprocessing_scale=model.get("tail_preprocessing_scale", 1.0),
            tail_preprocessing_epsilon=model.get("tail_preprocessing_epsilon", 1e-6),

            # optimizer
            lr=optimizer["lr"],
            weight_decay=optimizer["weight_decay"],
            eps=optimizer["eps"],

            # scheduler
            scheduler_step_size=scheduler["step_size"],
            scheduler_gamma=scheduler["gamma"],
            scheduler_factor=scheduler["factor"],
            scheduler_patience=scheduler["patience"],
            scheduler_threshold=scheduler["threshold"],
            scheduler_cooldown=scheduler["cooldown"],
            scheduler_min_lr=scheduler["min_lr"],
            scheduler_eps=scheduler["eps"],

            # classifier training (optional)
            label_smoothing=training.get("label_smoothing", 0.0),
        )
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

@dataclass
class RealNVP_config:
    # training
    bsize_train: int
    bsize_val: int
    bsize_test: int
    grad_clip: float
    n_epochs: int
    use_amp: bool
    s_scale_max: float

    # model
    n_layers: int
    hidden_dims: int
    s_scale: float

    # optimizer
    lr: float
    weight_decay: float
    eps: float

    # scheduler
    scheduler_step_size: int
    scheduler_gamma: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_threshold: float
    scheduler_cooldown: int
    scheduler_min_lr: float
    scheduler_eps: float

    # preprocessing (optional)
    use_cut_preprocessing: bool = True
    cut_preprocessing_index: CutIndexValue = (0, 1)
    cut_preprocessing_thresholds: CutFloatValue = (33.0, 30.0)
    cut_preprocessing_epsilon: float = 1e-6
    use_tail_preprocessing: bool = False
    tail_preprocessing_index: TailIndexValue = 2
    tail_preprocessing_type: TailStringValue = "asinh"
    tail_preprocessing_center: TailFloatValue = 0.0
    tail_preprocessing_scale: TailFloatValue = 1.0
    tail_preprocessing_epsilon: float = 1e-6

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "RealNVP_config":
        model_cfg = dict(cfg["model"])
        model_cfg.setdefault("use_cut_preprocessing", True)
        model_cfg.setdefault("cut_preprocessing_index", [0, 1])
        model_cfg.setdefault("cut_preprocessing_thresholds", [33.0, 30.0])
        model_cfg.setdefault("cut_preprocessing_epsilon", 1e-6)
        model_cfg.setdefault("use_tail_preprocessing", False)
        model_cfg.setdefault("tail_preprocessing_index", 2)
        model_cfg.setdefault("tail_preprocessing_type", "asinh")
        model_cfg.setdefault("tail_preprocessing_center", 0.0)
        model_cfg.setdefault("tail_preprocessing_scale", 1.0)
        model_cfg.setdefault("tail_preprocessing_epsilon", 1e-6)
        flat_cfg = {
            **cfg["training"],
            **model_cfg,
            **cfg["optimizer"],
            **{f"scheduler_{k}": v for k, v in cfg["scheduler"].items()},
        }
        return cls(**flat_cfg)
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)