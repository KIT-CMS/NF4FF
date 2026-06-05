from __future__ import annotations
import logging
import random
from typing import Tuple, Any, Union, Generator

from contextlib import contextmanager
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch as t
import torch.nn as nn
import yaml

from classes.Dataclasses import ModelConfig, _component_collection
from classes.Logging import setup_logging
from classes.NeuralNetworks import RealNVP, ConditionalRealNVP, GroupedNFRouter


FF_CLIP_TAIL_FRACTION = 0.04
logger = setup_logging(logger=logging.getLogger(__name__))


# ------- general stuff

@contextmanager
def rng_seed(seed: int) -> Generator[None, None, None]:
    np_rng_state, py_rng_state = np.random.get_state(), random.getstate()
    t_rng_state = t.get_rng_state()

    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_rng_state)
        random.setstate(py_rng_state)
        t.set_rng_state(t_rng_state)


# --------model loading

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_config(path: str) -> ModelConfig:
    cfg = load_config(path)
    model_cfg = cfg["model"]
    return ModelConfig(
        n_layers=model_cfg["n_layers"],
        hidden_dims=model_cfg["hidden_dims"],
        s_scale=model_cfg["s_scale"],
        use_cut_preprocessing=model_cfg.get("use_cut_preprocessing", True),
        cut_preprocessing_thresholds=tuple(model_cfg.get("cut_preprocessing_thresholds", [33.0, 30.0])),
        cut_preprocessing_epsilon=model_cfg.get("cut_preprocessing_epsilon", 1e-6),
        use_tail_preprocessing=model_cfg.get("use_tail_preprocessing", False),
        tail_preprocessing_index=model_cfg.get("tail_preprocessing_index", 2),
        tail_preprocessing_type=model_cfg.get("tail_preprocessing_type", "asinh"),
        tail_preprocessing_center=model_cfg.get("tail_preprocessing_center", 0.0),
        tail_preprocessing_scale=model_cfg.get("tail_preprocessing_scale", 1.0),
        tail_preprocessing_epsilon=model_cfg.get("tail_preprocessing_epsilon", 1e-6),
    )

def _get_backend_and_device(tensor_or_array: Union[np.ndarray, t.Tensor]) -> tuple[Any, Any]:
    if isinstance(tensor_or_array, t.Tensor):
        return t, tensor_or_array.device
    elif isinstance(tensor_or_array, np.ndarray):
        return np, None
    else:
        raise TypeError(f"Input must be a NumPy array or PyTorch tensor, got {type(tensor_or_array)}")

def load_flow(
    dim: int,
    cfg: ModelConfig,
    checkpoint_path: str,
    device: torch.device,
) -> RealNVP:

    model = RealNVP(
        dim=dim,
        n_layers=cfg.n_layers,
        hidden_dims=(cfg.hidden_dims,),
        s_scale=cfg.s_scale,
        use_cut_preprocessing=cfg.use_cut_preprocessing,
        cut_preprocessing_thresholds=cfg.cut_preprocessing_thresholds,
        cut_preprocessing_epsilon=cfg.cut_preprocessing_epsilon,
        use_tail_preprocessing=cfg.use_tail_preprocessing,
        tail_preprocessing_index=cfg.tail_preprocessing_index,
        tail_preprocessing_type=cfg.tail_preprocessing_type,
        tail_preprocessing_center=cfg.tail_preprocessing_center,
        tail_preprocessing_scale=cfg.tail_preprocessing_scale,
        tail_preprocessing_epsilon=cfg.tail_preprocessing_epsilon,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        logger.warning(
            "Loaded checkpoint %s with missing keys=%s unexpected keys=%s",
            checkpoint_path,
            load_result.missing_keys,
            load_result.unexpected_keys,
        )
    model.eval()

    return model


def load_conditional_flow(
    dim: int,
    cfg: ModelConfig,
    checkpoint_path: str,
    device: torch.device,
    cond_dim: int = 1,
) -> ConditionalRealNVP:

    model = ConditionalRealNVP(
        dim=dim,
        cond_dim=cond_dim,
        n_layers=cfg.n_layers,
        hidden_dims=(cfg.hidden_dims,),
        s_scale=cfg.s_scale,
        use_cut_preprocessing=cfg.use_cut_preprocessing,
        cut_preprocessing_thresholds=cfg.cut_preprocessing_thresholds,
        cut_preprocessing_epsilon=cfg.cut_preprocessing_epsilon,
        use_tail_preprocessing=cfg.use_tail_preprocessing,
        tail_preprocessing_index=cfg.tail_preprocessing_index,
        tail_preprocessing_type=cfg.tail_preprocessing_type,
        tail_preprocessing_center=cfg.tail_preprocessing_center,
        tail_preprocessing_scale=cfg.tail_preprocessing_scale,
        tail_preprocessing_epsilon=cfg.tail_preprocessing_epsilon,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        logger.warning(
            "Loaded checkpoint %s with missing keys=%s unexpected keys=%s",
            checkpoint_path,
            load_result.missing_keys,
            load_result.unexpected_keys,
        )
    model.eval()

    return model


def load_grouped_nf_router(
    checkpoint_dir: str,
    config_path: str,
    variables: list,
    RealNVP_class,
    device=None,
):
    """Load a saved `GroupedNFRouter` and its three grouped NF sub-models.

    Supports both constructor styles:
    - `RealNVP_NN`-style (`input_nodes`, `hidden_nodes`, ...)
    - `RealNVP`-style (`dim`, `hidden_dims`, ...)
    """

    import inspect

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_config = config["model"]
    cut_preprocessing_kwargs = {
        "use_cut_preprocessing": model_config.get("use_cut_preprocessing", True),
        "cut_preprocessing_thresholds": tuple(model_config.get("cut_preprocessing_thresholds", [33.0, 30.0])),
        "cut_preprocessing_epsilon": model_config.get("cut_preprocessing_epsilon", 1e-6),
        "use_tail_preprocessing": model_config.get("use_tail_preprocessing", False),
        "tail_preprocessing_index": model_config.get("tail_preprocessing_index", 2),
        "tail_preprocessing_type": model_config.get("tail_preprocessing_type", "asinh"),
        "tail_preprocessing_center": model_config.get("tail_preprocessing_center", 0.0),
        "tail_preprocessing_scale": model_config.get("tail_preprocessing_scale", 1.0),
        "tail_preprocessing_epsilon": model_config.get("tail_preprocessing_epsilon", 1e-6),
    }

    if device is None:
        device = t.device("cuda" if t.cuda.is_available() else "cpu")

    ckpt_path = f"{checkpoint_dir}/model_checkpoint.pth"
    ckpt = t.load(ckpt_path, map_location=device)
    router_state = ckpt["router_state_dict"]

    nf_dim = len(variables)

    def _build_nf_model() -> nn.Module:
        sig = inspect.signature(RealNVP_class.__init__)
        params = set(sig.parameters.keys())

        # RealNVP_NN-like signature
        if "input_nodes" in params:
            return RealNVP_class(
                input_nodes=nf_dim,
                hidden_nodes=(model_config["hidden_dims"],),
                n_layers=model_config["n_layers"],
                dropout=0.0,
                activation="ReLU",
                s_scale=model_config["s_scale"],
                **cut_preprocessing_kwargs,
            ).to(device)

        # RealNVP-like signature
        if "dim" in params:
            return RealNVP_class(
                dim=nf_dim,
                n_layers=model_config["n_layers"],
                hidden_dims=(model_config["hidden_dims"],),
                s_scale=model_config["s_scale"],
                **cut_preprocessing_kwargs,
            ).to(device)

        raise TypeError(
            "Unsupported RealNVP_class constructor. Expected either "
            "`input_nodes` (RealNVP_NN-style) or `dim` (RealNVP-style)."
        )

    model_0 = _build_nf_model()
    model_1 = _build_nf_model()
    model_2 = _build_nf_model()

    router = GroupedNFRouter().to(device)

    router._fallback_payload = model_2
    router._wrapped_delegate = model_2

    # Keep explicit registration so state_dict keys from training match on load.
    router.models.append(model_0)
    router.models.append(model_1)
    router.models.append(model_2)

    router._logic_pipeline = [
        ([(0, (0,))], model_0),
        ([(0, (1,))], model_1),
        ([(0, (2, 11000))], model_2),
    ]

    load_result = router.load_state_dict(router_state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        logger.warning(
            "Loaded router checkpoint %s with missing keys=%s unexpected keys=%s",
            ckpt_path,
            load_result.missing_keys,
            load_result.unexpected_keys,
        )
    router.eval()
    return router


def load_grouped_wjets_njets_router(
    checkpoint_dir: str,
    config_path: str,
    variables: list,
    device=None,
):
    """Load the grouped router checkpoint produced by NF_training_wjets_njets.py.

    This training uses `RealNVP` (not `RealNVP_NN`) as grouped payload model.
    """
    return load_grouped_nf_router(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        variables=variables,
        RealNVP_class=RealNVP,
        device=device,
    )


def load_grouped_qcd_njets_router(
    checkpoint_dir: str,
    config_path: str,
    variables: list,
    device=None,
):
    """Load the grouped router checkpoint produced by NF_training_qcd_njets.py.

    This training uses `RealNVP` (not `RealNVP_NN`) as grouped payload model.
    """
    return load_grouped_nf_router(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        variables=variables,
        RealNVP_class=RealNVP,
        device=device,
    )

# ------- data loading -------

def get_my_data_qcd(df, training_var):
    _df = df

    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        weights=_df["weight_qcd"].to_numpy(dtype=np.float32),

    )

# ------ data splitting -------

def split_even_odd(df: pd.DataFrame, SEED = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold1 = df[df.event_var  == 0].reset_index(drop=True)
    fold2 = df[df.event_var == 1].reset_index(drop=True)

    train1, val1 = train_test_split(
        fold1, test_size=0.5, random_state=SEED
    )
    train2, val2 = train_test_split(
        fold2, test_size=0.5, random_state=SEED
    )

    return train1.reset_index(drop=True), val1.reset_index(drop=True), train2.reset_index(drop=True), val2.reset_index(drop=True)


# ------- Fake Factor Stuff 

@torch.no_grad()
def evaluate_pdf(model: RealNVP, X: torch.Tensor) -> np.ndarray:
    """Returns PDF evaluated at events"""
    # Use model(X) so the model can apply the scaler and add the scaling Jacobian
    log_pdf = model(X)
    log_pdf = torch.clamp(log_pdf, min=-1e10)
    pdf = torch.exp(log_pdf).cpu().numpy()
    return pdf


def compute_eventwise_fake_factors(
    pdf_AR: np.ndarray,
    pdf_SR: np.ndarray,
    global_ff: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:

    # ----- safe ratio -----
    ratio = np.divide(
        pdf_SR,
        np.maximum(pdf_AR, 1e-100),
        out=np.zeros_like(pdf_SR, dtype=float),
        where=(pdf_AR > 0) & (pdf_SR > 0)
)

    # ---- determine adaptive clip value so the highest tail fraction is clipped,
    # while enforcing a minimum clip value of 1.0
    ff_eventwise_nominal = global_ff * ratio

    valid_ff_mask = np.isfinite(ff_eventwise_nominal)
    '''
    if np.any(valid_ff_mask):
        clip_quantile = 1.0 - FF_CLIP_TAIL_FRACTION
        clip_value = float(np.quantile(ff_eventwise_nominal[valid_ff_mask], clip_quantile))
        clip_value = max(clip_value, 1.0)
    else:
    '''
    clip_value = 2.0

    # ---- mask for clipping and nonzero PDFs
    clip_mask = (ff_eventwise_nominal <= clip_value) & (pdf_AR > 0) & (pdf_SR > 0)

    correction_factor = np.sum(clip_mask) / len(ratio)
    correction_factor = max(correction_factor, 1e-12)
    
    global_ff_cor = global_ff / correction_factor

    # ---- full FF for all events
    ff_eventwise_full = global_ff_cor * ratio
    #ff_eventwise_full = np.clip(ff_eventwise_full, a_min=0.0, a_max=clip_value)

    # ---- clipped FF
    
    ff_eventwise_clipped = ff_eventwise_full[clip_mask]

    return ff_eventwise_full, ff_eventwise_clipped, global_ff_cor, clip_mask, clip_value
