import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import logging
import yaml
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = WORKSPACE_ROOT / 'data'
MASKS_CONFIG_PATH = CONFIG_DIR / 'masks_MC.yaml'

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classes.NeuralNetworks import ConditionalRealNVP
from classes.Dataclasses import _component_collection
from classes.Collection import load_model_config, ModelConfig
from classes.Collection import compute_eventwise_fake_factors, load_conditional_flow
from CustomLogging import setup_logging, LogContext


logger = setup_logging(logger=logging.getLogger(__name__))

mode_dir = 'conditional_njets_input'
OUTPUT_ROOT = SCRIPT_DIR / 'Training_results_MC_even'

config_path = CONFIG_DIR / 'config_NF.yaml'


def get_my_data(df, training_var):
    _df = df
    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        weights=_df["weight"].to_numpy(dtype=np.float32),
    )


def load_saved_model_config(checkpoint_dir: str | Path, fallback_path: str | Path) -> ModelConfig:
    saved_config_path = Path(checkpoint_dir).parent / 'config.yaml'
    if saved_config_path.exists():
        with open(saved_config_path, 'r') as handle:
            raw = yaml.unsafe_load(handle)

        if isinstance(raw, ModelConfig):
            return raw

        values = vars(raw) if hasattr(raw, '__dict__') else raw
        # Handle nested format written by yaml.safe_dump: {'model': {...}, 'training': {...}, ...}
        if 'model' in values and isinstance(values['model'], dict):
            values = values['model']
        return ModelConfig(
            n_layers=values['n_layers'],
            hidden_dims=values['hidden_dims'],
            s_scale=values['s_scale'],
            use_cut_preprocessing=values.get('use_cut_preprocessing', True),
            cut_preprocessing_index=values.get('cut_preprocessing_index', [0, 1]),
            cut_preprocessing_thresholds=values.get('cut_preprocessing_thresholds', [33.0, 30.0]),
            cut_preprocessing_epsilon=values.get('cut_preprocessing_epsilon', 1e-6),
            use_tail_preprocessing=values.get('use_tail_preprocessing', False),
            tail_preprocessing_index=values.get('tail_preprocessing_index', 2),
            tail_preprocessing_type=values.get('tail_preprocessing_type', 'asinh'),
            tail_preprocessing_center=values.get('tail_preprocessing_center', 0.0),
            tail_preprocessing_scale=values.get('tail_preprocessing_scale', 1.0),
            tail_preprocessing_epsilon=values.get('tail_preprocessing_epsilon', 1e-6),
        )

    logger.warning('Saved model config not found for %s; falling back to %s', checkpoint_dir, fallback_path)
    return load_model_config(str(fallback_path))


def resolve_training_name(variables: list[str]) -> str:
    tail = variables[4:]
    tag = '_'.join(tail) if tail else 'none'
    return f"training_vars{len(variables)}_{tag}"

def load_masks_config(path: str | Path = MASKS_CONFIG_PATH) -> dict[str, list[str]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f'Mask configuration not found: {config_path}')

    with open(config_path, 'r') as handle:
        raw = yaml.safe_load(handle) or {}

    masks = raw.get('masks', raw)
    if not isinstance(masks, dict):
        raise ValueError(f'Invalid masks configuration in {config_path}: expected a mapping at top level.')

    normalized: dict[str, list[str]] = {}
    for name, expressions in masks.items():
        if isinstance(expressions, str):
            normalized[name] = [expressions]
        elif isinstance(expressions, list) and all(isinstance(expr, str) for expr in expressions):
            normalized[name] = expressions
        else:
            raise TypeError(
                f"Mask '{name}' in {config_path} must be a string or list[str], got {type(expressions).__name__}."
            )

    logger.info('Loaded %d masks from %s', len(normalized), config_path)
    return normalized

def _build_mask_from_config(
    df: pd.DataFrame,
    mask_name: str,
    masks_config: dict[str, list[str]],
) -> pd.Series:
    expressions = masks_config.get(mask_name)
    if not expressions:
        raise KeyError(f"Mask '{mask_name}' is not defined in masks config.")

    combined_expression = ' & '.join(f'({expr})' for expr in expressions)
    mask = df.eval(combined_expression, engine='python')
    return mask.fillna(False).astype(bool)

def _apply_config_mask(
    df: pd.DataFrame,
    mask_name: str,
    masks_config: dict[str, list[str]],
) -> pd.DataFrame:
    return df[_build_mask_from_config(df, mask_name, masks_config)].copy()

def compute_fake_factors(
    log_pdf_ar: np.ndarray,
    log_pdf_sr: np.ndarray,
    global_ff: float,
) -> np.ndarray:
    """Event-wise FF from log-density ratio: FF = global_ff * exp(log p_SR - log p_AR)."""
    log_ratio = np.clip(log_pdf_sr - log_pdf_ar, -700.0, 700.0)
    return np.clip(global_ff * np.exp(log_ratio), 0, 10)

@t.no_grad()
def evaluate_log_pdf(model: ConditionalRealNVP, X: t.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return log-pdf and preprocessing-valid mask for each event."""
    cond_dim = int(getattr(model, 'cond_dim', 0))
    x_features = X[:, cond_dim:]
    x_preprocessed, log_det_preprocess, valid_mask = model.apply_preprocessing(x_features)
    Xs = model.apply_scaler(x_preprocessed)

    n_invalid_preprocess = (~valid_mask).sum().item()
    n_nan_after_scale    = t.isnan(Xs[valid_mask]).any(dim=-1).sum().item() if valid_mask.any() else 0
    n_inf_after_scale    = t.isinf(Xs[valid_mask]).any(dim=-1).sum().item() if valid_mask.any() else 0
    logger.debug(
        'evaluate_log_pdf: n=%d  invalid_preprocess=%d  nan_after_scale=%d  inf_after_scale=%d',
        X.shape[0], n_invalid_preprocess, n_nan_after_scale, n_inf_after_scale,
    )

    # model(X) returns log-pdf; invalid events are -inf by construction.
    log_pdf = model(X)

    n_nan_logpdf = t.isnan(log_pdf).sum().item()
    n_inf_logpdf = t.isinf(log_pdf).sum().item()
    if n_nan_logpdf > 0:
        logger.warning('evaluate_log_pdf: %d NaN values in log_pdf (valid events with numerical issues)', n_nan_logpdf)
    if n_inf_logpdf > 0:
        logger.debug('evaluate_log_pdf: %d -inf values in log_pdf (invalid preprocess events)', n_inf_logpdf)

    log_pdf = t.nan_to_num(log_pdf, nan=-700.0, neginf=-700.0, posinf=700.0)
    return log_pdf.detach().cpu().numpy(), valid_mask.detach().cpu().numpy()



def plot_ff_distributions(
    ff_SR: np.ndarray,
    ff_DR: np.ndarray,
    plot_dir: Path = None,
    bins: int = 50,
    range: tuple[float, float] = (0.001, 10),
    title: str = "Fake Factor Distribution",
    xlabel: str = "Fake Factor",
    ylabel: str = "Frequency",
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    bins = np.logspace(np.log10(range[0]), np.log10(range[1]), bins + 1)
    plt.figure(figsize=(8, 6))
    plt.hist(ff_SR, bins=bins, histtype='step', alpha=0.5, color='blue', edgecolor='blue', label='SR')
    plt.hist(ff_DR, bins=bins, histtype='step', alpha=0.5, color='red', edgecolor='red', label='DR')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "fake_factor_distribution.png")

def plot_pdf_distribution(
        log_pdf_SR: np.ndarray,
        log_pdf_AR: np.ndarray,
        plot_dir: Path = None,
        bins: int = 50,
        range: tuple[float, float] = (-30, 1),
        title: str = "PDF Distribution",

):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    if title == 'DR':
        label_SR = 'SR-like'
        label_AR = 'AR-like'
    else:
        label_SR = 'SR'
        label_AR = 'AR'
    bins = np.linspace(range[0], range[1], bins + 1)
    plt.figure(figsize=(8, 6))
    plt.hist(log_pdf_SR, bins=bins, histtype='step', alpha=0.5, color='blue', edgecolor='blue', label=label_SR)
    plt.hist(log_pdf_AR, bins=bins, histtype='step', alpha=0.5, color='red', ls='--', edgecolor='red', label=label_AR)
    plt.title(title)
    plt.xlabel("log PDF Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if title == 'DR':
        plt.savefig(plot_dir / "pdf_distribution_DR.png")
    elif title == 'SR':
        plt.savefig(plot_dir / "pdf_distribution_SR.png")
    else:
        plt.savefig(plot_dir / "pdf_distribution.png")

def plt_control_plots(
        SR = pd.DataFrame,
        AR = pd.DataFrame,
        FF = np.ndarray,
        var = str,
        range = tuple[float, float],
        region = str,
        plot_dir: Path = None,
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    bins = np.linspace(range[0], range[1], 50)
    plt.figure(figsize=(8, 6))
    plt.hist(SR[var], bins=bins, weights=SR['weight'], histtype='step', color='blue', edgecolor='blue', label='MC')
    plt.hist(AR[var], bins=bins, weights=FF * AR['weight'], histtype='step', color='red', edgecolor='red', label='FF')
    plt.title(f"Distribution of {var} in {region}")
    plt.xlabel(var)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{var}_distribution_{region}.png")
    

def plot_correlation_FF_DR_SR(
        FF_SR: np.ndarray,
        FF_DR: np.ndarray,
        plot_dir: Path = None,
):
    if plot_dir is None:
        plot_dir = SCRIPT_DIR
    plt.figure(figsize=(8, 6))
    plt.scatter(FF_DR, FF_SR, alpha=0.5)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("FF_DR")
    plt.ylabel("FF_SR")
    plt.title("Correlation between FF_DR and FF_SR")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_dir / "correlation_FF_DR_SR.png")



with open(CONFIG_DIR / 'training_variables.yaml', 'r') as f:
    variables = yaml.safe_load(f)['variables_MC']

dim = len(variables)

resolved_tag = resolve_training_name(variables)
# Training models are in the original Training_results_MC directory
TRAINING_ROOT = SCRIPT_DIR / 'Training_results_MC_even' / mode_dir / resolved_tag
# Output results folder depends on trained variables
OUTPUT_ROOT = SCRIPT_DIR / f'Training_results_MC_even_{resolved_tag}'
PLOTS_DIR = OUTPUT_ROOT / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

masks_config = load_masks_config(MASKS_CONFIG_PATH)

training_root = TRAINING_ROOT

device = t.device("cuda:1" if t.cuda.is_available() else "cpu")

chk_pth_model_AR_like = training_root / 'Wjets' / 'DR' / 'AR-like' / 'latest'
chk_pth_model_SR_like = training_root / 'Wjets' / 'DR' / 'SR-like' / 'latest'
chk_pth_model_AR = training_root / 'Wjets' / 'antiDR' / 'AR' / 'latest'
chk_pth_model_SR = training_root / 'Wjets' / 'antiDR' / 'SR' / 'latest'

config_AR_like = load_saved_model_config(chk_pth_model_AR_like, config_path)
config_SR_like = load_saved_model_config(chk_pth_model_SR_like, config_path)
config_AR = load_saved_model_config(chk_pth_model_AR, config_path)
config_SR = load_saved_model_config(chk_pth_model_SR, config_path)

model_AR_like = load_conditional_flow(dim=dim, cfg=config_AR_like, checkpoint_path=chk_pth_model_AR_like / 'model_checkpoint.pth', device=device)
model_SR_like = load_conditional_flow(dim=dim, cfg=config_SR_like, checkpoint_path=chk_pth_model_SR_like / 'model_checkpoint.pth', device=device)
model_AR = load_conditional_flow(dim=dim, cfg=config_AR, checkpoint_path=chk_pth_model_AR / 'model_checkpoint.pth', device=device)
model_SR = load_conditional_flow(dim=dim, cfg=config_SR, checkpoint_path=chk_pth_model_SR / 'model_checkpoint.pth', device=device)


data_complete = pd.read_feather(DATA_DIR / 'data_complete.feather')
data_complete = data_complete[data_complete['event_var'] == 1]
data_0 = data_complete[data_complete['process'] == 1]
data_0 = data_0[data_0['OS'] == True]

data_presel = _apply_config_mask(data_0, 'mask_preselection_loose', masks_config)
data_DR = _apply_config_mask(data_presel, 'mask_DR_wjets', masks_config)
data_antiDR = _apply_config_mask(data_presel, 'mask_antiDR_wjets', masks_config)
data_SR = _apply_config_mask(data_antiDR, 'SR', masks_config)
data_AR = _apply_config_mask(data_antiDR, 'AR', masks_config)
data_SR_like = _apply_config_mask(data_DR, 'SR_like', masks_config)
data_AR_like = _apply_config_mask(data_DR, 'AR_like', masks_config)

logger.info('Selection summary (rows, weight_sum):')
logger.info('  SR      : %d, %.6f', len(data_SR), float(data_SR['weight'].sum()))
logger.info('  AR      : %d, %.6f', len(data_AR), float(data_AR['weight'].sum()))
logger.info('  SR_like : %d, %.6f', len(data_SR_like), float(data_SR_like['weight'].sum()))
logger.info('  AR_like : %d, %.6f', len(data_AR_like), float(data_AR_like['weight'].sum()))

input_variables = ['njets'] + list(variables)
X_AR = get_my_data(data_AR, input_variables).to_torch(device=None).X.to(device)

# DR-trained models applied to antiDR (SR region) events
log_pdf_AR_like, valid_AR_like = evaluate_log_pdf(model_AR_like, X_AR)
log_pdf_SR_like, valid_SR_like = evaluate_log_pdf(model_SR_like, X_AR)

# antiDR-trained models applied to DR region (AR-like) events
log_pdf_AR, valid_AR = evaluate_log_pdf(model_AR, X_AR)
log_pdf_SR, valid_SR = evaluate_log_pdf(model_SR, X_AR)

global_ff_DR = np.sum(data_SR_like['weight']) / np.sum(data_AR_like['weight'])
global_ff_SR = np.sum(data_SR['weight']) / np.sum(data_AR['weight'])

logger.info('Global FF DR: %f', global_ff_DR)
logger.info('Global FF SR: %f', global_ff_SR)
logger.info('Valid preprocessing fraction DR: AR-like=%.4f SR-like=%.4f', valid_AR_like.mean(), valid_SR_like.mean())
logger.info('Valid preprocessing fraction antiDR: AR=%.4f SR=%.4f', valid_AR.mean(), valid_SR.mean())

ff_DR = compute_fake_factors(log_pdf_AR_like, log_pdf_SR_like, global_ff_DR)
ff_SR = compute_fake_factors(log_pdf_AR, log_pdf_SR, global_ff_SR)

print(log_pdf_AR_like)
print(ff_DR)

plot_ff_distributions(ff_SR, ff_DR, plot_dir=PLOTS_DIR)

plot_pdf_distribution(log_pdf_SR_like, log_pdf_AR_like, plot_dir=PLOTS_DIR, title='DR')
plot_pdf_distribution(log_pdf_SR, log_pdf_AR, plot_dir=PLOTS_DIR, title='SR')


for var in variables:
    plt_control_plots(SR = data_SR, AR = data_AR, FF = ff_SR, var = var, range=(0, 150), region='SR', plot_dir=PLOTS_DIR)
    plt_control_plots(SR = data_SR, AR = data_AR, FF = ff_DR, var = var, range=(0, 150), region='DR', plot_dir=PLOTS_DIR)

plot_correlation_FF_DR_SR(ff_SR, ff_DR, plot_dir=PLOTS_DIR)