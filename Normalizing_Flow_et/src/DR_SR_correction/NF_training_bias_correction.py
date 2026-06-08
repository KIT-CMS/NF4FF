import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.distributions as D
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
DATA_DIR = WORKSPACE_ROOT / 'data'
MASKS_CONFIG_PATH = CONFIG_DIR / 'masks_MC.yaml'

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classes.Dataclasses import _component_collection
from classes.Collection import load_model_config, ModelConfig, load_conditional_flow
from classes.NeuralNetworks import ConditionalRealNVP
from CustomLogging import setup_logging


logger = setup_logging(logger=logging.getLogger(__name__))
MODE_DIR = 'conditional_njets_input'
CONFIG_PATH = CONFIG_DIR / 'config_NF.yaml'
BIAS_FLOW_RESULTS_DIR = SCRIPT_DIR / 'Training_results_bias_correction'


class ConditionalAffine1D(nn.Module):
    def __init__(self, cond_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # outputs mu and log_sigma
        )

    def forward(self, y, cond):
        """
        Forward map: y -> z
        """
        params = self.net(cond)
        mu, log_sigma = params[:, 0], params[:, 1]

        sigma = t.exp(log_sigma)
        z = (y - mu) / sigma
        log_det = -log_sigma

        return z, log_det

    def inverse(self, z, cond):
        """
        Inverse map: z -> y
        """
        params = self.net(cond)
        mu, log_sigma = params[:, 0], params[:, 1]

        sigma = t.exp(log_sigma)
        y = sigma * z + mu

        return y

class ConditionalFlow1D(nn.Module):
    def __init__(self, cond_dim, n_layers=4):
        super().__init__()

        self.layers = nn.ModuleList(
            [ConditionalAffine1D(cond_dim) for _ in range(n_layers)]
        )
        self.base_dist = D.Normal(0.0, 1.0)

    def log_prob(self, y, cond):
        """
        Compute log p(y | cond)
        """
        log_det_sum = 0.0
        z = y

        for layer in self.layers:
            z, log_det = layer(z, cond)
            log_det_sum += log_det

        log_pz = self.base_dist.log_prob(z)
        return log_pz + log_det_sum

    def sample(self, cond, n_samples=1):
        """
        Sample FF_SR given cond
        """
        z = self.base_dist.sample((n_samples, cond.shape[0]))
        z = z.view(-1)

        cond_rep = cond.repeat(n_samples, 1)

        y = z
        for layer in reversed(self.layers):
            y = layer.inverse(y, cond_rep)

        return y.view(n_samples, -1)

def train_flow(flow, optimizer, FF_SR, FF_DR, X, weights=None, n_epochs=50):
    flow.train()

    # Conditioning vector
    cond = t.cat([FF_DR, X], dim=1)

    if weights is not None:
        weights = weights.reshape(-1)
        if not t.isfinite(weights).all():
            raise ValueError('Training weights contain non-finite values.')
        if (weights < 0).any():
            raise ValueError('Training weights must be non-negative for weighted likelihood training.')
        weight_norm = weights.sum().clamp_min(1e-12)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        logp = flow.log_prob(FF_SR.squeeze(), cond)
        if weights is None:
            loss = -logp.mean()
        else:
            loss = -(weights * logp).sum() / weight_norm

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"epoch {epoch:03d} | NLL = {loss.item():.4f}")





def get_my_data_events(df, training_var):
    _df = df
    return _component_collection(
        X=_df[training_var].to_numpy(dtype=np.float32),
        weights=_df['weight_wjets'].to_numpy(dtype=np.float32),
    )


def load_training_variables(config_path: Path = CONFIG_DIR / 'training_variables.yaml') -> list[str]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['variables_MC']


def resolve_training_name(variables: list[str]) -> str:
    tail = variables[4:]
    tag = '_'.join(tail) if tail else 'none'
    return f"training_vars{len(variables)}_{tag}"


def build_conditioning_name(conditional_variables: list[str]) -> str:
    if not conditional_variables:
        return 'conditioning_none'

    readable = '_'.join(conditional_variables)
    readable = re.sub(r'[^A-Za-z0-9_]+', '_', readable).strip('_')
    return f'conditioning_{readable}' if readable else 'conditioning_none'


def _extract_hidden_dim(flow: nn.Module) -> int | None:
    first_layer = flow.layers[0] if hasattr(flow, 'layers') and len(flow.layers) > 0 else None
    if first_layer is None or not hasattr(first_layer, 'net'):
        return None
    first_linear = first_layer.net[0] if len(first_layer.net) > 0 else None
    return int(first_linear.out_features) if isinstance(first_linear, nn.Linear) else None


def save_bias_flow_and_config(
    flow: ConditionalFlow1D,
    optimizer: t.optim.Optimizer,
    training_variables: list[str],
    conditional_variables: list[str],
    n_epochs: int,
    output_root: Path = BIAS_FLOW_RESULTS_DIR,
) -> Path:
    training_tag = resolve_training_name(training_variables)
    conditioning_tag = build_conditioning_name(conditional_variables)

    output_dir = output_root / training_tag / conditioning_tag
    latest_dir = output_dir / 'latest'
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': flow.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_variables': list(training_variables),
        'conditioning_variables': list(conditional_variables),
    }

    t.save(checkpoint, output_dir / 'model_checkpoint.pth')
    t.save(checkpoint, latest_dir / 'model_checkpoint.pth')

    config_payload = {
        'model_class': type(flow).__name__,
        'n_layers': len(flow.layers),
        'hidden_dim': _extract_hidden_dim(flow),
        'base_distribution': 'Normal(0,1)',
        'cond_dim': int(getattr(flow.layers[0].net[0], 'in_features', 0)) if len(flow.layers) > 0 else 0,
        'training_variables': list(training_variables),
        'training_variables_tag': training_tag,
        'conditioning_variables': list(conditional_variables),
        'conditioning_tag': conditioning_tag,
        'n_epochs': int(n_epochs),
    }

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)
    with open(latest_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config_payload, f, sort_keys=False)

    logger.info('Saved bias-correction flow checkpoint and config to %s', output_dir)
    return output_dir


def load_saved_model_config(checkpoint_dir: str | Path, fallback_path: str | Path) -> ModelConfig:
    saved_config_path = Path(checkpoint_dir).parent / 'config.yaml'
    if saved_config_path.exists():
        with open(saved_config_path, 'r') as handle:
            raw = yaml.unsafe_load(handle)

        if isinstance(raw, ModelConfig):
            return raw

        values = vars(raw) if hasattr(raw, '__dict__') else raw
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
    log_ratio = np.clip(log_pdf_sr - log_pdf_ar, -700.0, 700.0)
    return np.clip(global_ff * np.exp(log_ratio), 0, 10)


@t.no_grad()
def evaluate_log_pdf(model: ConditionalRealNVP, X: t.Tensor) -> tuple[np.ndarray, np.ndarray]:
    cond_dim = int(getattr(model, 'cond_dim', 0))
    x_features = X[:, cond_dim:]
    x_preprocessed, _, valid_mask = model.apply_preprocessing(x_features)
    Xs = model.apply_scaler(x_preprocessed)

    n_invalid_preprocess = (~valid_mask).sum().item()
    n_nan_after_scale = t.isnan(Xs[valid_mask]).any(dim=-1).sum().item() if valid_mask.any() else 0
    n_inf_after_scale = t.isinf(Xs[valid_mask]).any(dim=-1).sum().item() if valid_mask.any() else 0
    logger.debug(
        'evaluate_log_pdf: n=%d  invalid_preprocess=%d  nan_after_scale=%d  inf_after_scale=%d',
        X.shape[0], n_invalid_preprocess, n_nan_after_scale, n_inf_after_scale,
    )

    log_pdf = model(X)
    log_pdf = t.nan_to_num(log_pdf, nan=-700.0, neginf=-700.0, posinf=700.0)
    return log_pdf.detach().cpu().numpy(), valid_mask.detach().cpu().numpy()


def load_ff_event_inputs(device: t.device, conditional_variables: list[str] = ['njets']) -> tuple[np.ndarray, np.ndarray, t.Tensor, t.Tensor]:
    variables = load_training_variables()

    dim = len(variables)
    resolved_tag = resolve_training_name(variables)
    training_root = SCRIPT_DIR / 'Training_results_MC' / MODE_DIR / resolved_tag

    chk_pth_model_AR_like = training_root / 'Wjets' / 'DR' / 'AR-like' / 'latest'
    chk_pth_model_SR_like = training_root / 'Wjets' / 'DR' / 'SR-like' / 'latest'
    chk_pth_model_AR = training_root / 'Wjets' / 'antiDR' / 'AR' / 'latest'
    chk_pth_model_SR = training_root / 'Wjets' / 'antiDR' / 'SR' / 'latest'

    config_AR_like = load_saved_model_config(chk_pth_model_AR_like, CONFIG_PATH)
    config_SR_like = load_saved_model_config(chk_pth_model_SR_like, CONFIG_PATH)
    config_AR = load_saved_model_config(chk_pth_model_AR, CONFIG_PATH)
    config_SR = load_saved_model_config(chk_pth_model_SR, CONFIG_PATH)

    model_AR_like = load_conditional_flow(dim=dim, cfg=config_AR_like, checkpoint_path=chk_pth_model_AR_like / 'model_checkpoint.pth', device=device)
    model_SR_like = load_conditional_flow(dim=dim, cfg=config_SR_like, checkpoint_path=chk_pth_model_SR_like / 'model_checkpoint.pth', device=device)
    model_AR = load_conditional_flow(dim=dim, cfg=config_AR, checkpoint_path=chk_pth_model_AR / 'model_checkpoint.pth', device=device)
    model_SR = load_conditional_flow(dim=dim, cfg=config_SR, checkpoint_path=chk_pth_model_SR / 'model_checkpoint.pth', device=device)

    masks_config = load_masks_config(MASKS_CONFIG_PATH)
    data_complete = pd.read_feather(DATA_DIR / 'data_complete.feather')
    data_MC = data_complete[data_complete['process'] == 1]
    data_events = data_complete[data_complete['process'] == 0]

    data_presel_MC = _apply_config_mask(data_MC, 'mask_preselection_loose', masks_config)
    data_presel_events = _apply_config_mask(data_events, 'mask_preselection_loose', masks_config)

    data_SR_MC = _apply_config_mask(data_presel_MC, 'SR', masks_config)
    data_AR_MC = _apply_config_mask(data_presel_MC, 'AR', masks_config)

    data_AR_events = _apply_config_mask(data_presel_events, 'AR', masks_config)

    data_SR_like_MC = _apply_config_mask(data_presel_MC, 'SR_like', masks_config)
    data_AR_like_MC = _apply_config_mask(data_presel_MC, 'AR_like', masks_config)

    input_variables = ['njets'] + list(variables)
    X_AR = get_my_data_events(data_AR_events, input_variables).to_torch(device=None).X.to(device)

    log_pdf_AR_like, _ = evaluate_log_pdf(model_AR_like, X_AR)
    log_pdf_SR_like, _ = evaluate_log_pdf(model_SR_like, X_AR)
    log_pdf_AR, _ = evaluate_log_pdf(model_AR, X_AR)
    log_pdf_SR, _ = evaluate_log_pdf(model_SR, X_AR)

    global_ff_DR_MC = np.sum(data_SR_like_MC['weight']) / np.sum(data_AR_like_MC['weight'])
    global_ff_SR_MC = np.sum(data_SR_MC['weight']) / np.sum(data_AR_MC['weight'])

    ff_DR_events = compute_fake_factors(log_pdf_AR_like, log_pdf_SR_like, global_ff_DR_MC)
    ff_SR_events = compute_fake_factors(log_pdf_AR, log_pdf_SR, global_ff_SR_MC)

    conditional_data = get_my_data_events(data_AR_events, conditional_variables).to_torch(device=None)
    X_conditional = conditional_data.X.to(device)
    weights_conditional = conditional_data.weights.to(device)

    # Diagnostic: check for non-finite weights
    n_nan_weights = t.isnan(weights_conditional).sum().item()
    n_inf_weights = t.isinf(weights_conditional).sum().item()
    n_total = weights_conditional.shape[0]
    
    if n_nan_weights > 0 or n_inf_weights > 0:
        logger.warning(
            'Non-finite weights detected: n_nan=%d, n_inf=%d (total=%d). '
            'Replacing with 1.0.',
            n_nan_weights, n_inf_weights, n_total
        )
        weights_conditional = t.where(
            t.isfinite(weights_conditional),
            weights_conditional,
            t.ones_like(weights_conditional)
        )

    return ff_DR_events, ff_SR_events, X_conditional, weights_conditional


def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    conditional_variables = ['njets', 'pt_1', 'pt_2']
    n_epochs = 50
    training_variables = load_training_variables()
    ff_DR_events, ff_SR_events, X_conditional, weights_conditional = load_ff_event_inputs(device=device, conditional_variables=conditional_variables)

    # Tensors for optional downstream bias-flow training.
    FF_DR = t.from_numpy(ff_DR_events.astype(np.float32)).unsqueeze(1).to(device)
    FF_SR = t.from_numpy(ff_SR_events.astype(np.float32)).unsqueeze(1).to(device)

    flow = ConditionalFlow1D(cond_dim=1 + X_conditional.shape[1], n_layers=4).to(device)
    optimizer = t.optim.Adam(flow.parameters(), lr=1e-3)

    logger.info('Loaded FF inputs: n_events=%d, X_dim=%d', X_conditional.shape[0], X_conditional.shape[1])
    logger.info('ff_DR_events shape=%s, ff_SR_events shape=%s', ff_DR_events.shape, ff_SR_events.shape)
    logger.info('conditional event weights: sum=%.6f, min=%.6f, max=%.6f', weights_conditional.sum().item(), weights_conditional.min().item(), weights_conditional.max().item())

    train_flow(flow, optimizer, FF_SR, FF_DR, X_conditional, weights=weights_conditional, n_epochs=n_epochs)

    save_bias_flow_and_config(
        flow=flow,
        optimizer=optimizer,
        training_variables=training_variables,
        conditional_variables=conditional_variables,
        n_epochs=n_epochs,
    )
    


if __name__ == '__main__':
    main()

