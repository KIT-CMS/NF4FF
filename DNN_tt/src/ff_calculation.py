from pathlib import Path
import copy
import logging
import random
import time
from typing import Literal, Union

import numpy as np
from tap import Tap
import torch as t

from classes.NeuralNetworks import load_fold_combined_model, FoldCombinedDNN
from classes.DataHandling import test_data
from classes.Loading import load_config, load_variables, load_labels, load_data
from classes.FF_calculation import calculate_fake_factors_incl, calculate_fake_factors, calculate_fake_factor_dnn, calculate_fake_factor_classic, calculate_fake_factors_in_DR_qcd


SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

class Args(Tap):
    taus: Literal['split', 'incl'] = 'split' # split: calc 2 FF for tau1 and tau2 | incl: calc only 1 FF
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables"
    dnn_grouped: bool = True

args = Args().parse_args()

cfg_path = load_config('/work/tapp/TauFF/NF4FF/DNN_tt/configs/config_path.yaml')

DATA_PATH = f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather'
DATA_CLASSIC_JV_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_jvoss.feather"
DATA_CLASSIC_SG_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_sgiappic.feather"
MASKS_PATH = cfg_path["masks"]
MASKS_PATH_INCL = cfg_path["masks_incl"]
TRAINING_VAR_PATH = cfg_path["train_var"]
NN_CONFIG_PATH = cfg_path["DNN"]
CHECKPOINT_DIR = cfg_path["traininfg_results"]

PLOTTING_CONFIG_PATH = cfg_path["cfg_plotting"]
LABELS_CONFIG_PATH = cfg_path["labels"]

#PLOTS_DIR = Path('../plots/layers_3/ReLU')
PLOTS_DIR = Path(cfg_path["plots"])
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_GROUPINGS = ('tau_decaymode', 'njets')
PLOT_SUBDIRS = ('closure_in_DR', 'FF_distribution_AR', 'FF_distribution_DR', 'closure_plots')
for subdir in PLOT_SUBDIRS:
    for grouping in PLOT_GROUPINGS:
        (PLOTS_DIR / subdir / grouping).mkdir(parents=True, exist_ok=True)






PLOTTING_CFG = load_config(PLOTTING_CONFIG_PATH) 
LABELS_CFG = load_labels(LABELS_CONFIG_PATH)

VARIABLES_SMALL = PLOTTING_CFG.get('variables_set_small', [])
VARIABLES_LARGE = PLOTTING_CFG.get('variables_set_large', [])


def get_bins(variable):
    bin_spec = PLOTTING_CFG.get('bins_by_variable', {}).get(variable)
    if bin_spec is None:
        raise KeyError(f'No bin specification found for variable: {variable}')

    if isinstance(bin_spec, (list, tuple)) and len(bin_spec) == 3:
        start, stop, num = bin_spec
        return np.linspace(float(start), float(stop), int(num))

    return np.asarray(bin_spec, dtype=float)


def get_label(variable, channel='et'):
    labels_by_channel = LABELS_CFG.get(channel, {}) if isinstance(LABELS_CFG, dict) else {}
    return labels_by_channel.get(variable, variable)


def get_bins_and_label(variable, channel='et'):
    return get_bins(variable), get_label(variable, channel)


def _prepare_input_tensor(model: t.nn.Module, X_tensor: t.Tensor, df_ar) -> t.Tensor:
    """Return the correctly shaped input tensor for the given model type."""
    if isinstance(model, FoldCombinedDNN):
        event_ids = t.from_numpy(np.asarray(df_ar['event'] % 2, dtype=np.float32))
        return t.cat([event_ids.unsqueeze(0), X_tensor.T], dim=0)  # [1 + n_features, N]
    return X_tensor  # [N, n_features]



def _build_group_masks(values, grouping_definition):
    masks = []

    for group in grouping_definition:
        if len(group) == 1:
            val = group[0]
            mask = values == val
            group_name = f"{val}"
        elif len(group) == 2:
            low, high = group
            mask = (values >= low) & (values <= high)
            group_name = f"{low}_{high}"
        else:
            raise ValueError(f"Invalid group definition: {group}")

        masks.append((group_name, mask))

    return masks



def _build_normalization_vector_for_views(
    target_view,
    sr_view,
    ar_view,
    grouping_variable,
    grouping_definition,
):
    """Build one normalization value per event in the target view."""
    normalization = np.zeros(target_view.n, dtype=np.float32)

    target_group_values = np.asarray(target_view[grouping_variable])
    target_masks = _build_group_masks(target_group_values, grouping_definition)

    sr_masks = dict(_build_group_masks(
        np.asarray(sr_view[grouping_variable]),
        grouping_definition,
    ))
    ar_masks = dict(_build_group_masks(
        np.asarray(ar_view[grouping_variable]),
        grouping_definition,
    ))

    for group_name, target_mask in target_masks:
        numerator = np.sum(sr_view.weight[sr_masks[group_name]])
        denominator = np.sum(ar_view.weight[ar_masks[group_name]])
        normalization[target_mask] = numerator / denominator if denominator > 0 else 0.0

    return normalization



def build_normalization_vector(
    df,
    grouping_variable,
    grouping_definition,
    process='wjets',
):
    """Build per-event normalization factors for df.AR."""
    process_views = {
        'wjets': (df.AR, df.data.SR_like_wjets, df.data.AR_like_wjets),
        'qcd': (df.AR, df.data.SR_like_qcd, df.data.AR_like_qcd),
        'ttbar': (df.AR, df.data.SR_like_ttbar, df.data.AR_like_ttbar),
    }

    if process not in process_views:
        raise ValueError(f"Unknown process '{process}'. Use 'wjets', 'qcd', or 'ttbar'.")

    target_view, sr_view, ar_view = process_views[process]
    return _build_normalization_vector_for_views(
        target_view,
        sr_view,
        ar_view,
        grouping_variable,
        grouping_definition,
    )

def build_normalization_vector_in_DR(
    df,
    process,
    grouping_variable,
    grouping_definition,
):
    """Build per-event normalization factors for df.AR_like_process."""
    target_view = getattr(df, f'AR_like_{process}')
    sr_view = getattr(df.data, f'SR_like_{process}')
    ar_view = getattr(df.data, f'AR_like_{process}')

    return _build_normalization_vector_for_views(
        target_view,
        sr_view,
        ar_view,
        grouping_variable,
        grouping_definition,
    )


def build_normalization_vector_in_DR_qcd(
    df,
    grouping_variable,
    grouping_definition,
):
    """Build per-event normalization factors for df.AR_like_qcd."""
    return _build_normalization_vector_for_views(
        df.AR_like_qcd,
        df.data.SR_like_qcd,
        df.data.AR_like_qcd,
        grouping_variable,
        grouping_definition,
    )



def predict_fake_factors(
    model,
    X_wjets,
    normalization,
    device: t.device | None = None,
):
    """Returns fake factors for one model."""
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    eps = 1e-6

    model = model.to(device)
    X_wjets = X_wjets.to(device)
    normalization = normalization.to(device)

    with t.inference_mode():
        f = model(X_wjets).squeeze()

    f = t.clamp(f, eps, 1 - eps)
    ratio = f / (1.0 - f)
    fake_factor = ratio * normalization
    fake_factor = t.clamp(fake_factor, 0, 1)

    return fake_factor.cpu()



def load_models(checkpoint_dir,
                seeds,
                process = 'wjets',
                ):
    models = []

    for seed in seeds:
        model = load_fold_combined_model(
            even_model_path=(
                Path(checkpoint_dir)
                / 'tau_decaymode'
                / process
                / str(seed)
                / 'fold_even'
            ),
            odd_model_path=(
                Path(checkpoint_dir)
                / 'tau_decaymode'
                / process
                / str(seed)
                / 'fold_odd'
            ),
        )

        model.eval()
        models.append(model)

    return models



def _calculate_fake_factor_mean_std_for_view_per_model(
    df_view,
    models,
    training_variables,
    normalization,
    device: t.device | None = None,
):
    """
    Compute FF mean/std by processing all events per model (no batching).
    """
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print(f'[INFO] Using device: {device}')
    print('[INFO] Building full input tensor...')

    X = test_data(df_view, training_variables)
    X_tensor = t.from_numpy(X.X).float()
    X = _prepare_input_tensor(models[0], X_tensor, df_view).to(device)

    normalization = t.from_numpy(normalization).float().to(device)

    n_events = df_view.n
    n_models = len(models)

    print(f'[INFO] Events: {n_events:,}')
    print(f'[INFO] Models: {n_models}')
    print('[INFO] Inference mode: full events per model')

    sum_ff = t.zeros(n_events, dtype=t.float32, device=device)
    sum_sq_ff = t.zeros(n_events, dtype=t.float32, device=device)

    start_time = time.time()

    with t.inference_mode():
        for i, model in enumerate(models, start=1):
            model.to(device)
            model.eval()

            f = model(X).squeeze()
            f = t.clamp(f, 1e-6, 1 - 1e-6)
            ratio = f / (1.0 - f)
            ff = ratio * normalization
            ff = t.clamp(ff, 0, 1)

            sum_ff += ff
            sum_sq_ff += ff * ff

            model.cpu()  # free GPU memory after each model

            elapsed = time.time() - start_time
            speed = i / elapsed if elapsed > 0 else 0
            remaining = n_models - i
            eta = remaining / speed if speed > 0 else 0

            print(
                f'\r[PROGRESS] Model {i}/{n_models} | '
                f'{100.0 * i / n_models:6.2f}% | '
                f'{speed:,.2f} models/s | '
                f'ETA {eta/60:.2f} min',
                end='',
                flush=True,
            )

    mean_ff = sum_ff / n_models
    var_ff = (sum_sq_ff / n_models) - mean_ff * mean_ff
    var_ff = t.clamp(var_ff, min=0)
    std_ff = t.sqrt(var_ff)

    print('\n[INFO] Inference complete.')

    return mean_ff.cpu().numpy(), std_ff.cpu().numpy()


def _enable_dropout_only(model: t.nn.Module) -> None:
    """
    Keep the model in eval mode, but activate dropout layers only.
    This avoids BatchNorm running-stat updates during MC-dropout inference.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, t.nn.Dropout):
            module.train()


def _calculate_fake_factor_mean_std_for_view_per_model_per_mask(
    df_view,
    model,
    training_variables,
    normalization,
    device: t.device | None = None,
):
    """
    Compute FF mean/std by processing all events per model (no batching).
    """
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print(f'[INFO] Using device: {device}')
    print('[INFO] Building full input tensor...')

    X = test_data(df_view, training_variables)
    X_tensor = t.from_numpy(X.X).float()
    X = _prepare_input_tensor(model, X_tensor, df_view).to(device)

    normalization = t.from_numpy(normalization).float().to(device)

    n_events = df_view.n
    n_masks = 100

    print(f'[INFO] Events: {n_events:,}')
    print(f'[INFO] Models: {n_masks}')
    print('[INFO] Inference mode: full events per model')

    sum_ff = t.zeros(n_events, dtype=t.float32, device=device)
    sum_sq_ff = t.zeros(n_events, dtype=t.float32, device=device)

    start_time = time.time()

    with t.inference_mode():
        for i in range(1, n_masks + 1):
            model.to(device)
            _enable_dropout_only(model)

            f = model(X).squeeze()
            f = t.clamp(f, 1e-6, 1 - 1e-6)
            ratio = f / (1.0 - f)
            ff = ratio * normalization
            ff = t.clamp(ff, 0, 1)

            sum_ff += ff
            sum_sq_ff += ff * ff

            model.cpu()  # free GPU memory after each model

            elapsed = time.time() - start_time
            speed = i / elapsed if elapsed > 0 else 0
            remaining = n_masks - i
            eta = remaining / speed if speed > 0 else 0

            print(
                f'\r[PROGRESS] Model {i}/{n_masks} | '
                f'{100.0 * i / n_masks:6.2f}% | '
                f'{speed:,.2f} models/s | '
                f'ETA {eta/60:.2f} min',
                end='',
                flush=True,
            )

    mean_ff = sum_ff / n_masks
    var_ff = (sum_sq_ff / n_masks) - mean_ff * mean_ff
    var_ff = t.clamp(var_ff, min=0)
    std_ff = t.sqrt(var_ff)

    print('\n[INFO] Inference complete.')

    return mean_ff.cpu().numpy(), std_ff.cpu().numpy()



def calculate_fake_factor_mean_std(
    df,
    models,
    training_variables,
    grouping_variable,
    grouping_definition,
    process='wjets',
    output_mean='fake_factor_mean',
    output_std='fake_factor_std',
    device: t.device | None = None,
):
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print('[INFO] Computing normalization...')
    normalization = build_normalization_vector(
        df,
        grouping_variable,
        grouping_definition,
        process,
    )

    mean_result, std_result = _calculate_fake_factor_mean_std_for_view_per_model(
        df.AR,
        models,
        training_variables,
        normalization,
        device=device,
    )

    df.AR[output_mean] = mean_result
    df.AR[output_std] = std_result
    return df


def calculate_fake_factor_mean_std_dropout_mask_variation(
    df,
    model,
    training_variables,
    grouping_variable,
    grouping_definition,
    process='wjets',
    output_mean='fake_factor_mean',
    output_std='fake_factor_std',
    device: t.device | None = None,
):
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print('[INFO] Computing normalization...')
    normalization = build_normalization_vector(
        df,
        grouping_variable,
        grouping_definition,
        process,
    )

    mean_result, std_result = _calculate_fake_factor_mean_std_for_view_per_model_per_mask(
        df.AR,
        model,
        training_variables,
        normalization,
        device=device,
    )

    df.AR[output_mean] = mean_result
    df.AR[output_std] = std_result
    return df


def calculate_fake_factor_mean_std_in_DR(
    df,
    models,
    training_variables,
    grouping_variable,
    grouping_definition,
    process='wjets',
    output_mean='fake_factor_mean',
    output_std='fake_factor_std',
    device: t.device | None = None,
):
    if process not in {'wjets', 'qcd', 'ttbar'}:
        raise ValueError("calculate_fake_factor_mean_std_batched_in_DR only supports process='wjets', 'qcd', or 'ttbar'.")

    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print('[INFO] Computing normalization...')
    normalization = build_normalization_vector_in_DR(
        df,
        process,
        grouping_variable,
        grouping_definition,
    )

    target_view = getattr(df, f'AR_like_{process}')

    mean_result, std_result = _calculate_fake_factor_mean_std_for_view_per_model(
        target_view,
        models,
        training_variables,
        normalization,
        device=device,
    )

    target_view[output_mean] = mean_result
    target_view[output_std] = std_result
    return df

	
def calculate_fake_factor_mean_std_in_DR_dropout_mask_variation(
    df,
    model,
    training_variables,
    grouping_variable,
    grouping_definition,
    process='wjets',
    output_mean='fake_factor_mean',
    output_std='fake_factor_std',
    device: t.device | None = None,
):
    if process not in {'wjets', 'qcd', 'ttbar'}:
        raise ValueError("calculate_fake_factor_mean_std_batched_in_DR only supports process='wjets', 'qcd', or 'ttbar'.")

    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    print('[INFO] Computing normalization...')
    normalization = build_normalization_vector_in_DR(
        df,
        process,
        grouping_variable,
        grouping_definition,
    )

    target_view = getattr(df, f'AR_like_{process}')

    mean_result, std_result = _calculate_fake_factor_mean_std_for_view_per_model_per_mask(
        target_view,
        model,
        training_variables,
        normalization,
        device=device,
    )

    target_view[output_mean] = mean_result
    target_view[output_std] = std_result
    return df


class FixedMaskDropout(t.nn.Module):
    def __init__(self, ensemble_size: int, feature_dim: int, p: float):
        super().__init__()
        self.ensemble_size = ensemble_size
        total_slots = ensemble_size + 1  # 0 always 1.0, 1 to N are random

        masks = t.ones(total_slots, feature_dim)
        rand_masks = (t.rand(ensemble_size, feature_dim) > p).float() / (1.0 - p)  # scale to keep same expected value magnitude
        masks[1:] = rand_masks

        self.register_buffer("masks", masks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        total_rows = x.shape[0]
        batch_size = total_rows // (self.ensemble_size + 1)
        feature_dim = x.shape[1]
        x = x.reshape(self.ensemble_size + 1, batch_size, feature_dim)
        x = x * self.masks.unsqueeze(1)
        return x.reshape(total_rows, feature_dim)




class EnsembleStatUncWrapper(t.nn.Module):
    def __init__(
        self,
        model: t.nn.Module,
        ensemble_size: int = 10,
        direction: Literal["Nominal", "Up", "Down"] = "Nominal",
        sigma: float = 1.0,
        vary_index: Union[int, None] = None,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.direction = direction
        self.sigma = sigma
        self.vary_index = vary_index
        self.wrapped_model = model

        self._input_nodes = getattr(model, "_input_nodes", None)
        self._input_names = getattr(model, "_input_names", None)
        self._fold_id_name = getattr(model, "_fold_id_name", "event_parity")

        self._replace_layers(self.wrapped_model)

    def _replace_layers(self, module):
        for name, child in module.named_children():
            if isinstance(child, t.nn.Dropout) and child.p > 0:
                parent_seq = module
                layer_list = list(parent_seq)
                layer_idx = layer_list.index(child)
                prev_linear = layer_list[layer_idx - 2]  # preceding Activation + Dropout

                new_dropout = FixedMaskDropout(
                    ensemble_size=self.ensemble_size,
                    feature_dim=prev_linear.out_features,
                    p=child.p
                )
                setattr(module, name, new_dropout)
            else:
                self._replace_layers(child)

    def forward(self, X: t.Tensor) -> t.Tensor:
        outputs = self.wrapped_model(X.repeat(1, self.ensemble_size + 1))
        outputs = outputs.reshape(self.ensemble_size + 1, X.shape[1], *outputs.shape[1:])

        nominal = outputs[0]
        std = t.std(outputs[1:], dim=0, unbiased=True)
        mean = t.mean(outputs[1:], dim=0)
        total_uncertainty = ((mean - nominal) ** 2 + std ** 2).sqrt()

        if self.vary_index is not None:
            idx_mask = t.zeros(nominal.shape[-1], device=nominal.device, dtype=nominal.dtype)
            idx_mask[self.vary_index] = 1.0

            nominal_value = nominal[..., self.vary_index: self.vary_index + 1]
            uncertainty_value = total_uncertainty[..., self.vary_index: self.vary_index + 1]

            if self.direction == "Up":
                shifted_value = t.clamp(nominal_value + self.sigma * uncertainty_value, max=1.0)
            elif self.direction == "Down":
                shifted_value = t.clamp(nominal_value - self.sigma * uncertainty_value, min=0.0)

            R_old, R_new = 1.0 - nominal_value, 1.0 - shifted_value
            scale_factor = t.where(R_old > 1e-6, R_new / R_old, t.zeros_like(R_old))  # if norm_v becomes rounding 1.0

            return (shifted_value * idx_mask) + (nominal * (1.0 - idx_mask) * scale_factor)

        if self.direction == "Up":
            return nominal + self.sigma * total_uncertainty / 2
        elif self.direction == "Down":
            return nominal - self.sigma * total_uncertainty / 2
        else:
            return nominal  # should actually never happen :)

    @property
    def _imports(self) -> str:
        base_imports = getattr(self.wrapped_model, "_imports", "")
        wrapper_imports = f"from {self.__class__.__module__} import {self.__class__.__name__}\n"
        return base_imports + wrapper_imports

    @property
    def model_name(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.wrapped_model.model_name}, "
            f"ensemble_size={self.ensemble_size}, "
            f"direction='{self.direction}'"
            f")"
        )

    def __recreate__(self) -> str:
        return f"{self._imports}__model = {self.model_name}\n\n"


def calculate_fake_factors_ensemble(
    df,
    model_wjets: t.nn.Module,
    training_variables,
    grouping_variable: str,
    grouping_definition,
):
    """Compute nominal/up/down FFs in AR_like_wjets using fixed-mask dropout ensemble wrapper."""

    # Use independent model copies so wrappers cannot interfere through in-place layer replacement.
    nominal_model = EnsembleStatUncWrapper(
        model=copy.deepcopy(model_wjets),
        ensemble_size=100,
        direction='Nominal',
    )
    up_model = EnsembleStatUncWrapper(
        model=copy.deepcopy(model_wjets),
        ensemble_size=100,
        direction='Up',
    )
    down_model = EnsembleStatUncWrapper(
        model=copy.deepcopy(model_wjets),
        ensemble_size=100,
        direction='Down',
    )

    X_data = test_data(df.AR_like_wjets, training_variables)
    X_tensor = t.from_numpy(X_data.X).float()
    X_wjets = _prepare_input_tensor(model_wjets, X_tensor, df.AR_like_wjets)

    with t.no_grad():
        f_nom = nominal_model(X_wjets)
        f_up = up_model(X_wjets)
        f_down = down_model(X_wjets)

    # Convert to numpy before numpy math.
    f_nom = f_nom.detach().cpu().numpy().squeeze()
    f_up = f_up.detach().cpu().numpy().squeeze()
    f_down = f_down.detach().cpu().numpy().squeeze()

    eps = 1e-6
    f_nom = np.clip(f_nom, eps, 1 - eps)
    f_up = np.clip(f_up, eps, 1 - eps)
    f_down = np.clip(f_down, eps, 1 - eps)

    ratio_nom = f_nom / (1.0 - f_nom)
    ratio_up = f_up / (1.0 - f_up)
    ratio_down = f_down / (1.0 - f_down)

    fake_factor_nominal = np.zeros_like(ratio_nom)
    fake_factor_up = np.zeros_like(ratio_up)
    fake_factor_down = np.zeros_like(ratio_down)

    # Keep masks in the same region where predictions were computed.
    ar_group_values = np.asarray(df.AR_like_wjets[grouping_variable])
    group_masks = _build_group_masks(ar_group_values, grouping_definition)

    sr_wjets_masks = dict(_build_group_masks(
        np.asarray(df.data.SR_like_wjets[grouping_variable]),
        grouping_definition,
    ))
    ar_wjets_masks = dict(_build_group_masks(
        np.asarray(df.data.AR_like_wjets[grouping_variable]),
        grouping_definition,
    ))

    for group_name, ar_mask in group_masks:
        sr_wjets_mask = sr_wjets_masks[group_name]
        ar_wjets_mask = ar_wjets_masks[group_name]

        denom = np.sum(df.data.AR_like_wjets.weight[ar_wjets_mask])
        norm_wjets = (
            np.sum(df.data.SR_like_wjets.weight[sr_wjets_mask]) / denom
            if denom > 0
            else 0.0
        )

        fake_factor_nominal[ar_mask] = norm_wjets * ratio_nom[ar_mask]
        fake_factor_up[ar_mask] = norm_wjets * ratio_up[ar_mask]
        fake_factor_down[ar_mask] = norm_wjets * ratio_down[ar_mask]

        print(f"[{group_name}] WJets norm = {norm_wjets:.4f}")

    fake_factor_nominal = np.clip(fake_factor_nominal, 0, 1)
    fake_factor_up = np.clip(fake_factor_up, 0, 1)
    fake_factor_down = np.clip(fake_factor_down, 0, 1)

    df.AR_like_wjets['ff_wjets_nominal_ensemble'] = fake_factor_nominal
    df.AR_like_wjets['ff_wjets_up_ensemble'] = fake_factor_up
    df.AR_like_wjets['ff_wjets_down_ensemble'] = fake_factor_down

    return df

def calculate_fake_factors_ensemble_2sigma(
    df,
    model_wjets: t.nn.Module,
    training_variables,
    grouping_variable: str,
    grouping_definition,
):
    """Compute nominal/up/down FFs in AR_like_wjets using fixed-mask dropout ensemble wrapper."""

    # Use independent model copies so wrappers cannot interfere through in-place layer replacement.
    nominal_model = EnsembleStatUncWrapper(
        model=copy.deepcopy(model_wjets),
        ensemble_size=100,
        direction='Nominal',
    )
    up_model = EnsembleStatUncWrapper(
        model=copy.deepcopy(model_wjets),
        ensemble_size=100,
        direction='Up',
        sigma=2.0,
    )
    down_model = EnsembleStatUncWrapper(
        model=copy.deepcopy(model_wjets),
        ensemble_size=100,
        direction='Down',
        sigma=2.0,
    )

    X_data = test_data(df.AR_like_wjets, training_variables)
    X_tensor = t.from_numpy(X_data.X).float()
    X_wjets = _prepare_input_tensor(model_wjets, X_tensor, df.AR_like_wjets)

    with t.no_grad():
        f_nom = nominal_model(X_wjets)
        f_up = up_model(X_wjets)
        f_down = down_model(X_wjets)

    # Convert to numpy before numpy math.
    f_nom = f_nom.detach().cpu().numpy().squeeze()
    f_up = f_up.detach().cpu().numpy().squeeze()
    f_down = f_down.detach().cpu().numpy().squeeze()

    eps = 1e-6
    f_nom = np.clip(f_nom, eps, 1 - eps)
    f_up = np.clip(f_up, eps, 1 - eps)
    f_down = np.clip(f_down, eps, 1 - eps)

    ratio_nom = f_nom / (1.0 - f_nom)
    ratio_up = f_up / (1.0 - f_up)
    ratio_down = f_down / (1.0 - f_down)

    fake_factor_nominal = np.zeros_like(ratio_nom)
    fake_factor_up = np.zeros_like(ratio_up)
    fake_factor_down = np.zeros_like(ratio_down)

    # Keep masks in the same region where predictions were computed.
    ar_group_values = np.asarray(df.AR_like_wjets[grouping_variable])
    group_masks = _build_group_masks(ar_group_values, grouping_definition)

    sr_wjets_masks = dict(_build_group_masks(
        np.asarray(df.data.SR_like_wjets[grouping_variable]),
        grouping_definition,
    ))
    ar_wjets_masks = dict(_build_group_masks(
        np.asarray(df.data.AR_like_wjets[grouping_variable]),
        grouping_definition,
    ))

    for group_name, ar_mask in group_masks:
        sr_wjets_mask = sr_wjets_masks[group_name]
        ar_wjets_mask = ar_wjets_masks[group_name]

        denom = np.sum(df.data.AR_like_wjets.weight[ar_wjets_mask])
        norm_wjets = (
            np.sum(df.data.SR_like_wjets.weight[sr_wjets_mask]) / denom
            if denom > 0
            else 0.0
        )

        fake_factor_nominal[ar_mask] = norm_wjets * ratio_nom[ar_mask]
        fake_factor_up[ar_mask] = norm_wjets * ratio_up[ar_mask]
        fake_factor_down[ar_mask] = norm_wjets * ratio_down[ar_mask]

        print(f"[{group_name}] WJets norm = {norm_wjets:.4f}")

    fake_factor_nominal = np.clip(fake_factor_nominal, 0, 1)
    fake_factor_up = np.clip(fake_factor_up, 0, 1)
    fake_factor_down = np.clip(fake_factor_down, 0, 1)

    df.AR_like_wjets['ff_wjets_up_ensemble_2sigma'] = fake_factor_up
    df.AR_like_wjets['ff_wjets_down_ensemble_2sigma'] = fake_factor_down

    return df






###################
# ----- main -----#
###################

def main():

    # ----- load models from training.py -----
    if args.dnn_grouped:
        logger.info("Loading models from training.py for grouped DNN...")

        # tau decay mode
        model_tau1_tdm = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'tau1' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'tau1' / 'fold_odd',
        )
        model_tau2_tdm = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'tau2' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'tau_decaymode' / 'tau2' / 'fold_odd',
        )

        # njets
        model_tau1_njets = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'tau1' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'tau1' / 'fold_odd',
        )
        model_tau2_njets = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'tau2' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'njets' / 'tau2' / 'fold_odd',
        )
    else:
        logger.info("Loading models from training.py for ungrouped DNN...")

        model_tau1 = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau1' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau1' / 'fold_odd',
        )
        model_tau2 = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau2' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau2' / 'fold_odd',
        )
        model_incl = load_fold_combined_model(
            even_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau_incl' / 'fold_even',
            odd_model_path=Path(CHECKPOINT_DIR) / 'ungrouped' / 'tau_incl' / 'fold_odd',
        )


    # ----- grouping definitions -----

    grouping_njets = (
        (0,),
        (1,),
        (2, 1000),
    )

    grouping_tdm = (
        (0,),
        (1,),
        (10,),
        (11,),
    )

    # ----- execution -----
    logger.info("Loading data...")
    df = load_data(DATA_PATH, MASKS_PATH)
    df_incl = load_data(DATA_PATH, MASKS_PATH_INCL)
    df_classic_jv = load_data(DATA_CLASSIC_JV_PATH, MASKS_PATH)
    df_classic_sg = load_data(DATA_CLASSIC_SG_PATH, MASKS_PATH)
    #print(df_classic.columns)
    #exit()

    training_variables = load_variables(TRAINING_VAR_PATH, args.var)

   

    # ----- calculate fake factors -----
    # classic: at the moment from jvoss smhtt ul v12
    calculate_fake_factor_classic(df_classic_jv, 'jv')
    calculate_fake_factor_classic(df_classic_sg, 'sg')


    if args.dnn_grouped:
        # tau decay mode
        logger.info("Calculating fake factors for tau decay mode...")
        calculate_fake_factors(
            df=df,
            model_tau1=model_tau1_tdm,
            model_tau2=model_tau2_tdm,
            training_variables=training_variables,
            grouping_variable = ['tau_decaymode_1', 'tau_decaymode_2'],
            grouping_definition = grouping_tdm,
            output_suffix = 'tau_dm',
        )

        # njets
        logger.info("Calculating fake factors for njets...")
        calculate_fake_factors(
            df=df,
            model_tau1=model_tau1_njets,
            model_tau2=model_tau2_njets,
            training_variables=training_variables,
            grouping_variable = 'njets',
            grouping_definition = grouping_njets,
            output_suffix = 'njets',
        )


        logger.info("Saving fake factors to df in ff_dnn_tau_dm and ff_dnn_njets columns...")
        calculate_fake_factor_dnn(
            df1 = df.AR_tau1,
            df2 = df.AR_tau2,
            grouping = 'tau_decaymode',
        )

        calculate_fake_factor_dnn(
            df1 = df.AR_tau1,
            df2 = df.AR_tau2,
            grouping = 'njets',
        )

        #the equivalent would be saving the tau1 and tau2 FF seperately
        
        # ----- calculate fake factors in DR -----
        logger.info("Calculating fake factors in DR...")
        calculate_fake_factors_in_DR_qcd(
            df=df,
            model_tau1=model_tau1_tdm,
            model_tau2=model_tau2_tdm,
            training_variables=training_variables,
            grouping_variable = ['tau_decaymode_1', 'tau_decaymode_2'],
            grouping_definition = grouping_tdm,
            output_suffix = 'tau_dm',
        )

        calculate_fake_factors_in_DR_qcd(
            df=df,
            model_tau1=model_tau1_njets,
            model_tau2=model_tau2_njets,
            training_variables=training_variables,
            grouping_variable = 'njets',
            grouping_definition = grouping_njets,
            output_suffix = 'njets',
        )

    else:
        if args.taus == 'split':
            logger.info("Calculating fake factors...")
            calculate_fake_factors(
                df=df,
                model_tau1=model_tau1,
                model_tau2=model_tau2,
                training_variables=training_variables,
            )
        elif args.taus == 'incl':
            logger.info("Calculating fake factors inclusive...")
            calculate_fake_factors_incl(
                df=df_incl,
                model=model_incl,
                training_variables=training_variables,
            )    
    

    #print(list(df.columns))
    logger.info(f"Saving main dataframe to feather file: {DATA_PATH}")
    df.to_feather(DATA_PATH)

    logger.info(f"Saving tau inclusive dataframe to feather file: {DATA_PATH}")
    df_incl.to_feather(DATA_PATH)

    logger.info(f"Saving classic dataframe with fake-factor columns to feather file: {DATA_CLASSIC_JV_PATH} and {DATA_CLASSIC_SG_PATH}")
    df_classic_jv.to_feather(DATA_CLASSIC_JV_PATH)
    df_classic_sg.to_feather(DATA_CLASSIC_SG_PATH)


if __name__ == '__main__':
    main()
    logger.info("Done.")