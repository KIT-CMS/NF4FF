from __future__ import annotations
import logging
import hashlib
from contextlib import contextmanager
from copy import deepcopy
import correctionlib as cr
from pathlib import Path
import os
import random 
import re
from typing import Literal, Tuple, Iterable

import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep
import numpy as np
import pandas as pd
from tap import Tap
from tayloranalysis import extend_model as _ta_extend
import torch
import torch as t
import yaml

from classes.Logging import setup_logging
from classes.NeuralNetworks import RealNVP
from classes.Dataclasses import ModelConfig
from classes.Collection import load_model_config, load_flow, load_conditional_flow, evaluate_pdf, compute_eventwise_fake_factors, get_my_data_qcd
from classes.Collection import load_config, load_grouped_qcd_njets_router

logger = setup_logging(logger=logging.getLogger(__name__))

SEED = 42

class Args(Tap):
    model_mode: Literal['grouped_njets_split', 'single_nf', 'conditional_nf'] = 'single_nf'  # Same default as plot_results_njets.py.
    training_tag: str = ''  # Empty resolves the same training as plot_results_njets.py; use DNNplotting for that explicit checkpoint.
    classifier_training_tag: str = ''  # Optional classifier training folder suffix after 'training_'. Empty -> pick most recent.
    classifier_hidden_layers: int = 2  # Binary-classifier selection helper: pick the most recent training with this number of hidden layers.

    taus = 1 #[1, 2] #[1, 2, 12] # list of tau fakes
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables_61"
    


# Runtime context (initialized in `initialize_runtime_context()` and consumed by plotting functions)
cfg_path = load_config('/work/tapp/TauFF/NF4FF/Normalizing_Flow_tt/configs/config_path.yaml')
args = Args().parse_args()

config_path = cfg_path['config_NF']
plot_root_dir = Path('plots')
MASKS_CONFIG_PATH = Path('/work/tapp/TauFF/NF4FF/Normalizing_Flow_tt/configs/masks.yaml')
MASKS_CONFIG: dict[str, list[str]] = {}

def build_training_variables_tag(variables: list[str]) -> str:
    variables_joined = "|".join(variables)
    variables_hash = hashlib.sha1(variables_joined.encode("utf-8")).hexdigest()[:8]
    tail_variables = variables[4:]
    if tail_variables:
        readable_tail = "_".join(tail_variables)
        readable_tail = re.sub(r"[^A-Za-z0-9_]+", "_", readable_tail).strip("_")
    else:
        readable_tail = "none"
    
    if 'deltaR_ditaupair' in variables and 'pt_1' in variables:
        return f"vars{len(variables)}_{readable_tail}_{variables_hash}"
    elif 'pt_1' not in variables:
        return f"vars{len(variables)}2_{readable_tail}_{variables_hash}"
    elif 'deltaR_ditaupair' not in variables and 'pt_1' in variables: 
        return f"vars{len(variables)}1_{readable_tail}_{variables_hash}"
    else:
        return f"vars{len(variables)}a_{readable_tail}_{variables_hash}"

def _build_training_variables_prefix(variables: list[str]) -> str:
    """Return the hash-free readable prefix, e.g. 'vars5_pt_vis'."""
    tail_variables = variables[4:]
    if tail_variables:
        readable_tail = "_".join(tail_variables)
        readable_tail = re.sub(r"[^A-Za-z0-9_]+", "_", readable_tail).strip("_")
    else:
        readable_tail = "none"

    if 'deltaR_ditaupair' in variables and 'pt_1' in variables:
        return f"vars{len(variables)}_{readable_tail}"
    elif 'pt_1' not in variables:
        return f"vars{len(variables)}2_{readable_tail}"
    elif 'deltaR_ditaupair' not in variables and 'pt_1' in variables:
        return f"vars{len(variables)}1_{readable_tail}"
    else:
        return f"vars{len(variables)}a_{readable_tail}"

def resolve_training_tag(variables: list[str], mode_dir: str, base_dir: str = cfg_path['NF_results']) -> str:
    """
    Glob for a training folder whose name starts with 'training_<prefix>'
    (ignoring the trailing hash).  Returns the folder-name suffix that follows
    'training_', so the caller can use it identically to training_variables_tag.

    If exactly one matching folder is found it is used.  If several match the
    most-recently-modified one is chosen.  If none match, fall back to the
    exact computed tag (old behaviour).
    """
    exact_tag = build_training_variables_tag(variables)
    prefix = _build_training_variables_prefix(variables)
    search_root = Path(base_dir) / mode_dir
    if not search_root.exists():
        logger.warning('Training base dir not found: %s — falling back to exact tag', search_root)
        return exact_tag

    candidates = sorted(search_root.glob(f'training_{prefix}*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        logger.warning(
            'No training folder matching training_%s* found in %s — falling back to exact tag %s',
            prefix, search_root, exact_tag,
        )
        return exact_tag
    if len(candidates) > 1:
        logger.warning(
            'Multiple training folders match training_%s* in %s: %s — using most recent: %s',
            prefix, search_root,
            [c.name for c in candidates],
            candidates[0].name,
        )
    chosen = candidates[0].name  # e.g. 'training_vars5_pt_vis_137f51a0'
    resolved_tag = chosen.removeprefix('training_')
    logger.info('Resolved training tag: %s -> %s', exact_tag, resolved_tag)
    return resolved_tag

def resolve_latest_training_tag(base_dir: str, prefix: str = 'training_') -> str:
    search_root = Path(base_dir)
    if not search_root.exists():
        raise FileNotFoundError(f'Training base dir not found: {search_root}')
    candidates = sorted(search_root.glob(f'{prefix}*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f'No {prefix}* folders found in {search_root}')
    return candidates[0].name.removeprefix('training_')

def load_masks_config(path: str | Path = MASKS_CONFIG_PATH) -> dict[str, list[str]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f'Masks config not found: {config_path}')

    with open(config_path, 'r') as handle:
        raw = yaml.safe_load(handle) or {}

    masks = raw.get('masks', raw)

    if not isinstance(masks, dict):
        raise ValueError(f'Invalid masks config format in {config_path}: expected a mapping at root or under "masks"')

    normalized: dict[str, list[str]] = {}
    for name, expressions in masks.items():        
        if isinstance(expressions, str):
            normalized[name] = [expressions]
            continue
        if isinstance(expressions, list) and all(isinstance(expr, str) for expr in expressions):
            normalized[name] = expressions
            continue
        raise ValueError(f'Invalid expression list for mask "{name}" in {config_path}')
    
    logger.info('Loaded %d masks from %s', len(normalized), config_path)
    return normalized

def load_saved_model_config(checkpoint_dir: str | Path, fallback_path: str | Path) -> ModelConfig:
    saved_config_path = Path(checkpoint_dir).parent / 'config.yaml'
    if saved_config_path.exists():
        with open(saved_config_path, 'r') as handle:
            raw = yaml.unsafe_load(handle)

        if isinstance(raw, ModelConfig):
            return raw

        values = vars(raw) if hasattr(raw, '__dict__') else raw
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

def select_njets_category(df, category_name):
    if category_name == 'njets_0':
        return df[df.njets == 0].copy()
    if category_name == 'njets_1':
        return df[df.njets == 1].copy()
    if category_name == 'njets_geq_2':
        return df[df.njets >= 2].copy()
    if category_name == 'njets_inclusive':
        return df[df.njets >= 0].copy()
    raise ValueError(f'Unknown njets category: {category_name}')

# ----- masks -----
def load_masks_config(path: str | Path = MASKS_CONFIG_PATH) -> dict[str, list[str]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f'Masks config not found: {config_path}')

    with open(config_path, 'r') as handle:
        raw = yaml.safe_load(handle) or {}

    masks = raw.get('masks', raw)

    if not isinstance(masks, dict):
        raise ValueError(f'Invalid masks config format in {config_path}: expected a mapping at root or under "masks"')

    normalized: dict[str, list[str]] = {}
    for name, expressions in masks.items():        
        if isinstance(expressions, str):
            normalized[name] = [expressions]
            continue
        if isinstance(expressions, list) and all(isinstance(expr, str) for expr in expressions):
            normalized[name] = expressions
            continue
        raise ValueError(f'Invalid expression list for mask "{name}" in {config_path}')
    
    logger.info('Loaded %d masks from %s', len(normalized), config_path)
    return normalized

def _build_mask_from_config(df: pd.DataFrame, mask_name: str) -> pd.Series:
    global MASKS_CONFIG

    if not MASKS_CONFIG:
        MASKS_CONFIG = load_masks_config()

    expressions = MASKS_CONFIG.get(mask_name)

    if not expressions:
        raise KeyError(f'Mask "{mask_name}" not found in {MASKS_CONFIG_PATH}')

    combined_expression = ' & '.join(f'({expr})' for expr in expressions)
    mask = df.eval(combined_expression, engine='python')
    result = mask.fillna(False).astype(bool)
    return result

def _apply_config_mask(df: pd.DataFrame, mask_name: str) -> pd.DataFrame:
    return df[_build_mask_from_config(df, mask_name)].copy()

def mask_preselection_tight(df):
    return _apply_config_mask(df, 'mask_preselection_tight')

def mask_preselection_tight_binary_classifier(df):
    return _apply_config_mask(df, 'mask_preselection_tight_binary_classifier')

def mask_preselection_for_estimator(df):
    return mask_preselection_tight(df)

def mask_DR(df):
    return _apply_config_mask(df, 'mask_DR')

def AR_like_tau1(df):
    return _apply_config_mask(df, 'AR_like_tau1')

def AR_like_tau2(df):
    return _apply_config_mask(df, 'AR_like_tau2')

def SR_like(df):
    return _apply_config_mask(df, 'SR_like')

def AR_tau1(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    return _apply_config_mask(df, 'AR_tau1')

def AR_tau2(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    return _apply_config_mask(df, 'AR_tau2')

def SR(df):                 # without SS/OS conditions !!!!!!!!!!!!11
    return _apply_config_mask(df, 'SR')


def normalizing_flow_ff(
    df1,
    df2,
    variables,
    model_AR_like_tau1,
    model_SR_like_tau1,
    model_AR_like_tau2,
    model_SR_like_tau2,
    global_ff_tau1,
    global_ff_tau2,
    device,
    include_njets=True,
    grouped=True,
):
    """Evaluate one process group, preserving event indices for Feather export.

    Use the reference clipping/acceptance correction. Rejected events receive
    zero prediction weight and an explicit keep flag for FF distributions.
    The stored weights already contain the tau-combination factor of 0.5.
    """
    input_variables = (['njets'] + variables) if include_njets else list(variables)
    suffix = '_njets' if grouped else ''
    results = []
    for tau, frame, model_ar, model_sr, global_ff in (
        (1, df1, model_AR_like_tau1, model_SR_like_tau1, global_ff_tau1),
        (2, df2, model_AR_like_tau2, model_SR_like_tau2, global_ff_tau2),
    ):
        frame = frame.copy()
        ff_column = f'ff_nf_tau{tau}{suffix}'
        full_column = f'ff_unclipped_nf_tau{tau}{suffix}'

        if frame.empty:
            frame[full_column] = pd.Series(index=frame.index, dtype=float)
            frame[ff_column] = pd.Series(index=frame.index, dtype=float)
            frame[f'{ff_column}_kept'] = pd.Series(index=frame.index, dtype=bool)
            results.append(frame)
            continue

        inputs = get_my_data_qcd(frame, input_variables).to_torch().to(device)
        # Promote before the reference division: 1e-100 underflows in float32.
        pdf_ar = np.asarray(evaluate_pdf(model_ar, inputs.X), dtype=float)
        pdf_sr = np.asarray(evaluate_pdf(model_sr, inputs.X), dtype=float)

        if not (np.isfinite(pdf_ar).all() and np.isfinite(pdf_sr).all()):
            raise ValueError(f'Non-finite NF densities for tau {tau}')
        
        full, _, _, keep, _ = compute_eventwise_fake_factors(pdf_ar, pdf_sr, global_ff)

        frame[full_column] = 0.5 * full
        frame[ff_column] = 0.5 * np.where(keep, full, 0.0)
        frame[f'{ff_column}_kept'] = keep
        logger.info('Tau %d: retained %d/%d events after FF clipping', tau, keep.sum(), len(keep))
        results.append(frame)
    return tuple(results)


def nf_output_columns(grouped):
    suffix = '_njets' if grouped else ''
    return [
        column
        for tau in (1, 2)
        for column in (f'ff_nf_tau{tau}{suffix}',
                       f'ff_unclipped_nf_tau{tau}{suffix}',
                       f'ff_nf_tau{tau}{suffix}_kept')
    ]


def nf_process_groups(process_map):
    """Match the background groups used by plot_results_njets.py."""
    groups = {'data': [process_map['data']]}
    for name in ('diboson', 'DYjets', 'ST', 'ttbar'):
        groups[name] = [process_map[f'{name}_{kind}']
                        for kind in ('J', 'L', 'T') if f'{name}_{kind}' in process_map]
    if 'embedding' in process_map:
        groups['embedding'] = [process_map['embedding']]
    groups['wjets'] = [process_map['Wjets']]
    return groups


def main():
    # Step 1: parse runtime arguments and core variable list
    with open(cfg_path['variables'], 'r') as f:
        variables = yaml.safe_load(f)[args.var]

    cfg_set = load_config(cfg_path['config_settings'])

    dim = len(variables)
    training_variables_tag = build_training_variables_tag(variables)
    variables_with_njets = ['njets'] + variables
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    MASKS_CONFIG = load_masks_config(MASKS_CONFIG_PATH)

    

    # Step 2: resolve model mode and load model checkpoints
    _MODE_DIR = {
        'grouped_njets_split': 'split_njets_0_1_ge2',
        'single_nf':           'no_njets_split',
        'conditional_nf':      'conditional_njets_input',
    }
    mode_dir = _MODE_DIR[args.model_mode]
    include_njets_feature = args.model_mode in ('grouped_njets_split', 'conditional_nf')    

    resolved_tag = args.training_tag or resolve_training_tag(variables, mode_dir)
    logger.info('Using training tag: %s (exact computed: %s)', resolved_tag, training_variables_tag)

    chk_pth_model_AR_like_tau1 = f'{cfg_path["NF_results"]}/{mode_dir}/training_DNNplotting/tau1/all/AR-like/latest'
    chk_pth_model_SR_like_tau1 = f'{cfg_path["NF_results"]}/{mode_dir}/training_DNNplotting/tau1/all/SR-like/latest'
    chk_pth_model_AR_like_tau2 = f'{cfg_path["NF_results"]}/{mode_dir}/training_DNNplotting/tau2/all/AR-like/latest'
    chk_pth_model_SR_like_tau2 = f'{cfg_path["NF_results"]}/{mode_dir}/training_DNNplotting/tau2/all/SR-like/latest'

    config_AR_like_tau1 = load_saved_model_config(chk_pth_model_AR_like_tau1, config_path)
    config_SR_like_tau1 = load_saved_model_config(chk_pth_model_SR_like_tau1, config_path)
    config_AR_like_tau2 = load_saved_model_config(chk_pth_model_AR_like_tau2, config_path)
    config_SR_like_tau2 = load_saved_model_config(chk_pth_model_SR_like_tau2, config_path)
    
    logger.info('Loading models from RNVP checkpoints:')
    if args.model_mode == 'grouped_njets_split':
        model_AR_like_tau1 = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_AR_like_tau1,
            config_path=config_path,
            variables=variables,
            device=device,
        )
        model_SR_like_tau1 = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_SR_like_tau1,
            config_path=config_path,
            variables=variables,
            device=device,
        )
        model_AR_like_tau2 = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_AR_like_tau2,
            config_path=config_path,
            variables=variables,
            device=device,
        )
        model_SR_like_tau2 = load_grouped_qcd_njets_router(
            checkpoint_dir=chk_pth_model_SR_like_tau2,
            config_path=config_path,
            variables=variables,
            device=device,
        )
    elif args.model_mode == 'conditional_nf':
        model_AR_like_tau1 = load_conditional_flow(dim=dim, cfg=config_AR_like_tau1, checkpoint_path=f'{chk_pth_model_AR_like_tau1}/model_checkpoint.pth', device=device)
        model_SR_like_tau1 = load_conditional_flow(dim=dim, cfg=config_SR_like_tau1, checkpoint_path=f'{chk_pth_model_SR_like_tau1}/model_checkpoint.pth', device=device)
        model_AR_like_tau2 = load_conditional_flow(dim=dim, cfg=config_AR_like_tau2, checkpoint_path=f'{chk_pth_model_AR_like_tau2}/model_checkpoint.pth', device=device)
        model_SR_like_tau2 = load_conditional_flow(dim=dim, cfg=config_SR_like_tau2, checkpoint_path=f'{chk_pth_model_SR_like_tau2}/model_checkpoint.pth', device=device)
    elif args.model_mode == 'single_nf':
        model_AR_like_tau1 = load_flow(dim=dim, cfg=config_AR_like_tau1, checkpoint_path=f'{chk_pth_model_AR_like_tau1}/model_checkpoint.pth', device=device)
        model_SR_like_tau1 = load_flow(dim=dim, cfg=config_SR_like_tau1, checkpoint_path=f'{chk_pth_model_SR_like_tau1}/model_checkpoint.pth', device=device)
        model_AR_like_tau2 = load_flow(dim=dim, cfg=config_AR_like_tau2, checkpoint_path=f'{chk_pth_model_AR_like_tau2}/model_checkpoint.pth', device=device)
        model_SR_like_tau2 = load_flow(dim=dim, cfg=config_SR_like_tau2, checkpoint_path=f'{chk_pth_model_SR_like_tau2}/model_checkpoint.pth', device=device)
    else:
        raise ValueError(f'Unknown model mode: {args.model_mode}. Supported: "grouped_njets_split", "conditional_nf", "single_nf".')

    
    # Step 3: load data and plotting labels/binning
    data_complete = pd.read_feather(f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather')

    # Single NFs export inclusive FFs; conditional NFs export category-normalized
    # FFs. Grouped NFs retain both exports.
    categories = {
        'single_nf': [('njets_inclusive', False)],
        'conditional_nf': [('njets_0', True), ('njets_1', True),
                           ('njets_geq_2', True)],
        'grouped_njets_split': [('njets_inclusive', False), ('njets_0', True),
                                ('njets_1', True), ('njets_geq_2', True)],
    }[args.model_mode]
    process_groups = nf_process_groups(load_config(cfg_path['process_map'][args.embedding]))
    
    for grouped in dict.fromkeys(grouped for _, grouped in categories):
        for column in nf_output_columns(grouped):
            # Reset only this export's columns so previous runs cannot fill gaps.
            data_complete[column] = False if column.endswith('_kept') else np.nan

    for category_name, grouped in categories:
        selected = mask_preselection_for_estimator(select_njets_category(data_complete, category_name))
        ar1 = AR_tau1(selected)
        ar1 = ar1[ar1.OS == True]
        ar2 = AR_tau2(selected)
        ar2 = ar2[ar2.OS == True]
        if ar1.empty and ar2.empty:
            continue

        ar_like1 = AR_like_tau1(selected)
        ar_like2 = AR_like_tau2(selected)
        sr_like = SR_like(selected)
        data_process = process_groups['data'][0]
        samples = [frame[(frame.process == data_process) & (frame.SS == True)]
                   for frame in (ar_like1, ar_like2, sr_like)]
        if any(frame.empty for frame in samples):
            raise ValueError(f'{category_name}: empty same-sign normalization sample; export aborted')
        global_ff_tau1 = len(samples[2]) / len(samples[0])
        global_ff_tau2 = len(samples[2]) / len(samples[1])
        logger.info('%s: global FF tau1=%.6g, tau2=%.6g',
                    category_name, global_ff_tau1, global_ff_tau2)

        known_processes = [value for values in process_groups.values() for value in values]
        for frame in (ar1, ar2):
            if not frame.process.isin(known_processes).all():
                raise ValueError(f'{category_name}: AR contains unmapped processes')

        # The acceptance correction depends on the sample. Do not pool data,
        # embedding and MC, or split a background's J/L/T components here.
        for process, ids in process_groups.items():
            logger.info('%s: evaluating %s', category_name, process)
            frames = normalizing_flow_ff(
                ar1[ar1.process.isin(ids)], ar2[ar2.process.isin(ids)], variables,
                model_AR_like_tau1, model_SR_like_tau1,
                model_AR_like_tau2, model_SR_like_tau2,
                global_ff_tau1, global_ff_tau2, device,
                include_njets=include_njets_feature, grouped=grouped,
            )
            for tau, frame in enumerate(frames, start=1):
                # Copy only the newly evaluated tau's columns, never old FFs
                # carried in the source dataframe.
                columns = [column for column in nf_output_columns(grouped)
                           if f'_tau{tau}' in column]
                if not frame.empty:
                    data_complete.loc[frame.index, columns] = frame[columns]

    output_path = Path(cfg_path['datasets']) / args.embedding / 'combined_data_updated.feather'
    logger.info('Saving main dataframe to %s', output_path)
    data_complete.to_feather(output_path)

if __name__ == '__main__':
    main()
