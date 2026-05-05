import logging
from pathlib import Path
import sys

# ensure the src/ directory is on the path so that 'classes' and 'CustomLogging' are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
FF_FACTORS_DIR = SCRIPT_DIR / 'Fake_Factors'

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.distributions as D
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from classes.NeuralNetworks import ConditionalFlow1D

logger = logging.getLogger(__name__)

PATIENCE = 20
OUTPUT_ROOT = SCRIPT_DIR / 'FF_flow_results'


def resolve_training_name(variables: list[str]) -> str:
    tail = variables[4:]
    tag = '_'.join(tail) if tail else 'none'
    return f'training_vars{len(variables)}_{tag}'

def train_flow(
    flow,
    optimizer,
    FF_SR: t.Tensor,
    cond: t.Tensor,
    n_epochs: int = 200,
    batch_size: int = 1024,
    val_fraction: float = 0.25,
    patience: int = PATIENCE,
    device: t.device = t.device('cpu'),
    scheduler_factor: float = 0.2,
    scheduler_patience: int = 10,
    scheduler_threshold: float = 1e-4,
    scheduler_cooldown: int = 2,
    scheduler_min_lr: float = 1e-6,
):
    indices = t.arange(len(FF_SR))
    idx_train, idx_val = train_test_split(indices.numpy(), test_size=val_fraction, shuffle=True)
    idx_train = t.tensor(idx_train)
    idx_val = t.tensor(idx_val)

    train_loader = DataLoader(
        TensorDataset(FF_SR[idx_train], cond[idx_train]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(FF_SR[idx_val], cond[idx_val]),
        batch_size=batch_size,
        shuffle=False,
    )

    best_val_loss = float('inf')
    counter = 0
    best_checkpoint = None
    log_rows = []

    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        threshold_mode='rel',
        cooldown=scheduler_cooldown,
        min_lr=scheduler_min_lr,
    )

    for epoch in range(1, n_epochs + 1):
        flow.train()
        train_loss = 0.0
        for ff_batch, cond_batch in train_loader:
            ff_batch = ff_batch.to(device)
            cond_batch = cond_batch.to(device)
            optimizer.zero_grad()
            logp = flow.log_prob(ff_batch.squeeze(), cond_batch)
            loss = -logp.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(ff_batch)
        train_loss /= len(idx_train)

        flow.eval()
        val_loss = 0.0
        with t.no_grad():
            for ff_batch, cond_batch in val_loader:
                ff_batch = ff_batch.to(device)
                cond_batch = cond_batch.to(device)
                logp = flow.log_prob(ff_batch.squeeze(), cond_batch)
                val_loss += (-logp.mean().item()) * len(ff_batch)
        val_loss /= len(idx_val)

        log_rows.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': scheduler.get_last_lr()[0]})

        if epoch % 1 == 0:
            logger.info('epoch %03d | train NLL = %.4f | val NLL = %.4f | lr = %.2e', epoch, train_loss, val_loss, scheduler.get_last_lr()[0])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_checkpoint = {
                'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
            }
        else:
            counter += 1
            if counter >= patience:
                logger.info('Early stopping at epoch %d (best val NLL = %.4f)', epoch, best_val_loss)
                break

    if best_checkpoint is None:
        raise RuntimeError('No checkpoint was saved during training.')

    return best_checkpoint, pd.DataFrame(log_rows)

def save_flow(
    checkpoint: dict,
    log_df: pd.DataFrame,
    output_dir: Path,
):
    output_dir = Path(output_dir)
    latest_dir = output_dir / 'latest'
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    t.save(checkpoint, output_dir / 'model_checkpoint.pth')
    t.save(checkpoint, latest_dir / 'model_checkpoint.pth')

    log_df.to_pickle(str(output_dir / 'training_logs.pkl'))
    log_df.to_pickle(str(latest_dir / 'training_logs.pkl'))

    logger.info(
        'Saved best checkpoint (epoch %d, val NLL = %.4f) to %s',
        checkpoint['epoch'],
        checkpoint['val_loss'],
        output_dir,
    )

# ------- main -------

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', device)

    with open(CONFIG_DIR / 'training_variables.yaml', 'r') as fh:
        training_variables_cfg = yaml.safe_load(fh)
    nf_variables = training_variables_cfg['variables_MC']
    variables = training_variables_cfg['variables_correction']

    resolved_tag = resolve_training_name(nf_variables)
    correction_tag = resolve_training_name(variables)
    ff_path = FF_FACTORS_DIR / f'fake_factors_{resolved_tag}.feather'
    if not ff_path.exists():
        raise FileNotFoundError(
            f'Fake factors file not found: {ff_path}\nRun FF_calculation.py first.'
        )
    df_AR = pd.read_feather(ff_path)
    logger.info('Loaded %d AR events from %s', len(df_AR), ff_path)

    cols_to_check = ['FF_SR', 'FF_DR'] + variables
    valid_mask = np.ones(len(df_AR), dtype=bool)
    for col in cols_to_check:
        valid_mask &= np.isfinite(df_AR[col].to_numpy(dtype='float32'))
    # also drop events with extreme FF values that cause flow instability
    ff_sr_np = df_AR['FF_SR'].to_numpy(dtype='float32')
    ff_dr_np = df_AR['FF_DR'].to_numpy(dtype='float32')
    ff_clip_max = 20.0
    valid_mask &= (ff_sr_np > 0) & (ff_sr_np <= ff_clip_max)
    valid_mask &= (ff_dr_np > 0) & (ff_dr_np <= ff_clip_max)
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.warning('Dropping %d events with non-finite or extreme FF values', n_dropped)
    df_AR = df_AR[valid_mask].reset_index(drop=True)

    # --- training dataset: log(FF_SR) — log-transform for stability with heavy-tailed distribution
    FF_SR_log = t.log(t.tensor(df_AR['FF_SR'].to_numpy(dtype='float32'))).unsqueeze(1)

    # --- conditional dataset: log(FF_DR) + variables
    cond_np = np.column_stack([
        np.log(df_AR['FF_DR'].to_numpy(dtype='float32')),
        df_AR[variables].to_numpy(dtype='float32'),
    ])
    cond_data = t.tensor(cond_np)

    cond_dim = 1 + len(variables)

    shift_training = FF_SR_log.mean(dim=0)
    scale_training = FF_SR_log.std(dim=0, unbiased=False).clamp_min(1e-12)
    shift_cond = cond_data.mean(dim=0)
    scale_cond = cond_data.std(dim=0, unbiased=False).clamp_min(1e-12)

    model = ConditionalFlow1D(cond_dim).to(device)
    model.initialize_scaler(shift_training, scale_training)
    model.initialize_cond_scaler(shift_cond, scale_cond)

    # save the scaler stats alongside the checkpoint for reproducibility
    output_dir = OUTPUT_ROOT / resolved_tag / correction_tag / 'FF_SR'
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_meta = {
        'shift_training': shift_training.numpy().tolist(),
        'scale_training': scale_training.numpy().tolist(),
        'shift_cond': shift_cond.numpy().tolist(),
        'scale_cond': scale_cond.numpy().tolist(),
        'variables': variables,
        'cond_dim': cond_dim,
        'variables_mc_tag': resolved_tag,
        'variables_correction_tag': correction_tag,
        'use_log_transform': True,
        'ff_clip_max': ff_clip_max,
    }
    with open(output_dir / 'scaler_meta.yaml', 'w') as fh:
        yaml.dump(scaler_meta, fh)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    checkpoint, log_df = train_flow(
        flow=model,
        optimizer=optimizer,
        FF_SR=FF_SR_log,
        cond=cond_data,
        device=device,
    )

    save_flow(
        checkpoint=checkpoint,
        log_df=log_df,
        output_dir=output_dir,
    )


# -------

if __name__ == '__main__':
    main()
