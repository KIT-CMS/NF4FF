import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from classes.path_managment import StorePathHelper
import logging
from CustomLogging import setup_logging, LogContext
import yaml
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import torch as t
import random
import math
from classes.NeuralNetworks import RealNVP_NN, GroupedNFRouter, RealNVP, AffineCoupling, MLP
from classes.Dataclasses import _component_collection, RealNVP_config, _same_sign_opposite_sign_split
from classes.Collection import get_my_data_qcd

from typing import (Any, Callable, Dict, Iterable, List, Protocol, Tuple,
                    Union, runtime_checkable)
import time


SEED = 42

t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

t.set_num_threads(8)

with open('../configs/training_variables.yaml', 'r') as f:
    variables = yaml.safe_load(f)['variables']


dim = len(variables)

# ------ logger -----

logger = setup_logging(logger=logging.getLogger(__name__))
log = LogContext(logger)
# ----- constants -----

N_EPOCHS_MAX = 50
PATIENCE = 30
N_SAMPLES = 1000000


# ------ variables definition -------

with open('../configs/training_variables.yaml', 'r') as f:
    variables = yaml.safe_load(f)['variables']

dim = len(variables)


# ------ model ------


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        last = in_dim

        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h

        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class AffineCoupling(nn.Module):
    def __init__(self, dim, mask, hidden_dims=(128, 128), s_scale=2.0):
        super().__init__()

        self.dim = dim
        self.register_buffer("mask", mask)

        self.st_net = MLP(in_dim=dim, out_dim=2 * dim, hidden_dims=hidden_dims)
        self.s_scale = s_scale

    def forward(self, x):
        x_masked = x * self.mask

        s, shift = t.chunk(self.st_net(x_masked), 2, dim=-1)
        s = t.tanh(s) * self.s_scale

        y = x_masked + (1 - self.mask) * (x * t.exp(s) + shift)
        log_det = ((1 - self.mask) * s).sum(dim=-1)

        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask

        s, t = t.chunk(self.st_net(y_masked), 2, dim=-1)
        s = t.tanh(s) * self.s_scale

        x = y_masked + (1 - self.mask) * ((y - t) * t.exp(-s))
        return x

class RealNVP(nn.Module):
    """
    Stack of affine coupling layers with alternating masks.
    Base distribution: standard Normal.
    """
    def __init__(self, dim, n_layers=6, hidden_dims=(128, 128), s_scale=2.0, device= t.device("cuda" if t.cuda.is_available() else "cpu")):
        super().__init__()
        t.device(device)
        self.dim = dim
        logger.info(f"Dimension of RealNVP input: {self.dim}")
        base_mask = t.tensor([i % 2 for i in range(dim)], dtype=t.float32)

        masks = []
        for i in range(n_layers):
            if i % 2 == 0:
                masks.append(base_mask)
            else:
                masks.append(1 - base_mask)

        # create list of layers, 
        self.couplings = nn.ModuleList([
            AffineCoupling(dim=dim, mask=m, hidden_dims=hidden_dims, s_scale=s_scale)
            for m in masks
        ])

        # Learnable base distribution parameters (optional; here fixed to standard normal)
        self.register_buffer('base_mean', t.zeros(dim))
        self.register_buffer('base_log_std', t.zeros(dim))
                # StandardScaler or RobustScaler
        self.register_buffer("_scaler_shift", t.full((dim,), 0.0))
        self.register_buffer("_scaler_scale", t.full((dim,), 1.0))

    def _init_permute(self, m):
        # For invertible mixing layers, orthogonal init is robust
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
        elif isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_fn(self, module):
        # Generic global init (use cautiously; don't override the s,t zero head)
        if isinstance(module, nn.Linear) and module not in [self.permute]:
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    def f(self, x):
        """Forward through all couplings: x -> z. Returns z and sum log-dets."""
        log_det_total = t.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.couplings:
            z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def f_inv(self, z):
        """Inverse through all couplings: z -> x."""
        x = z
        # inverse in reverse order
        for layer in reversed(self.couplings):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        """Compute log p(x) via change of variables."""
        z, log_det = self.f(x)
        # base log prob (diagonal Normal)
        std = t.exp(self.base_log_std)
        log_pz = (-0.5 * (((z - self.base_mean) / std) ** 2).sum(dim=-1)
                  - 0.5 * self.dim * math.log(2 * math.pi)
                  - self.base_log_std.sum())
        return log_pz + log_det

    def sample(self, n):
        """Sample x by drawing z from base and mapping through inverse."""
        std = t.exp(self.base_log_std)
        z = self.base_mean + std * t.randn(n, self.dim, device=self.base_mean.device)
        x_scaled = self.f_inv(z)
        # map back to original (un-scaled) space: x = x_scaled * scale + shift
        x = x_scaled * self._scaler_scale.to(x_scaled.device) + self._scaler_shift.to(x_scaled.device)
        return x
    

    @property
    def _is_initialized(self):
        initialized = (t.isnan(self._scaler_shift) | t.isnan(self._scaler_scale)).sum() == 0
        initialized &= (self._scaler_scale != 1).all() & (self._scaler_shift != 0).all()
        return initialized

    def initialize_scaler(
        self,
        shift: Union[np.ndarray, t.Tensor, None] = None,
        scale: Union[np.ndarray, t.Tensor, None] = None,
    ):
        # Both shift and scale must be provided (no partial initialization)
        if shift is None or scale is None:
            msg = (
                "shift and scale must be both provided (not None)."
                " To reset to defaults explicitly set shift=0 and scale=1."
            )
            logger.error(msg)
            raise ValueError(msg)

        # convert numpy arrays to tensors if needed
        shift = t.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
        scale = t.from_numpy(scale) if isinstance(scale, np.ndarray) else scale

        if self._is_initialized:
            logger.warning("Scaler already initialized. Overwriting the current values.")

        self._scaler_shift.data[:] = shift
        self._scaler_scale.data[:] = scale

    def apply_scaler(self, x):
        return (x - self._scaler_shift.to(x.device)) / self._scaler_scale.to(x.device)



    def forward(self, X):
        Xs = self.apply_scaler(X)
        # log prob of the scaled input
        logp_scaled = self.log_prob(Xs)
        # account for the scaling Jacobian: Xs = (X - shift)/scale -> det = prod(1/scale)
        # so log|det| = -sum(log(scale)). Add this constant per-sample.
        log_det_scale = -t.log(self._scaler_scale.to(X.device)).sum()
        return logp_scaled + log_det_scale


# ------ functions and masks --------

def mask_DR(df):

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.q_1 * df.q_2 > 0)
    mask_a4 = ((df.iso_1 > 0.02) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 < 50)
    mask_DR = (mask_a1 & mask_a2 & mask_a4 & mask_a5 & mask_a6)

    return df[mask_DR].copy()

def mask_preselection_loose(df):
    mask_eta = (df.eta_1 <= 2.1) & (df.eta_2 <= 2.3)
    mask_pt = (df.pt_1 >= 33) & (df.pt_2 >= 30)
    #mask_m_vis = (df.m_vis >= 35)
    mask_tau_decay_mode = (df.tau_decaymode_2 == 0) | (df.tau_decaymode_2 == 1) | (df.tau_decaymode_2 == 10) | (df.tau_decaymode_2 == 11)
    return df[mask_eta & mask_pt & mask_tau_decay_mode]

def SR_like(df):
    mask_s1 = (df.id_tau_vsJet_Tight_2 > 0.5)
    return(df[mask_s1])

def AR_like(df):
    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5) & (df.id_tau_vsJet_Tight_2 < 0.5))

    return(df[mask_a1])


# ----- main -----

def main():

    t.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    config_path = '../configs/config_NF.yaml'

    config = RealNVP_config.from_yaml(config_path)

    # --- load device ---

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- load data ---

    data_complete = pd.read_feather('../../data/data_complete.feather')

    data_DR = mask_DR(data_complete)

    data_DR = data_DR[(data_DR.process == 0) & (data_DR.SS == True)].reset_index(drop=True)

    train1, val1 = train_test_split(data_DR)

    train1_AR_like = mask_preselection_loose(AR_like(train1))
    val1_AR_like = mask_preselection_loose(AR_like(val1))
    train1_SR_like = mask_preselection_loose(SR_like(train1))
    val1_SR_like = mask_preselection_loose(SR_like(val1))    

    weight_corr_factor = (
        pd.concat([train1_AR_like['weight_qcd'], val1_AR_like['weight_qcd']]).sum()
        /
        pd.concat([train1_SR_like['weight_qcd'], val1_SR_like['weight_qcd']]).sum()
    )

    for region, val1, train1 in zip(['AR-like', 'SR-like'], [val1_AR_like, val1_SR_like], [train1_AR_like, train1_SR_like]):
        variables_with_njets = ['njets'] + variables
        train1 = get_my_data_qcd(train1, variables_with_njets).to_torch(device=None)
        val1 = get_my_data_qcd(val1, variables_with_njets).to_torch(device=None)


        X_train = train1.X
        X_val = val1.X
        weights_train = train1.weights
        weights_val = val1.weights  

        weights_train = weights_train / t.sum(weights_train)
        weights_val   = weights_val / t.sum(weights_val)


        # ----------- Normalize weights -----------

        if region == "SR-like":
            weights_train = weights_train * weight_corr_factor
            weights_val = weights_val * weight_corr_factor
        elif region == "AR-like":
            pass 

        # -------------- DataLoaders ------------

        train_loader = DataLoader(
            TensorDataset(X_train, weights_train),
            batch_size=config.bsize_train,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )

        val_loader = DataLoader(
            TensorDataset(X_val, weights_val),
            batch_size=config.bsize_val,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )


        # --- model setup ---

        dim  = len(variables)

        shift = X_train[:, 1:].mean(dim=0)
        scale  = X_train[:, 1:].std(dim=0, unbiased=False).clamp_min(1e-12) 
        
        model_0 = RealNVP(
            dim = dim,
            n_layers = config.n_layers,
            hidden_dims=(config.hidden_dims,),
            s_scale = config.s_scale,
        ).to(device)
        
        model_1 = RealNVP(
            dim = dim,
            n_layers = config.n_layers,
            hidden_dims=(config.hidden_dims,),
            s_scale = config.s_scale,
        ).to(device)
        
        model_2 = RealNVP(
            dim = dim,
            n_layers = config.n_layers,
            hidden_dims=(config.hidden_dims,),
            s_scale = config.s_scale,
        ).to(device)
        
        '''
        model_0 = RealNVP_NN(
            input_nodes=dim, 
            hidden_nodes = (config.hidden_dims,),
            n_layers=config.n_layers, 
            dropout = 0.0,
            activation = 'ReLU',
            s_scale = config.s_scale,
            ).to(device)
        model_1 = RealNVP_NN(
            input_nodes=dim, 
            hidden_nodes = (config.hidden_dims,),
            n_layers=config.n_layers, 
            dropout = 0.0,
            activation = 'ReLU',
            s_scale = config.s_scale,
            ).to(device)
        model_2 = RealNVP_NN(
            input_nodes=dim, 
            hidden_nodes = (config.hidden_dims,),
            n_layers=config.n_layers, 
            dropout = 0.0,
            activation = 'ReLU',
            s_scale = config.s_scale,
            ).to(device)
        '''
        model_0.initialize_scaler(shift, scale)
        model_1.initialize_scaler(shift, scale)
        model_2.initialize_scaler(shift, scale)

        router = GroupedNFRouter()

        router._fallback_payload = model_2
        router._wrapped_delegate = model_2
        
        # Register all models
        router.models.append(model_0)
        router.models.append(model_1)
        router.models.append(model_2)
        
        router._logic_pipeline = [
            ([(0, (0,))], model_0),
            ([(0, (1,))], model_1),
            ([(0, (2, 11000))], model_2),
        ]
       
       
        #optimizer = t.optim.NAdam(model.parameters(), lr=config.lr)
        optimizer= t.optim.AdamW(router.parameters(), lr=config.lr)
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            threshold=config.scheduler_threshold,
            threshold_mode='rel',
            cooldown=config.scheduler_cooldown,
            min_lr=config.scheduler_min_lr,
            eps=config.scheduler_eps
        )

        # AMP
        scaler = t.amp.GradScaler('cuda',enabled=config.use_amp)

        # -------------------------------------
        #  Training bookkeeping
        # -------------------------------------
        NLL_training = []
        NLL_validation = []
        best_val_nll = float('inf')
        counter = 0
        log_rows = []

        # -------------------------------------
        #  Training Loop
        # -------------------------------------
        with log.training_dashboard() as dash:
            for epoch in range(1, config.n_epochs + 1):
                epoch_start = time.time()



                # --------------------
                #  TRAIN
                # --------------------
                router.train()
                train_loss_sum = 0.0
                train_weight_sum = 0.0

                for Xb, Wb in train_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    Wb = Wb.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    #if (Z is None) or (step % 20 == 0):
                    #    Z = estimate_Z(model, T1, T2, nsamples=512)


                    with t.amp.autocast('cuda', enabled=False):


                        log_px = router(Xb)      # [N,1]
                        log_px = log_px.squeeze(1)   # [N]
                        loss = (-(log_px) * Wb).sum()


                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(router.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_sum += loss.item()
                    train_weight_sum += Wb.sum().item()

                avg_train_nll = train_loss_sum / train_weight_sum
                NLL_training.append(avg_train_nll)

                # --------------------

                router.eval()
                val_loss_sum = 0.0
                val_weight_sum = 0.0

                with t.no_grad():
                    for Xb, Wb in val_loader:
                        Xb = Xb.to(device, non_blocking=True)
                        Wb = Wb.to(device, non_blocking=True)

                        with t.amp.autocast('cuda', enabled=False):
                            #log_p_obs = truncated_log_prob(model, Xb, T1, T2, Z)
                            log_px = router(Xb)      # [N,1]
                            log_px = log_px.squeeze(1)   # [N]
                            vloss = (-(log_px) * Wb).sum()

                        val_loss_sum += vloss.item()
                        val_weight_sum += Wb.sum().item()

                avg_val_nll = val_loss_sum / val_weight_sum
                NLL_validation.append(avg_val_nll)

                scheduler.step(avg_val_nll)
                epoch_time = time.time() - epoch_start

                log_rows.append({
                "epoch": epoch,
                "train_loss": avg_train_nll,
                "val_loss": avg_val_nll,
                "lr": scheduler.get_last_lr(),
                "time_s": epoch_time,
                "type": "epoch",
                })


                # --------------------
                #  Early stopping
                # --------------------
                if avg_val_nll < best_val_nll:
                    best_val_nll = avg_val_nll
                    counter = 0

                    checkpoint = {
                        "router_state_dict": router.state_dict(),       # <- preferred key
                        "optimizer_state_dict": optimizer.state_dict(),
                        # store a plain dict, not the config object itself
                        #"config": config.to_dict() if hasattr(config, "to_dict") else dict(config),
                        "variables": list(variables),                   # <- features only (NO 'njets')
                        "schema": "grouped_nf_router_v1"
                    }

                else:
                    counter += 1

                dash.update(epoch = epoch, train_loss=np.round(avg_train_nll, 6), val_loss=np.round(avg_val_nll, 6), lr = scheduler.get_last_lr(), region = region)

                if counter >= PATIENCE:
                    logger.info("Early stopping triggered.")
                    break



        # -------------------------------------
        #  Save training artifacts
        # -------------------------------------
        paths_training = StorePathHelper(directory=f"Training_results_new/QCD/all/{region}")

        t.save(checkpoint, paths_training.autopath.joinpath("model_checkpoint.pth"))
        t.save(checkpoint, f"Training_results_new/QCD/all/{region}/latest/model_checkpoint.pth")

        pd.DataFrame(log_rows).to_pickle(str(paths_training.autopath.joinpath('training_logs.pkl')))
        pd.DataFrame(log_rows).to_pickle(str(f"Training_results_new/QCD/all/{region}/latest/training_logs.pkl"))


        with open(paths_training.autopath.joinpath("config.yaml"), "w") as f:
            yaml.dump(config, f)

        logger.info("Model saved successfully")

# --------------

if __name__ == "__main__":
    main()