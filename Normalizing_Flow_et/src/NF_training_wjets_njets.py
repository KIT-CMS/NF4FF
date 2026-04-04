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
import numpy as np
import pandas as pd
import torch as t
import logging
import random
from rich.live import Live
from rich.table import Table
from classes.NeuralNetworks import RealNVP, GroupedNFRouter, AffineCoupling, MLP
from classes.Dataclasses import _component_collection, RealNVP_config, _same_sign_opposite_sign_split
from classes.Collection import get_my_data_wjets

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

# ------ functions and masks --------


class TrainingDashboard:

    def __init__(self, console):
        self.console = console
        self.live = Live(self.render(), console=console, refresh_per_second=5)

        self.epoch = 0
        self.train_loss = 0
        self.val_loss = 0
        self.lr = 0
        self.region = ""

    def render(self):
        table = Table(title="Training")

        table.add_column("Epoch")
        table.add_column("Train Loss")
        table.add_column("Val Loss")
        table.add_column("LR")
        table.add_column("Region")

        table.add_row(
            str(self.epoch),
            f"{self.train_loss:.4f}",
            f"{self.val_loss:.4f}",
            f"{self.lr:.2e}",
            self.region
        )

        return table

    def update(self, epoch, train_loss, val_loss, lr, region):
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.lr = lr
        self.region = region
        self.live.update(self.render())

def evaluate_loader(model, loader, device):
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0

    with t.no_grad():
        for Xb, Wb in loader:
            Xb = Xb.to(device, non_blocking=True)
            Wb = Wb.to(device, non_blocking=True)

            with t.amp.autocast('cuda', enabled=False):
                log_px = model(Xb).squeeze(1)
                loss = (-(log_px) * Wb).sum()

            loss_sum += loss.item()
            weight_sum += Wb.sum().item()

    return loss_sum / max(weight_sum, 1e-12)

def model_in_optimizer(model, optimizer):
    opt_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    return any(id(p) in opt_params for p in model.parameters())



# ----- masks -----

def mask_DR(df):

    mask_a1 = ((df.id_tau_vsJet_VLoose_2 > 0.5))
    mask_a2 = (df.nbtag == 0)
    mask_a4 = ((df.iso_1 > 0.0) & (df.iso_1 < 0.15))
    mask_a5 = ( (df.extramuon_veto < 0.5) & df.extraelec_veto < 0.5 )
    mask_a6 = (df.mt_1 > 70)
    mask_DR = (mask_a1 & mask_a2  & mask_a4 & mask_a5 & mask_a6)

    return df[mask_DR].copy()

def mask_preselection_loose(df):
    mask_eta = (df.eta_1 <= 2.1) & (df.eta_2 <= 2.3)
    mask_pt = (df.pt_1 >= 33) & (df.pt_2 >= 30)
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

    data_DR = data_DR[(data_DR.process == 0) & (data_DR.OS == True)].reset_index(drop=True)

    train1, val1 = train_test_split(data_DR)

    train1_AR_like = mask_preselection_loose(AR_like(train1))
    val1_AR_like = mask_preselection_loose(AR_like(val1))
    train1_SR_like = mask_preselection_loose(SR_like(train1))
    val1_SR_like = mask_preselection_loose(SR_like(val1))    

    weight_corr_factor = (
        pd.concat([train1_AR_like['weight_wjets'], val1_AR_like['weight_wjets']]).sum()
        /
        pd.concat([train1_SR_like['weight_wjets'], val1_SR_like['weight_wjets']]).sum()
    )

    for region, val1, train1 in zip(['AR-like', 'SR-like'], [val1_AR_like, val1_SR_like], [train1_AR_like, train1_SR_like]):
        variables_with_njets = ['njets'] + variables
        train1 = get_my_data_wjets(train1, variables_with_njets).to_torch(device=None)
        val1 = get_my_data_wjets(val1, variables_with_njets).to_torch(device=None)


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
            use_tail_preprocessing=config.use_tail_preprocessing,
            tail_preprocessing_index=config.tail_preprocessing_index,
            tail_preprocessing_type=config.tail_preprocessing_type,
            tail_preprocessing_center=config.tail_preprocessing_center,
            tail_preprocessing_scale=config.tail_preprocessing_scale,
            tail_preprocessing_epsilon=config.tail_preprocessing_epsilon,
        ).to(device)
        
        model_1 = RealNVP(
            dim = dim,
            n_layers = config.n_layers,
            hidden_dims=(config.hidden_dims,),
            s_scale = config.s_scale,
            use_tail_preprocessing=config.use_tail_preprocessing,
            tail_preprocessing_index=config.tail_preprocessing_index,
            tail_preprocessing_type=config.tail_preprocessing_type,
            tail_preprocessing_center=config.tail_preprocessing_center,
            tail_preprocessing_scale=config.tail_preprocessing_scale,
            tail_preprocessing_epsilon=config.tail_preprocessing_epsilon,
        ).to(device)
        
        model_2 = RealNVP(
            dim = dim,
            n_layers = config.n_layers,
            hidden_dims=(config.hidden_dims,),
            s_scale = config.s_scale,
            use_tail_preprocessing=config.use_tail_preprocessing,
            tail_preprocessing_index=config.tail_preprocessing_index,
            tail_preprocessing_type=config.tail_preprocessing_type,
            tail_preprocessing_center=config.tail_preprocessing_center,
            tail_preprocessing_scale=config.tail_preprocessing_scale,
            tail_preprocessing_epsilon=config.tail_preprocessing_epsilon,
        ).to(device)

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

        print("model_0 in optimizer:", model_in_optimizer(model_0, optimizer))
        print("model_1 in optimizer:", model_in_optimizer(model_1, optimizer))
        print("model_2 in optimizer:", model_in_optimizer(model_2, optimizer))

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

                    with t.amp.autocast('cuda', enabled=False):
                        log_px = router(Xb).squeeze(1)
                        loss = (-(log_px) * Wb).sum()

                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(router.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_sum += loss.item()
                    train_weight_sum += Wb.sum().item()

                avg_train_opt_nll = train_loss_sum / max(train_weight_sum, 1e-12)

                # --------------------
                #  EVAL: comparable train/val losses
                # --------------------
                avg_train_nll = evaluate_loader(router, train_loader, device)
                avg_val_nll = evaluate_loader(router, val_loader, device)

                NLL_training.append(avg_train_nll)
                NLL_validation.append(avg_val_nll)

                scheduler.step(avg_val_nll)
                epoch_time = time.time() - epoch_start
                current_lr = scheduler.get_last_lr()[0]

                log_rows.append({
                    "epoch": epoch,
                    "train_loss": avg_train_nll,
                    "train_loss_optim": avg_train_opt_nll,
                    "val_loss": avg_val_nll,
                    "lr": current_lr,
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
                        "router_state_dict": router.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "variables": list(variables),
                        "schema": "grouped_nf_router_v1"
                    }
                else:
                    counter += 1

                dash.update(
                    epoch=epoch,
                    train_loss=np.round(avg_train_nll, 6),
                    val_loss=np.round(avg_val_nll, 6),
                    lr=current_lr,
                    region=region
                )

                if counter >= PATIENCE:
                    logger.info("Early stopping triggered.")
                    break
        # -------------------------------------
        #  Save training artifacts
        # -------------------------------------
        paths_training = StorePathHelper(directory=f"Training_results_new/Wjets/all/{region}")

        t.save(checkpoint, paths_training.autopath.joinpath("model_checkpoint.pth"))
        t.save(checkpoint, f"Training_results_new/Wjets/all/{region}/latest/model_checkpoint.pth")

        pd.DataFrame(log_rows).to_pickle(str(paths_training.autopath.joinpath('training_logs.pkl')))
        pd.DataFrame(log_rows).to_pickle(str(f"Training_results_new/Wjets/all/{region}/latest/training_logs.pkl"))


        with open(paths_training.autopath.joinpath("config.yaml"), "w") as f:
            yaml.dump(config, f)

        logger.info("Model saved successfully")

# --------------

if __name__ == "__main__":
    main()