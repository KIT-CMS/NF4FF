import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from CustomLogging import setup_logging
from classes.path_managment import StorePathHelper
from classes.config_loader import load_config
import CODE.HELPER as helper
import torch as t
from tap import Tap
from typing import Literal, Generator
import time


# ----- seed -----

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ------ logger -----

logger = setup_logging(logger=logging.getLogger(__name__))

# ----- TAP Arguments -----

class Args(Tap):
    njets: Literal['inclusive', 'njets0', 'njets1', 'njets2']
    fold: Literal['fold1', 'fold2']

# ----- constants ------

INPUT_DIM = 20
CONFIG_MODEL_PATH = '../configs/config_NN.yaml'
PATIENCE = 5

# ----- data classes -----
@dataclass
class Config:
    # training
    bsize_train: int     # not used in full-batch (kept for compatibility)
    bsize_val: int       # not used in full-batch
    bsize_test: int      # not used in full-batch
    grad_clip: float
    n_epochs: int
    use_amp: bool
    s_scale_max: float

    # optimizer
    lr: float

    # scheduler
    scheduler_step_size: int
    scheduler_gamma: float
    scheduler_factor: float
    scheduler_patience: int
    scheduler_threshold: float
    scheduler_cooldown: int
    scheduler_min_lr: float
    scheduler_eps: float

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "Config":
        training = cfg["training"]
        optimizer = cfg["optimizer"]
        scheduler = cfg["scheduler"]

        return Config(
            bsize_train=training["bsize_train"],
            bsize_val=training["bsize_val"],
            bsize_test=training["bsize_test"],
            grad_clip=training["grad_clip"],
            n_epochs=training["n_epochs"],
            use_amp=training["use_amp"],
            s_scale_max=training["s_scale_max"],
            lr=optimizer["lr"],
            scheduler_step_size=scheduler["step_size"],
            scheduler_gamma=scheduler["gamma"],
            scheduler_factor=scheduler["factor"],
            scheduler_patience=scheduler["patience"],
            scheduler_threshold=scheduler["threshold"],
            scheduler_cooldown=scheduler["cooldown"],
            scheduler_min_lr=scheduler["min_lr"],
            scheduler_eps=scheduler["eps"],
        )
    
@dataclass
class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[t.Tensor, pd.DataFrame, np.ndarray]
    os: Union[t.Tensor, pd.DataFrame, np.ndarray]

@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    X: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    Y: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None

# ------ lists -----
dim = 20
variables = [
    "pt_1","pt_2","eta_1","eta_2","jpt_1","jpt_2","jeta_1","jeta_2",
    "m_fastmtt","pt_fastmtt","met","njets","nbtag","mt_tot","m_vis",
    "pt_tt","pt_vis","mjj","pt_dijet","pt_ttjj","pzetamissvis","deltaEta_jj",
    "deltaEta_ditaupair","deltaR_jj","deltaR_ditaupair","deltaR_1j1","deltaR_1j2",
    "deltaR_2j1","deltaR_2j2","deltaR_12j1","deltaR_12j2","deltaEta_1j1",
    "deltaEta_1j2","deltaEta_2j1","deltaEta_2j2","deltaEta_12j1","deltaEta_12j2",
][:dim]

# ----- model -----
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 200, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # registered buffers (no-op because we pre-scale; left for compatibility)
        self.register_buffer("_scaler_shift", torch.full((input_dim,), 0.0, dtype=torch.float32))
        self.register_buffer("_scaler_scale", torch.full((input_dim,), 1.0, dtype=torch.float32))

    @property
    def _is_initialized(self):
        initialized = (torch.isnan(self._scaler_shift) | torch.isnan(self._scaler_scale)).sum() == 0
        initialized &= (self._scaler_scale != 1).all() & (self._scaler_shift != 0).all()
        return initialized

    def initialize_scaler(
        self,
        shift: Union[np.ndarray, torch.Tensor, None] = None,
        scale: Union[np.ndarray, torch.Tensor, None] = None,
    ):
        if self._is_initialized:
            logger.warning("Scaler already initialized. Overwriting the current values.")
        elif shift is not None and scale is not None:
            shift = torch.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
            scale = torch.from_numpy(scale) if isinstance(scale, np.ndarray) else scale
        else:
            msg = """
                shift and scale must be both None
                (falling to default of shift=0.0, scale=1.0)
                or both not None raise ValueError
            """
            logger.error(msg)
            raise ValueError(msg)

        self._scaler_shift.data[:] = shift
        self._scaler_scale.data[:] = scale

    def apply_scaler(self, x):
        return (x - self._scaler_shift.to(x.device)) / self._scaler_scale.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.apply_scaler(x)
        return self.net(x)

# ------------------
def main():

    # ----

    dim = 20

    # --- load config

    raw = load_config(CONFIG_MODEL_PATH)
    config = Config.from_dict(raw)
    args = Args().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(torch.cuda.get_device_name())

    # --- load data 


    if args.fold == 'fold1':
        train_pt = torch.load(f'../data/{args.njets}/train1.pt')
        val_pt = torch.load(f'../data/{args.njets}/val1.pt')
        w_qcd_train = np.load(f'../data/{args.njets}/qcd_weights_train1.npy')
        w_qcd_val = np.load(f'../data/{args.njets}/qcd_weights_val1.npy')
    elif args.fold == 'fold2':
        train_pt = torch.load(f'../data/{args.njets}/train2.pt')
        val_pt = torch.load(f'../data/{args.njets}/val2.pt')
        w_qcd_train = np.load(f'../data/{args.njets}/qcd_weights_train2.npy')
        w_qcd_val = np.load(f'../data/{args.njets}/qcd_weights_val2.npy')
    else:
        logger.warning('Invalid --fold input: Choose from [fold1, fold2] ')

    # ------ 

    # train data loading

    # QCD data

    X_train_qcd = train_pt.X.ss[train_pt.Y.ss == 2].to(device)
    Y_train_qcd = torch.zeros(len(train_pt.Y.ss == 2)).to(device)
    w_train_qcd = torch.from_numpy(w_qcd_train).to(device)

    logger.info(f'len X_train_qcd: {len(X_train_qcd)}')
    logger.info(f'len w_train_qcd: {len(w_train_qcd)}')

    # non FF data

    X_train_nFF = train_pt.X.os[train_pt.Y.os == 0].to(device)
    Y_train_nFF = train_pt.Y.os[train_pt.Y.os == 0].to(device)
    w_train_nFF = train_pt.weights.os[train_pt.Y.os == 0].to(device)

    logger.info(f'len X_train_nFF: {len(X_train_nFF)}')
    logger.info(f'len w_train_nFF: {len(w_train_nFF)}')

    # Wjets

    X_train_FF = train_pt.X.os[train_pt.Y.os == 1].to(device)
    Y_train_FF = train_pt.Y.os[train_pt.Y.os == 1].to(device)
    w_train_FF = train_pt.weights.os[train_pt.Y.os == 1].to(device)

    logger.info(f'len X_train_FF: {len(X_train_FF)}')
    logger.info(f'len w_train_FF: {len(w_train_FF)}')

    # train data loading

    # QCD data

    X_val_qcd = val_pt.X.ss[val_pt.Y.ss == 2].to(device)
    Y_val_qcd = torch.zeros(len(val_pt.Y.ss == 2)).to(device)
    w_val_qcd = torch.from_numpy(w_qcd_val).to(device)

    logger.info(f'len X_val_qcd: {len(X_val_qcd)}')
    logger.info(f'len w_val_qcd: {len(w_val_qcd)}')

    # non FF data

    X_val_nFF = val_pt.X.os[val_pt.Y.os == 0].to(device)
    Y_val_nFF = val_pt.Y.os[val_pt.Y.os == 0].to(device)
    w_val_nFF = val_pt.weights.os[val_pt.Y.os == 0].to(device)

    logger.info(f'len X_val_nFF: {len(X_val_nFF)}')
    logger.info(f'len w_val_nFF: {len(w_val_nFF)}')

    # Wjets

    X_val_FF = val_pt.X.os[val_pt.Y.os == 1].to(device)
    Y_val_FF = val_pt.Y.os[val_pt.Y.os == 1].to(device)
    w_val_FF = val_pt.weights.os[val_pt.Y.os == 1].to(device)

    logger.info(f'len X_val_FF: {len(X_val_FF)}')
    logger.info(f'len w_val_FF: {len(w_val_FF)}')


    X_train = torch.cat([X_train_FF, X_train_nFF, X_train_qcd], dim = 0)
    y_train = torch.cat([Y_train_FF, Y_train_nFF, Y_train_qcd], dim = 0)
    w_train = torch.cat([w_train_FF, w_train_nFF, w_train_qcd], dim = 0)

    X_val = torch.cat([X_val_FF, X_val_nFF, X_val_qcd], dim = 0)
    y_val = torch.cat([Y_val_FF, Y_val_nFF, Y_val_qcd], dim = 0)
    w_val = torch.cat([w_val_FF, w_val_nFF, w_val_qcd], dim = 0)

    perm_train = torch.randperm(X_train.shape[0], device=X_train.device)
    perm_val   = torch.randperm(X_val.shape[0],   device=X_val.device)

    X_train = X_train[perm_train]
    y_train = y_train[perm_train]
    w_train = w_train[perm_train]

    X_val = X_val[perm_val]
    y_val = y_val[perm_val]
    w_val = w_val[perm_val]


    shift = X_train.mean(dim=0)
    scale  = X_train.std(dim=0, unbiased=False).clamp_min(1e-12)

    dataset_train = TensorDataset(X_train, y_train, w_train)
    dataset_val = TensorDataset(X_val, y_val, w_val)

    train_loader = DataLoader(
        dataset_train,
        batch_size = 20_000,
        shuffle = True,
        drop_last = False
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size = 20_000,
        shuffle= True,
        drop_last = False
    )


    # model, optimizer, scheduler
    model = BinaryClassifier(input_dim=dim, hidden_dim=200, p=0.15).to(device)
    model.initialize_scaler(shift = shift, scale = scale)
    criterion = nn.BCELoss(reduction='none')                                  # there
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.scheduler_threshold,
        threshold_mode='rel',
        cooldown=config.scheduler_cooldown,
        min_lr=config.scheduler_min_lr,
        eps=config.scheduler_eps
    )

    use_amp = (device.type == device) and bool(config.use_amp)
    scaler_amp = torch.amp.GradScaler(device, enabled=use_amp)

    # training loop (full-batch)
    best_val = float('inf')
    counter = 0
    checkpoint = None

    log_rows = []
    logger.info("Start training (full-batch)")

    for epoch in range(config.n_epochs):

        # ------- train

        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        epoch_start = time.time()
        num_samples_loaded = 0 

        for Xb, yb, wb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(Xb)
                logits_f32 = logits.float()
                yb_f32= yb.float().view(-1,1)
                wb_f32 = wb.float().view(-1,1) 
                if not torch.all((yb_f32 >= 0) & (yb_f32 <= 1)):
                    print("BAD TARGETS:", torch.unique(yb_f32))
                    print("min/max:", yb_f32.min(), yb_f32.max())
                    raise RuntimeError("Invalid targets for BCE")                        # (B, 1)
                loss_per_sample = criterion(logits_f32, yb_f32)   # (B, 1)
                wb_ = wb.view(-1, 1)

                batch_loss = (loss_per_sample * wb_f32).sum()
                batch_weight = wb_.sum()

                loss = batch_loss / batch_weight

            scaler_amp.scale(loss).backward()

            if config.grad_clip and config.grad_clip > 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            scaler_amp.step(optimizer)
            scaler_amp.update()

            # accumulate for epoch average
            train_loss_sum += batch_loss.item()
            train_weight_sum += batch_weight.item()

        train_loss = train_loss_sum / train_weight_sum

        # ------ validation

        model.eval()
        val_loss_sum = 0.0
        val_weight_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for Xb, yb, wb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(Xb)
                    logits_f32 = logits.float()
                    yb_f32= yb.float().view(-1,1)
                    wb_f32 = wb.float().view(-1,1)                         # (B, 1)
                    if not torch.all((yb_f32 >= 0) & (yb_f32 <= 1)):
                        print("BAD TARGETS:", torch.unique(yb_f32))
                        print("min/max:", yb_f32.min(), yb_f32.max())
                    loss_per_sample = criterion(logits_f32, yb_f32)   # (B, 1)
                    wb_ = wb.view(-1, 1)

                    batch_loss = (loss_per_sample * wb_).sum()
                    batch_weight = wb_.sum()

                    val_loss_sum += batch_loss.item()
                    val_weight_sum += batch_weight.item()

                    probs = logits
                    preds = (probs >= 0.5).float()

                    correct += (preds == yb).sum().item()
                    total += yb.numel()

        val_loss = val_loss_sum / val_weight_sum
        val_acc = correct / total
        epoch_time = time.time() - epoch_start


        # ----- scheduler and logging -----

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch}: train={train_loss:.6f}, "
            f"val={val_loss:.6f}, acc={val_acc:.4f}, LR={current_lr:.6e}"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "time_s": epoch_time,
            "type": "epoch",
        })


        # ----- early stopping -----



        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'variables': variables,
            }
        else:
            counter += 1
            if counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break


    # save checkpoint
    paths_training = StorePathHelper(directory=f"Categorizer_results/{args.njets}/training/{args.fold}")
    if checkpoint is None:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_mean': torch.from_numpy(shift.astype(np.float32)),
            'scaler_scale': torch.from_numpy(scale.astype(np.float32)),
            'variables': variables,
        }
    torch.save(checkpoint, paths_training.autopath.joinpath('model_checkpoint.pth'))

    # save log file
    pd.DataFrame(log_rows).to_pickle(str(paths_training.autopath.joinpath('training_logs.pkl')))


# ----------------
if __name__ == "__main__":
    main()