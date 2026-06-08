import numpy as np
import torch as t
from classes.helper import _component_collection
from classes.DataHandling import load_data, load_variables
from classes.NeuralNetworks import DNN
from sklearn import train_test_split
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as f
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import random
from torch.utils import data
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from CustomLogging import setup_logging
from classes.path_managment import StorePathHelper
from classes.config_loader import load_config
from classes.helper import get_class_weights, _same_sign_opposite_sign_split, _collection, _component_collection
import CODE.HELPER as helper
import torch as t
from tap import Tap
from typing import Any, Callable, Dict, Generator, List, Literal, Tuple, Union
from copy import deepcopy

import time

RANDOM_SEED = 42
TEST_SIZE = 0.25
DATA_PATH = 'data/data_complete.feather'
VARIABLE_PATH = 'configs/training_variables.yaml'
MASKS_PATH = 'configs/masks_new.yaml'
LEARNING_RATE = 1e-3





def calculate_scaled_event_weights_torch(
    event_values: torch.Tensor,
    event_original_weights: torch.Tensor,
    bins: torch.Tensor,
    total_subtraction_per_bin: torch.Tensor,
) -> torch.Tensor:

    device = event_values.device
    dtype = event_values.dtype

    # --- infer shapes ---
    n_bins = bins.numel() - 1
    n_events = event_values.shape[-1]

    # --- flatten batch dims ---
    flat_values = event_values.reshape(-1, n_events).contiguous()
    flat_weights = event_original_weights.reshape(-1, n_events).contiguous()
    flat_hist = total_subtraction_per_bin.reshape(-1, n_bins).contiguous()

    batch_size = flat_values.shape[0]

    # --- broadcast if needed ---
    if flat_weights.shape[0] == 1 and batch_size > 1:
        flat_weights = flat_weights.expand(batch_size, n_events)
    if flat_hist.shape[0] == 1 and batch_size > 1:
        flat_hist = flat_hist.expand(batch_size, n_bins)

    # --- bin assignment ---
    event_bin_indices = torch.bucketize(flat_values, bins, right=False) - 1

    is_out_of_bounds = (event_bin_indices < 0) | (event_bin_indices >= n_bins)
    event_bin_indices_clamped = event_bin_indices.clamp(0, n_bins - 1)

    # --- mask out-of-bounds contributions ---
    weights_for_sum = flat_weights.clone()
    weights_for_sum[is_out_of_bounds] = 0.0

    # --- sum weights per bin ---
    sum_weights = torch.zeros(
        (batch_size, n_bins),
        device=device,
        dtype=dtype
    )

    sum_weights.scatter_add_(
        1,
        event_bin_indices_clamped.long(),
        weights_for_sum
    )

    # --- compute scaling factors ---
    scale_factors = torch.ones_like(sum_weights)

    nonzero_mask = sum_weights != 0
    scale_factors[nonzero_mask] = (
        1.0 - flat_hist[nonzero_mask] / sum_weights[nonzero_mask]
    )

    zero_sum_nonzero_subtraction = (sum_weights == 0) & (flat_hist != 0)
    scale_factors[zero_sum_nonzero_subtraction] = 0.0

    # --- gather per-event scale factors ---
    scale_factors_events = scale_factors.gather(
        1,
        event_bin_indices_clamped.long()
    )

    # --- apply correction ---
    corrected = flat_weights * scale_factors_events
    corrected[is_out_of_bounds] = flat_weights[is_out_of_bounds]

    return corrected.reshape(event_original_weights.shape)

def get_my_data(df, training_var):

    ss_region = df.data
    os_region = df[(df.OS & (df.Label != 2)) | (df.SS & (df.Label == 2))]

    ss_os_split = _same_sign_opposite_sign_split(
        ss = ss_region,
        os = os_region
    )

    return _component_collection(
        X = ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype=np.float32)),
        Y = ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype=np.float32)),
        weights = ss_os_split.apply_func(lambda x: x["weight"].to_numpy(dtype=np.float32)),
        class_weights = ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
        process = ss_os_split.apply_func(lambda x: x["process"].to_numpy(dtype=np.float32)),
        SR_like = ss_os_split.apply_func(lambda x: x["id_tau_vsJet_Tight_2"].to_numpy(dtype=np.float32)),
    )




def training_data(
    df,
    label,
    training_var,
    weight_column="weight",
):

    X = df[training_var].to_numpy(dtype=np.float32)
    weights = df[weight_column].to_numpy(dtype=np.float32)

    if label == 0:

        Y = np.ones(df.shape[0], dtype=np.float32)
    elif label == 1:
        Y = np.zeros(df.shape[0], dtype=np.float32)
    else:
        'label must be 0 or 1'

    return _component_collection(
        X=X,
        Y=Y,
        weights=weights,
    )



def create_training_dataset(
    df,
    training_var,
    weight_column="weight",
    test_size=0.25,
    random_state=42,
):

    dataset = training_data(
        df_sig=df,
        training_var=training_var,
        weight_column=weight_column,
    )

    X = dataset.X
    Y = dataset.Y
    w = dataset.weights

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, Y, w,
        test_size=test_size,
        random_state=random_state
    )

    train = _component_collection(
        X=X_train,
        Y=y_train,
        weights=w_train,
    ).to_torch(device=None)

    val = _component_collection(
        X=X_val,
        Y=y_val,
        weights=w_val,
    ).to_torch(device=None)

    return train, val



def main():

    df = load_data(DATA_PATH, MASKS_PATH)
    training_variables = load_variables(VARIABLE_PATH)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    df_DR = pd.concatenate([df.wjets.DR_wjets, df.wjets_bkg.DR_wjets])

    train_sig, val_sig = create_training_dataset(
        df = df.wjets.DR_wjets,
        training_var = training_variables,
        weight_column='weight',
        test_soze = 0.25,
        random_state=RANDOM_SEED
    )

    train_bkg1, val_bkg1 = create_training_dataset(
        df = df.wjets_bkg.DR_wjets,
        training_var = training_variables,
        weight_column='weight',
        test_soze = 0.25,
        random_state=RANDOM_SEED
    )

    train_bkg2, val_bkg2 = create_training_dataset(
        df = df.data.DR_wjets_SS,
        training_var = training_variables,
        weight_column='weight',
        test_soze = 0.25,
        random_state=RANDOM_SEED
    )

    train_X = np.concatenate([train_sig.X, train_bkg1.X, train_bkg2.X], axis = 0).to_torch(device)
    train_Y = np.concatenate([train_sig.Y, train_bkg1.Y, train_bkg2.Y], axis = 0).to_torch(device)
    train_weights1 = np.concatenate([train_sig.W, train_bkg1.W], axis = 0).to_torch(device)
    train_weights2 = np.concatenate([train_bkg2.W], axis = 0).to_torch(device)
    train_weights = np.concatenate([train_weights1, train_weights2], axis = 0).to_torch(device)

    model = DNN(
        input_nodes=len(training_variables),
        hidden_nodes=[200, 200, 200],
        output_nodes= 1,
        dropout=0.15,
        activation = 'ReLU',
        output_activation='Sigmoid',
        input_names=training_variables,
    )

    model.initialize_scaler(
        shift = train_X.mean(dim = 0),
        scale = train_X.std(dim = 0) + 1e-6
    )
    
    model.to(device)

    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = t.nn.BCELoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None

    def run_epoch(X, Y, W, training=True):

        if training:
            model.train()
        else:
            model.eval()


        with t.set_grad_enabled(training):

            preds = model(X).squeeze(-1)
            Y = Y.squeeze(-1) if Y.dim() > 1 else Y
            W = W.squeeze(-1) if W.dim() > 1 else W
            # per-sample loss

            assert t.isfinite(X).all()
            assert t.isfinite(Y).all()
            assert t.isfinite(W).all()
            assert t.isfinite(preds).all()
            assert ((Y >= 0) & (Y <= 1)).all()
            assert ((preds >= 0) & (preds <= 1)).all()

            loss = loss_fn(preds, Y)

            # apply weights
            loss = (loss * W).sum() / (W.sum() + 1e-12)


            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    run_epoch(train_X, train_Y, train_weights, training = True)

    model.eval()

if __name__ == '__main__':
    main()