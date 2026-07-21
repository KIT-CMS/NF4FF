import torch as t
import copy
import os
from typing import Dict, List

from pathlib import Path
import copy
import os
import torch as t
import numpy as np

from .NeuralNetworks import save_model
from .Loss import PretrainingSqueezedLossWeightNormalized, PretrainingBCELossWeightNormalized
from .Logging import TrainingDashboard
from rich.live import Live

def train_dnn(
    model,
    train,
    val,
    epochs: int = 50,
    lr: float = 1e-3,
    loss_fn=None,
    checkpoint_dir: str = "./checkpoints",
    device: str = "cpu",
    scheduler_patience: int = 10,
    early_stopping_patience: int = 20,
    scheduler_factor: float = 0.1,
    min_delta = 1.0e-4,
    min_lr: float = 1.0e-6,
):
    dashboard = TrainingDashboard()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )


    if loss_fn is None:
        loss_fn = t.nn.BCELoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None

    def run_epoch(X, Y, W, training=True):

        if training:
            model.train()
        else:
            model.eval()

        X = X.to(device)
        Y = Y.to(device)
        W = W.to(device)

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




        return loss.item()
    with Live(dashboard.render(0, epochs, 0, 0, lr, 0, 0), refresh_per_second=4) as live:
        for epoch in range(epochs):

            train_loss = run_epoch(train.X, train.Y, train.weights, training=True)

            val_loss = run_epoch(val.X, val.Y, val.weights, training=False)

            old_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(val_loss)

            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != old_lr:
                print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

            current_lr = optimizer.param_groups[0]["lr"]


            if val_loss < best_val_loss - min_delta:

                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            live.update(
                dashboard.render(
                    epoch=epoch,
                    epochs=epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=current_lr,
                    patience=epochs_without_improvement,
                    best_val=best_val_loss,
                )
            )




            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

            if current_lr <= min_lr:
                print("\nStopping: minimum learning rate reached")
                break

    model.load_state_dict(best_state)

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    return model, best_val_loss

def train_dnn_new(
    model,
    train,
    val,
    epochs: int = 50,
    lr: float = 1e-3,
    loss_fn=None,
    checkpoint_dir: str = "./checkpoints",
    device: str = "cpu",
    scheduler_patience: int = 10,
    early_stopping_patience: int = 20,
    scheduler_factor: float = 0.1,
    min_delta = 1.0e-4,
    min_lr: float = 1.0e-6,
):
    dashboard = TrainingDashboard()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )


    if loss_fn is None:
        loss_fn = PretrainingBCELossWeightNormalized()
        
    best_val_loss = float("inf")
    best_state = None

    def run_epoch(X, Y, W, training=True):

        if training:
            model.train()
        else:
            model.eval()

        X = X.to(device)
        Y = Y.to(device)
        W = W.to(device)

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

            loss = loss_fn(
                device = device,
                input=preds,
                target=Y,
                weights_class=W,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()




        return loss.item()
    with Live(dashboard.render(0, epochs, 0, 0, lr, 0, 0), refresh_per_second=4) as live:
        for epoch in range(epochs):

            train_loss = run_epoch(train.X, train.Y, train.weights, training=True)

            val_loss = run_epoch(val.X, val.Y, val.weights, training=False)

            old_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(val_loss)

            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != old_lr:
                print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

            current_lr = optimizer.param_groups[0]["lr"]


            if val_loss < best_val_loss - min_delta:

                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            live.update(
                dashboard.render(
                    epoch=epoch,
                    epochs=epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=current_lr,
                    patience=epochs_without_improvement,
                    best_val=best_val_loss,
                )
            )




            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

            if current_lr <= min_lr:
                print("\nStopping: minimum learning rate reached")
                break

    model.load_state_dict(best_state)

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    return model, best_val_loss


def train_dnn_squeezed_loss(
    model,
    train,
    val,
    squeezing,
    epochs: int = 50,
    lr: float = 1e-3,
    checkpoint_dir: str = "./checkpoints",
    device: str = "cpu",
    scheduler_patience: int = 10,
    early_stopping_patience: int = 20,
    scheduler_factor: float = 0.1,
    min_delta = 1.0e-4,
    min_lr: float = 1.0e-6,
):
    dashboard = TrainingDashboard()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )
    if squeezing >= 1.0:
        penalty_upper_bound = 1000
    elif squeezing <= 0.0:
        raise ValueError(f"squeezing must be > 0, got {squeezing}")
    else:
        penalty_upper_bound = np.log(squeezing / (1 - squeezing))

    loss_fn = PretrainingSqueezedLossWeightNormalized(
        penalty_upper_bound = penalty_upper_bound,
        device = device,
        )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    def run_epoch(X, Y, W, training=True):

        if training:
            model.train()
        else:
            model.eval()

        X = X.to(device)
        Y = Y.to(device)
        W = W.to(device)

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
            #import ipdb;ipdb.set_trace()
            loss = loss_fn(
                input=preds,
                target=Y,
                weights_class=W,
            )



            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()




        return {
            "total_loss": float(loss.item()),
            "base_loss": float(loss_fn.tracked_base_loss),
            "penalty_loss": float(loss_fn.tracked_penalty_loss),
        }
    

    with Live(dashboard.render(0, epochs, 0, 0, lr, 0, 0), refresh_per_second=4) as live:

        for epoch in range(epochs):

            train_metrics = run_epoch(train.X, train.Y, train.weights, training=True)
            train_loss = train_metrics["total_loss"]

            val_metrics = run_epoch(val.X, val.Y, val.weights, training=False)
            val_loss = val_metrics["total_loss"]

            base_loss = val_metrics["base_loss"]
            penalty_loss = val_metrics["penalty_loss"]

            old_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(val_loss)

            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr != old_lr:
                print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

            current_lr = optimizer.param_groups[0]["lr"]


            if val_loss < best_val_loss - min_delta:

                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

                epochs_without_improvement = 0

            else:
                epochs_without_improvement += 1

            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_total_loss": train_metrics["total_loss"],
                    "train_base_loss": train_metrics["base_loss"],
                    "train_penalty_loss": train_metrics["penalty_loss"],
                    "val_total_loss": val_metrics["total_loss"],
                    "val_base_loss": val_metrics["base_loss"],
                    "val_penalty_loss": val_metrics["penalty_loss"],
                    "lr": float(current_lr),
                    "best_val_loss": float(best_val_loss),
                }
            )

            live.update(
                dashboard.render(
                    epoch=epoch,
                    epochs=epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=current_lr,
                    patience=epochs_without_improvement,
                    best_val=best_val_loss,

                    # NEW DEBUG SIGNALS
                    #base_loss=base_loss,
                    #penalty_loss=penalty_loss,
                )
            )




            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

            if current_lr <= min_lr:
                print("\nStopping: minimum learning rate reached")
                break

    model.load_state_dict(best_state)
    model.training_history = history

    print(f"\nBest validation loss: {best_val_loss:.6f}")

    return model, best_val_loss, history

