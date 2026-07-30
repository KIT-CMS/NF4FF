"""Process-fraction classifier training."""

import json
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Union
from pathlib import Path
from models.networks import DNN, FoldCombinedDNN, save_model
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from data.handling import load_data, load_variables
from inference.yields import calculate_qcd_yield_corrections
from core.paths import CONFIG_ROOT, PROJECT_ROOT


DATA_PATH = PROJECT_ROOT / "data" / "dataframe_complete.feather"
MASKS_PATH = CONFIG_ROOT / "selections.yaml"
TRAINING_VAR_PATH = CONFIG_ROOT / "variables_fake_factor.yaml"
MODEL_RESULT_DIR = PROJECT_ROOT / "Law_workflow_results" / "training_fraction"
QCD_FRACTION_WEIGHT_COLUMN = "weight_qcd_fraction_njets"
CLASS_WEIGHT_COLUMNS = (
    QCD_FRACTION_WEIGHT_COLUMN,
    "weight",
    "weight",
)


def qcd_fraction_data_frame(df):
    df.data[QCD_FRACTION_WEIGHT_COLUMN]
    frame = df.data.DR_qcd_fractions.events.copy()
    validate_qcd_fraction_weights(frame)

    return frame


def validate_qcd_fraction_weights(frame, weight_column=QCD_FRACTION_WEIGHT_COLUMN):
    if weight_column not in frame.columns:
        raise ValueError(f"Missing QCD fraction weight column: {weight_column}")

    finite_weights = np.isfinite(frame[weight_column].to_numpy(dtype=np.float64))
    if not finite_weights.all():
        raise ValueError(
            f"{weight_column} contains non-finite weights for "
            f"{int((~finite_weights).sum())}/{len(frame)} "
            "preselected DR_qcd_fractions events."
        )


def _weight_array(frame, weight_column, label):
    if weight_column not in frame.columns:
        raise ValueError(f"{label}: missing event-weight column {weight_column}.")

    weights = frame[weight_column].to_numpy(dtype=np.float32)
    finite_weights = np.isfinite(weights)
    if not finite_weights.all():
        raise ValueError(
            f"{label}: {weight_column} contains non-finite weights for "
            f"{int((~finite_weights).sum())}/{len(frame)} events."
        )

    weight_sum = weights.sum(dtype=np.float64)
    if weight_sum <= 0:
        raise ValueError(
            f"{label}: {weight_column} has non-positive total weight "
            f"{weight_sum}."
        )

    return weights


def _qcd_yield_correction_array(frame, corrections):
    correction_values = np.ones(len(frame), dtype=np.float32)

    correction_specs = (
        ("njets_0", frame["njets"] == 0),
        ("njets_1", frame["njets"] == 1),
        ("njets_ge_2", frame["njets"] >= 2),
    )
    for bin_name, mask in correction_specs:
        correction = float(corrections[bin_name]["correction"])
        if not np.isfinite(correction) or correction <= 0:
            raise ValueError(
                f"{bin_name}: invalid QCD OS/SS yield correction {correction}."
            )
        correction_values[np.asarray(mask, dtype=bool)] = correction

    return correction_values


def _normalize_qcd_weights_to_ss_yields(frame, weights, corrections):
    """Normalize non-negative QCD shape weights to subtracted SS yields."""
    normalized = np.asarray(weights, dtype=np.float64).copy()
    bin_specs = (
        ("njets_0", frame["njets"] == 0),
        ("njets_1", frame["njets"] == 1),
        ("njets_ge_2", frame["njets"] >= 2),
    )
    normalization = {}

    for bin_name, mask in bin_specs:
        mask = np.asarray(mask, dtype=bool)
        observed_yield = float(normalized[mask].sum(dtype=np.float64))
        expected_yield = float(corrections[bin_name]["yield_SS"]["qcd"])
        if not np.isfinite(expected_yield) or expected_yield <= 0:
            raise ValueError(
                f"{bin_name}: invalid subtracted SS QCD yield {expected_yield}."
            )
        if not np.isfinite(observed_yield) or observed_yield <= 0:
            raise ValueError(
                f"{bin_name}: QCD enrichment weights have invalid total "
                f"yield {observed_yield}."
            )

        scale = expected_yield / observed_yield
        normalized[mask] *= scale
        normalization[bin_name] = {
            "pre_normalization_yield": observed_yield,
            "expected_subtracted_ss_yield": expected_yield,
            "normalization_factor": scale,
            "post_normalization_yield": float(
                normalized[mask].sum(dtype=np.float64)
            ),
        }

    return normalized.astype(np.float32), normalization


def calculate_qcd_training_yield_closure(
    frame,
    corrections,
    *,
    weight_column=QCD_FRACTION_WEIGHT_COLUMN,
    relative_tolerance=1.0e-3,
    absolute_tolerance=1.0,
):
    """Compare final SS-derived QCD training yields with subtracted OS targets."""
    raw_weights = _weight_array(frame, weight_column, "qcd")
    normalized_weights, normalization = _normalize_qcd_weights_to_ss_yields(
        frame,
        raw_weights,
        corrections,
    )
    correction_values = _qcd_yield_correction_array(frame, corrections)
    corrected_weights = normalized_weights * correction_values
    bin_specs = (
        ("njets_0", frame["njets"] == 0),
        ("njets_1", frame["njets"] == 1),
        ("njets_ge_2", frame["njets"] >= 2),
    )

    report = {}
    failures = []
    for bin_name, mask in bin_specs:
        mask = np.asarray(mask, dtype=bool)
        raw_ss_yield = float(raw_weights[mask].sum(dtype=np.float64))
        normalized_ss_yield = float(
            normalized_weights[mask].sum(dtype=np.float64)
        )
        observed_os_yield = float(
            corrected_weights[mask].sum(dtype=np.float64)
        )
        expected_ss_yield = float(corrections[bin_name]["yield_SS"]["qcd"])
        expected_os_yield = float(corrections[bin_name]["yield_OS"]["qcd"])
        absolute_difference = observed_os_yield - expected_os_yield
        relative_difference = (
            absolute_difference / expected_os_yield
            if expected_os_yield != 0.0
            else np.inf
        )
        passed = bool(
            np.isclose(
                observed_os_yield,
                expected_os_yield,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        )
        report[bin_name] = {
            "raw_ss_training_yield": raw_ss_yield,
            "ss_normalization_factor": normalization[bin_name][
                "normalization_factor"
            ],
            "normalized_ss_training_yield": normalized_ss_yield,
            "expected_subtracted_ss_yield": expected_ss_yield,
            "os_ss_correction": float(corrections[bin_name]["correction"]),
            "corrected_training_yield": observed_os_yield,
            "expected_subtracted_os_yield": expected_os_yield,
            "absolute_difference": absolute_difference,
            "relative_difference": relative_difference,
            "passed": passed,
        }
        if not passed:
            failures.append(bin_name)

    report["configuration"] = {
        "weight_column": weight_column,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }
    report["passed"] = not failures

    return report


def build_multiclass_training_arrays(df, training_var, qcd_yield_corrections=None):
    data_qcd_fraction = qcd_fraction_data_frame(df)

    class_frames = [
        data_qcd_fraction,
        df.wjets.AR.events,
        df.ttbar_J.AR.events,
    ]
    class_labels = ("qcd", "wjets", "ttbar")

    X = np.concatenate(
        [
            frame[training_var].to_numpy(dtype=np.float32)
            for frame in class_frames
        ],
        axis=0,
    )
    y = np.concatenate(
        [
            np.full(len(frame), class_index, dtype=np.int64)
            for class_index, frame in enumerate(class_frames)
        ],
        axis=0,
    )
    parity = np.concatenate(
        [
            (frame["event"].to_numpy(dtype=np.int64) % 2)
            for frame in class_frames
        ],
        axis=0,
    )
    class_weights = []
    for frame, weight_column, label in zip(
        class_frames,
        CLASS_WEIGHT_COLUMNS,
        class_labels,
    ):
        weights = _weight_array(frame, weight_column, label)
        if label == "qcd" and qcd_yield_corrections is not None:
            weights, _ = _normalize_qcd_weights_to_ss_yields(
                frame,
                weights,
                qcd_yield_corrections,
            )
            weights = (
                weights
                * _qcd_yield_correction_array(frame, qcd_yield_corrections)
            ).astype(np.float32)
        class_weights.append(weights)

    weights_event = np.concatenate(class_weights, axis=0)

    return X, y, parity, weights_event



class MulticlassFoldDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parity: np.ndarray,
        weights_event: Union[np.ndarray, None] = None,
    ) -> None:
        """
        X:
            Shape (n_events, n_features)

        y:
            Shape (n_events,)
            Integer labels:
                0, 1, 2, ...

        parity:
            Shape (n_events,)
            Values 0 or 1.

        weights_event:
            Optional event weights.
        """

        self.X = t.tensor(X, dtype=t.float32)
        self.y = t.tensor(y, dtype=t.long)

        self.parity = t.tensor(parity, dtype=t.float32).view(-1, 1)

        if weights_event is None:
            self.weights_event = t.ones(len(self.y), dtype=t.float32)
        else:
            self.weights_event = t.tensor(weights_event, dtype=t.float32)

        # Normal DataLoader shape:
        # (n_events, n_features + 1)
        #
        # Column 0  = parity
        # Columns 1 = physics input features
        self.X_with_parity = t.cat(
            [self.parity, self.X],
            dim=1,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_with_parity[idx], self.y[idx], self.weights_event[idx]


class FoldDataModule(L.LightningDataModule):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parity: np.ndarray,
        weights_event: Union[np.ndarray, None] = None,
        batch_size: int = 4096,
        train_fraction: float = 0.8,
        seed: int = 42,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.X = X
        self.y = y
        self.parity = parity
        self.weights_event = weights_event

        self.batch_size = batch_size
        self.train_fraction = train_fraction
        self.seed = seed
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None

        self.train_indices = None
        self.val_indices = None

    def setup(self, stage=None) -> None:
        full_dataset = MulticlassFoldDataset(
            X=self.X,
            y=self.y,
            parity=self.parity,
            weights_event=self.weights_event,
        )

        n_train = int(self.train_fraction * len(full_dataset))
        n_val = len(full_dataset) - n_train

        train_dataset, val_dataset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=t.Generator().manual_seed(self.seed),
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_indices = np.array(train_dataset.indices)
        self.val_indices = np.array(val_dataset.indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class LitFoldMulticlassClassifier(L.LightningModule):
    def __init__(
        self,
        even_model: t.nn.Module,
        odd_model: t.nn.Module,
        n_classes: int,
        weights_class: Union[t.Tensor, None] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        fold_id_name: str = "event_parity",
    ) -> None:
        super().__init__()

        self.even_model = even_model
        self.odd_model = odd_model

        # Your original FoldCombinedDNN, unchanged.
        #
        # Important convention:
        #
        # even_model is trained on ODD events
        # and used for EVEN events.
        #
        # odd_model is trained on EVEN events
        # and used for ODD events.
        self.combined_model = FoldCombinedDNN(
            even_model=self.even_model,
            odd_model=self.odd_model,
            fold_id_name=fold_id_name,
        )

        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay

        if weights_class is not None:
            self.register_buffer("weights_class", weights_class.float())
        else:
            self.weights_class = None

        self.save_hyperparameters(ignore=["even_model", "odd_model"])

    def forward(self, X_with_parity: t.Tensor) -> t.Tensor:
        """
        X_with_parity comes from the DataLoader with shape:

            (batch_size, n_features + 1)

        Your original FoldCombinedDNN expects:

            (n_features + 1, batch_size)

        Therefore we transpose here.
        """

        return self.combined_model(X_with_parity.T)

    def _weighted_ce_loss(
        self,
        logits: t.Tensor,
        y: t.Tensor,
        weights_event: t.Tensor,
    ) -> t.Tensor:
        loss_per_event = F.cross_entropy(
            logits,
            y.long(),
            weight=self.weights_class,
            reduction="none",
        )

        return (weights_event * loss_per_event).sum() / weights_event.sum()

    def training_step(self, batch, batch_idx):
        X_with_parity, y, weights_event = batch

        parity = X_with_parity[:, 0].long()
        features = X_with_parity[:, 1:]

        even_events = parity == 0
        odd_events = parity == 1

        losses = []

        # --------------------------------------------------------
        # even_model is trained on odd events
        # --------------------------------------------------------

        if odd_events.any():
            logits_even_model = self.even_model(features[odd_events])

            loss_even_model = self._weighted_ce_loss(
                logits=logits_even_model,
                y=y[odd_events],
                weights_event=weights_event[odd_events],
            )

            losses.append(loss_even_model)

            self.log(
                "train_loss_even_model",
                loss_even_model,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=int(odd_events.sum()),
            )

        # --------------------------------------------------------
        # odd_model is trained on even events
        # --------------------------------------------------------

        if even_events.any():
            logits_odd_model = self.odd_model(features[even_events])

            loss_odd_model = self._weighted_ce_loss(
                logits=logits_odd_model,
                y=y[even_events],
                weights_event=weights_event[even_events],
            )

            losses.append(loss_odd_model)

            self.log(
                "train_loss_odd_model",
                loss_odd_model,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=int(even_events.sum()),
            )

        if len(losses) == 0:
            loss = t.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss = sum(losses) / len(losses)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        X_with_parity, y, weights_event = batch

        # Uses FoldCombinedDNN:
        #
        # parity == 0 events use even_model
        # parity == 1 events use odd_model
        logits = self.forward(X_with_parity)

        loss = self._weighted_ce_loss(
            logits=logits,
            y=y,
            weights_event=weights_event,
        )

        pred_class = t.argmax(logits, dim=1)

        acc = (pred_class == y).float().mean()

        correct = (pred_class == y).float()
        weighted_acc = (weights_event * correct).sum() / weights_event.sum()

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )

        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )

        self.log(
            "val_weighted_acc",
            weighted_acc,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=len(y),
        )

        return loss

    def configure_optimizers(self):
        optimizer = t.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return optimizer

    def predict_proba(self, X_with_parity: t.Tensor) -> t.Tensor:
        logits = self.forward(X_with_parity)
        return t.softmax(logits, dim=1)

    def predict_class(self, X_with_parity: t.Tensor) -> t.Tensor:
        return t.argmax(self.predict_proba(X_with_parity), dim=1)


def train_fraction_classifier(
    data_path=DATA_PATH,
    masks_path=MASKS_PATH,
    training_var_path=TRAINING_VAR_PATH,
    output_dir=MODEL_RESULT_DIR,
    yield_closure_report_path=None,
):

    # ------------------------------------------------------------
    # Real training data
    # ------------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path, masks_path)
    training_var = load_variables(training_var_path)
    qcd_yield_corrections = calculate_qcd_yield_corrections(df)
    qcd_frame = qcd_fraction_data_frame(df)
    yield_closure_report = calculate_qcd_training_yield_closure(
        qcd_frame,
        qcd_yield_corrections,
    )
    if yield_closure_report_path is None:
        yield_closure_report_path = output_dir / "qcd_yield_closure.json"
    yield_closure_report_path = Path(yield_closure_report_path)
    yield_closure_report_path.parent.mkdir(parents=True, exist_ok=True)
    yield_closure_report_path.write_text(
        json.dumps(yield_closure_report, indent=2) + "\n"
    )
    print("QCD yield closure:", yield_closure_report)
    if not yield_closure_report["passed"]:
        failed_bins = [
            bin_name
            for bin_name in ("njets_0", "njets_1", "njets_ge_2")
            if not yield_closure_report[bin_name]["passed"]
        ]
        details = ", ".join(
            f"{bin_name}: corrected="
            f"{yield_closure_report[bin_name]['corrected_training_yield']:.8g}, "
            f"expected="
            f"{yield_closure_report[bin_name]['expected_subtracted_os_yield']:.8g}, "
            f"relative_difference="
            f"{yield_closure_report[bin_name]['relative_difference']:.3%}"
            for bin_name in failed_bins
        )
        raise ValueError(f"QCD training yield closure failed ({details}).")

    n_classes = 3
    X, y, parity, weights_event = build_multiclass_training_arrays(
        df,
        training_var,
        qcd_yield_corrections=qcd_yield_corrections,
    )
    n_features = X.shape[1]

    # ------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------

    data_module = FoldDataModule(
        X=X,
        y=y,
        parity=parity,
        weights_event=weights_event,
        batch_size=4096,
        train_fraction=0.8,
        seed=42,
        num_workers=0,
    )

    # Need setup once here so we can compute the scaler and class counts
    # from training indices only.
    data_module.setup()

    train_indices = data_module.train_indices

    X_train = X[train_indices]
    y_train = y[train_indices]
    weights_train = weights_event[train_indices]

    # ------------------------------------------------------------
    # Scaler from training data only
    # ------------------------------------------------------------

    scaler_shift = X_train.mean(axis=0)
    scaler_scale = X_train.std(axis=0)
    scaler_scale[scaler_scale == 0] = 1.0

    # ------------------------------------------------------------
    # Class counts from training data only
    # ------------------------------------------------------------

    class_counts = np.bincount(y_train, minlength=n_classes)
    class_weight_sums = np.bincount(
        y_train,
        weights=weights_train.astype(np.float64),
        minlength=n_classes,
    )

    if np.any(class_counts == 0):
        raise ValueError(
            f"At least one class has zero entries in the training set: "
            f"class_counts={class_counts}"
        )

    print("Class counts:", class_counts)
    print("Class weight columns:", CLASS_WEIGHT_COLUMNS)
    print(
        "QCD yield corrections:",
        {
            bin_name: result["correction"]
            for bin_name, result in qcd_yield_corrections.items()
        },
    )
    print("Class weighted yields:", class_weight_sums)

    # ------------------------------------------------------------
    # Create the two normal DNNs
    # ------------------------------------------------------------

    even_model = DNN(
        input_nodes=n_features,
        hidden_nodes=(200, 200),
        output_nodes=n_classes,
        dropout=0.1,
        activation="ReLU",
        output_activation="Linear",
    )

    odd_model = DNN(
        input_nodes=n_features,
        hidden_nodes=(200, 200),
        output_nodes=n_classes,
        dropout=0.1,
        activation="ReLU",
        output_activation="Linear",
    )

    even_model.initialize_scaler(
        shift=scaler_shift,
        scale=scaler_scale,
    )

    odd_model.initialize_scaler(
        shift=scaler_shift,
        scale=scaler_scale,
    )

    # ------------------------------------------------------------
    # Lightning model
    # ------------------------------------------------------------

    lit_model = LitFoldMulticlassClassifier(
        even_model=even_model,
        odd_model=odd_model,
        n_classes=n_classes,
        # Intentionally no class-balancing weights: the classifier should learn
        # process fractions from the physical event yields.
        weights_class=None,
        lr=1e-3,
        weight_decay=1e-4,
        fold_id_name="event_parity",
    )

    # ------------------------------------------------------------
    # Callbacks and logger
    # ------------------------------------------------------------

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "lightning_checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-fold-combined-{epoch:03d}-{val_loss:.5f}",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
    )

    logger = CSVLogger(
        save_dir=output_dir / "logs",
        name="fold_multiclass_classifier",
    )

    # ------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------

    accelerator = "gpu" if t.cuda.is_available() else "cpu"

    trainer = L.Trainer(
        max_epochs=150,
        accelerator=accelerator,
        devices=1,
        precision="32-true",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(
        model=lit_model,
        datamodule=data_module,
    )

    print("Best checkpoint:")
    print(checkpoint_callback.best_model_path)

    if checkpoint_callback.best_model_path:
        checkpoint = t.load(
            checkpoint_callback.best_model_path,
            map_location="cpu",
        )
        lit_model.load_state_dict(checkpoint["state_dict"])

    save_model(lit_model.even_model, output_dir / "fold_even")
    save_model(lit_model.odd_model, output_dir / "fold_odd")
    save_model(lit_model.combined_model, output_dir)

    # ------------------------------------------------------------
    # Inference example
    # ------------------------------------------------------------

    X_new = X[:10]
    parity_new = parity[:10]

    X_new_with_parity = np.concatenate(
        [
            parity_new.reshape(-1, 1).astype(np.float32),
            X_new.astype(np.float32),
        ],
        axis=1,
    )

    X_new_with_parity = t.tensor(X_new_with_parity, dtype=t.float32)

    lit_model.eval()

    with t.no_grad():
        logits = lit_model(X_new_with_parity)
        probs = lit_model.predict_proba(X_new_with_parity)
        pred_class = lit_model.predict_class(X_new_with_parity)

    print("Logits:")
    print(logits.cpu().numpy())

    print("Probabilities:")
    print(probs.cpu().numpy())

    print("Predicted classes:")
    print(pred_class.cpu().numpy())

    return output_dir


def main():
    train_fraction_classifier()
