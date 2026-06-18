from pathlib import Path
import fcntl
import os
import yaml
import pandas as pd
import operator
from functools import reduce
from classes.helper import _component_collection, _same_sign_opposite_sign_split
import numpy as np
import torch as t
from sklearn.model_selection import train_test_split
from typing import Union

class SelectionManager:

    def __init__(self, yaml_path):

        self.yaml_path = Path(yaml_path)

        with open(self.yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        self.raw_masks = raw.get("masks", {})
        self.regions = raw.get("regions", {})
        self.processes = raw.get("processes", {})

        self.masks = {
            name: self._normalize(conds)
            for name, conds in self.raw_masks.items()
        }
        self.features = raw.get("features", {})
        self.feature_columns = {}

        for feature_name, info in self.features.items():
            path = info["path"]

            for col in info["columns"]:
                self.feature_columns[col] = path

    @staticmethod
    def _normalize(conditions):

        fixed = []

        for c in conditions:
            c = (
                c.replace("&gt;", ">")
                 .replace("&lt;", "<")
                 .replace("&&", "&")
            )
            fixed.append(f"({c})")

        return " & ".join(fixed)


    def get_mask(self, df, name):
        return df.eval(self.masks[name], engine="python")


    def get_region_mask(self, df, region):

        combined = pd.Series(True, index=df.index)

        for m in self.regions[region]:
            combined &= self.get_mask(df, m)

        return combined


    def get_process_mask(self, df, process):

        expr = self.processes[process]
        return df.eval(expr, engine="python")


class RegionView:

    def __init__(self, df, mask, parent):
        self._df = df
        self._mask = mask
        self._parent = parent

    def _current_df(self):
        """Always use the latest dataframe held by the parent."""
        return self._parent._df

    @property
    def events(self):
        return self._current_df().loc[self._mask]

    @property
    def n(self):
        return self._mask.sum()

    def __getitem__(self, key):

        if isinstance(key, str):
            self._parent.ensure_column(key)

        return self._current_df().loc[self._mask, key]

    def __getattr__(self, name):

        #
        # Trigger lazy feature loading
        #
        self._parent.ensure_column(name)

        current_df = self._current_df()

        if name in current_df.columns:
            return current_df.loc[self._mask, name]

        return getattr(current_df.loc[self._mask], name)

    def __setitem__(self, key, value):
        """Assign values to the underlying dataframe for masked rows."""
        self._current_df().loc[self._mask, key] = value

    def copy(self):
        """Return a plain DataFrame copy of the masked events."""
        return self._current_df().loc[self._mask].copy()


class ProcessView:

    def __init__(self, df, process_mask, manager, parent):
        self._df = df
        self._process_mask = process_mask
        self._manager = manager
        self._parent = parent
        self._cache = {}

    def mask(self, region):

        if region not in self._cache:

            region_mask = self._manager.get_region_mask(self._df, region)

            self._cache[region] = self._process_mask & region_mask

        return self._cache[region]

    def __getattr__(self, name):

        #
        # Region access
        #
        if name in self._manager.regions:
            return RegionView(
                self._df,
                self.mask(name),
                self._parent
            )

        #
        # Lazy feature loading
        #
        self._parent.ensure_column(name)

        current_df = self._parent._df
        if name in current_df.columns:
            return current_df.loc[self._process_mask, name]

        return getattr(current_df.loc[self._process_mask], name)

    @property
    def events(self):
        return self._parent._df.loc[self._process_mask]

    def __getitem__(self, key):

        if isinstance(key, str):
            self._parent.ensure_column(key)

        return self._parent._df.loc[self._process_mask, key]


class AnalysisDataFrame:
    """DataFrame wrapper with configured region and process views.

    Process attributes always apply their mask: ``df.data`` contains only
    process 0, while ``df.full`` is the explicit all-process view.
    """

    def __init__(self, df, manager, resolver):
        self._df = df
        self._manager = manager
        self._resolver = resolver

        self._region_cache = {}
        self._process_cache = {}
        self._loaded_feature_files = set()
    def mask(self, region):

        if region not in self._region_cache:

            self._region_cache[region] = (
                self._manager.get_region_mask(self._df, region)
            )

        return self._region_cache[region]


    def process(self, name):

        if name not in self._process_cache:

            pmask = self._manager.get_process_mask(self._df, name)

            self._process_cache[name] = ProcessView(
                self._df,
                pmask,
                self._manager,
                self)

        return self._process_cache[name]

    def __getattr__(self, name):
        if name in self._manager.regions:
            return RegionView(
                self._df,
                self.mask(name),
                self
            )

        if name in self._manager.processes:
            return self.process(name)

        if hasattr(self._df, name):
            return getattr(self._df, name)

        self.ensure_column(name)

        return self._df[name]
    

    def __getitem__(self, key):

        #
        # Process access
        #
        if key in self._manager.processes:
            return self.process(key)

        #
        # Region access
        #
        if key in self._manager.regions:
            return RegionView(
                self._df,
                self.mask(key),
                self
            )
        #
        # Lazy feature loading
        #
        if isinstance(key, str):
            self.ensure_column(key)

        return self._df[key]


    @property
    def events(self):
        return self._df

    def subset(self, mask):
        """Return a filtered analysis dataframe with fresh region/process views."""
        if isinstance(mask, pd.Series):
            mask = (
                mask.reindex(self._df.index, fill_value=False)
                .fillna(False)
                .astype(bool)
            )
        else:
            mask = np.asarray(mask, dtype=bool)
            if len(mask) != len(self._df):
                raise ValueError(
                    "Subset mask length does not match the dataframe length."
                )

        subset_df = self._df.loc[mask].copy()
        return AnalysisDataFrame(
            subset_df,
            self._manager,
            self._resolver,
        )

    def load_feature_file(self, path):

        #
        # Avoid duplicate loading
        #
        if path in self._loaded_feature_files:
            return

        feat = pd.read_feather(path)

        #
        # Prevent duplicate columns
        #
        new_cols = [
            c for c in feat.columns
            if c not in self._df.columns
            or c == "event"
        ]

        feat = feat[new_cols]

        key_column = "row_index" if "row_index" in feat.columns else "event"
        feature_cols = [
            c for c in feat.columns
            if c not in ("event", "row_index")
        ]
        if len(feature_cols) == 0:
            self._loaded_feature_files.add(path)
            return

        #
        # Duplicate-safe lazy loading:
        # collapse feature file to one row per event and map values to the
        # existing dataframe without changing row count/order.
        #
        feat_compact = (
            feat[[key_column] + feature_cols]
            .groupby(key_column, as_index=False, sort=False)
            .last()
            .set_index(key_column)
        )

        for col in feature_cols:
            if key_column == "row_index":
                self._df[col] = self._df.index.to_series().map(feat_compact[col])
            else:
                self._df[col] = self._df["event"].map(feat_compact[col])

        self._loaded_feature_files.add(path)


    def ensure_column(self, column):

        if column in self._df.columns:
            return

        self._df[column] = self._resolver.resolve(column, self._df)

def load_variables(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    yaml_vars = config.get("variables", [])
    return yaml_vars


def load_data(feather_file, config_file, feature_registry_path=None):

    df = pd.read_feather(feather_file)

    manager = SelectionManager(config_file)

    if feature_registry_path is None:
        feature_registry_path = (
            Path(feather_file).resolve().parent
            / "features"
            / "feature_registry.json"
        )
    registry = FeatureRegistry(feature_registry_path)
    resolver = FeatureResolver(registry)

    return AnalysisDataFrame(df, manager, resolver)


import json
from pathlib import Path
from collections import defaultdict


class FeatureRegistry:

    def __init__(self, path="feature_registry.json"):
        self.path = Path(path)

        if self.path.exists():
            self.index = json.loads(self.path.read_text())
        else:
            self.index = {}  # column -> file
        self._updates = {}
        self._removed = {}

    def get_file(self, column):
        return self.index.get(column)

    def register(self, columns, file_path):
        for c in columns:
            path = str(file_path)
            self.index[c] = path
            self._updates[c] = path
            self._removed.pop(c, None)

    def remove(self, column, expected_path=None):
        self.index.pop(column, None)
        self._updates.pop(column, None)
        self._removed[column] = (
            None if expected_path is None else str(expected_path)
        )

    def replace_file_columns(self, file_path, columns):
        stored_path = str(file_path)
        columns = set(columns)
        for column, path in tuple(self.index.items()):
            if path == stored_path and column not in columns:
                self.remove(column, expected_path=stored_path)
        self.register(columns, stored_path)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        temporary_path = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if self.path.exists():
                current = json.loads(self.path.read_text())
            else:
                current = {}
            for column, expected_path in self._removed.items():
                if (
                    expected_path is None
                    or current.get(column) == expected_path
                ):
                    current.pop(column, None)
            current.update(self._updates)
            temporary_path.write_text(json.dumps(current, indent=2))
            os.replace(temporary_path, self.path)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        self.index = current
        self._updates.clear()
        self._removed.clear()

class FeatureStore:

    def __init__(self, path, registry: FeatureRegistry):
        self.path = Path(path)
        self.registry = registry

        if self.path.exists():
            self.df = pd.read_feather(self.path)
        else:
            self.df = pd.DataFrame(columns=["event"])

    def upsert(self, df):
        df = self._normalize(df)
        key_column = "row_index" if "row_index" in df.columns else "event"

        if self.df.empty:
            self.df = df.copy()
        else:
            current = self.df.set_index(key_column)
            incoming = df.set_index(key_column)

            for col in incoming.columns:
                if col not in current.columns:
                    current[col] = np.nan

            current.update(incoming)

            missing_idx = incoming.index.difference(current.index)
            if len(missing_idx) > 0:
                current = pd.concat([current, incoming.loc[missing_idx]], axis=0)

            self.df = current.reset_index()

        feature_columns = [
            c for c in df.columns
            if c not in ("event", "row_index")
        ]
        self.registry.register(feature_columns, self.path)

    def write(self, df):
        df = self._normalize(df)
        self.df = df

        feature_columns = [
            c for c in df.columns
            if c not in ("event", "row_index")
        ]
        self.registry.replace_file_columns(self.path, feature_columns)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_feather(self.path)

    def _normalize(self, df):
        key_column = "row_index" if "row_index" in df.columns else "event"
        return df.groupby(key_column, as_index=False).last()

class FeatureResolver:

    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self.file_cache = {}   # file → dataframe
        self.col_cache = {}    # column → series

    def resolve(self, column, base_df):
        if column in base_df.columns:
            return base_df[column]

        if column in self.col_cache:
            series = self.col_cache[column]
            if series.index.name == "row_index":
                return base_df.index.to_series().map(series)
            return base_df["event"].map(series)

        file = self.registry.get_file(column)

        if file is None:
            raise KeyError(f"Unknown feature: {column}")

        if file not in self.file_cache:
            feature_df = pd.read_feather(file)
            key_column = (
                "row_index"
                if "row_index" in feature_df.columns
                else "event"
            )
            self.file_cache[file] = feature_df.set_index(key_column)

        df = self.file_cache[file]

        if column not in df.columns:
            raise KeyError(f"{column} not in {file}")

        series = df[column]

        self.col_cache[column] = series

        if df.index.name == "row_index":
            return base_df.index.to_series().map(series)
        return base_df["event"].map(series)


def write_features(
    base_df,
    output_file,
    columns_dict
):

    feat_df = pd.DataFrame({
        "event": base_df.event
    })

    for name, values in columns_dict.items():
        feat_df[name] = values

    feat_df.to_feather(output_file)


def append_features(
    feature_file,
    base_df,
    new_columns
):

    feat = pd.read_feather(feature_file)

    if "event" not in feat.columns:
        raise ValueError("feature_file must contain event")

    if "event" not in base_df.columns:
        raise ValueError("base_df must contain event")

    #
    # Create dataframe from new columns
    #
    new_df = pd.DataFrame({
        "event": base_df["event"]
    })

    for name, values in new_columns.items():
        new_df[name] = values

    #
    # Nothing to append
    #
    append_columns = [c for c in new_df.columns if c != "event"]
    if len(append_columns) == 0:
        return

    #
    # Reduce new_df to one value per event to avoid many-to-many row explosion
    # when event is duplicated in either dataframe.
    #
    new_compact = (
        new_df[["event"] + append_columns]
        .groupby("event", as_index=False, sort=False)
        .last()
        .set_index("event")
    )

    #
    # Append/overwrite columns by mapping through event.
    # Unmatched events remain NaN (same behavior as left-merge append).
    #
    for col in append_columns:
        feat[col] = feat["event"].map(new_compact[col])

    feat.to_feather(feature_file)

def update_features(feature_file, update_df):

    feat = pd.read_feather(feature_file)

    #
    # Ensure event exists
    #
    if "event" not in feat.columns:
        raise ValueError("feature_file must contain event")

    if "event" not in update_df.columns:
        raise ValueError("update_df must contain event")

    #
    # Nothing to update
    #
    update_columns = [c for c in update_df.columns if c != "event"]
    if len(update_columns) == 0:
        return

    #
    # Reduce update_df to one value per event to avoid ambiguous duplicate-index updates.
    # For duplicate events, keep the last non-null value per column.
    #
    update_compact = (
        update_df[["event"] + update_columns]
        .groupby("event", as_index=False, sort=False)
        .last()
    )

    update_compact = update_compact.set_index("event")

    #
    # Update only rows whose event exists in update_df.
    # This is duplicate-safe for both feat.event and update_df.event.
    #
    for col in update_columns:
        mapped_values = feat["event"].map(update_compact[col])

        if col in feat.columns:
            feat[col] = mapped_values.where(mapped_values.notna(), feat[col])
        else:
            feat[col] = mapped_values

    #
    # Save
    #
    feat.to_feather(feature_file)


def training_data(
    df_sig,
    df_bkg,
    training_var,
    weight_column="weight",
    balance=True,
    balance_column=None,
    balance_groups=None,
    balance_with_absolute_yields=False,
):

    X_sig = df_sig[training_var].to_numpy(dtype=np.float32)
    w_sig = df_sig[weight_column].to_numpy(dtype=np.float64)

    X_bkg = df_bkg[training_var].to_numpy(dtype=np.float32)
    w_bkg = df_bkg[weight_column].to_numpy(dtype=np.float64)

    y_sig = np.ones(df_sig.shape[0], dtype=np.float32)
    y_bkg = np.zeros(df_bkg.shape[0], dtype=np.float32)

    if not np.isfinite(X_sig).all() or not np.isfinite(X_bkg).all():
        raise ValueError("Training features contain non-finite values.")
    if not np.isfinite(w_sig).all() or not np.isfinite(w_bkg).all():
        raise ValueError(
            f"Training weight column '{weight_column}' contains "
            "non-finite values."
        )

    if balance:
        if (
            balance_column is not None
            and balance_column in df_sig.columns
            and balance_column in df_bkg.columns
        ):
            group_values_sig = df_sig[balance_column].to_numpy()
            group_values_bkg = df_bkg[balance_column].to_numpy()
            if balance_groups is not None:
                group_defs = tuple(
                    (
                        (lambda values, value=group[0]: values == value)
                        if len(group) == 1
                        else (
                            lambda values, low=group[0], high=group[1]:
                            (values >= low) & (values <= high)
                        )
                    )
                    for group in balance_groups
                )
            elif balance_column == "tau_decaymode_2":
                group_defs = (
                    lambda values: values == 0,
                    lambda values: values == 1,
                    lambda values: values == 10,
                    lambda values: values == 11,
                )
            elif balance_column == "njets":
                group_defs = (
                    lambda values: values == 0,
                    lambda values: values == 1,
                    lambda values: values >= 2,
                )
            else:
                unique_values = np.union1d(
                    group_values_sig,
                    group_values_bkg,
                )
                group_defs = tuple(
                    lambda values, current=value: values == current
                    for value in unique_values
                )

            w_sig_scaled = w_sig.copy()

            for group_fn in group_defs:
                sig_mask = group_fn(group_values_sig)
                bkg_mask = group_fn(group_values_bkg)
                if not sig_mask.any():
                    continue

                if balance_with_absolute_yields:
                    sig_yield = np.abs(w_sig[sig_mask]).sum(dtype=np.float64)
                    bkg_yield = np.abs(w_bkg[bkg_mask]).sum(dtype=np.float64)
                else:
                    sig_yield = w_sig[sig_mask].sum(dtype=np.float64)
                    bkg_yield = w_bkg[bkg_mask].sum(dtype=np.float64)

                if sig_yield != 0:
                    sig_scale = bkg_yield / sig_yield
                    w_sig_scaled[sig_mask] = (
                        w_sig[sig_mask] * sig_scale
                    )

            w_sig = w_sig_scaled

        else:
            if balance_with_absolute_yields:
                sig_yield = np.abs(w_sig).sum(dtype=np.float64)
                bkg_yield = np.abs(w_bkg).sum(dtype=np.float64)
            else:
                sig_yield = w_sig.sum(dtype=np.float64)
                bkg_yield = w_bkg.sum(dtype=np.float64)

            if sig_yield != 0:
                sig_scale = bkg_yield / sig_yield
                w_sig = w_sig * sig_scale

    if not np.isfinite(w_sig).all() or not np.isfinite(w_bkg).all():
        raise ValueError(
            f"Balancing produced non-finite values in '{weight_column}'."
        )
    float32_max = np.finfo(np.float32).max
    max_absolute_weight = max(
        np.abs(w_sig).max(initial=0.0),
        np.abs(w_bkg).max(initial=0.0),
    )
    if max_absolute_weight > float32_max:
        raise ValueError(
            f"Balancing '{weight_column}' exceeded the float32 range."
        )

    X = np.concatenate([X_sig, X_bkg], axis=0)
    Y = np.concatenate([y_sig, y_bkg], axis=0)
    weights = np.concatenate([w_sig, w_bkg], axis=0).astype(np.float32)

    idx = np.random.permutation(len(X))

    return _component_collection(
        X=X[idx],
        Y=Y[idx],
        weights=weights[idx],
    )


def test_data(
    df_test,
    training_var,
    ):

    _df = df_test[training_var].to_numpy(dtype=np.float32)
    
    return _component_collection(X = _df)


def create_training_dataset(
    df_sig,
    df_bkg,
    training_var,
    weight_column="weight",
    balance=True,
    balance_column=None,
    balance_groups=None,
    balance_with_absolute_yields=False,
    test_size=0.25,
    random_state=42,
):
    df_sig_train, df_sig_val = train_test_split(
        df_sig,
        test_size=test_size,
        random_state=random_state,
    )

    df_bkg_train, df_bkg_val = train_test_split(
        df_bkg,
        test_size=test_size,
        random_state=random_state,
    )

    train_dataset = training_data(
        df_sig=df_sig_train,
        df_bkg=df_bkg_train,
        training_var=training_var,
        weight_column=weight_column,
        balance=balance,
        balance_column=balance_column,
        balance_groups=balance_groups,
        balance_with_absolute_yields=balance_with_absolute_yields,
    )

    val_dataset = training_data(
        df_sig=df_sig_val,
        df_bkg=df_bkg_val,
        training_var=training_var,
        weight_column=weight_column,
        balance=False,
        balance_column=balance_column,
        balance_groups=balance_groups,
        balance_with_absolute_yields=balance_with_absolute_yields,
    )

    X_train = train_dataset.X
    y_train = train_dataset.Y
    w_train = train_dataset.weights

    X_val = val_dataset.X
    y_val = val_dataset.Y
    w_val = val_dataset.weights

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


def estimate_qcd_in_bins(
	df,
	var: str,
	bins = np.ndarray,
):
	data = np.histogram(df.data.AR_SS[var], weights = df.data.AR_SS.weight, bins = bins)
	wjets = np.histogram(df.wjets.AR_SS[var], weights = df.wjets.AR_SS.weight, bins = bins )
	diboson = np.histogram(df.diboson.AR_SS[var], weights = df.diboson.AR_SS.weight, bins = bins )
	DYjets = np.histogram(df.DYjets.AR_SS[var], weights = df.DYjets.AR_SS.weight, bins = bins )
	ST = np.histogram(df.ST.AR_SS[var], weights = df.ST.AR_SS.weight, bins = bins)
	ttbar = np.histogram(df.ttbar.AR_SS[var], weights = df.ttbar.AR_SS.weight, bins = bins )
	embedding = np.histogram(df.embedding.AR_SS[var], weights = df.embedding.AR_SS.weight, bins = bins )

	qcd = data - wjets - diboson - DYjets - ST - ttbar - embedding

	return qcd


def get_class_weights(
    weights: Union[pd.Series, np.ndarray, t.Tensor],
    Y: Union[pd.Series, np.ndarray, t.Tensor],
    classes: tuple = (0, 1),
    class_weighted: bool = True,
) -> Union[pd.Series, np.ndarray, t.Tensor]:
    _weights = np.zeros_like(weights)
    for _class in classes:
        _weights[Y == _class] = weights.sum() / weights[Y == _class].sum()
    return _weights * (weights if class_weighted else 1.0)
