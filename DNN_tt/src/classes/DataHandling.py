from pathlib import Path
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

    def __len__(self):
        return int(self._mask.sum())

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

    def __len__(self):
        return int(self._process_mask.sum())

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
        return self._df.loc[self._process_mask]

    def __getitem__(self, key):

        if isinstance(key, str):
            self._parent.ensure_column(key)

        return self._df.loc[self._process_mask, key]


class AnalysisDataFrame:

    def __init__(self, df, manager):

        self._df = df
        self._manager = manager

        self._region_cache = {}
        self._process_cache = {}
        self._loaded_feature_files = set()

    def __len__(self):
        return len(self._df)
        
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

        feature_cols = [c for c in feat.columns if c not in {"event", "row_index"}]
        if len(feature_cols) == 0:
            self._loaded_feature_files.add(path)
            return

        #
        # Prefer exact row-index keyed assignment when the feature file was
        # produced for this dataframe. Fall back to duplicate-safe event
        # mapping for legacy feature files.
        #
        if "row_index" in feat.columns and feat["row_index"].is_unique:
            feat_indexed = feat.set_index("row_index")
            for col in feature_cols:
                self._df[col] = self._df.index.to_series().map(feat_indexed[col])
        else:
            feat_compact = (
                feat[["event"] + feature_cols]
                .groupby("event", as_index=False, sort=False)
                .last()
                .set_index("event")
            )

            for col in feature_cols:
                self._df[col] = self._df["event"].map(feat_compact[col])

        self._loaded_feature_files.add(path)


    def ensure_column(self, column):

        #
        # Already present
        #
        if column in self._df.columns:
            return

        #
        # Feature column?
        #
        if column in self._manager.feature_columns:

            path = self._manager.feature_columns[column]

            self.load_feature_file(path)



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
):

    X_sig = df_sig[training_var].to_numpy(dtype=np.float32)
    w_sig = df_sig[weight_column].to_numpy(dtype=np.float32)

    X_bkg = df_bkg[training_var].to_numpy(dtype=np.float32)
    w_bkg = df_bkg[weight_column].to_numpy(dtype=np.float32)

    y_sig = np.ones(df_sig.shape[0], dtype=np.float32)
    y_bkg = np.zeros(df_bkg.shape[0], dtype=np.float32)


    if balance:
        # Normalize SR to AR, preferably per njets category like in FF calculation.
        if "tau_decaymode_2" in df_sig.columns and "tau_decaymode_2" in df_bkg.columns:

            njets_sig = df_sig["tau_decaymode_2"].to_numpy(dtype=np.float32)
            njets_bkg = df_bkg["tau_decaymode_2"].to_numpy(dtype=np.float32)

            # Categories:
            group_defs = (
                (lambda x: x == 0),
                (lambda x: x == 1),
                (lambda x: x == 10),
                (lambda x: x == 11),
            )

            w_sig_scaled = w_sig.copy()

            for group_fn in group_defs:
                sig_mask = group_fn(njets_sig)
                bkg_mask = group_fn(njets_bkg)

                sig_yield = np.sum(w_sig[sig_mask])
                bkg_yield = np.sum(w_bkg[bkg_mask])

                if sig_yield > 0:
                    sig_scale = bkg_yield / sig_yield if bkg_yield > 0 else 0.0
                    w_sig_scaled[sig_mask] = w_sig[sig_mask] * sig_scale

            w_sig = w_sig_scaled

        else:
            sig_yield = np.sum(w_sig)
            bkg_yield = np.sum(w_bkg)

            if sig_yield > 0:
                sig_scale = bkg_yield / sig_yield if bkg_yield > 0 else 0.0
                w_sig = w_sig * sig_scale


    X = np.concatenate([X_sig, X_bkg], axis=0)
    Y = np.concatenate([y_sig, y_bkg], axis=0)
    weights = np.concatenate([w_sig, w_bkg], axis=0)

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


#used
def create_training_dataset(
    df_sig,
    df_bkg,
    training_var,
    weight_column="weight",
    balance=True,
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
    )

    val_dataset = training_data(
        df_sig=df_sig_val,
        df_bkg=df_bkg_val,
        training_var=training_var,
        weight_column=weight_column,
        balance=False,
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
