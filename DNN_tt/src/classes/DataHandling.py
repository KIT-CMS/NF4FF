from pathlib import Path
import yaml
import pandas as pd
import operator
from functools import reduce
from classes.helper import _component_collection, _same_sign_opposite_sign_split
import numpy as np
from sklearn.model_selection import train_test_split


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

    def __init__(self, df, mask):
        self._df = df
        self._mask = mask

    @property
    def events(self):
        return self._df.loc[self._mask]

    @property
    def n(self):
        return self._mask.sum()

    def __getitem__(self, key):
        return self._df.loc[self._mask, key]

    def __setitem__(self, key, value):
        """Assign values to the underlying dataframe for masked rows."""
        self._df.loc[self._mask, key] = value

    def copy(self):
        """Return a plain DataFrame copy of the masked events."""
        return self._df.loc[self._mask].copy()

    def __getattr__(self, name):

        if name in self._df.columns:
            return self._df.loc[self._mask, name]

        return getattr(self._df.loc[self._mask], name)


class ProcessView:

    def __init__(self, df, process_mask, manager):
        self._df = df
        self._process_mask = process_mask
        self._manager = manager
        self._cache = {}

    def mask(self, region):

        if region not in self._cache:

            region_mask = self._manager.get_region_mask(self._df, region)

            self._cache[region] = self._process_mask & region_mask

        return self._cache[region]

    def __getattr__(self, name):

        if name in self._manager.regions:
            return RegionView(self._df, self.mask(name))

        return getattr(self._df, name)

    @property
    def events(self):
        return self._df.loc[self._process_mask]

    def __getitem__(self, key):
        return self._df.loc[self._process_mask, key]


class AnalysisDataFrame:

    def __init__(self, df, manager):

        self._df = df
        self._manager = manager

        self._region_cache = {}
        self._process_cache = {}

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
                self._manager
            )

        return self._process_cache[name]

    def __getattr__(self, name):


        if name in self._manager.regions:
            return RegionView(self._df, self.mask(name))


        if name in self._manager.processes:
            return self.process(name)

        return getattr(self._df, name)
    

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
            return RegionView(self._df, self.mask(key))

        #
        # Fall back to pandas column access
        #
        return self._df[key]

    @property
    def events(self):
        return self._df


def load_variables(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    yaml_vars = config.get("variables", [])
    return yaml_vars


def load_data(feather_file, config_file):

    df = pd.read_feather(feather_file)

    manager = SelectionManager(config_file)

    return AnalysisDataFrame(df, manager)


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

        sig_yield = np.sum(w_sig)
        bkg_yield = np.sum(w_bkg)


        sig_scale = 1.0
        bkg_scale = 1.0

        if sig_yield > 0:
            sig_scale = 1.0 / sig_yield
        if bkg_yield > 0:
            bkg_scale = 1.0 / bkg_yield


        w_sig = w_sig * sig_scale
        w_bkg = w_bkg * bkg_scale


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

def create_training_dataset(
    df_sig,
    df_bkg,
    training_var,
    weight_column="weight",
    balance=True,
    test_size=0.25,
    random_state=42,
):

    dataset = training_data(
        df_sig=df_sig,
        df_bkg=df_bkg,
        training_var=training_var,
        weight_column=weight_column,
        balance=balance,
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