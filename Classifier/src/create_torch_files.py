import numpy as np
import pandas as pd
import torch as t
import CODE.HELPER as helper
import logging
import random
from dataclasses import KW_ONLY, dataclass
from typing import (Any, Callable, Dict, Iterable, List, Protocol, Tuple,
                    Union, runtime_checkable)
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple, Type, Union, get_args, get_origin)
from sklearn.model_selection import train_test_split

from CustomLogging import setup_logging

# ----- seed -----

SEED = 42

t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ------ logger -----

logger = setup_logging(logger=logging.getLogger(__name__))

# ----- data classes -----

@dataclass
class _same_sign_opposite_sign_split(metaclass=helper.CollectionMeta):
    ss: Union[t.Tensor, pd.DataFrame, np.ndarray]
    os: Union[t.Tensor, pd.DataFrame, np.ndarray]

@dataclass
class _component_collection(metaclass=helper.CollectionMeta):
    _: KW_ONLY
    X: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    Y: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    process: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None


# ----- functions -----

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainval, test = train_test_split(
        df, test_size=0.5, random_state=SEED
    )
    train, val = train_test_split(
        trainval, test_size=0.5, random_state=SEED
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def get_my_data(df, training_var):
    _df = df  # fold/fold_train/fold_val to load, should contain SS/OS columns
    ss_os_split = _same_sign_opposite_sign_split(
            ss=_df[(_df.SS)],
            os=_df[(_df.OS)],
        )

    return _component_collection(
            X=ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype = np.float32)),
            Y=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype = np.float32)),  # or ss_os_split.apply_func(extract_label)
            # instead of _same_sign_opposite_sign_split.apply(lambda x: x["Label"].to_numpy()).to_collection(ss_os_split)
            weights=ss_os_split.apply_func(lambda __df: __df["weight"].to_numpy(dtype = np.float32)),
            #class_weights=ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
            process=ss_os_split.apply_func(lambda x: x['process'].to_numpy(dtype = np.float32))
        )


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

# ----- lists -----

dim = 20
variables = [
    "pt_1","pt_2","eta_1","eta_2","jpt_1","jpt_2","jeta_1","jeta_2",
    "m_fastmtt","pt_fastmtt","met","njets","nbtag","mt_tot","m_vis",
    "pt_tt","pt_vis","mjj","pt_dijet","pt_ttjj","pzetamissvis","deltaEta_jj",
    "deltaEta_ditaupair","deltaR_jj","deltaR_ditaupair","deltaR_1j1","deltaR_1j2",
    "deltaR_2j1","deltaR_2j2","deltaR_12j1","deltaR_12j2","deltaEta_1j1",
    "deltaEta_1j2","deltaEta_2j1","deltaEta_2j2","deltaEta_12j1","deltaEta_12j2",
][:dim]

# ---------------

def main():

    logger.info('Loading pd.Dataframes ...')

    train1_df = pd.read_feather('../data/train1.feather')
    val1_df = pd.read_feather('../data/val1.feather')
    train2_df = pd.read_feather('../data/train2.feather')
    val2_df = pd.read_feather('../data/val2.feather')


    train1_n0 = train1_df[train1_df.njets == 0]
    val1_n0   = val1_df[val1_df.njets == 0]
    train2_n0 = train2_df[train2_df.njets == 0]
    val2_n0   = val2_df[val2_df.njets == 0]

    train1_n1 = train1_df[train1_df.njets == 1]
    val1_n1   = val1_df[val1_df.njets == 1]
    train2_n1 = train2_df[train2_df.njets == 1]
    val2_n1   = val2_df[val2_df.njets == 1]

    train1_n2 = train1_df[train1_df.njets >= 2]
    val1_n2   = val1_df[val1_df.njets >= 2]
    train2_n2 = train2_df[train2_df.njets >= 2]
    val2_n2   = val2_df[val2_df.njets >= 2]

    logger.info('Creating torch files')

    train1_pt_inclusive = get_my_data(train1_df, variables).to_torch(device="cuda")
    val1_pt_inclusive   = get_my_data(val1_df, variables).to_torch(device="cuda")
    train2_pt_inclusive = get_my_data(train2_df, variables).to_torch(device="cuda")
    val2_pt_inclusive   = get_my_data(val2_df, variables).to_torch(device="cuda")

    train1_pt_n0    = get_my_data(train1_n0 , variables).to_torch(device="cuda")
    val1_pt_n0      = get_my_data(val1_n0   , variables).to_torch(device="cuda")
    train2_pt_n0    = get_my_data(train2_n0 , variables).to_torch(device="cuda")
    val2_pt_n0      = get_my_data(val2_n0   , variables).to_torch(device="cuda")

    train1_pt_n1    = get_my_data(train1_n1 , variables).to_torch(device="cuda")
    val1_pt_n1      = get_my_data(val1_n1 , variables).to_torch(device="cuda")
    train2_pt_n1    = get_my_data(train2_n1 , variables).to_torch(device="cuda")
    val2_pt_n1      = get_my_data(val2_n1 , variables).to_torch(device="cuda")

    train1_pt_n2    = get_my_data(train1_n2 , variables).to_torch(device="cuda")
    val1_pt_n2      = get_my_data(val1_n2 , variables).to_torch(device="cuda")
    train2_pt_n2    = get_my_data(train2_n2 , variables).to_torch(device="cuda")
    val2_pt_n2      = get_my_data(val2_n2 , variables).to_torch(device="cuda")


    logger.info('Saving torch files: n inclusve')

    t.save(train1_pt_inclusive  , '../data/inclusive/train1.pt')
    t.save(val1_pt_inclusive    , '../data/inclusive/val1.pt')
    t.save(train2_pt_inclusive  , '../data/inclusive/train2.pt')
    t.save(val2_pt_inclusive    , '../data/inclusive/val2.pt')

    logger.info('Saving torch files: n = 0')

    t.save(train1_pt_n0  , '../data/njets0/train1.pt')
    t.save(val1_pt_n0    , '../data/njets0/val1.pt')
    t.save(train2_pt_n0  , '../data/njets0/train2.pt')
    t.save(val2_pt_n0    , '../data/njets0/val2.pt')

    logger.info('Saving torch files: n = 1')

    t.save(train1_pt_n1  , '../data/njets1/train1.pt')
    t.save(val1_pt_n1    , '../data/njets1/val1.pt')
    t.save(train2_pt_n1  , '../data/njets1/train2.pt')
    t.save(val2_pt_n1    , '../data/njets1/val2.pt')

    logger.info('Saving torch files: n >= 2')

    t.save(train1_pt_n2  , '../data/njets2/train1.pt')
    t.save(val1_pt_n2    , '../data/njets2/val1.pt')
    t.save(train2_pt_n2  , '../data/njets2/train2.pt')
    t.save(val2_pt_n2    , '../data/njets2/val2.pt')

# ------------------

if __name__ == "__main__":
    main()
