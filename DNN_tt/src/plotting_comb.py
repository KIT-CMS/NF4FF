from pathlib import Path
import logging
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tap import Tap
import torch as t
from typing import Literal

from classes.Loading import load_config, load_labels, load_data



SEED = 42
logger = logging.getLogger(__name__)


t.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
t.set_num_threads(8)

class Args(Tap):
    embedding: Literal["embedding", "no_embedding"] = "embedding"
    var = "variables"

    taus: Literal['split', 'incl'] = 'split' # split: calc 2 FF for tau1 and tau2 | incl: calc only 1 FF
    incl: Literal['and', 'or'] = 'or' # Combine tau1 and tau2 AR with and or or
    dnn_grouped: bool = False
    classic: bool = False

    closure_DR: bool = True
    FF_dist: bool = True
    closure_AR: bool = True

args = Args().parse_args()

cfg_path = load_config('/work/tapp/TauFF/NF4FF/DNN_tt/configs/config_path.yaml')

DATA_PATH = f'{cfg_path["datasets"]}/{args.embedding}/combined_data_updated.feather'
DATA_CLASSIC_JV_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_jvoss.feather"
DATA_CLASSIC_SG_PATH = "/work/tapp/TauFF/NF4FF/Data/datasets/classic/combined_data_sgiappic.feather"
MASKS_PATH = cfg_path["masks"]
MASKS_PATH_INCL = [cfg_path["masks_incl_and"], cfg_path["masks_incl_or"]]
TRAINING_VAR_PATH = cfg_path["train_var"]
NN_CONFIG_PATH = cfg_path["DNN"]
CHECKPOINT_DIR = cfg_path["traininfg_results"]

PLOTTING_CONFIG_PATH = cfg_path["cfg_plotting"]
LABELS_CONFIG_PATH = cfg_path["labels"]

#PLOTS_DIR = Path('../plots/layers_3/ReLU')
PLOTS_DIR = Path(cfg_path["plots"])
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_GROUPINGS = ('njets', 'tau_decaymode', 'ungrouped', 'classic')
PLOT_SUBDIRS = ('closure_in_DR', 'FF_distribution_AR', 'FF_distribution_DR', 'closure_plots')
for subdir in PLOT_SUBDIRS:
    for grouping in PLOT_GROUPINGS:
        (PLOTS_DIR / 'combinations' / subdir / grouping).mkdir(parents=True, exist_ok=True)

PLOT_GROUPINGS = ('njets', 'tau_decaymode')


matplotlib.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

PLOTTING_CFG = load_config(PLOTTING_CONFIG_PATH)
LABELS_CFG = load_labels(LABELS_CONFIG_PATH)

VARIABLES_SMALL = PLOTTING_CFG.get('variables_set_small', [])
VARIABLES_LARGE = PLOTTING_CFG.get('variables_set_large', [])


def get_bins(variable):
    bin_spec = PLOTTING_CFG.get('bins_by_variable', {}).get(variable)
    if bin_spec is None:
        raise KeyError(f'No bin specification found for variable: {variable}')

    if isinstance(bin_spec, (list, tuple)) and len(bin_spec) == 3:
        start, stop, num = bin_spec
        return np.linspace(float(start), float(stop), int(num))

    return np.asarray(bin_spec, dtype=float)


def get_label(variable, channel='et'):
    labels_by_channel = LABELS_CFG.get(channel, {}) if isinstance(LABELS_CFG, dict) else {}
    return labels_by_channel.get(variable, variable)


def get_bins_and_label(variable, channel='et'):
    return get_bins(variable), get_label(variable, channel)


# ------------------------------------------------------

def main():
    df = load_data(DATA_PATH, MASKS_PATH)

    if args.incl=='and': incl = 0
    elif args.incl=='or': incl = 1
    else:
        logger.error(f'Value Error: args.incl = {args.incl}, but only accepts "and or "or".')
        exit()
    df_incl = load_data(DATA_PATH, MASKS_PATH_INCL[incl])
    

# -------------------------------------------

if __name__ == '__main__':
    main()