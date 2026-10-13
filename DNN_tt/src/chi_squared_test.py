import numpy as np
import torch as t
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import chi2
from copy import deepcopy
from classes import (
    load_data,
    load_variables,
    load_model,
    test_data,
    LikelihoodRatioCalculation,
    FoldCombinedDNN,
    FF_closure,
    EnsembleStatUncWrapper
)


TRAINING_DATASETS = Path('../training_datasets')
CHECKPOINT_DIR = Path('../Training_results_squeezed')

DATA_PATH = '../../data/data_complete.feather'
MASKS_PATH = '../configs/masks.yaml'
TRAINING_VAR_PATH = '../configs/training_variables.yaml'

PLOTTING_CONFIG_PATH = '../configs/plotting.yaml'
LABELS_CONFIG_PATH = '../configs/labels.yaml'

# Available methods:
# - per_fold_lrc: LikelihoodRatioCalculation on each fold model, then FoldCombinedDNN
# - post_fold_lrc: FoldCombinedDNN on score models, then LikelihoodRatioCalculation on top
FF_BUILD_METHOD = 'post_fold_lrc'

# Same normalization constants as in delme5.ipynb
NC_WJETS_NOTEBOOK = {
    ((0,),): 0.3950,
    ((1,),): 0.3284,
    ((10,),): 0.2163,
    ((11,),): 0.1050,
    'fallback': 1.0,
}


def _read_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _read_labels_yaml(path):
    labels_by_channel = {}
    current_channel = None

    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(' '))

            if not stripped or stripped.startswith('#'):
                continue

            # Be tolerant if the channel key is accidentally indented by one space.
            if stripped.endswith(':') and ':' not in stripped[:-1] and indent <= 1:
                current_channel = stripped[:-1]
                labels_by_channel.setdefault(current_channel, {})
                continue

            if current_channel is None:
                continue

            if indent < 4:
                continue

            key_value = line.strip().split(':', 1)
            if len(key_value) != 2:
                continue

            key, value = key_value
            labels_by_channel[current_channel][key] = value.strip().strip('"').strip("'")

    return labels_by_channel


PLOTTING_CFG = _read_yaml(PLOTTING_CONFIG_PATH)
LABELS_CFG = _read_labels_yaml(LABELS_CONFIG_PATH)


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


def chi2_func(d, m, d_unc, m_unc, ndof):
    variance = d_unc**2 + m_unc**2
    valid = variance > 0
    if not np.any(valid):
        return np.nan
    return float(np.sum((d[valid] - m[valid])**2 / variance[valid]) / ndof)

def chi2_to_pvalue(chi2_value, ndof):
    return chi2.sf(chi2_value, ndof)


def build_ff_model(
    even_model_path,
    odd_model_path,
    constants_even,
    constants_odd,
    method='per_fold_lrc',
    notebook_norm_constants=None,
):
    score_even = load_model(even_model_path).eval()
    score_odd = load_model(odd_model_path).eval()

    if method == 'per_fold_lrc':
        ff_even = LikelihoodRatioCalculation(
            model=score_even,
            normalization_constants=constants_even,
        )
        ff_odd = LikelihoodRatioCalculation(
            model=score_odd,
            normalization_constants=constants_odd,
        )

        return FoldCombinedDNN(
            even_model=ff_even,
            odd_model=ff_odd,
            fold_id_name='event',
        )

    if method == 'post_fold_lrc':
        combined_score_model = FoldCombinedDNN(
            even_model=score_even,
            odd_model=score_odd,
            fold_id_name='event',
        )
        return LikelihoodRatioCalculation(
            model=combined_score_model,
            normalization_constants=notebook_norm_constants,
            clip=(1e-4, 10.0),
        )

    raise ValueError(f'Unknown FF build method: {method}')


def equi_populated_bins(data, n_bins):

    data = np.asarray(data)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(data, quantiles)
    return bin_edges




def main():
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    df = load_data(DATA_PATH, MASKS_PATH)
    training_variables = load_variables(TRAINING_VAR_PATH)

    region = 'wjets'
    grouping_variable = 'tau_decaymode_2'
    squeezing_values = np.round(np.arange(0.85, 1.001, 0.01), 2)

    constants_even = t.load(TRAINING_DATASETS / region / 'fold_even' / 'constants.pt')
    constants_odd = t.load(TRAINING_DATASETS / region / 'fold_odd' / 'constants.pt')

    region_view_ar = df.data.AR_like_wjets
    region_view_sr = df.data.SR_like_wjets
    ar_events = region_view_ar.events
    sr_events = region_view_sr.events

    X_features = test_data(region_view_ar, training_variables).X.astype(np.float32)
    event_parity = (ar_events['event'].to_numpy() % 2).astype(np.float32)
    X_np = np.concatenate([event_parity.reshape(1, -1), X_features.T], axis=0)
    X_t = t.from_numpy(X_np).to(device)


    
    vars_to_test = ['pt_1', 'pt_2', 'pt_fastmtt', 'met', 'm_vis', 'm_fastmtt', # 'eta_1', 'eta_2', 'jeta_1', 'jeta_2',
                    'pt_vis', 'pt_tt', 'deltaR_ditaupair', 'jpt_1', 'jpt_2', 'mt_tot', 'njets', 'tau_decaymode_2']


    pvalue_list = []
    clipff_list = []
    clipff_list_up = []
    clipff_list_down = []
    ff_wjets_list = []
    chi2_by_var = {var: [] for var in vars_to_test}
    pvalue_by_var = {var: [] for var in vars_to_test}



    X_ar_like = t.from_numpy(test_data(df.data.AR_like_wjets, training_variables).X).to(device)
    event_parity = t.from_numpy((df.data.AR_like_wjets.event.to_numpy() % 2).astype(np.float32)).to(device)

    for squeezing in squeezing_values:
        print(f'current squeezing limit: {squeezing}')
        squeezing_label = f"{float(squeezing):.2f}"

        even_model_path = CHECKPOINT_DIR / squeezing_label / grouping_variable / region / 'fold_even'
        odd_model_path = CHECKPOINT_DIR / squeezing_label / grouping_variable / region / 'fold_odd'

        ff_model = build_ff_model(
            even_model_path=even_model_path,
            odd_model_path=odd_model_path,
            constants_even=constants_even,
            constants_odd=constants_odd,
            method=FF_BUILD_METHOD,
            notebook_norm_constants=NC_WJETS_NOTEBOOK,
        ).to(device).eval()

        with t.no_grad():
            ff_wjets_nominal = ff_model(X_t).detach().cpu().numpy().reshape(-1)

        batch_size = 50_000
        n_events = X_ar_like.shape[0]
        n_variations = 100
        ff_dropout_chunks = []


        ff_model_dropout = EnsembleStatUncWrapper(
            model=deepcopy(ff_model),
            ensemble_size=n_variations,
            direction="Nominal",
        ).to(device)
        ff_model_dropout.eval()

        with t.no_grad():
            for start in range(0, n_events, batch_size):
                end = min(start + batch_size, n_events)

                model_input_batch = t.cat([
                    event_parity[start:end].unsqueeze(0),
                    X_ar_like[start:end].T,
                ], dim=0)

                # Use the wrapped model outputs directly to keep all dropout-mask members.
                batch_all = ff_model_dropout.wrapped_model(
                    model_input_batch.repeat(1, n_variations + 1)
                )
                batch_all = batch_all.reshape(n_variations + 1, end - start, *batch_all.shape[1:])

                # Drop index 0 (deterministic nominal mask) and keep the 100 stochastic mask outputs.
                ff_dropout_chunks.append(batch_all[1:].detach().cpu())

        ff_wjets_models_dmv = t.cat(ff_dropout_chunks, dim=1).cpu().numpy()
        ff_wjets_mean_dmv = ff_wjets_models_dmv.mean(axis=0)
        ff_wjets_std_dmv = ff_wjets_models_dmv.std(axis=0)

        ff_wjets_unc = np.sqrt((ff_wjets_nominal - ff_wjets_mean_dmv)**2 + (ff_wjets_std_dmv/2)**2)
        ff_wjets_up = ff_wjets_nominal + ff_wjets_unc
        ff_wjets_down = ff_wjets_nominal - ff_wjets_unc
        

        
        ff_wjets_list.append(ff_wjets_nominal)
        

        clipff_list.append(int(np.sum(ff_wjets_nominal >= 1.0)))
        clipff_list_up.append(int(np.sum(ff_wjets_up >= 1.0)))
        clipff_list_down.append(int(np.sum(ff_wjets_down >= 1.0)))
        
        # -----------------------------
        # COLLECT p-values per variable
        # -----------------------------
        p_values = []

        for var in vars_to_test:

            print(var)


            ar_values = ar_events[var].to_numpy()
            ar_weights = ar_events['weight_wjets'].to_numpy()

            sr_values = sr_events[var].to_numpy()
            sr_weights = sr_events['weight_wjets'].to_numpy()


            if var in ['njets', 'tau_decaymode_2']:
                bins, _ = get_bins_and_label(var)
            else:
                bins = equi_populated_bins(sr_values, 20)
                
            counts_data, _ = np.histogram(sr_values, weights=sr_weights, bins=bins)
            var_data, _ = np.histogram(sr_values, weights=sr_weights**2, bins=bins)


            ar_values = region_view_ar.events[var].to_numpy()
            sr_values = region_view_sr.events[var].to_numpy()

            ar_weights = region_view_ar.events['weight_wjets'].to_numpy()
            sr_weights = region_view_sr.events['weight_wjets'].to_numpy()

            counts_data, _ = np.histogram(sr_values, weights=sr_weights, bins=bins)

            counts_nominal, _ = np.histogram(ar_values, weights=ff_wjets_nominal * ar_weights, bins=bins)

            # Poisson uncertainties (simple version)
            var_data, _ = np.histogram(
                sr_values,
                weights=sr_weights**2,
                bins=bins
            )

            d_unc = np.sqrt(var_data)
            var_sys, _ = np.histogram(ar_values, weights=(ff_wjets_nominal**2) * ar_weights**2, bins=bins)
            var_ff, _ = np.histogram(ar_values,weights=(ff_wjets_unc * ar_weights)**2, bins=bins)
            
            m_unc = 2 * np.sqrt(var_sys + var_ff)

            variance = d_unc**2 + m_unc**2

            valid = variance > 0

            pulls = np.zeros_like(counts_data, dtype=float)

            pulls[valid] = (
                counts_data[valid] - counts_nominal[valid]
            ) / np.sqrt(variance[valid])

            ndof = max(len(counts_data) - 1, 1)

            chi2_val = chi2_func(
                d=counts_data,
                m=counts_nominal,
                d_unc=d_unc,
                m_unc=m_unc,
                ndof=ndof,
            )

            print(f"{var}")
            print(f"  mean pull  = {np.mean(pulls[valid]):.3f}")
            print(f"  width pull = {np.std(pulls[valid]):.3f}")
            print(f"  max |pull| = {np.max(np.abs(pulls[valid])):.3f}")

            p = chi2.sf(chi2_val * ndof, ndof)  # if your chi2 returns reduced χ²

            p_values.append(p)
            chi2_by_var[var].append(chi2_val)
            pvalue_by_var[var].append(p)

        # -----------------------------
        # Combine p-values (Fisher)
        # -----------------------------
        p_values = np.clip(np.array(p_values), 1e-300, 1.0)

        fisher_stat = -2 * np.sum(np.log(p_values))
        dof_fisher = 2 * len(p_values)

        global_p = chi2.sf(fisher_stat, dof_fisher)

        pvalue_list.append(global_p)

        print("reduced chi2 =", chi2_val)
        print("chi2 =", chi2_val * ndof)
        print("ndof =", ndof)
        print("p =", p)

        print("mean data unc =", np.mean(d_unc))
        print("mean model unc =", np.mean(m_unc))

   
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top panel: p-values ---
    ax[0].plot(
        squeezing_values,
        pvalue_list,
        marker='o',
        linewidth=2,
        markersize=5
    )

    ax[0].set_ylabel('Combined p-value (Fisher)', fontsize=12)
    ax[0].set_yscale('log')

    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    ax[0].minorticks_on()

    # --- Bottom panel: clipff ---
    ax[1].plot(
        squeezing_values,
        clipff_list,
        marker='o',
        linewidth=2,
        markersize=5
    )

    ax[1].set_xlabel('Squeezing limit', fontsize=12)
    ax[1].set_ylabel(r'$N(F_\mathrm{F} \geq 1)$', fontsize=12)

    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax[1].minorticks_on()

    # --- Shared styling tweaks ---
    for a in ax:
        a.tick_params(axis='both', which='major', labelsize=11)
        a.tick_params(axis='both', which='minor', length=3)

    plt.tight_layout(h_pad=1.2)

    plt.savefig('pvalue_test.png', dpi=300, bbox_inches='tight')
    plt.show()

    n_vars = len(vars_to_test)
    n_cols = 2
    n_rows = int(np.ceil(n_vars / n_cols))



    fig_var, axes_var = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 3.5 * n_rows),
        sharex=True,
        constrained_layout=True
    )

    axes_var = np.atleast_1d(axes_var).ravel()

    for idx, var in enumerate(vars_to_test):
        axv = axes_var[idx]

        axv.plot(
            squeezing_values,
            chi2_by_var[var],
            marker='o',
            linewidth=2,
            markersize=4
        )

        axv.set_title(var, fontsize=11)
        axv.set_ylabel(r'$\chi^2 / \mathrm{ndof}$', fontsize=11)

        axv.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        axv.minorticks_on()

        axv.tick_params(axis='both', which='major', labelsize=10)
        axv.tick_params(axis='both', which='minor', length=3)

    # turn off unused axes
    for idx in range(n_vars, len(axes_var)):
        axes_var[idx].axis('off')

    # only bottom row gets x-labels
    for idx in range(len(axes_var)):
        if idx < n_vars:
            axes_var[idx].set_xlabel('Squeezing limit', fontsize=11)

    plt.savefig(
        'chi2_per_variable_test.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig_var)
    
    # Mean chi2 trend across all tested variables for each squeezing point.
    chi2_matrix = np.array([chi2_by_var[var] for var in vars_to_test], dtype=float)
    mean_chi2 = np.nanmean(chi2_matrix, axis=0)
    std_chi2 = np.nanstd(chi2_matrix, axis=0)

    fig_mean, ax_mean = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax_mean.plot(
        squeezing_values,
        mean_chi2,
        marker='o',
        linewidth=2,
        markersize=5,
        label='Mean reduced chi2',
    )
    ax_mean.fill_between(
        squeezing_values,
        mean_chi2 - std_chi2,
        mean_chi2 + std_chi2,
        alpha=0.2,
        label='1 sigma across variables',
    )
    ax_mean.set_xlabel('Squeezing limit', fontsize=12)
    ax_mean.set_ylabel('Mean chi2 / ndof', fontsize=12)
    ax_mean.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_mean.minorticks_on()
    ax_mean.legend(fontsize=10)
    fig_mean.savefig('chi2_mean_over_variables_test.png', dpi=300, bbox_inches='tight')
    plt.close(fig_mean)

    fig_pvar, axes_pvar = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 3.5 * n_rows),
        sharex=True,
        constrained_layout=True
    )

    axes_pvar = np.atleast_1d(axes_pvar).ravel()

    # small floor to avoid log(0)
    p_floor = 1e-300

    for idx, var in enumerate(vars_to_test):
        axp = axes_pvar[idx]

        pvals = np.clip(pvalue_by_var[var], p_floor, None)

        axp.plot(
            squeezing_values,
            pvals,
            marker='o',
            linewidth=2,
            markersize=4
        )

        axp.set_title(f'{var} p-value', fontsize=11)
        axp.set_ylabel('p-value', fontsize=11)
        axp.set_yscale('log')

        axp.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        axp.minorticks_on()

        axp.tick_params(axis='both', which='major', labelsize=10)
        axp.tick_params(axis='both', which='minor', length=3)

    # turn off unused panels
    for idx in range(n_vars, len(axes_pvar)):
        axes_pvar[idx].axis('off')

    # only label bottom plots (cleaner)
    for idx in range(len(axes_pvar)):
        if idx < n_vars:
            axes_pvar[idx].set_xlabel('Squeezing limit', fontsize=11)

    plt.savefig(
        'pvalue_per_variable_test.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close(fig_pvar)

    out_dir = Path("closure_for_chi2")
    out_dir.mkdir(parents=True, exist_ok=True)

    bins, label = get_bins_and_label("m_vis")

    for i, s in enumerate(squeezing_values):

        fig_w, _ = FF_closure(
            data=region_view_sr["m_vis"],
            data_weights=region_view_sr.weight,
            closure=region_view_ar["m_vis"],
            closure_weights=ff_wjets_list[i] * region_view_ar.weight,
            bins=bins,
            label=label,
        )

        # safer filename (avoids '.' issues in floats)
        tag = f"s{i:03d}_val{s:.3f}".replace(".", "p")

        fig_w.savefig(out_dir / f"FF_closure_DR_{tag}.png", dpi=300, bbox_inches="tight")
        fig_w.savefig(out_dir / f"FF_closure_DR_{tag}.pdf", bbox_inches="tight")

        plt.close(fig_w)

    print("Saved closure plots")
    
    return squeezing_values, pvalue_list, clipff_list, vars_to_test, chi2_by_var, chi2_matrix, mean_chi2, std_chi2, pvals, clipff_list_up, clipff_list_down

if __name__ == '__main__':
    main()
