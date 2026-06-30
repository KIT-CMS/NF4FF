# `dataframe_complete.feather` and Lazy Features

## Overview

The workflow stores data in two layers:

1. `dataframe_complete.feather` contains the complete base event table.
2. `data/features/**/*.feather` contains derived columns keyed to rows of the
   base table.

The default Law paths are:

```text
DNN_et/Law_workflow_results/data/dataframe_complete.feather
DNN_et/Law_workflow_results/data/features/feature_registry.json
DNN_et/Law_workflow_results/data/features/**/*.feather
```

Derived columns are not physically added to `dataframe_complete.feather`.
`load_data()` reads the base table and uses `feature_registry.json` to attach a
derived column when that column is first requested.

## Loading the data

Run code from `DNN_et/pipeline`, or add that directory to `PYTHONPATH`:

```python
from pathlib import Path

from classes import load_data

project_root = Path("..").resolve()
data_root = project_root / "Law_workflow_results" / "data"

df = load_data(
    data_root / "dataframe_complete.feather",
    project_root / "configs" / "masks.yaml",
)
```

`load_data()` automatically looks for the registry at:

```text
<directory containing dataframe_complete.feather>/features/feature_registry.json
```

An alternative registry can be selected explicitly:

```python
df = load_data(
    data_root / "dataframe_complete.feather",
    project_root / "configs" / "masks.yaml",
    feature_registry_path=data_root / "features" / "feature_registry.json",
)
```

## Access patterns

Access a base or lazy feature column through the wrapper:

```python
values = df["ff_dnn_98"]
values = df.AR["ff_dnn_98"]
values = df.data.AR["ff_dnn_98"]
values = df.wjets.AR_like_wjets["ff_dnn_wjets_98"]
```

The first access to a registered feature calls `ensure_column()`, reads its
feature Feather file, aligns the values to the base rows, and adds the column
to `df.events`.

You can also load every column from one feature file explicitly:

```python
feature_path = (
    data_root / "features" / "fake_factors" / "0.9800"
    / "fake_factors.feather"
)
df.load_feature_file(feature_path)
```

After a feature has been loaded:

```python
assert "ff_dnn_98" in df.events.columns
```

### Important: `events` does not trigger lazy loading

`df.events` is the underlying pandas DataFrame. This fails if the feature has
not already been loaded:

```python
df.events["ff_dnn_98"]
```

Trigger loading first:

```python
df.ensure_column("ff_dnn_98")
values = df.events["ff_dnn_98"]
```

or use:

```python
values = df["ff_dnn_98"]
```

## Processes and regions

Processes and regions are configured in `configs/masks.yaml`.

Examples:

```python
all_events = df.full.events
data_events = df.data.events
wjets_events = df.wjets.events

signal_region = df.SR.events
application_region = df.AR.events
wjets_control_region = df.data.AR_like_wjets.events
```

Common process views are:

```text
full, data, wjets, diboson, DYjets, ST, ttbar, embedding
```

Common region views are:

```text
SR, AR, AR_SS
SR_like_wjets, AR_like_wjets
SR_like_qcd, AR_like_qcd
SR_like_ttbar, AR_like_ttbar
DR_wjets, DR_wjets_without_signs
DR_qcd, DR_qcd_without_signs
```

Use `subset()` when a filtered `AnalysisDataFrame` with working process and
region views is needed:

```python
high_ff = df.subset(df["ff_dnn_98"] > 1.0)
```

## Rows and alignment

`dataframe_complete.feather` contains all rows imported from these ROOT
ntuples:

```text
Wjets, data, diboson_J, diboson_L, DYjets_J, DYjets_L,
ST_J, ST_L, ttbar_J, ttbar_L, embedding
```

The inputs are concatenated, shuffled with random seed 42, and assigned a new
zero-based pandas index. This index is the canonical `row_index` used by new
feature files.

Feature files normally contain:

```text
row_index
event
one or more derived feature columns
```

Feature files may contain only rows in the regions where their values were
calculated. When attached to the complete dataframe, rows without a stored
value become `NaN`.

`row_index` is preferred over `event` because event numbers can be duplicated.
Do not reset, reorder, or otherwise change the base index before attaching
row-index-keyed features. Use `AnalysisDataFrame.subset()` for filtering.

## Base columns

All branches from each input ROOT `ntuple` are copied into the base Feather
file. Therefore the complete physics-column list depends on the ROOT input
files and is not fixed in Python source.

`BuildDataset.py` additionally creates:

| Column | Meaning |
| --- | --- |
| `process` | Integer process identifier |
| `Label` | Training label: W+jets = 1, data = 2, others = 0 |
| `SS` | `q_1 * q_2 > 0` |
| `OS` | `q_1 * q_2 < 0` |
| `parity` | `event % 2` |
| `class_weights` | Weight derived from the clipped `njets` category |

Print the authoritative columns and row count:

```python
import pandas as pd

base = pd.read_feather(data_root / "dataframe_complete.feather")
print(f"rows: {len(base)}")
print(f"columns: {len(base.columns)}")
print(*base.columns, sep="\n")
```

## Feature inventory

The registry is the authoritative map from a lazy column name to its Feather
file:

```python
import json

registry_path = data_root / "features" / "feature_registry.json"
registry = json.loads(registry_path.read_text())

for column, path in sorted(registry.items()):
    print(f"{column}: {path}")
```

### Enrichment features

For every configured grouping, enrichment produces patterns such as:

```text
qcd_weight_<grouping>
weight_qcd_fraction_<grouping>
```

The current grouping names are:

```text
tau_decaymode_2
tau_decaymode_2_alt
njets
```

### Reduced-dataset features

For `process` equal to `wjets` or `qcd`, the reduced feature files register:

```text
nn_output_<process>_<grouping>
reduced_weight_<process>_<grouping>_nominal
```

Examples:

```text
nn_output_wjets_tau_decaymode_2_alt
reduced_weight_wjets_tau_decaymode_2_alt_nominal
nn_output_qcd_njets
reduced_weight_qcd_njets_nominal
```

### Grouped DNN fake factors

Unsqueezed grouped fake factors keep their original names:

```text
ff_dnn
ff_dnn_wjets
ff_dnn_qcd
ff_dnn_ttbar

ff_dnn_tau_decaymode_2_alt
ff_dnn_wjets_tau_decaymode_2_alt
ff_dnn_qcd_tau_decaymode_2_alt
ff_dnn_ttbar_tau_decaymode_2_alt

ff_dnn_njets
ff_dnn_wjets_njets
ff_dnn_qcd_njets
ff_dnn_ttbar_njets
```

For squeezed models, the two decimal digits of the squeezing probability are
appended to the complete unsqueezed name:

```text
squeezing = 0.90 -> _90
squeezing = 0.95 -> _95
squeezing = 0.98 -> _98
squeezing = 0.99 -> _99
```

Examples for `squeezing = 0.98`:

```text
ff_dnn_98
ff_dnn_wjets_98
ff_dnn_tau_decaymode_2_alt_98
ff_dnn_wjets_tau_decaymode_2_alt_98
ff_dnn_njets_98
ff_dnn_wjets_njets_98
```

### How `tau_decaymode_2_alt` is handled

`tau_decaymode_2_alt` uses the source column `tau_decaymode_2`, but combines
the decay modes into two inclusive groups:

```text
0 through 2
10 through 12
```

The grouping suffix is always `_tau_decaymode_2_alt`. Squeezing is appended
after that suffix:

| Model | Combined column | W+jets column |
| --- | --- | --- |
| Unsqueezed | `ff_dnn_tau_decaymode_2_alt` | `ff_dnn_wjets_tau_decaymode_2_alt` |
| Squeezing 0.98 | `ff_dnn_tau_decaymode_2_alt_98` | `ff_dnn_wjets_tau_decaymode_2_alt_98` |

Example:

```python
alt_ff = df["ff_dnn_tau_decaymode_2_alt_98"]
alt_wjets_ff = df["ff_dnn_wjets_tau_decaymode_2_alt_98"]
alt_weight = df["reduced_weight_wjets_tau_decaymode_2_alt_nominal"]
```

The combined `ff_dnn_tau_decaymode_2_alt[_XX]` column exists for SR and AR
rows. Process-specific columns are calculated over the configured FF
application regions, including the corresponding AR-like and SR-like regions.
Other rows are `NaN`.

### Single-DNN fake factors

For the default reduced-weight grouping `tau_decaymode_2_alt`, unsqueezed
single-DNN columns keep their original names:

```text
ff_dnn_single
ff_dnn_single_wjets
ff_dnn_single_qcd
ff_dnn_single_ttbar
```

With squeezing 0.98:

```text
ff_dnn_single_98
ff_dnn_single_wjets_98
ff_dnn_single_qcd_98
ff_dnn_single_ttbar_98
```

For a non-default reduced-weight grouping, the grouping is also added to avoid
registry collisions:

```text
ff_dnn_single_njets
ff_dnn_single_wjets_njets
ff_dnn_single_njets_98
ff_dnn_single_wjets_njets_98
```

### Classic fake factor

The classic fake-factor task registers:

```text
ff_classic
```

It has no squeezing parameter and therefore needs no squeezing suffix.

## Inspecting every feature file

Use PyArrow to print schemas without converting the tables to pandas:

```python
from pathlib import Path
import pyarrow.feather as feather

feature_root = data_root / "features"

for path in sorted(feature_root.rglob("*.feather")):
    table = feather.read_table(path)
    print(path.relative_to(feature_root))
    print(f"  rows: {table.num_rows}")
    print(f"  columns: {', '.join(table.column_names)}")
```

To inspect one feature file with pandas:

```python
feature = pd.read_feather(feature_path)
print(feature.shape)
print(feature.columns.tolist())
print(feature.head())
```

## Finding available lazy keys

Before requesting a column, check whether it is registered:

```python
available = sorted(registry)
print(*available, sep="\n")

name = "ff_dnn_tau_decaymode_2_alt_98"
if name not in registry:
    raise KeyError(f"{name} has not been produced by the workflow")
```

If a squeezed key is absent, run the corresponding
`CalculateFakeFactors --squeezing <value>` Law task. Feature tasks created
before the squeezing-aware naming change must be rerun so their Feather schema
and registry entries are updated.
