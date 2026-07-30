# Dataframes and Features

This document describes the dataframe and feature interface used by
`DNN_et/pipeline`. It covers the current simplified law workflow: the only
supported fake-factor grouping is `njets`, ML process fractions are the
default, and the default final fake factor includes both DRSR and non-closure
corrections.

## 1. Storage layout

All generated workflow artifacts are stored below:

```text
DNN_et/Law_workflow_results/
├── data/
│   ├── dataframe_complete.feather
│   ├── dataframe_complete_no_embedding.feather
│   └── features/
│       ├── feature_registry.json
│       ├── process_fake_factors/
│       └── fake_factors/
├── models/
└── plots/
```

The paths are defined centrally in `pipeline/core/paths.py`:

| Constant | Path below `DNN_et/Law_workflow_results` |
|---|---|
| `DATA_ROOT` | `data/` |
| `FEATURE_ROOT` | `data/features/` |
| `MODEL_ROOT` | `models/` |
| `PLOT_ROOT` | `plots/` |

The correctionlib inputs are kept separately below:

```text
DNN_et/corrections/fake_factors/
```

Do not hard-code these paths in new pipeline code. Import the corresponding
constant or path helper from `core.paths`.

## 2. Base dataframes

`BuildDataset` creates:

```text
Law_workflow_results/data/dataframe_complete.feather
```

This is the standard input for training, inference, and plotting.

`BuildDatasetNoEmbedding` creates:

```text
Law_workflow_results/data/dataframe_complete_no_embedding.feather
```

The no-embedding dataframe is used by QCD extrapolation/DRSR tasks. In
addition to its Feather file, the task writes a schema marker and verifies
that the required true-tau process IDs 11, 12, 13, and 14 are present.

The base Feather files contain event variables and bookkeeping columns, but
not every feature produced later in the workflow. Derived columns are stored
in separate feature files and loaded lazily.

## 3. Loading a dataframe

Run examples from `DNN_et/pipeline`, where the pipeline packages are directly
importable:

```python
from pathlib import Path

from data.handling import load_data

data_path = Path("../Law_workflow_results/data/dataframe_complete.feather")
masks_path = Path("config/selections.yaml")

df = load_data(data_path, masks_path)
```

`load_data` returns an `AnalysisDataFrame`, not a plain pandas dataframe. The
underlying pandas object is available as:

```python
events = df.events
```

By default, `load_data` uses this registry:

```text
<directory containing the base Feather file>/features/feature_registry.json
```

For the standard dataset, that resolves to:

```text
DNN_et/Law_workflow_results/data/features/feature_registry.json
```

An explicit registry can be supplied when needed:

```python
df = load_data(
    data_path,
    masks_path,
    feature_registry_path="../Law_workflow_results/data/features/feature_registry.json",
)
```

`load_data_no_embedding` currently provides the same interface and delegates
to `load_data`.

## 4. Regions and processes

Regions, processes, and primitive masks are configured in
`DNN_et/pipeline/config/selections.yaml`. They are evaluated by
`SelectionManager`.

### Region access

Regions can be accessed by attribute or item:

```python
ar = df.AR
sr = df["SR"]

ar_events = df.AR.events
ar_copy = df.AR.copy()
number_of_ar_rows = df.AR.n
```

A `RegionView` keeps the selection mask but always reads from the current
parent dataframe. Features loaded after the view was created are therefore
visible through the existing view.

### Process access

Configured process names can also be accessed by attribute or item:

```python
data = df.data
wjets = df["wjets"]
```

Process views always apply their process mask. Use `df.full` only if `full` is
defined as the explicit all-process view in the mask configuration.

Processes and regions can be combined:

```python
data_ar = df.data.AR
wjets_ar_like = df.wjets.AR_like_wjets

values = df.data.AR["weight"]
```

### Subsets

Use `subset` to preserve the analysis wrapper and create fresh region/process
views:

```python
positive_weight = df.subset(df.events["weight"] > 0)
high_ff = df.subset(df["ff_99"] > 1.0)
```

A pandas `Series` mask is reindexed to the dataframe index. An array-like mask
must have exactly the same length as the dataframe.

## 5. Lazy feature loading

Derived features are resolved when a column is requested:

```python
ff = df["ff_99"]
ff_ar = df.AR["ff_99"]
ff_data_ar = df.data.AR["ff_99"]
```

The resolution sequence is:

1. Return the column immediately if it is already in the base dataframe.
2. Look up the column in `feature_registry.json`.
3. Read the registered Feather file.
4. Align its values to the base dataframe.
5. Add the resolved column to `df.events`.

The following forms trigger lazy loading:

```python
df["ff_99"]
df.ff_99
df.AR["ff_99"]
df.data.AR["ff_99"]
df.ensure_column("ff_99")
```

Accessing `df.events` alone does not load anything. Consequently, this fails
unless the feature was already requested:

```python
# KeyError if ff_99 has not yet been resolved
df.events["ff_99"]
```

Use one of these instead:

```python
values = df["ff_99"]

# or
df.ensure_column("ff_99")
values = df.events["ff_99"]
```

Each feature file is cached after loading. Requesting another registered
column from the same file does not reread the Feather file.

## 6. Row identity and alignment

New workflow feature files should use `row_index` as their identity column:

```text
row_index | feature_a | feature_b | ...
```

`row_index` refers to the index of the corresponding row in the base
dataframe. It is the preferred key because an `event` value is not guaranteed
to be unique.

Legacy event-keyed files are still readable:

```text
event | feature_a | feature_b | ...
```

When loading an event-keyed file, all base rows with the same event receive
the same feature value. Duplicate keys in a feature file are compacted with
`groupby(...).last()` before alignment. Loading never changes the number or
order of base-dataframe rows.

Important consequences:

- Prefer `row_index` for every new artifact.
- Do not reset or reorder the base dataframe before constructing
  `row_index`-keyed output.
- Missing keys produce `NaN`.
- Multiple values for the same key are ambiguous; the last one wins.
- A final fake-factor feature is normally defined only for AR rows. `NaN`
  outside its declared region is therefore not automatically an error.

## 7. Feature registry

The registry is a JSON mapping from a column name to the Feather file that
owns it:

```json
{
  "ff_99": "/absolute/path/to/ff_99.feather",
  "ff_dnn_wjets_njets_99": "/absolute/path/to/fake_factors.feather"
}
```

It is managed by `FeatureRegistry`:

```python
from data.handling import FeatureRegistry

registry = FeatureRegistry(
    "../Law_workflow_results/data/features/feature_registry.json"
)

path = registry.get_file("ff_99")
```

`FeatureRegistry.save()` uses a file lock and an atomic replacement, allowing
independent law tasks to update the shared registry safely. Registering a
column again changes its owner to the new file.

Feature files should not be moved manually after registration. The stored
path would become stale. Regenerate or re-register the artifact instead.

## 8. Writing features

Use `FeatureStore` for new workflow code. It writes the feature artifact and
updates the registry consistently:

```python
import pandas as pd

from data.handling import FeatureRegistry, FeatureStore

registry = FeatureRegistry(
    "../Law_workflow_results/data/features/feature_registry.json"
)
store = FeatureStore(
    "../Law_workflow_results/data/features/example/features.feather",
    registry,
)

feature_frame = pd.DataFrame({
    "row_index": selected_events.index,
    "my_feature": calculated_values,
})

store.write(feature_frame)
store.save()
registry.save()
```

`write` replaces the contents of the feature file. It also removes registry
entries for columns that used to belong to that file but are no longer
written.

Use `upsert` only when an artifact is intentionally assembled in multiple
steps:

```python
store.upsert(first_feature_frame)
store.upsert(second_feature_frame)
store.save()
registry.save()
```

Both operations compact duplicate keys and keep their last values. Feature
columns are all columns except `row_index` and `event`.

The older helpers `write_features`, `append_features`, and `update_features`
operate on event-keyed files. They remain useful for existing code, but new
row-index-keyed artifacts should use `FeatureStore`.

## 9. Metadata contract

Canonical final and process fake-factor tasks write `metadata.json` next to
their Feather output. The helper is `core.metadata.write_feature_metadata`.

Example:

```json
{
  "schema_version": 1,
  "columns": ["ff_99"],
  "method": "drsr_nonclosure",
  "grouping": "njets",
  "squeezing": 0.99,
  "squeezing_loss_limit": 0.1,
  "regions": ["AR"],
  "index_column": "row_index"
}
```

Fields:

| Field | Meaning |
|---|---|
| `schema_version` | Metadata format version |
| `columns` | Feature columns owned by the artifact |
| `method` | Construction or correction method |
| `grouping` | Fake-factor grouping; currently always `njets` |
| `squeezing` | Model squeezing value, or `null` |
| `squeezing_loss_limit` | DRSR loss limit, or `null` |
| `regions` | Regions on which the features are defined |
| `index_column` | Alignment key; canonical outputs use `row_index` |

When adding a canonical feature task, write both the Feather feature file and
its metadata. Law's output target should include both files.

## 10. Current fake-factor convention

### Grouping

Only `njets` is trained, calculated, and plotted by the simplified law
workflow. The bins are:

```text
W+jets and QCD: 0, 1, >=2 jets
ttbar:          0–1, >=2 jets
```

Older grouping definitions remain internal implementation data where needed,
but they are not choices exposed by the current law tasks.

### Process fake factors

The three DNN process fake factors are:

```text
ff_dnn_wjets_njets[_XX]
ff_dnn_qcd_njets[_XX]
ff_dnn_ttbar_njets[_XX]
```

For the default squeezing of `0.99`, these become:

```text
ff_dnn_wjets_njets_99
ff_dnn_qcd_njets_99
ff_dnn_ttbar_njets_99
```

They are stored together below:

```text
data/features/process_fake_factors/<squeezing>/fake_factors.feather
```

### ML process fractions

ML fractions are the default combination method. The relevant fraction
columns are:

```text
fraction_wjets
fraction_qcd
fraction_ttbar
```

Before corrections, the combined fake factor is conceptually:

```text
fraction_wjets * process_ff_wjets
+ fraction_qcd * process_ff_qcd
+ fraction_ttbar * process_ff_ttbar
```

The old `mlf`/`MLFraction` marker is deliberately absent from both public task
names and final feature names.

### Final feature names

| Method | Law `--correction` | Feature at squeezing 0.99 |
|---|---|---|
| DRSR + non-closure, default | `drsr_nonclosure` | `ff_99` |
| No correction | `none` | `ff_uncorrected_99` |
| Non-closure only | `nonclosure` | `ff_nonclosure_99` |
| DRSR only | `drsr` | `ff_drsr_99` |
| Classic fractions | separate task | `ff_cf_99` |

The short, unqualified `ff[_XX]` name is reserved for the default product:
ML fractions with both DRSR and non-closure corrections.

Classic fractions are opt-in and use the `cf`/`ClassicFraction` marker. They
combine the same process fake factors with classic process fractions.

### DRSR correction features

Per-process DRSR correction factors use:

```text
correction_drsr_wjets[_XX]
correction_drsr_qcd[_XX]
correction_drsr_ttbar[_XX]
```

At the default squeezing:

```text
correction_drsr_wjets_99
correction_drsr_qcd_99
correction_drsr_ttbar_99
```

### Squeezing suffixes and directories

Feature names represent squeezing with a two-digit suffix:

| Squeezing | Feature suffix |
|---|---|
| `None` | no suffix |
| `0.95` | `_95` |
| `0.99` | `_99` |

Named features accept squeezing values with at most two decimal places and
strictly between zero and one.

Artifact directories use four decimal places:

| Squeezing | Directory |
|---|---|
| `None` | `unsqueezed/` |
| `0.95` | `0.9500/` |
| `0.99` | `0.9900/` |

DRSR-based products include the DRSR loss limit as an additional directory.
The default limit is `0.1`:

```text
loss_squeeze_pm0p1/
```

### Final feature directories

```text
data/features/fake_factors/
├── none/<squeezing>/
├── nonclosure/<squeezing>/
├── drsr/<squeezing>/<loss-limit>/
├── drsr_nonclosure/<squeezing>/<loss-limit>/
└── classic_fraction/<squeezing>/
```

At the defaults, the main final artifact is:

```text
data/features/fake_factors/drsr_nonclosure/0.9900/loss_squeeze_pm0p1/
├── ff_99.feather
└── metadata.json
```

Plot directories follow the same method, squeezing, and loss-limit hierarchy
below:

```text
plots/fake_factors/<plot-kind>/<method>/<squeezing>/<loss-limit>/
```

The loss-limit component is present only for DRSR-based methods.

## 11. Law entry points

Run law from `DNN_et/pipeline`. The public task registry is `logic.py`, while
implementations are organized below `tasks/`.

Calculate the default final fake factor:

```bash
law run CalculateFF
```

This means:

```text
grouping = njets
squeezing = 0.99
correction = drsr_nonclosure
squeezing-loss-limit = 0.1
fraction method = ML fractions
```

Explicit correction variants:

```bash
law run CalculateFF --correction none
law run CalculateFF --correction nonclosure
law run CalculateFF --correction drsr
law run CalculateFF --correction drsr_nonclosure
```

Calculate the opt-in classic-fraction product:

```bash
law run CalculateClassicFractionFF
```

Plotting uses the same default and correction parameter:

```bash
law run PlotFF
law run PlotFF --correction none
law run PlotFF --correction nonclosure
law run PlotFF --correction drsr
```

`CalculateProcessFakeFactors` produces the three process-level inputs. In
normal use it does not need to be invoked manually because it is part of the
dependency graph.

## 12. Inspecting and validating features

Resolve and inspect the default fake factor:

```python
import numpy as np

ff = df["ff_99"]
ar_ff = df.AR["ff_99"]

assert len(ff) == len(df.events)
assert np.isfinite(ar_ff).all()
```

Inspect the registry owner:

```python
from data.handling import FeatureRegistry

registry = FeatureRegistry(
    "../Law_workflow_results/data/features/feature_registry.json"
)
print(registry.get_file("ff_99"))
```

Inspect artifact metadata:

```python
import json
from pathlib import Path

metadata_path = Path(
    "../Law_workflow_results/data/features/fake_factors/"
    "drsr_nonclosure/0.9900/loss_squeeze_pm0p1/metadata.json"
)
metadata = json.loads(metadata_path.read_text())
print(metadata["columns"])
print(metadata["regions"])
```

Check row-index coverage manually:

```python
import pandas as pd

feature = pd.read_feather(registry.get_file("ff_99"))
ar_indices = df.events.index[df.mask("AR")]

stored = (
    feature[["row_index", "ff_99"]]
    .drop_duplicates("row_index", keep="last")
    .set_index("row_index")
)
aligned = stored.reindex(ar_indices)

assert aligned["ff_99"].notna().all()
```

Canonical final-FF law tasks perform an equivalent AR coverage and finiteness
check in their `complete()` implementation.

## 13. Adding a new feature-producing task

Use this checklist:

1. Depend on the task that provides the base dataframe and all required
   upstream features.
2. Select rows through `AnalysisDataFrame` so that dataframe indices retain
   their meaning.
3. Create a dataframe containing `row_index` and the new feature columns.
4. Write it with `FeatureStore`.
5. Save the shared `FeatureRegistry`.
6. Write `metadata.json` with the column names, method, grouping, parameters,
   regions, and `row_index` identity.
7. Declare both the Feather file and metadata file as law outputs.
8. Validate required-region coverage and finite values in `complete()` when
   the feature is needed for physics output.

Minimal pattern:

```python
from pathlib import Path

import pandas as pd

from core.metadata import write_feature_metadata
from data.handling import FeatureRegistry, FeatureStore

output_dir = Path("../Law_workflow_results/data/features/example")
feature_path = output_dir / "example.feather"
registry_path = Path(
    "../Law_workflow_results/data/features/feature_registry.json"
)

selected = df.AR.events
feature_frame = pd.DataFrame({
    "row_index": selected.index,
    "example_score": calculate_score(selected),
})

registry = FeatureRegistry(registry_path)
store = FeatureStore(feature_path, registry)
store.write(feature_frame)
store.save()
registry.save()

write_feature_metadata(
    output_dir / "metadata.json",
    columns=("example_score",),
    method="example",
    grouping="njets",
    regions=("AR",),
)
```

## 14. Common problems

### `KeyError: Unknown feature`

The column is absent from both the base dataframe and registry. Check the
spelling, confirm that its producing law task completed, and inspect
`feature_registry.json`.

### The registered file does not exist

The artifact was moved or removed without updating the registry. Rerun the
producing task or deliberately register the new path and save the registry.

### A feature is missing from `df.events`

`events` does not trigger lazy loading. Request the feature through
`df["name"]` or call `df.ensure_column("name")` first.

### Values are unexpectedly duplicated

The file probably uses `event`, while event IDs occur more than once in the
base dataframe. Recreate the artifact using `row_index`.

### Values are `NaN`

Check the artifact's declared regions in `metadata.json`. Region-local
features are expected to be missing elsewhere. If values are missing inside a
required region, compare the stored keys with the selected dataframe indices.

### A task appears complete after naming or schema changes

Law completion is target-based. Rename/version the target or strengthen the
task's `complete()` validation when the content contract changes. Do not rely
only on the existence of an old Feather file.

## 15. Source map

The main implementation locations are:

| Concern | Module |
|---|---|
| Dataframe, regions, processes, registry, store | `data/handling.py` |
| Embedded dataset construction | `data/build.py` |
| No-embedding dataset construction | `data/build_no_embedding.py` |
| Grouping definitions | `core/groupings.py` |
| Canonical feature names | `core/names.py` |
| FF correction methods | `core/methods.py` |
| Artifact path construction | `core/paths.py` |
| Feature metadata | `core/metadata.py` |
| ML-fraction combination | `inference/fractions.py` |
| DRSR correction and combination | `inference/drsr.py` |
| Process and classic-fraction FFs | `inference/process.py` |
| Public law task registry | `logic.py` |
| Law task implementations | `tasks/workflow.py` |

The removed top-level compatibility modules and old `classes` package should
not be used in new code. Import from the domain package shown above.
