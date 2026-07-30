# Using the NF4FF workflow

This guide explains how to run the `DNN_et` law workflow. For dataframe and
feature details, see `DATAFRAME_AND_FEATURES.md`. For the code layout, see
`ARCHITECTURE.md`.

## 1. Before running

The workflow expects a Python environment containing its analysis
dependencies, including `law`, `luigi`, PyTorch, pandas, NumPy, pyarrow,
scikit-learn, correctionlib, matplotlib, and the ROOT-reading dependencies
used by dataset construction.

Run commands from:

```bash
cd DNN_et/pipeline
```

The local Luigi scheduler is enabled in `luigi.cfg`, so a separate central
scheduler is not required.

Before the first run, check:

```text
config/data_sources.yaml
```

It contains the site-specific input ROOT directories for the embedded and
no-embedding datasets.

The workflow also expects the correctionlib inputs below:

```text
DNN_et/corrections/fake_factors/
├── classic/2018/
├── non_drsr/
└── drsr/
```

The public task registry is `logic.py`. Use `run_workflow.py` as the command
entry point:

```bash
python run_workflow.py <TaskName> [task options]
```

## 2. The normal workflow

For normal usage, the two main commands are:

```bash
python run_workflow.py CalculateFF
python run_workflow.py PlotFF
```

Law follows the dependency graph automatically. It builds missing datasets,
trains missing models, exports models, calculates intermediate features, and
then produces the requested final output. It does not rerun dependencies whose
declared outputs are already complete.

The default configuration is:

| Setting | Default |
|---|---|
| Grouping | `njets` |
| Fraction method | ML process fractions |
| Squeezing | `0.99` |
| Corrections | DRSR and non-closure |
| DRSR loss limit | `0.1` |
| Plot variable set | `variables_set_large` |

The default final feature is:

```text
ff_99
```

and is written below:

```text
../Law_workflow_results/data/features/fake_factors/
└── drsr_nonclosure/0.9900/loss_squeeze_pm0p1/
    ├── ff_99.feather
    └── metadata.json
```

The default closure plots are written below:

```text
../Law_workflow_results/plots/fake_factors/
└── closure/drsr_nonclosure/0.9900/loss_squeeze_pm0p1/
    └── variables_set_large/
```

## 3. Building the input dataframes

Build the standard embedded dataset:

```bash
python run_workflow.py BuildDataset
```

Output:

```text
../Law_workflow_results/data/dataframe_complete.feather
```

Build the no-embedding dataset used by extrapolation and DRSR training:

```bash
python run_workflow.py BuildDatasetNoEmbedding
```

Output:

```text
../Law_workflow_results/data/dataframe_complete_no_embedding.feather
```

Normally these commands do not need to be run separately. They are
dependencies of the relevant training tasks.

To use a different data-source file with the same schema:

```bash
python run_workflow.py BuildDataset \
    --config-path /absolute/path/to/data_sources.yaml
```

The expected schema is:

```yaml
directories:
  data_input_directory: /path/to/embedded/input/
  data_input_directory_no_embedding: /path/to/no-embedding/input/
```

## 4. Calculating fake factors

### Default: ML fractions with both corrections

```bash
python run_workflow.py CalculateFF
```

This is equivalent to:

```bash
python run_workflow.py CalculateFF \
    --grouping njets \
    --squeezing 0.99 \
    --correction drsr_nonclosure \
    --squeezing-loss-limit 0.1
```

Only `njets` is supported by the simplified workflow.

### Correction variants

Uncorrected:

```bash
python run_workflow.py CalculateFF --correction none
```

Non-closure only:

```bash
python run_workflow.py CalculateFF --correction nonclosure
```

DRSR only:

```bash
python run_workflow.py CalculateFF --correction drsr
```

DRSR and non-closure:

```bash
python run_workflow.py CalculateFF --correction drsr_nonclosure
```

The resulting columns are:

| Correction | Feature at squeezing `0.99` |
|---|---|
| `none` | `ff_uncorrected_99` |
| `nonclosure` | `ff_nonclosure_99` |
| `drsr` | `ff_drsr_99` |
| `drsr_nonclosure` | `ff_99` |

### Different squeezing

```bash
python run_workflow.py CalculateFF --squeezing 0.95
```

This creates the corresponding `_95` feature and stores artifacts under the
`0.9500` directory.

DRSR-based methods require a numeric squeezing value. For an unsqueezed
non-DRSR calculation, Luigi's optional-float syntax can be inspected with:

```bash
python run_workflow.py CalculateFF --help
```

### Different DRSR loss limit

```bash
python run_workflow.py CalculateFF \
    --correction drsr_nonclosure \
    --squeezing-loss-limit 0.05
```

The loss limit is included in the output directory, so products from
different limits do not overwrite one another.

### Classic process fractions

Classic fractions are an explicit alternative to the default ML fractions:

```bash
python run_workflow.py CalculateClassicFractionFF
```

At the default squeezing this creates:

```text
ff_cf_99
```

To supply another correctionlib process-fraction file:

```bash
python run_workflow.py CalculateClassicFractionFF \
    --process-fractions-path /absolute/path/to/fake_factors_et.json.gz
```

### Intermediate process fake factors

To calculate only the three process-specific DNN fake factors:

```bash
python run_workflow.py CalculateProcessFakeFactors
```

The resulting columns at the default squeezing are:

```text
ff_dnn_wjets_njets_99
ff_dnn_qcd_njets_99
ff_dnn_ttbar_njets_99
```

This task is already an automatic dependency of both final fraction methods.

## 5. Plotting fake factors

Create the default closure plots:

```bash
python run_workflow.py PlotFF
```

Select another correction method with the same option used by `CalculateFF`:

```bash
python run_workflow.py PlotFF --correction none
python run_workflow.py PlotFF --correction nonclosure
python run_workflow.py PlotFF --correction drsr
python run_workflow.py PlotFF --correction drsr_nonclosure
```

Use the smaller configured variable set:

```bash
python run_workflow.py PlotFF --variable-set variables_set_small
```

The available sets, variables, and binning are defined in:

```text
config/plotting.yaml
```

Axis labels are defined in:

```text
config/labels.yaml
```

`PlotFF` depends on the corresponding calculation task. Requesting a plot is
therefore sufficient even when the feature has not yet been calculated.

## 6. Running individual workflow stages

The high-level commands are preferred, but individual stages are useful for
development and diagnostics.

### Enrichment

```bash
python run_workflow.py TrainEnrichmentWjetsV2
python run_workflow.py TrainEnrichmentQCDV2
python run_workflow.py TrainEnrichmentQCDFractions
python run_workflow.py TrainEnrichmentQCDExtrapolation
```

Training diagnostics:

```bash
python run_workflow.py PlotTrainingResultsWjets
python run_workflow.py PlotTrainingResultsQCD
python run_workflow.py PlotTrainingResultsQCD2
python run_workflow.py PlotTrainingResultsQCDExtrapolation
```

### Process fractions

```bash
python run_workflow.py TrainFractionClassifier
python run_workflow.py PlotFractions
```

`PlotFractions` also evaluates and stores the trained fraction outputs used by
the final ML-fraction calculation.

### Process fake-factor models

```bash
python run_workflow.py TrainSqueezedModels
python run_workflow.py ConvertFFModelsToONNX
python run_workflow.py CalculateProcessFakeFactors
```

### DRSR

```bash
python run_workflow.py TrainDRSRSqueezedModels
python run_workflow.py CalculateDRSRCorrectionFactors
```

These stages are pulled in automatically by `CalculateFF` when
`--correction drsr` or `--correction drsr_nonclosure` is selected.

### Normalizing flow

```bash
python run_workflow.py TrainConditionalNF
```

### Optional single-DNN studies

```bash
python run_workflow.py TrainSqueezedSingleDNNModels
python run_workflow.py ConvertSingleDNNModels
python run_workflow.py CalculateSingleDNNFakeFactors
python run_workflow.py PlotSingleDNNFakeFactorDistributions
```

These are study tasks and are not part of the default fake-factor path.

### Optional uncertainty and interpretation studies

The public tasks include:

```text
TrainUncertaintyModels
SaveUncertaintyCombinedModels
CalculateFakeFactorModelUncertainty
CalculateFakeFactorDropoutMaskVariation
CalculateWjetsGradientCovarianceUncertainty
CalculateGroupedDNNTaylorCoefficients
PlotGroupedDNNTaylorCoefficients
CalculateUncertaintyModelTaylorCoefficients
PlotUncertaintyModelTaylorCoefficients
```

Run a task's help command before using a study-specific task:

```bash
python run_workflow.py CalculateFakeFactorModelUncertainty --help
```

## 7. Configuration

All configuration used by the pipeline is owned by:

```text
DNN_et/pipeline/config/
```

The most commonly edited files are:

| File | Change this when |
|---|---|
| `data_sources.yaml` | Input ROOT production changes |
| `selections.yaml` | A region, mask, or process definition changes |
| `variables_fake_factor.yaml` | FF/fraction model inputs change |
| `model_fake_factor.yaml` | FF DNN architecture or training changes |
| `variables_enrichment.yaml` | Enrichment model inputs change |
| `model_enrichment.yaml` | Enrichment training changes |
| `plotting.yaml` | Plotted variables or bins change |
| `labels.yaml` | Plot labels change |

See `config/README.md` for the full configuration contract.

After changing configuration, remember that law determines completeness from
declared output targets. An existing target is not automatically invalidated
just because a YAML file changed. Use a new output location/parameter or
deliberately remove only the affected generated artifact before rerunning.

## 8. Output locations

All generated artifacts live below:

```text
DNN_et/Law_workflow_results/
```

The main categories are:

```text
Law_workflow_results/
├── data/
│   ├── dataframe_complete.feather
│   ├── dataframe_complete_no_embedding.feather
│   └── features/
│       └── feature_registry.json
├── models/
└── plots/
```

Final fake-factor paths encode:

```text
method / squeezing / DRSR-loss-limit
```

The DRSR-loss-limit directory is present only for methods using DRSR.

## 9. Checking task parameters and status

Display the parameters of any task:

```bash
python run_workflow.py CalculateFF --help
```

Law prints the dependency status before execution. Typical states are:

| State | Meaning |
|---|---|
| `complete` | All declared outputs already exist and pass custom checks |
| `pending` | The task or one of its dependencies still needs to run |
| `failed` | The task raised an exception |

For the final fake-factor tasks, completion additionally checks that the
feature contains finite values for every required AR row.

## 10. Common issues

### `law` or another package is not found

The analysis environment has not been activated or does not contain the
workflow dependencies. Activate the intended environment before running
`run_workflow.py`.

### Input ROOT files are not found

Check both paths in `config/data_sources.yaml`. The no-embedding input must
contain the required true-tau samples.

### A correctionlib file is missing

Check the corresponding directory below
`DNN_et/corrections/fake_factors/`, or provide the task's path override such
as `--process-fractions-path`.

### CUDA runs out of memory

Inspect the requested task's parameters for a batch-size option and reduce it
where available. Training batch sizes are configured in the relevant
`config/model_*.yaml` file.

### A task is reported complete after configuration changed

Law tracks outputs, not arbitrary input-file contents. Remove only the
specific affected generated target or change a significant task parameter.
Do not delete the complete workflow-results directory.

### A final feature contains missing values

Final fake factors are defined on AR rows. Missing values outside AR can be
expected. Missing or non-finite values inside AR cause the canonical final
calculation task to remain incomplete.

## 11. Recommended command summary

```bash
# Default final FF: ML fractions + DRSR + non-closure
python run_workflow.py CalculateFF

# Default closure plots
python run_workflow.py PlotFF

# Explicit alternatives
python run_workflow.py CalculateFF --correction none
python run_workflow.py CalculateFF --correction nonclosure
python run_workflow.py CalculateFF --correction drsr

# Classic-fraction alternative
python run_workflow.py CalculateClassicFractionFF

# Inspect parameters
python run_workflow.py CalculateFF --help
```
