# Pipeline configuration

These files are the complete configuration interface of `DNN_et/pipeline`.
Pipeline code must use `core.paths.CONFIG_ROOT`; it must not load files from
the parent `DNN_et/configs` directory.

The configuration is split by responsibility because these values have
different consumers and change at different rates:

| File | Purpose |
|---|---|
| `data_sources.yaml` | Input ROOT directories |
| `selections.yaml` | Primitive masks, composed regions, and process selectors |
| `variables_fake_factor.yaml` | Inputs for FF, fraction, uncertainty, and interpretation models |
| `variables_enrichment.yaml` | Inputs for enrichment models |
| `variables_normalizing_flow.yaml` | Inputs for normalizing flows |
| `model_fake_factor.yaml` | Shared DNN architecture and training settings |
| `model_enrichment.yaml` | Enrichment training settings |
| `model_normalizing_flow.yaml` | Normalizing-flow settings |
| `grouping_enrichment.yaml` | QCD enrichment grouping; currently `njets` only |
| `plotting.yaml` | Plot variable sets and binning |
| `labels.yaml` | Human-readable plot labels |

## Conventions

- YAML keys consumed by Python are part of the pipeline interface. Rename
  them only together with their loader.
- Boolean values use `true` and `false`.
- Scientific notation is preferred for very small floating-point values.
- Model inputs belong in a `variables_*.yaml` file, not in model settings.
- Dataset locations belong only in `data_sources.yaml`.
- Selection expressions use pandas `DataFrame.eval` syntax.
- New generated-feature paths do not belong here; they are constructed by
  `core.paths` and registered in `feature_registry.json`.

## Data sources

The two paths in `data_sources.yaml` are site-specific. Change them for a new
input production or override `BuildDataset.config_path` /
`BuildDatasetNoEmbedding.config_path` with another file using the same schema.

## Selection schema

`selections.yaml` has three top-level mappings:

```yaml
masks:
  mask_name:
    - boolean expression

regions:
  region_name:
    - mask_name

processes:
  process_name: boolean expression
```

Every mask listed for a region is combined with logical AND. Process
expressions are evaluated independently of regions.
