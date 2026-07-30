# Pipeline architecture

`logic.py` is the stable law task-discovery entry point. It contains no task
implementations.

For instructions on running the workflow, see `WORKFLOW_USAGE.md`.

## Packages

- `core/`: canonical paths, feature names, FF methods, and artifact metadata.
- `tasks/`: law tasks grouped by workflow domain. `tasks/workflow.py` currently
  contains the shared implementation graph; the smaller domain modules define
  the supported public surfaces.
- `data/`: dataset, region, feature registry, and feature store primitives.
- `models/`: neural-network definitions, serialization, and losses.
- `training/`: reusable model-training loops.
- `inference/`: model evaluation and FF feature construction.
- `visualization/`: enrichment, closure, distribution, and diagnostic plots.
- `logging_utils/`: workflow-specific log rendering.

## Fake-factor dependency graph

```text
TrainSqueezedModels
        |
ConvertFFModelsToONNX
        |
CalculateProcessFakeFactors
        +--------------------------+
        |                          |
CalculateClassicFractionFF   ML process fractions
                                   |
                       optional correction branches
                                   |
                              CalculateFF
```

`CalculateFF` and `PlotFF` are the preferred public commands. Their default
method is `drsr_nonclosure`.

## Dependency rules

1. `core` must not import task or analysis modules.
2. Tasks may orchestrate training, inference, and visualization, but numerical
   modules must not import tasks.
3. Feature and plot names must be constructed by `core.names` and `core.paths`.
4. New feature artifacts use `metadata.json`; do not introduce version numbers
   in marker filenames.
5. Compatibility modules are intentionally not retained on this branch.
