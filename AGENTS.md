# AGENTS.md

## Scope
Instructions for AI coding agents working in this repository.

## First Steps
- Install Python deps from [requiremets.txt](requiremets.txt).
- Prefer running scripts from their project root to avoid relative-path breakage.
- Respect duplicate code areas: changes in one classifier tree often need same change in sibling tree.

## Project Map
- [Classifier_qcd/](Classifier_qcd): main classifier workflow (dataset, training, plotting).
- [Classifier/](Classifier): near-duplicate classifier tree.
- [Normalizing_Flow_et/](Normalizing_Flow_et): e-tau normalizing flow workflows.
- [Normalizing_Flow_mt/](Normalizing_Flow_mt): m-tau normalizing flow workflows.
- Shared root data artifacts in [data/](data).

## Common Run Commands
- Classifier dataset build: `python create_dataset.py` from [Classifier_qcd/](Classifier_qcd).
- Classifier training: `python training_qcd.py` or `python training_wjets.py` from [Classifier_qcd/](Classifier_qcd).
- NF correction training: `python src/DR_SR_correction_2.0/FF_correction_flow.py` from [Normalizing_Flow_et/](Normalizing_Flow_et).

## Conventions And Pitfalls
- Config-driven behavior from YAML files in [Classifier_qcd/configs/](Classifier_qcd/configs) and [Normalizing_Flow_et/configs/](Normalizing_Flow_et/configs).
- Some scripts manipulate import path with `sys.path.insert`; keep cwd and path assumptions stable.
- Several configs include environment-specific absolute paths (for example CEHP/CERN paths); do not silently rewrite without request.
- No single unified test harness detected at repo root; validate by running touched scripts or focused checks.

## Logging
- Custom logging usage documented in [Classifier_qcd/CustomLogging/README.md](Classifier_qcd/CustomLogging/README.md).
- Similar logging module also exists in sibling directories; keep behavior aligned if changing shared logging API.

## Link-First Rule
- Do not duplicate long documentation into instructions.
- Point to source docs/config files and keep agent notes short and actionable.
