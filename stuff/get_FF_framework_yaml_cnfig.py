import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, ".")
sys.path.insert(0, "../..")

from CODE.NN import load_model

ALLOWED_NORMAL_PROCESSES = {"Wjets"}#, "ttbar", "QCD"}



def load_input_names_from_model(torch_model_dir: Path):
    model = None
    try:
        model = load_model(torch_model_dir, device="cpu")
    except Exception:
        model = load_model(torch_model_dir, device="cpu")

    input_names = getattr(model, "_input_names", None)
    if not input_names:
        raise ValueError(f"Model at '{torch_model_dir}' has no _input_names metadata")
    return list(input_names)


def build_entry(process: str, era: str, channel: str, combined_models: Path):
    torch_model_dir = combined_models / process / "torch_model" 
    model_input = load_input_names_from_model(torch_model_dir)

    return {
        "model_path": str((combined_models / process / "onnx_model" / "model.onnx").resolve()),
        "model_input": model_input,
        "define_columns": {"event_parity": "event % 2"},
    }


def base_process_name(process: str) -> str:
    if process.endswith("_DR_SR_applied_correction"):
        return process.replace("_DR_SR_applied_correction", "")
    if process.endswith("_applied_correction"):
        return process.replace("_applied_correction", "")
    return process.replace("_DR_SR", "") if process.endswith("_DR_SR") else process


def choose_normal_processes(all_processes, corrected: bool):
    selected = []
    for base in sorted(ALLOWED_NORMAL_PROCESSES):
        # ttbar has no correction variants by design.
        if base == "ttbar":
            expected = base
        else:
            expected = f"{base}_applied_correction" if corrected else base

        if expected not in all_processes:
            mode = "corrected" if corrected else "uncorrected"
            raise ValueError(
                f"Requested {mode} normal model '{expected}' is missing in '{sorted(all_processes)}'"
            )
        selected.append(expected)

    return selected


def choose_drsr_processes(all_processes, corrected: bool):
    preferred = []
    bases = sorted({
        p.replace("_DR_SR_applied_correction", "").replace("_DR_SR", "")
        for p in all_processes
        if p.endswith("_DR_SR") or p.endswith("_DR_SR_applied_correction")
    })

    for base in bases:
        applied = f"{base}_DR_SR_applied_correction"
        drsr = f"{base}_DR_SR"
        expected = applied if corrected else drsr
        if expected not in all_processes:
            mode = "corrected" if corrected else "uncorrected"
            raise ValueError(
                f"Requested {mode} DR_SR model '{expected}' is missing in '{sorted(all_processes)}'"
            )
        preferred.append(expected)

    return preferred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-models", default ='/work/mmoser/NF4FF/DNN_et/CombinedModels' )
    ap.add_argument("--outdir", default = '.')
    ap.add_argument("--era", default = '2018')
    ap.add_argument("--channel", default = 'et')
    args = ap.parse_args()

    combined_models = Path(args.combined_models)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_processes = sorted([p.name for p in combined_models.iterdir() if p.is_dir()])
    normal = choose_normal_processes(set(all_processes), corrected=False)


    normal_yaml = {"target_processes": {}}
    drsr_yaml = {"target_processes": {}}

    for p in normal:
        key = base_process_name(p)
        normal_yaml["target_processes"][key] = build_entry(p, args.era, args.channel, combined_models)

    with open(outdir / f"fake_factors_models_{args.channel}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(normal_yaml, f, sort_keys=False)


    print("Done.")


if __name__ == "__main__":
    main()
