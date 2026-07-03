import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from classes import (
    DRSRCorrectedFakeFactorModel,
    FoldCombinedDNN,
    LikelihoodRatioCalculation,
    convert_models_to_onnx,
    load_data,
    load_model,
    save_model,
)
from groupings import GROUPING_NAMES, grouping_bounds, grouping_source


logger = logging.getLogger(__name__)

PROCESSES = ("wjets", "qcd", "ttbar")
FOLD_PARITIES = {
    "fold_even": 0,
    "fold_odd": 1,
}
PROCESS_OUTPUT_NAMES = {
    "wjets": "Wjets",
    "qcd": "QCD",
    "ttbar": "ttbar",
}

# Reference values from the previous fallback implementation.
TAU_DECAYMODE_REFERENCE_CONSTANTS = {
    "wjets": {
        ((0,),): 0.3950,
        ((1,),): 0.3284,
        ((10,),): 0.2163,
        ((11,),): 0.1050,
    },
    "qcd": {
        ((0,),): 0.2278,
        ((1,),): 0.2085,
        ((10,),): 0.1483,
        ((11,),): 0.0647,
    },
    "ttbar": {
        ((0,),): 0.3495,
        ((1,),): 0.3068,
        ((10,),): 0.2051,
        ((11,),): 0.1012,
    },
}


def _group_mask(values: np.ndarray, bounds: Tuple[int, ...]) -> np.ndarray:
    if len(bounds) == 1:
        return np.isclose(values, bounds[0], rtol=0.0, atol=1e-4)
    if len(bounds) == 2:
        return (values >= bounds[0]) & (values <= bounds[1])
    raise ValueError(f"Unsupported grouping bounds: {bounds}")


def _normalization_constants(
    df,
    process: str,
    grouping: str,
    parity: Union[int, None] = None,
) -> Dict[Union[str, Tuple[Tuple[int, ...], ...]], float]:
    constants, _ = _normalization_result(
        df,
        process,
        grouping,
        parity=parity,
    )
    return constants


def _normalization_result(
    df,
    process: str,
    grouping: str,
    parity: Union[int, None] = None,
) -> Tuple[Dict, Dict]:
    if process == "ttbar":
        signal = df.ttbar.SR_like_ttbar.events
        application = df.ttbar.AR_like_ttbar.events
        weight_column = "weight"
    else:
        signal = getattr(df.data, f"SR_like_{process}").events
        application = getattr(df.data, f"AR_like_{process}").events
        weight_column = f"reduced_weight_{process}_{grouping}_nominal"

    constants = {}
    diagnostics = {}
    source_column = grouping_source(grouping)
    process_bounds = grouping_bounds(grouping, process)
    for bounds in process_bounds:
        signal_mask = _group_mask(
            signal[source_column].to_numpy(dtype=np.float64),
            bounds,
        )
        application_mask = _group_mask(
            application[source_column].to_numpy(dtype=np.float64),
            bounds,
        )
        if parity is not None:
            signal_mask &= signal["event"].to_numpy() % 2 == parity
            application_mask &= application["event"].to_numpy() % 2 == parity

        numerator = float(signal.loc[signal_mask, weight_column].sum())
        denominator = float(
            application.loc[application_mask, weight_column].sum()
        )
        if not np.isfinite(denominator) or denominator == 0:
            raise ValueError(
                f"Cannot normalize {process}/{grouping}/{bounds}: "
                f"application-region yield is {denominator}."
            )

        value = float(numerator / denominator)
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite normalization for {process}/{grouping}/{bounds}."
            )
        constants[(bounds,)] = value
        diagnostics[_json_group_key((bounds,))] = {
            "normalization": value,
            "sr_yield": numerator,
            "ar_yield": denominator,
            "sr_events": int(signal_mask.sum()),
            "ar_events": int(application_mask.sum()),
        }

    constants["fallback"] = 1.0
    return constants, diagnostics


def _json_group_key(key: Tuple[Tuple[int, ...], ...]) -> str:
    bounds = key[0]
    return str(bounds[0]) if len(bounds) == 1 else f"{bounds[0]}-{bounds[1]}"


def _write_normalization_constants(
    constants: Dict[str, Dict[str, Dict[str, Dict]]],
    validation: Dict,
    path: Path,
) -> None:
    serializable_constants = {
        grouping: {
            process: {
                fold: {
                    (
                        key
                        if isinstance(key, str)
                        else _json_group_key(key)
                    ): value
                    for key, value in fold_constants.items()
                }
                for fold, fold_constants in process_constants.items()
            }
            for process, process_constants in grouping_constants.items()
        }
        for grouping, grouping_constants in constants.items()
    }
    serializable = {
        "constants": serializable_constants,
        "validation": validation,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


def _log_tau_decaymode_reference_comparison(
    calculated: Dict[str, Dict],
) -> Dict:
    comparison = {}
    for process, reference_values in TAU_DECAYMODE_REFERENCE_CONSTANTS.items():
        comparison[process] = {}
        for key, reference in reference_values.items():
            value = calculated[process][key]
            relative_difference = (
                (value - reference) / reference
                if reference != 0
                else float("nan")
            )
            group_key = _json_group_key(key)
            comparison[process][group_key] = {
                "calculated": value,
                "reference": reference,
                "relative_difference": relative_difference,
            }
            log = (
                logger.warning
                if abs(relative_difference) > 0.05
                else logger.info
            )
            log(
                "Normalization reference comparison %s/tau_decaymode_2/%s: "
                "calculated=%.8g, old_reference=%.8g, difference=%+.3f%%",
                process,
                group_key,
                value,
                reference,
                100.0 * relative_difference,
            )
    return comparison


def _normalization_validation(df) -> Tuple[Dict, Dict]:
    constants = {}
    validation = {
        "method": (
            "The union normalization must equal the AR-yield-weighted "
            "combination of the even- and odd-event normalizations."
        ),
        "fold_definition": {
            "fold_even": "event % 2 == 0",
            "fold_odd": "event % 2 == 1",
        },
        "fold_recombination": {},
    }

    for grouping in GROUPING_NAMES:
        constants[grouping] = {}
        validation["fold_recombination"][grouping] = {}
        for process in PROCESSES:
            _, union_diagnostics = _normalization_result(
                df,
                process,
                grouping,
            )
            fold_results = {
                fold: _normalization_result(
                    df,
                    process,
                    grouping,
                    parity=parity,
                )
                for fold, parity in FOLD_PARITIES.items()
            }
            constants[grouping][process] = {
                fold: result[0]
                for fold, result in fold_results.items()
            }

            process_validation = {}
            for bounds in grouping_bounds(grouping, process):
                group_key = _json_group_key((bounds,))
                union = union_diagnostics[group_key]
                even = fold_results["fold_even"][1][group_key]
                odd = fold_results["fold_odd"][1][group_key]

                sr_sum = even["sr_yield"] + odd["sr_yield"]
                ar_sum = even["ar_yield"] + odd["ar_yield"]
                recombined = (
                    even["normalization"] * even["ar_yield"]
                    + odd["normalization"] * odd["ar_yield"]
                ) / ar_sum

                yields_match = (
                    np.isclose(sr_sum, union["sr_yield"], rtol=1e-10, atol=1e-10)
                    and np.isclose(
                        ar_sum,
                        union["ar_yield"],
                        rtol=1e-10,
                        atol=1e-10,
                    )
                )
                normalization_matches = np.isclose(
                    recombined,
                    union["normalization"],
                    rtol=1e-10,
                    atol=1e-10,
                )
                if not yields_match or not normalization_matches:
                    raise ValueError(
                        "Fold normalization validation failed for "
                        f"{process}/{grouping}/{group_key}: "
                        f"union={union['normalization']}, "
                        f"recombined={recombined}, "
                        f"SR union/sum={union['sr_yield']}/{sr_sum}, "
                        f"AR union/sum={union['ar_yield']}/{ar_sum}."
                    )

                process_validation[group_key] = {
                    "union": union,
                    "fold_even": even,
                    "fold_odd": odd,
                    "recombined_normalization": recombined,
                    "fold_spread": (
                        odd["normalization"] - even["normalization"]
                    ),
                    "fold_spread_relative_to_union": (
                        (
                            odd["normalization"] - even["normalization"]
                        )
                        / union["normalization"]
                        if union["normalization"] != 0
                        else None
                    ),
                    "valid": True,
                }
                logger.info(
                    "Normalization validation %s/%s/%s: "
                    "union=%.8g, even=%.8g (SR=%.8g, AR=%.8g, n=%d/%d), "
                    "odd=%.8g (SR=%.8g, AR=%.8g, n=%d/%d), "
                    "recombined=%.8g",
                    process,
                    grouping,
                    group_key,
                    union["normalization"],
                    even["normalization"],
                    even["sr_yield"],
                    even["ar_yield"],
                    even["sr_events"],
                    even["ar_events"],
                    odd["normalization"],
                    odd["sr_yield"],
                    odd["ar_yield"],
                    odd["sr_events"],
                    odd["ar_events"],
                    recombined,
                )

            validation["fold_recombination"][grouping][
                process
            ] = process_validation

    full_tau_constants = {
        process: _normalization_constants(
            df,
            process,
            "tau_decaymode_2",
        )
        for process in PROCESSES
    }
    validation["old_tau_decaymode_reference"] = (
        _log_tau_decaymode_reference_comparison(full_tau_constants)
    )
    return constants, validation


def _model_paths(
    trained_models_dir: Path,
    grouping: str,
    process: str,
) -> Tuple[Path, Path]:
    model_dir = trained_models_dir / grouping / process
    even_path = model_dir / "fold_even"
    odd_path = model_dir / "fold_odd"
    for path in (even_path, odd_path):
        if not (path / "model_weights.pth").is_file():
            raise FileNotFoundError(f"Missing trained fold model: {path}")
    return even_path, odd_path


def ff_models_to_onnx(
    *,
    data_path: Union[str, Path],
    masks_path: Union[str, Path],
    trained_models_dir: Union[str, Path],
    reduced_weight_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> List[Path]:
    """Build likelihood-ratio FF models and export Torch and ONNX versions."""
    trained_models_dir = Path(trained_models_dir)
    reduced_weight_dir = Path(reduced_weight_dir)
    output_dir = Path(output_dir)

    df = load_data(data_path, masks_path)
    for process in ("wjets", "qcd"):
        for grouping in GROUPING_NAMES:
            df.load_feature_file(
                reduced_weight_dir
                / process
                / f"reduced_weight_{grouping}.feather"
            )

    constants, validation = _normalization_validation(df)
    normalization_path = output_dir / "normalization_constants.json"
    _write_normalization_constants(
        constants,
        validation,
        normalization_path,
    )

    outputs = [normalization_path]
    for grouping in GROUPING_NAMES:
        for process in PROCESSES:
            even_path, odd_path = _model_paths(
                trained_models_dir,
                grouping,
                process,
            )
            even_ff_model = LikelihoodRatioCalculation(
                model=load_model(even_path).eval(),
                normalization_constants=(
                    constants[grouping][process]["fold_even"]
                ),
                clip=(1e-4, 10.0),
            )
            odd_ff_model = LikelihoodRatioCalculation(
                model=load_model(odd_path).eval(),
                normalization_constants=(
                    constants[grouping][process]["fold_odd"]
                ),
                clip=(1e-4, 10.0),
            )
            combined_ff_model = FoldCombinedDNN(
                even_model=even_ff_model,
                odd_model=odd_ff_model,
                fold_id_name="event",
            )

            model_output_dir = (
                output_dir / PROCESS_OUTPUT_NAMES[process] / grouping
            )
            save_model(combined_ff_model, model_output_dir / "torch_model")
            onnx_path = model_output_dir / "onnx_model" / "model.onnx"
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            convert_models_to_onnx(
                torch_model=combined_ff_model,
                onnx_model_path=onnx_path,
            )
            outputs.append(onnx_path)

    logger.info(
        "Exported %d combined FF models and fold normalization constants.",
        len(outputs) - 1,
    )
    return outputs


def corrected_ff_models_to_onnx(
    *,
    combined_model_dir: Union[str, Path],
    drsr_model_dir: Union[str, Path],
    output_dir: Union[str, Path],
    grouping: str = "njets",
) -> List[Path]:
    """Build DRSR-corrected FF models and export Torch and ONNX versions."""
    combined_model_dir = Path(combined_model_dir)
    drsr_model_dir = Path(drsr_model_dir)
    output_dir = Path(output_dir)

    if grouping != "njets":
        raise ValueError(
            "DRSR-corrected combined models are only defined for the "
            f"njets grouping, got {grouping!r}."
        )

    outputs = []
    for process in PROCESSES:
        process_name = PROCESS_OUTPUT_NAMES[process]
        fake_factor_model_dir = (
            combined_model_dir
            / process_name
            / grouping
            / "torch_model"
        )
        correction_model_dir = drsr_model_dir / process
        for path, description in (
            (fake_factor_model_dir, "base combined FF model"),
            (correction_model_dir, "DRSR correction model"),
        ):
            if not (path / "model_weights.pth").is_file():
                raise FileNotFoundError(f"Missing {description}: {path}")

        fake_factor_model = load_model(fake_factor_model_dir).eval()
        correction_model = LikelihoodRatioCalculation(
            load_model(correction_model_dir).eval(),
            normalization_constants=1.0,
            clip=(1.0e-8, 1.0e30),
        )
        corrected_model = DRSRCorrectedFakeFactorModel(
            fake_factor_model=fake_factor_model,
            correction_model=correction_model,
        ).eval()

        model_output_dir = output_dir / process_name / grouping
        save_model(corrected_model, model_output_dir / "torch_model")
        torch_path = model_output_dir / "torch_model" / "model_weights.pth"
        outputs.append(torch_path)

        onnx_path = model_output_dir / "onnx_model" / "model.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        convert_models_to_onnx(
            torch_model=corrected_model,
            onnx_model_path=onnx_path,
        )
        outputs.append(onnx_path)

    metadata = {
        "combined_model_dir": str(combined_model_dir),
        "drsr_model_dir": str(drsr_model_dir),
        "grouping": grouping,
        "processes": list(PROCESSES),
        "correction": (
            "corrected_model(x) = combined_fake_factor_model(x) * "
            "DRSR_model(x)/(1 - DRSR_model(x))"
        ),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    outputs.append(metadata_path)

    logger.info(
        "Exported %d DRSR-corrected combined FF models to %s.",
        len(PROCESSES),
        output_dir,
    )
    return outputs
