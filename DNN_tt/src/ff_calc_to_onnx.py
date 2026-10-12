"""Export grouped, two-tau fake-factor models after running training.py.

Uses the saved fold-combined classifiers and existing ff_normalization.yaml;
no training data or normalization recalculation is needed.
"""

import logging
from pathlib import Path

from tap import Tap
import torch as t

from classes.Loading import load_config
from classes.NeuralNetworks import (
    LikelihoodRatioCalculation,
    convert_models_to_onnx,
    load_model,
)

logger = logging.getLogger(__name__)


class Args(Tap):
    config_path: str = str(Path(__file__).resolve().parents[1] / 'configs' / 'config_path.yaml')
    output_dir: str = ''  # Defaults to training_results/onnx.


def main():
    args = Args().parse_args()
    cfg_path = load_config(args.config_path)
    checkpoint_dir = Path(cfg_path['training_results'])
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / 'onnx'
    normalization = load_config(cfg_path['ff_normalization'])
    t.set_num_threads(8)

    groupings = (
        ('tau_decaymode', 'tau_dm', ((0,), (1,), (10,), (11,))),
        ('njets', 'njets', ((0,), (1,), (2, 1000))),
    )

    for group_label, normalization_key, groups in groupings:
        for process in ('tau1', 'tau2'):
            # YAML stores strings such as '0' and '2_1000'; the wrapper
            # matches the tuple bounds used by the trained GroupedDNN.
            constants = {
                bounds: float(normalization[process][normalization_key][
                    '_'.join(str(value) for value in bounds)
                ])
                for bounds in groups
            }

            # ----- calculate likelihood ratio -----
            model_path = checkpoint_dir / group_label / '2split' / process
            logger.info('Loading trained model from %s', model_path)
            model = load_model(model_path, device='cpu').eval()
            ff_model = LikelihoodRatioCalculation(
                model=model,
                normalization_constants=constants,
                clip=(0.0, 3.0),
            ).eval()

            # ----- export to ONNX -----
            onnx_path = output_dir / group_label / '2split' / process / 'model.onnx'
            convert_models_to_onnx(
                torch_model=ff_model,
                onnx_model_path=onnx_path,
            )
            logger.info('Exported fake-factor model to %s', onnx_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
