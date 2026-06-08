import itertools
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union, Literal
from torch.nn.modules.loss import _Loss
#import CODE.HELPER as helper
import dill
import more_itertools as mit
import numpy as np
import torch as t
import torch.nn.functional as F
import classes.CustomLogging as log
import onnx

logger = log.setup_logging(logger=logging.getLogger(__name__))

class DNN(t.nn.Module):
    def __init__(
        self,
        input_nodes: int,
        hidden_nodes: Tuple[int, ...],
        output_nodes: int,
        dropout: Union[float, Tuple[float, ...]] = 0.0,
        activation: Union[Callable, str] = "ReLU",
        output_activation: Union[Callable, str] = "Sigmoid",
        input_names: Union[List[str], None] = None
    ) -> None:
        super().__init__()
        self._input_names = input_names
        self._input_nodes = input_nodes
        self._hidden_nodes = hidden_nodes
        self._output_nodes = output_nodes
        self._dropout = dropout if isinstance(dropout, tuple) else (dropout,) * (len(hidden_nodes) - 1)
        self._activation = activation
        self._output_activation = output_activation

        self._forward_auto_to_device = True


        if self._input_names and len(self._input_names) != input_nodes:
            logger.warning(f"Input names count ({len(self._input_names)}) does not match input_nodes ({input_nodes})")

        if isinstance(activation, str) and hasattr(t.nn, activation):
            activation = getattr(t.nn, activation)()
        elif callable(activation):
            activation = activation
        else:
            msg = "Not a valid layer activation function"
            logger.error(msg)
            raise TypeError(msg)

        if isinstance(output_activation, str) and hasattr(t.nn, output_activation):
            if output_activation == "Softmax":
                output_activation = t.nn.Softmax(dim=1)
            elif output_activation == "Sigmoid":
                output_activation = t.nn.Sigmoid()
            elif output_activation == "Linear":
                output_activation = t.nn.Identity()
            else:
                output_activation = getattr(t.nn, output_activation)()
        elif callable(output_activation):
            output_activation = output_activation
        else:
            msg = "Not a valid final activation function"
            logger.error(msg)
            raise TypeError(msg)


        layers = []
        for (n1, n2), drop in zip(
            mit.pairwise([input_nodes] + list(hidden_nodes) + [output_nodes]),
            [0.0] + list(self._dropout),
        ):
            layers.extend(
                [
                    t.nn.Linear(n1, n2),
                    activation,
                    t.nn.Dropout(drop),
                ]
            )

        layers.extend(
            [
                t.nn.Linear(hidden_nodes[-1], output_nodes),
                output_activation,
            ]
        )

        self.layers = t.nn.Sequential(*layers)

        # StandardScaler or RobustScaler
        self.register_buffer("_scaler_shift", t.full((input_nodes,), 0.0))
        self.register_buffer("_scaler_scale", t.full((input_nodes,), 1.0))


    @property
    def _is_initialized(self) -> bool:
        initialized = (t.isnan(self._scaler_shift) | t.isnan(self._scaler_scale)).sum() == 0
        initialized &= (self._scaler_scale != 1).all() & (self._scaler_shift != 0).all()
        return initialized

    def initialize_scaler(
        self,
        shift: Union[np.ndarray, t.Tensor, None] = None,
        scale: Union[np.ndarray, t.Tensor, None] = None,
        safety_epsilon: float = 1e-6,
    ) -> None:
        if self._is_initialized:
            logger.warning("Scaler already initialized. Overwriting the current values.")
        elif shift is not None and scale is not None:
            shift = t.from_numpy(shift) if isinstance(shift, np.ndarray) else shift
            scale = t.from_numpy(scale) if isinstance(scale, np.ndarray) else scale
            if safety_epsilon is not None:
                scale = scale.clamp(min=safety_epsilon)
        else:
            msg = """
                shift and scale must be both None
                (falling to default of shift=0.0, scale=1.0)
                or both not None raise ValueError
            """
            logger.error(msg)
            raise ValueError(msg)

        if (scale == 0).any():
            _idx = (scale == 0).nonzero(as_tuple=True)[0].tolist()
            msg = f"Scaler initialization failed: Features at input indices {_idx} have a scale/std-dev of 0.0.\n"
            logger.error(msg)
            raise ValueError(msg)

        self._scaler_shift.data[:] = shift
        self._scaler_scale.data[:] = scale

    def apply_scaler(self, x: t.Tensor) -> t.Tensor:
        return (x - self._scaler_shift.to(x.device)) / self._scaler_scale.to(x.device)

    def forward(self, X: t.Tensor) -> t.Tensor:
        if self._forward_auto_to_device:
            return self.layers.to(X.device)(self.apply_scaler(X))
        else:
            return self.layers(self.apply_scaler(X))

    @property
    def _imports(self) -> str:
        imports = ""
        if not isinstance(self._activation, str):
            imports += f"from {self._activation.__class__.__module__} import {self._activation.__class__.__name__}\n"
        if not isinstance(self._output_activation, str):
            imports += f"from {self._output_activation.__class__.__module__} import {self._output_activation.__class__.__name__}\n"
        return f"{imports}\n"

    @property
    def model_name(self) -> str:
        activation = f"'{self._activation}'" if isinstance(self._activation, str) else self._activation
        output_activation = f"'{self._output_activation}'" if isinstance(self._output_activation, str) else self._output_activation
        args = [
            f"input_nodes={self._input_nodes}",
            f"hidden_nodes={self._hidden_nodes}",
            f"output_nodes={self._output_nodes}",
            f"dropout={self._dropout}",
            f"activation={activation}",
            f"output_activation={output_activation}",
        ]

        if self._input_names is not None:
            args.append(f"input_names={self._input_names}")
        else:
            args.append("input_names=None")

        return f"{self.__class__.__name__}({', '.join(args)})"

    def __recreate__(self) -> str:
        return f"{self._imports}__model = {self.model_name}\n\n"


class GroupedLayerABC(t.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self._logic_pipeline: List[Tuple[Any, Any]] = []
        self._fallback_payload: Any = None
        self._wrapped_delegate: Any = None

    @abstractmethod
    def _execute_group(self, X: t.Tensor, payload: Any) -> t.Tensor:
        pass

    def forward(self, X: t.Tensor) -> t.Tensor:
        fallback_out = self._execute_group(X, self._fallback_payload)

        if fallback_out.dim() == 1:
            fallback_out = fallback_out.unsqueeze(1)

        if not self._logic_pipeline:
            return fallback_out

        batch_size = X.shape[0]

        output = t.zeros_like(fallback_out)
        processed_mask = t.zeros(batch_size, dtype=t.bool, device=X.device)

        for conditions, payload in self._logic_pipeline:
            current_mask = t.ones(batch_size, dtype=t.bool, device=X.device)
            for colume_idx, bounds in conditions:
                vals = X[:, colume_idx]

                if len(bounds) == 1:  # checks at trace time, not ONNX run time
                    current_mask = current_mask & (t.abs(vals - bounds[0]) < 1e-4)
                elif len(bounds) == 2:
                    lower, upper = bounds
                    current_mask = current_mask & (vals >= lower - 1e-4)
                    if upper != float("inf"):
                        current_mask = current_mask & (vals <= upper + 1e-4)
                else:
                    raise ValueError(f"Invalid bound: {bounds}")

            group_out = self._execute_group(X, payload)

            if group_out.dim() == 1:
                group_out = group_out.unsqueeze(1)

            mask_float = current_mask.to(dtype=output.dtype).unsqueeze(1)

            output = output + (group_out * mask_float)
            processed_mask = processed_mask | current_mask

        unprocessed_mask = ~processed_mask
        unprocessed_mask_float = unprocessed_mask.to(dtype=output.dtype).unsqueeze(1)
        output = output + (fallback_out * unprocessed_mask_float)

        return output

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self._wrapped_delegate is not None:
                return getattr(self._wrapped_delegate, name)
            raise

    @property
    def _imports(self) -> str:
        imports = set()

        if self.__class__.__module__ != "builtins":
            imports.add(f"from {self.__class__.__module__} import {self.__class__.__name__}")

        imports.add("from math import inf")

        def extract_recursive(obj: Any):
            if hasattr(obj, "__class__"):
                cls = obj.__class__
                if cls.__module__ != "builtins":
                    imports.add(f"from {cls.__module__} import {cls.__name__}")

            if hasattr(obj, "_imports"):
                for line in obj._imports.split("\n"):
                    if line.strip():
                        imports.add(line)

        extract_recursive(self._fallback_payload)

        for _, payload in self._logic_pipeline:
            extract_recursive(payload)

        if self._wrapped_delegate is not None:
            extract_recursive(self._wrapped_delegate)

        return "\n".join(sorted(list(imports))) + "\n"

    def __recreate__(self) -> str:
        return f"{self._imports}__model = {self.model_name}\n\n"


class InputSlicer(t.nn.Module):
    def __init__(
        self,
        model: t.nn.Module,
        indices: List[int],
        input_names: List[str],
    ):
        super().__init__()
        self.wrapped_model = model

        self.register_buffer("indices", t.tensor(indices, dtype=t.long))
        self._input_names = input_names

        self._forward_auto_to_device = True

    def forward(self, X: t.Tensor) -> t.Tensor:
        if self._forward_auto_to_device:
            return self.wrapped_model(X[:, self.indices.to(X.device)])
        else:
            return self.wrapped_model(X[:, self.indices])

    @property
    def model_name(self) -> str:
        return self.wrapped_model.model_name

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_model, name)


class GroupedDNN(GroupedLayerABC):
    def __init__(
        self,
        grouping: Dict[int, Tuple[Tuple[int, ...], ...]],
        default_model: DNN,
        specific_models: Union[Dict[Tuple[Tuple[int, ...], ...], DNN], None] = None,
        allowed_variables: Union[Dict[Tuple[Tuple[int, ...], ...], List[str]], None] = None,
    ) -> None:
        super().__init__()

        self.grouping = grouping
        self.sub_models = t.nn.ModuleDict()
        self.fallback_payload = default_model
        self.allowed_variables = allowed_variables

        self._fallback_payload = default_model
        self._wrapped_delegate = default_model

        if default_model._input_names is None:  # Ensure a complete DNN-like interface
            default_model._input_names = [f"x{i}" for i in range(default_model._input_nodes)]

        global_input_names = default_model._input_names
        sorted_col_indices = sorted(grouping.keys())
        group_lists = [grouping[k] for k in sorted_col_indices]
        specific_models = specific_models or {}
        allowed_variables = allowed_variables or {}

        for _comb in itertools.product(*group_lists):
            if (str_key := str(_comb)) in self.sub_models:
                continue

            group_var_names = allowed_variables[_comb] if _comb in allowed_variables else global_input_names

            try:
                indices = [global_input_names.index(name) for name in group_var_names]
            except ValueError as e:
                raise ValueError(f"Variable in group {_comb} not found in global inputs: {e}")

            base_model = (
                specific_models[_comb]
                if _comb in specific_models
                else self._create_reduced_model(default_model, indices, group_var_names)
            )
            model = InputSlicer(base_model, indices, group_var_names) if len(indices) < len(global_input_names) else base_model

            self.sub_models[str_key] = model

            conds = list(zip(sorted_col_indices, _comb))
            self._logic_pipeline.append((conds, model))

    def _create_reduced_model(self, original_model: DNN, feature_indices: List[int], new_input_names: List[str]) -> DNN:
        assert isinstance(original_model, DNN), "Original model must be an instance of DNN"

        new_model = original_model.__class__(
            input_nodes=len(feature_indices),
            hidden_nodes=original_model._hidden_nodes,
            output_nodes=original_model._output_nodes,
            dropout=original_model._dropout,
            activation=original_model._activation,
            output_activation=original_model._output_activation,
            input_names=list(new_input_names),
        )

        new_model._forward_auto_to_device = original_model._forward_auto_to_device

        idx = t.tensor(feature_indices, dtype=t.long, device=original_model._scaler_shift.device)
        shift = t.index_select(original_model._scaler_shift.detach(), 0, idx).clone()
        scale = t.index_select(original_model._scaler_scale.detach(), 0, idx).clone()

        with t.no_grad():
            new_model._scaler_shift.copy_(shift.to(new_model._scaler_shift.device))
            new_model._scaler_scale.copy_(scale.to(new_model._scaler_scale.device))

        return new_model

    def _execute_group(self, X, payload) -> t.Tensor:
        return payload(X)  # payload is a model

    @property
    def model_name(self) -> str:
        spec_models_parts = []
        for conds, model in self._logic_pipeline:
            key_tuple = tuple(c[1] for c in conds)
            spec_models_parts.append(f"{key_tuple}: {model.model_name}")

        spec_models_str = "{" + ", ".join(spec_models_parts) + "}"

        return (
            f"{self.__class__.__name__}("
            f"grouping={repr(self.grouping)}, "
            f"default_model={self._fallback_payload.model_name}, "
            f"specific_models={spec_models_str}, "
            f"allowed_variables={self.allowed_variables},"
            ")"
        )


class FoldCombinedDNN(t.nn.Module):
    def __init__(
        self,
        even_model: Union[DNN, GroupedDNN, t.nn.Module],
        odd_model: Union[DNN, GroupedDNN, t.nn.Module],
        fold_id_name: str = "event_parity",
    ) -> None:
        super(FoldCombinedDNN, self).__init__()
        self.even_model = even_model  # Even Model: Trained on ODD -> Use for EVEN events
        self.odd_model = odd_model  # Odd Model: Trained on EVEN -> Use for ODD events
        self._fold_id_name = fold_id_name
        self._input_nodes = even_model._input_nodes + 1  # +1 for event ID

        if even_model._input_names is not None:
            self._input_names = [fold_id_name] + even_model._input_names

    def forward(self, X: t.Tensor) -> t.Tensor:
        even_event_ids, features = (X[0, ...].long() % 2 == 0).squeeze(), X[1:, ...].T

        output_even = self.even_model(features).squeeze()
        output_odd = self.odd_model(features).squeeze()

        condition = even_event_ids.view(-1, 1) if output_even.dim() > 1 else even_event_ids

        return t.where(condition, output_even, output_odd)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.even_model, name)

    @property
    def _imports(self) -> str:
        imports = set()

        if self.__class__.__module__ != "builtins":
            imports.add(f"from {self.__class__.__module__} import {self.__class__.__name__}")

        def add_model_imports(model):
            if hasattr(model, "__class__") and model.__class__.__module__ != "builtins":
                imports.add(f"from {model.__class__.__module__} import {model.__class__.__name__}")

            if hasattr(model, "_imports"):
                for line in model._imports.split("\n"):
                    if line.strip():
                        imports.add(line)

        add_model_imports(self.even_model)
        add_model_imports(self.odd_model)

        return "\n".join(sorted(list(imports))) + "\n"

    @property
    def model_name(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"even_model={self.even_model.model_name}, "
            f"odd_model={self.odd_model.model_name}, "
            f"fold_id_name='{self._fold_id_name}', "
            ")"
        )

    def __recreate__(self) -> str:
        return f"{self._imports}__model = {self.model_name}\n\n"


def save_model(item: t.nn.Module, path: Path) -> None:
    path = Path(path)
    assert not path.suffix, "Provide a directory, not a file name"
    path.mkdir(parents=True, exist_ok=True)

    full_model_path = path.joinpath("model_full.dill")
    weights_path = path.joinpath("model_weights.pth")
    model_recreation_path = path.joinpath("model_recreation.py")

    # Move model to CPU before saving
    item = item.to('cpu')
    for name, buf in item.named_buffers():
        if "_scaler_shift" in name:
            logger.debug(f"{name} before saving: {buf}")

    try:
        with open(full_model_path, "wb") as f:
            dill.dump(item, f)
            logger.info(f"Model {item.__class__.__name__} saved to {full_model_path}")
    except Exception as e:
        logger.error(f"Error saving complete model {item.__class__.__name__} to {path} failed")
        logger.error(f"Error: {e}")

    t.save(item.state_dict(), str(weights_path))
    logger.info(f"Model {item.__class__.__name__} weights saved to {weights_path}")
    with open(model_recreation_path, "w") as f:
        f.write(f"from {item.__class__.__module__} import {item.__class__.__name__}\n{str(item.__recreate__())}")
        logger.info(f"Model {item.__class__.__name__} recreation snippet saved to {model_recreation_path}")


def load_model(
    path: Path,
    device: str = "cpu",
    transverse_path: bool = False,
    force_recreate: bool = False,
) -> t.nn.Module:
    path = Path(path).resolve()

    if transverse_path:
        _path = path
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

        if len((date_folders := [p for p in path.iterdir() if p.is_dir()])) != 1:
            with log.LogContext(logger).logging_raised_Error():
                raise ValueError(f"transverse_path=True: Found {len(date_folders)}: {[p.name for p in date_folders]}, not unique")

        if len((time_folders := [p for p in date_folders[0].iterdir() if p.is_dir()])) != 1:
            with log.LogContext(logger).logging_raised_Error():
                raise ValueError(f"transverse_path=True: Found {len(time_folders)}: {[p.name for p in time_folders]}, not unique")

        path = path / date_folders[0] / time_folders[0] / "model"
        logger.info(f"Auto-traversed path from '{_path}' to '{path}'")

    assert path.is_dir(), "Provide a directory, not a file name"

    full_model_path = path.joinpath("model_full.dill")
    weights_path = path.joinpath("model_weights.pth")
    model_recreation_path = path.joinpath("model_recreation.py")

    def log_nested_scaler_shifts(model: t.nn.Module, label: str) -> None:
        for name, buffer in model.named_buffers():
            if name.endswith("_scaler_shift"):
                logger.debug(f"{name} on {label}: {buffer}")

    if not force_recreate:
        try:
            with open(full_model_path, "rb") as f:
                __model = dill.load(f)
                #logger.info(f"Model {__model.__class__.__name__} loaded from {full_model_path}")
                __model = __model.to('cpu')
                log_nested_scaler_shifts(__model, "cpu")
                __model = __model.to(device)
                log_nested_scaler_shifts(__model, device)
                return __model
        except Exception as e:
            logger.warning(f"Loading complete model from {full_model_path} failed\nError: {e}\nTrying to load model weights and recreation snippet")
    else:
        logger.info("force_recreate=True. Skipping complete model dill file and forcing code recreation.")

    model_weights = t.load(weights_path, map_location=device)
    logger.info(f"Model weights loaded from {weights_path}")

    scaler_entries = {k: v for k, v in model_weights.items() if k.endswith("_scaler_shift")}
    if scaler_entries:
        logger.debug(f"State dict _scaler_shift entries: {list(scaler_entries.keys())}")

    with open(model_recreation_path, "r") as f:
        code = f.read()
        logger.info(f"Running recreation snippet from {model_recreation_path}")
        local_vars = {}
        exec(code, globals(), local_vars)
        __model = local_vars['__model']
        __model.load_state_dict(model_weights)
        logger.info(f"Model {__model.__class__.__name__} recreated from {model_recreation_path}")
        log_nested_scaler_shifts(__model, device)
        return __model.to(device)


def load_fold_combined_model(
    even_model_path: Path,  # usually fold0
    odd_model_path: Path,   # usually fold1
) -> FoldCombinedDNN:
    return FoldCombinedDNN(
        even_model=load_model(even_model_path).eval(),
        odd_model=load_model(odd_model_path).eval(),
    )




@contextmanager
def eval_state(model: t.nn.Module) -> Iterable[t.nn.Module]:
    was_training = model.training
    try:
        model.eval()
        yield model
    finally:
        if was_training:
            model.train()


@contextmanager
def train_state(model: t.nn.Module) -> Iterable[t.nn.Module]:
    was_training = model.training
    try:
        model.train()
        yield model
    finally:
        if not was_training:
            model.eval()


def build_manual_scaler(model: t.nn.Module) -> Callable[[t.Tensor], t.Tensor]:

    if isinstance(model, FoldCombinedDNN):
        even_scaler = build_manual_scaler(model.even_model)
        odd_scaler = build_manual_scaler(model.odd_model)

        def _scaler_function(X: t.Tensor) -> t.Tensor:
            even_mask, features = (X[0, ...].long() % 2 == 0).squeeze(), X[1:, ...].T
            scaled_even, scaled_odd = even_scaler(features), odd_scaler(features)

            condition = even_mask.view(-1, 1) if scaled_even.dim() > 1 else even_mask
            scaled_features = t.where(condition, scaled_even, scaled_odd)

            out_X = X.clone()
            out_X[1:, ...] = scaled_features.T

            return out_X

        return _scaler_function

    elif isinstance(model, GroupedDNN):
        pipeline = [(conds, build_manual_scaler(payload)) for conds, payload in model._logic_pipeline]
        fallback_scaler = build_manual_scaler(model._fallback_payload)

        def _scaler_function(X: t.Tensor) -> t.Tensor:
            output, batch_size = t.zeros_like(X), X.shape[0]
            processed_mask = t.zeros(batch_size, dtype=t.bool, device=X.device)

            for conditions, payload_scaler in pipeline:
                current_mask = t.ones(batch_size, dtype=t.bool, device=X.device)
                for colume_idx, bounds in conditions:
                    vals = X[:, colume_idx]
                    if len(bounds) == 1:
                        current_mask = current_mask & (t.abs(vals - bounds[0]) < 1e-4)
                    elif len(bounds) == 2:
                        current_mask = current_mask & (vals >= bounds[0])
                        if bounds[1] != float("inf"):
                            current_mask = current_mask & (vals <= bounds[1])

                group_out, mask_float = payload_scaler(X), current_mask.to(dtype=X.dtype).unsqueeze(1)
                output = output + (group_out * mask_float)
                processed_mask = processed_mask | current_mask

            unprocessed_mask = ~processed_mask
            output = output + (fallback_scaler(X) * unprocessed_mask.to(dtype=X.dtype).unsqueeze(1))

            return output

        return _scaler_function

    elif hasattr(model, "wrapped_model"):
        return build_manual_scaler(model.wrapped_model)

    elif hasattr(model, "_scaler_shift") and hasattr(model, "_scaler_scale"):
        shift = model._scaler_shift.clone().detach()
        scale = model._scaler_scale.clone().detach()

        def _scaler_function(X: t.Tensor) -> t.Tensor:
            return (X - shift.to(X.device)) / scale.to(X.device)

        return _scaler_function

    else:
        raise ValueError(f"Cannot build manual scaler for type {type(model)}")


@contextmanager
def temporary_extract_scaler_callable(
    model: t.nn.Module,
) -> Iterable[Tuple[t.nn.Module, Callable[[t.Tensor], t.Tensor]]]:
    manual_scaler, originals = build_manual_scaler(model), {}

    try:
        for name, buffer in model.named_buffers():
            if name.endswith("_scaler_shift"):
                originals[name] = buffer.clone()
                buffer.fill_(0.0)
            elif name.endswith("_scaler_scale"):
                originals[name] = buffer.clone()
                buffer.fill_(1.0)

        yield model, manual_scaler

    finally:
        for name, buffer in model.named_buffers():
            if name in originals:
                buffer.copy_(originals[name])


def convert_models_to_onnx(
    torch_model: Union[t.nn.Module, None] = None,
    torch_model_dir: Union[str, List[str], None] = None,
    onnx_model_path: Union[str, Path] = "__model.onnx",
) -> None:
    assert (torch_model is not None) ^ (torch_model_dir is not None), "Provide either torch_model or torch_model_dir, not both"

    if torch_model is not None:
        model = torch_model.eval()
        logger.info("Model provided directly as torch_model argument, using it for ONNX conversion.")

    elif torch_model_dir is not None:
        if isinstance(torch_model_dir, str):
            model = load_model(torch_model_dir).eval().to("cpu")
        elif isinstance(torch_model_dir, list):
            logger.info("Assuming list of fold models is provided [fold0, fold1] > [even_model_path, odd_model_path]")
            assert len(torch_model_dir) == 2, "Provide exactly two model paths for fold combined model"
            model = load_fold_combined_model(*torch_model_dir).eval().to("cpu")
    else:
        raise ValueError("No model provided for ONNX conversion")

    toggled = []
    for m in model.modules():
        if hasattr(m, "_forward_auto_to_device"):
            toggled.append((m, "_forward_auto_to_device", m._forward_auto_to_device))
            m._forward_auto_to_device = False

    try:
        logger.info(f"Model loaded successfully from {torch_model_dir}")

        example_inputs = (t.randn(model._input_nodes, 1),)
        logger.info(f"Example inputs created with shape: {example_inputs[0].shape}")
        logger.info("Exporting model to ONNX format...")

        onnx_program = t.onnx.export(
            model,
            example_inputs,
            input_names=["input_tensor"],
            dynamo=True,
            optimize=True,
            verify=True,
            profile=True,
        )

        if (input_names := getattr(model, "_input_names", None)) is not None:
            logger.info(f"Adding input names metadata to ONNX model: {input_names}")
            model_proto = onnx_program.model_proto
            meta = model_proto.metadata_props.add()
            meta.key = "input_tensor"
            meta.value = "Input features in order:\n\n" + "\n".join(input_names)

        onnx.save(
            model_proto,
            onnx_model_path,
            save_as_external_data=False,
        )

        logger.info(f"Model successfully converted to ONNX format at {onnx_model_path}")
    finally:
        for m, attr, old in toggled:
            setattr(m, attr, old)



class LikelihoodRatioCalculation(GroupedLayerABC):
    def __init__(
        self,
        model: Union[t.nn.Module, DNN, GroupedDNN],
        normalization_constants: Union[Dict[Any, float], float] = 1.0,
        clip: Tuple[float, float] = (1e-4, 1.0),
    ) -> None:
        super().__init__()
        self.wrapped_model = model
        self._wrapped_delegate = model
        self.clip = clip
        self._init_constants = normalization_constants

        self._is_grouped_norm = isinstance(normalization_constants, dict)
        self._norm_dict = normalization_constants if self._is_grouped_norm else {}
        self._fallback_norm = float(self._norm_dict.get("fallback", 1.0)) if self._is_grouped_norm else float(normalization_constants)
        self._group_specs: List[Tuple[List[Tuple[int, Tuple[Any, ...]]], float]] = []

        if self._is_grouped_norm and hasattr(model, "_logic_pipeline"):
            for conditions, _ in model._logic_pipeline:
                key = tuple(cond[1] for cond in conditions)
                norm = self._resolve_norm_value(key)
                self._group_specs.append((conditions, norm))

    def _resolve_norm_value(self, key: Any) -> float:
        if not self._is_grouped_norm:
            return self._fallback_norm

        if key in self._norm_dict:
            return float(self._norm_dict[key])

        if isinstance(key, tuple) and len(key) == 1 and key[0] in self._norm_dict:
            return float(self._norm_dict[key[0]])

        return self._fallback_norm

    def _extract_values(self, X: t.Tensor, col_idx: int) -> t.Tensor:
        # FoldCombinedDNN consumes [1 + n_features, N] where row 0 is event parity.
        if isinstance(self.wrapped_model, FoldCombinedDNN):
            feature_row_idx = col_idx + 1
            if feature_row_idx >= X.shape[0]:
                raise IndexError(
                    f"Feature index {col_idx} out of bounds for FoldCombined input shape {tuple(X.shape)}"
                )
            return X[feature_row_idx, ...]

        if col_idx >= X.shape[1]:
            raise IndexError(f"Feature index {col_idx} out of bounds for input shape {tuple(X.shape)}")
        return X[:, col_idx]

    def _build_condition_mask(self, X: t.Tensor, conditions: List[Tuple[int, Tuple[Any, ...]]], n_events: int) -> t.Tensor:
        mask = t.ones(n_events, dtype=t.bool, device=X.device)

        for col_idx, bounds in conditions:
            vals = self._extract_values(X, col_idx)

            if len(bounds) == 1:
                target = float(bounds[0])
                target_tensor = t.tensor(target, dtype=vals.dtype, device=vals.device)
                mask = mask & t.isclose(vals, target_tensor, atol=1e-4, rtol=0.0)
            elif len(bounds) == 2:
                low, high = float(bounds[0]), float(bounds[1])
                mask = mask & (vals >= (low - 1e-4))
                if high != float("inf"):
                    mask = mask & (vals <= (high + 1e-4))
            else:
                raise ValueError(f"Invalid bound: {bounds}")

        return mask

    def forward(self, X: t.Tensor) -> t.Tensor:
        if not isinstance(X, t.Tensor):
            raise TypeError(f"LikelihoodRatioCalculation expects torch.Tensor input, got {type(X)}")

        output = self.wrapped_model(X)
        if output.dim() == 0:
            output = output.unsqueeze(0)
        fraction = output / (1 - output + 1e-8)

        if not self._is_grouped_norm:
            return (fraction * self._fallback_norm).clamp(*self.clip)

        n_events = fraction.shape[0]
        norm_vec = t.full((n_events,), self._fallback_norm, dtype=fraction.dtype, device=fraction.device)

        for conditions, norm in self._group_specs:
            cond_mask = self._build_condition_mask(X, conditions, n_events)
            norm_vec = t.where(cond_mask, t.full_like(norm_vec, float(norm)), norm_vec)

        if fraction.dim() > 1:
            norm_vec = norm_vec.unsqueeze(1)

        return (fraction * norm_vec).clamp(*self.clip)

    def _execute_group(self, X, payload) -> t.Tensor:
        output = self.wrapped_model(X)
        fraction = output / (1 - output + 1e-8)
        return (fraction * float(payload)).clamp(*self.clip)

    @property
    def model_name(self) -> str:
        delegate_name = getattr(self._wrapped_delegate, "model_name", self._wrapped_delegate.__class__.__name__)
        return (
            f"{self.__class__.__name__}("
            f"model={delegate_name}, "
            f"normalization_constants={repr(self._init_constants)}, "
            f"clip={repr(self.clip)}"
            ")"
        )



class FixedMaskDropout(t.nn.Module):
    def __init__(self, ensemble_size: int, feature_dim: int, p: float):
        super().__init__()
        self.ensemble_size = ensemble_size
        total_slots = ensemble_size + 1  # 0 always 1.0, 1 to N are random

        masks = t.ones(total_slots, feature_dim)
        rand_masks = (t.rand(ensemble_size, feature_dim) > p).float() / (1.0 - p)  # scale to keep same expected value magnitude
        masks[1:] = rand_masks

        self.register_buffer("masks", masks)

    def forward(self, x: t.Tensor) -> t.Tensor:
        total_rows = x.shape[0]
        batch_size = total_rows // (self.ensemble_size + 1)
        feature_dim = x.shape[1]
        x = x.reshape(self.ensemble_size + 1, batch_size, feature_dim)
        x = x * self.masks.unsqueeze(1)
        return x.reshape(total_rows, feature_dim)



class EnsembleStatUncWrapper(t.nn.Module):
    def __init__(
        self,
        model: t.nn.Module,
        ensemble_size: int = 10,
        direction: Literal["Nominal", "Up", "Down"] = "Nominal",
        sigma: float = 1.0,
        vary_index: Union[int, None] = None,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.direction = direction
        self.sigma = sigma
        self.vary_index = vary_index
        self.wrapped_model = model

        self._input_nodes = getattr(model, "_input_nodes", None)
        self._input_names = getattr(model, "_input_names", None)
        self._fold_id_name = getattr(model, "_fold_id_name", "event_parity")

        self._replace_layers(self.wrapped_model)

    def _replace_layers(self, module):
        for name, child in module.named_children():
            if isinstance(child, t.nn.Dropout) and child.p > 0:
                parent_seq = module
                layer_list = list(parent_seq)
                layer_idx = layer_list.index(child)
                prev_linear = layer_list[layer_idx - 2]  # preceding Activation + Dropout

                new_dropout = FixedMaskDropout(
                    ensemble_size=self.ensemble_size,
                    feature_dim=prev_linear.out_features,
                    p=child.p
                )
                setattr(module, name, new_dropout)
            else:
                self._replace_layers(child)

    def forward(self, X: t.Tensor) -> t.Tensor:
        outputs = self.wrapped_model(X.repeat(1, self.ensemble_size + 1))
        outputs = outputs.reshape(self.ensemble_size + 1, X.shape[1], *outputs.shape[1:])

        nominal = outputs[0]
        std = t.std(outputs[1:], dim=0, unbiased=True)
        mean = t.mean(outputs[1:], dim=0)
        total_uncertainty = ((mean - nominal) ** 2 + std ** 2).sqrt()

        if self.vary_index is not None:
            idx_mask = t.zeros(nominal.shape[-1], device=nominal.device, dtype=nominal.dtype)
            idx_mask[self.vary_index] = 1.0

            nominal_value = nominal[..., self.vary_index: self.vary_index + 1]
            uncertainty_value = total_uncertainty[..., self.vary_index: self.vary_index + 1]

            if self.direction == "Up":
                shifted_value = t.clamp(nominal_value + self.sigma * uncertainty_value, max=1.0)
            elif self.direction == "Down":
                shifted_value = t.clamp(nominal_value - self.sigma * uncertainty_value, min=0.0)

            R_old, R_new = 1.0 - nominal_value, 1.0 - shifted_value
            scale_factor = t.where(R_old > 1e-6, R_new / R_old, t.zeros_like(R_old))  # if norm_v becomes rounding 1.0

            return (shifted_value * idx_mask) + (nominal * (1.0 - idx_mask) * scale_factor)

        if self.direction == "Up":
            return nominal + self.sigma * total_uncertainty / 2
        elif self.direction == "Down":
            return nominal - self.sigma * total_uncertainty / 2
        else:
            return nominal  # should actually never happen :)

    @property
    def _imports(self) -> str:
        base_imports = getattr(self.wrapped_model, "_imports", "")
        wrapper_imports = f"from {self.__class__.__module__} import {self.__class__.__name__}\n"
        return base_imports + wrapper_imports

    @property
    def model_name(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.wrapped_model.model_name}, "
            f"ensemble_size={self.ensemble_size}, "
            f"direction='{self.direction}'"
            f")"
        )

    def __recreate__(self) -> str:
        return f"{self._imports}__model = {self.model_name}\n\n"

