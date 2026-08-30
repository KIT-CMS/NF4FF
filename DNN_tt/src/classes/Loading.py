from dataclasses import is_dataclass, fields
from typing import Tuple

import numpy as np
import pandas as pd
import uproot
import yaml

from classes.DataHandling import SelectionManager, AnalysisDataFrame

def load_root_file_as_pd(file_path):
    with uproot.open(file_path) as file:
        data = file["ntuple"].arrays(file["ntuple"].keys(), library="pd")
    return data

def _to_yaml_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _to_yaml_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(value) for value in obj]
    return obj

def write_yaml_to_file(py_obj, filename):
    with open(f'{filename}', 'w',) as f :
        yaml.safe_dump(_to_yaml_safe(py_obj), f, sort_keys=False, default_flow_style=False) 

def load_data(feather_file, config_file):

    df = pd.read_feather(feather_file)
    #print('len df process == 0', len(df[df.process == 0]))

    manager = SelectionManager(config_file)

    return AnalysisDataFrame(df, manager)

def load_variables(yaml_path, vars: str):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    yaml_vars = config.get(vars, [])
    return yaml_vars

def load_config(path: str, cls=None):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if cls == None:
        return data

    return _from_dict(data, cls)

def load_labels(path):
    labels_by_channel = {}
    current_channel = None

    with open(path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n')
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(' '))

            if not stripped or stripped.startswith('#'):
                continue

            # Be tolerant if the channel key is accidentally indented by one space.
            if stripped.endswith(':') and ':' not in stripped[:-1] and indent <= 1:
                current_channel = stripped[:-1]
                labels_by_channel.setdefault(current_channel, {})
                continue

            if current_channel is None:
                continue

            if indent < 4:
                continue

            key_value = line.strip().split(':', 1)
            if len(key_value) != 2:
                continue

            key, value = key_value
            labels_by_channel[current_channel][key] = value.strip().strip('"').strip("'")

    return labels_by_channel

def _from_dict(data: dict, cls):
    """
    Minimal recursive dict → dataclass converter
    """

    if not is_dataclass(cls):
        return data

    kwargs = {}

    for field in fields(cls):
        value = data.get(field.name)

        if value is None:
            kwargs[field.name] = None
            continue

        # tuple conversion (important for hidden_nodes)
        if field.type == tuple or field.type == Tuple[int, ...]:
            kwargs[field.name] = tuple(value)

        # nested dataclass
        elif is_dataclass(field.type):
            kwargs[field.name] = _from_dict(value, field.type)

        else:
            kwargs[field.name] = value

    return cls(**kwargs)