from dataclasses import is_dataclass, fields
from typing import Tuple
import yaml

def load_variables(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    yaml_vars = config.get("variables", [])
    return yaml_vars

def load_config(path: str, cls):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return _from_dict(data, cls)

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