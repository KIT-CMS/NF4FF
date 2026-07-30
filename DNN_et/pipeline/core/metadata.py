import json
from pathlib import Path


SCHEMA_VERSION = 1


def write_feature_metadata(
    path,
    *,
    columns,
    method,
    grouping="njets",
    squeezing=None,
    squeezing_loss_limit=None,
    regions=("AR",),
):
    """Write machine-readable metadata for a row-index keyed feature artifact."""
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "columns": list(columns),
        "method": method,
        "grouping": grouping,
        "squeezing": squeezing,
        "squeezing_loss_limit": squeezing_loss_limit,
        "regions": list(regions),
        "index_column": "row_index",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path
