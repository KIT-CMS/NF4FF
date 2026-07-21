import argparse
import os
from datetime import datetime
from pathlib import Path

import nbformat
import papermill as pm

os.environ["NB_EXECUTION_MODE"] = "batch"


def strip_output(notebook_path: str):
    nb = nbformat.read(notebook_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(nb, notebook_path)


def main(files, clear_output: bool):
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    out_dir = Path("executed_notebooks") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    for nb in files:
        nb = Path(nb)
        out_nb = out_dir / nb.name

        print(f"▶ Running {nb} → {out_nb}")

        pm.execute_notebook(
            input_path=str(nb),
            output_path=str(out_nb),
            log_output=True,
            env={"NB_EXECUTION_MODE": "batch"},
        )

        if clear_output:
            print(f"🧹 Stripping outputs from {out_nb}")
            strip_output(out_nb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove all output cells from executed notebooks",
    )

    args = parser.parse_args()
    main(args.files, args.clear_output)