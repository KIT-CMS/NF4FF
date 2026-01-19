import pathlib
from pathlib import Path
from typing import Any, List, Union
import datetime
import os
import pickle
import logging
from logging_setup_configs import setup_logging
logger = setup_logging(logger=logging.getLogger(__name__))
import numpy as np
from dataclasses import dataclass
from typing import Any, Union, Generator
from contextlib import contextmanager
import random


class AutoPath(object):
    def __init__(self, base: pathlib.Path) -> None:
        self.base = base

    def __getattr__(self, attr: str):
        if hasattr(self.base, attr):  # If the attribute exists in pathlib.Path
            value = getattr(self.base, attr)
            if callable(value):
                def wrapper(*args, **kwargs):
                    result = value(*args, **kwargs)
                    if isinstance(result, pathlib.Path):
                        return AutoPath(result)  # If method returns a Path, wrap it
                    return result
                return wrapper
            return value

        new_path = self.base / attr  # Fallback behavior: missing attribute = folder name.
        new_path.mkdir(parents=True, exist_ok=True)
        return AutoPath(new_path)

    def __getitem__(self, key: str) -> "AutoPath":
        new_path = self.base / key
        new_path.mkdir(parents=True, exist_ok=True)
        return AutoPath(new_path)

    def __truediv__(self, other: str) -> "AutoPath":
        new_path = self.base / other
        if not Path(other).suffix:
            new_path.mkdir(parents=True, exist_ok=True)
        return AutoPath(new_path)

    def __fspath__(self) -> str:
        return str(self.base)

    def __str__(self) -> str:
        return str(self.base)

    def __repr__(self) -> str:
        return repr(self.base)

class StorePathHelper(object):
    def __init__(
        self,
        directory: Union[str, Path],
        date_directory: Union[str, Path] = None,
        time_directory: Union[str, Path] = None,
        shortcuts: Union[dict, None] = None,
        filename: str = "_.spm",
        **kwargs: Any,
    ) -> None:
        self._directory = Path(directory)

        now = datetime.datetime.now()
        self.date = Path(date_directory or now.strftime("%Y-%m-%d"))
        self.time = Path(
            time_directory or self._unique_time(
                Path(directory) / self.date,
                now.strftime("%H-%M-%S"),
            )
        )

        self.directory = Path(directory) / self.date / self.time
        pathlib.Path.mkdir(self.directory, exist_ok=True, parents=True)

        self.filename = filename
        self.shortcuts = shortcuts or {}
        self._name_mappings = {
            "time": {self.time: None},
            "date": {self.date: None},
        }

        self.make_snapshot()

    @property
    def autopath(self) -> AutoPath:
        return AutoPath(self.directory)

    def _unique_time(self, base: Union[str, Path], name: Union[str, Path]) -> Path:
        idx = 0
        while True:
            try:
                current_name = Path(f"{idx}_{name}")
                (base / current_name).mkdir(parents=True, exist_ok=False)
                return current_name
            except FileExistsError:
                idx += 1
            except OSError as e:
                raise OSError(f"Error creating directory {(base / current_name)}: {e}") from e

    def make_snapshot(self) -> None:
        with open(Path(self.filename), "wb") as pickle_file:
            pickle.dump(self, pickle_file, protocol=1)

    @classmethod
    def from_pickle(cls, file: Union[str, Path]) -> "StorePathHelper":
        try:
            with open(file, "rb") as f:
                return cls(**pickle.load(f))
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file} not found.")
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading pickle file {file}: {e}")
            raise e

    @classmethod
    def like(cls, other: "StorePathHelper", **kwargs: Any) -> "StorePathHelper":
        return cls(
            directory=other.directory,
            date_directory=other.date,
            time_directory=other.time,
            filename=other.filename,
        )

    def update_directory_names(self, end_time: bool = True, suffix: Union[str, None] = "complete") -> None:
        if end_time:
            now = datetime.datetime.now().strftime("%H-%M-%S")
            times_directory = f"{self.time}_{now}"

        if suffix:
            times_directory = f"{times_directory}_{suffix}"

        new_directory = self.directory.parent / times_directory

        os.rename(self.directory.absolute(), new_directory.absolute())
        self.directory = new_directory
        self.make_snapshot()

    def find(self, item: str, as_string: bool = False) -> Union[str, List[str]]:
        results = list(self.autopath.rglob(item))
        if as_string:
            results = [str(it) for it in results]
        return results[0] if results and len(results) == 1 else results

