import contextlib
import copy
import gzip
import json
import logging
import lzma
import math
import os
import random
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import MISSING
from dataclasses import fields
from dataclasses import fields as dc_fields
from dataclasses import is_dataclass
from dataclasses import is_dataclass as dc_is_dataclass
from functools import reduce
from typing import (Any, Callable, Dict, Generator, Iterable, Iterator, List,
                    Optional, Tuple, Type, Union, get_args, get_origin)
from dataclasses import dataclass, KW_ONLY

import fsspec

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch as t
from tap import Tap
from tqdm import tqdm
from CustomLogging import setup_logging, LogContext

logger = setup_logging(logger=logging.getLogger(__name__))

@contextmanager
def rng_seed(seed: int) -> Generator[None, None, None]:
    np_rng_state, py_rng_state = np.random.get_state(), random.getstate()
    t_rng_state = t.get_rng_state()

    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(np_rng_state)
        random.setstate(py_rng_state)
        t.set_rng_state(t_rng_state)

def get_compression_lib_and_suffix(compression: Union[str, None]) -> Tuple[Union[str, None], str]:
    if compression is None:
        return None, ""
    elif compression == "gzip":
        return gzip, ".gz"
    elif compression == "lzma":
        return lzma, ".xz"
    else:
        raise NotImplementedError(f"Compression '{compression}' is not implemented.")

def custom_chunking(iterable_obj: Any, /, chunks: Optional[int] = None, dim: Union[int, Any] = 0) -> Generator:
    if isinstance(iterable_obj, (BatchLoader, LazyBatchLoader)):
        if chunks is None:
            chunks = iterable_obj.num_batches
            for i in range(chunks):
                yield iterable_obj[i]
        else:
            if not iterable_obj.lengths:
                return None  # empty dataset
            total_length = next(iter(iterable_obj.lengths.values()))

            quotient, remainder = divmod(total_length, chunks)
            chunk_sizes = [quotient + 1 if i < remainder else quotient for i in range(chunks)]

            buffer, current_buffer_len = None, 0

            def slicer(obj, start, end):
                return obj[start:end]

            target_idx = 0
            stored_iterator = iter(iterable_obj)
            while target_idx < len(chunk_sizes):
                target_size = chunk_sizes[target_idx]
                while current_buffer_len < target_size:
                    try:
                        next_batch = next(stored_iterator)
                        next_len = next(iter(iterable_obj.collection_class.to_collection(next_batch).get_leaves().values())).shape[0]
                        if buffer is None:
                            buffer = next_batch
                        else:
                            buffer = iterable_obj.collection_class.cat(buffer, next_batch)
                        current_buffer_len += next_len
                    except StopIteration:
                        break
                if current_buffer_len == 0:
                    break
                actual_slice_size = min(target_size, current_buffer_len)

                yield Recursevly._apply(lambda x: x[0: actual_slice_size], buffer)

                if actual_slice_size < current_buffer_len:
                    buffer = Recursevly._apply(lambda x: x[actual_slice_size:], buffer)
                    current_buffer_len -= actual_slice_size
                else:
                    buffer = None
                    current_buffer_len = 0
                target_idx += 1

    if chunks is None:
        raise ValueError("chunks must be specified for non-lazy datasets")

    if iterable_obj is None:
        for _ in range(chunks):
            yield None

    elif isinstance(iterable_obj, t.Tensor):
        if not isinstance(dim, int):
            raise TypeError(f"For Tensor input, 'dim' must be an int, got {type(dim)}")
        for chunk_tensor in t.chunk(iterable_obj, chunks, dim=dim):
            yield chunk_tensor

    elif isinstance(iterable_obj, (list, tuple)) and all(isinstance(it, t.Tensor) for it in iterable_obj):
        if not isinstance(dim, int):
            raise TypeError("For list/tuple input of Tensors, 'dim' must be an int.")
        for chunk_parts in zip(*[custom_chunking(it, chunks, dim) for it in iterable_obj]):
            yield type(iterable_obj)(chunk_parts)

    elif isinstance(iterable_obj, (list, tuple)) and all(is_dataclass(it) for it in iterable_obj):
        if isinstance(dim, int) or (is_dataclass(dim) and isinstance(type(dim), CollectionMeta)):
            for chunk_parts in zip(*[custom_chunking(it, chunks, dim) for it in iterable_obj]):
                yield type(iterable_obj)(chunk_parts)
        else:
            raise TypeError(
                f"For a {type(iterable_obj).__name__} of dataclasses, 'dim' must be an int or a "
                f"dataclass specification (with CollectionMeta). Got 'dim' type {type(dim)}."
            )

    elif is_dataclass(iterable_obj) and isinstance(type(iterable_obj), CollectionMeta):
        field_names, field_generators = [f.name for f in fields(iterable_obj)], []

        for field_name in field_names:
            field_iterable = getattr(iterable_obj, field_name)

            if field_iterable is None:
                field_iterable_dim = 0
            elif isinstance(dim, int):
                field_iterable_dim = dim
            elif is_dataclass(dim) and isinstance(dim, type(iterable_obj)):
                if not hasattr(dim, field_name):
                    raise AttributeError(
                        f"'dim' dataclass of type '{type(iterable_obj).__name__}' is missing dimension specification "
                        f"for non-None field '{field_name}' of iterable."
                    )
                field_iterable_dim = getattr(dim, field_name)
            else:
                raise TypeError(
                    f"The 'dim' argument for dataclass '{type(iterable_obj).__name__}' must be an int or "
                    f"an instance of '{type(iterable_obj).__name__}' if its fields (e.g., '{field_name}') are not all None. "
                    f"Got dim type {type(dim)}."
                )

            field_generators.append(custom_chunking(field_iterable, chunks, field_iterable_dim))

        for field_chunks_tuple in zip(*field_generators):
            yield type(iterable_obj)(
                **{
                    name: chunk_val
                    for name, chunk_val in zip(field_names, field_chunks_tuple)
                }
            )
    else:
        raise NotImplementedError(
            f"Chunking for type {type(iterable_obj)} is not implemented. "
            "Only torch.Tensor, dataclasses with CollectionMeta, lists, and tuples and None are supported."
        )

class LazyDataset:
    def __init__(self, loader, path: str = ''):
        self.loader = loader
        self.path = path
        self._cached_full = None

    def is_leaf(self):
        return self.path in self.loader.lengths

    def _get_full(self):
        if self._cached_full is not None:
            return self._cached_full

        if self.is_leaf():
            def _get_file(idx: int) -> str:
                return self.loader._join(self.loader.storage_path, self.path.replace('.', '/'), f'chunk_{idx}.pt')

            def _load_chunk(idx: int) -> t.Tensor:
                chunk = self.loader._load_tensor_file(_get_file(idx))
                if chunk is not None and self.loader.pin_memory and chunk.device.type == "cpu":
                    chunk = chunk.pin_memory()
                return chunk

            chunks = []
            if self.loader.n_prefetch <= 0:
                for idx in range(self.loader.num_batches):
                    chunk = self.loader._load_tensor_file(_get_file(idx))
                    if chunk is not None:
                        chunk = chunk.to(self.loader.device, non_blocking=True)
                        chunks.append(chunk)
            else:
                for cpu_chunk in self.loader._generate_with_prefetch(_load_chunk):
                    if cpu_chunk is not None:
                        chunk = cpu_chunk.to(self.loader.device, non_blocking=True)
                        chunks.append(chunk)

            self._cached_full = t.cat(chunks, dim=0) if chunks else None
        else:
            sub_loader = BatchLoader(
                self.loader.collection_class,
                self.loader.storage_path,
                self.loader.metadata,
                self.loader.skeleton,
                self.loader.free_after_use,
                device=self.loader.device,
                path_prefix=self.path + '.' if self.path else '',
                fs=self.loader.fs,
                storage_options=self.loader.storage_options,
            )
            batches = list(sub_loader)
            path_parts = self.path.split('.') if self.path else []
            subs = []
            for batch in batches:
                try:
                    sub = reduce(getattr, path_parts, batch)
                    if sub is not None:
                        subs.append(sub)
                except AttributeError:
                    continue
            if subs:
                sub_type = type(subs[0])
                self._cached_full = sub_type.cat(*subs)
            else:
                self._cached_full = None
        return self._cached_full

    def __getattr__(self, name):
        if name == 'device':
            return t.device(self.loader.device)
        if name in ('shape', 'size', 'dtype', 'ndim') and self.is_leaf():
            leaves_meta = self.loader.metadata.get('leaves_metadata', {})
            if self.path in leaves_meta:
                meta = leaves_meta[self.path]
                if name == 'ndim':
                    return meta['ndim']
                if name == 'shape' or name == 'size':
                    return t.Size(meta['shape'])
                if name == 'dtype':
                    dtype_name = meta['dtype']
                    return getattr(t, dtype_name)
        if name.startswith('_'):
            return super().__getattr__(name)
        full = self._get_full()
        if full is not None and hasattr(full, name):
            attr = getattr(full, name)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    return attr(*args, **kwargs)
                return wrapper
            return attr
        new_path = self.path + '.' + name if self.path else name
        return LazyDataset(self.loader, new_path)

    def __repr__(self):
        return f"LazyDataset(path='{self.path}')"

class Recursevly:
    @staticmethod
    def _apply(func: Callable, item: Any, *args: Any, **kwargs: Any) -> Any:
        if item is None:
            return item
        if isinstance(item, LazyDataset):
            item = item._get_full()  # Resolve LazyDataset to its full data
            if item is None:
                return None
        if isinstance(item, t.Tensor):
            return func(item, *args, **kwargs)
        if isinstance(item, (list, tuple)):
            return type(item)(Recursevly._apply(func, it, *args, **kwargs) for it in item)
        if is_dataclass(item):
            return type(item)(
                **{f.name: Recursevly._apply(func, getattr(item, f.name), *args, **kwargs)
                   for f in fields(item)}
            )
        return func(item, *args, **kwargs)

    @staticmethod
    def apply_func(func: Callable, item: Any, *args: Any, **kwargs: Any) -> Any:
        return Recursevly._apply(func, item, *args, **kwargs)

    @staticmethod
    def _leaf_to_numpy(item: Any) -> Any:
        if isinstance(item, t.Tensor):
            return item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            return item
        if hasattr(item, "to_numpy"):
            return item.to_numpy()
        return np.array(item)

    @staticmethod
    def to_numpy(item: Any) -> Any:
        return Recursevly._apply(Recursevly._leaf_to_numpy, item)

    @staticmethod
    def _leaf_to_torch(item: Any, dtype: Union[t.dtype, None], device: Union[str, None]) -> t.Tensor:
        if isinstance(item, t.Tensor):
            tensor = item
        elif isinstance(item, np.ndarray):
            tensor = t.from_numpy(item)
        elif hasattr(item, "values") and isinstance(item.values, np.ndarray):
            tensor = t.from_numpy(item.values)
        else:
            tensor = t.as_tensor(item)

        if dtype is not None or device is not None:
            return tensor.to(dtype=dtype, device=device)

        return tensor

    @staticmethod
    def to_torch(item: Any, dtype: Union[t.dtype, None] = None, device: Union[str, None] = None) -> Any:
        return Recursevly._apply(Recursevly._leaf_to_torch, item, dtype=dtype, device=device)

    @staticmethod
    def squeeze(item: Any) -> Any:
        return Recursevly._apply(t.squeeze, item)

    @staticmethod
    def to_device(item: Any, device: str) -> Any:
        return Recursevly._apply(lambda x: x.to(device, non_blocking=True), item)

    @staticmethod
    def ones_like(item: Any, device: Union[str, None] = None) -> Any:
        if device is None:
            return Recursevly._apply(lambda x: t.ones_like(x).to(x.device), item)
        else:
            return Recursevly._apply(t.ones_like, item, device=device)

    @staticmethod
    def pow(item: Any, exponent: Union[int, float]) -> Any:
        return Recursevly._apply(lambda x: t.pow(x, exponent), item)

    @staticmethod
    def passtrough(item: Any) -> Any:
        return Recursevly._apply(lambda x: x, item)

    @staticmethod
    def apply_mask(item: Any, mask: Any) -> Any:
        if item is None:
            return item
        if isinstance(item, t.Tensor):
            return item[mask]
        if isinstance(item, (list, tuple)):
            return type(item)(Recursevly.apply_mask(it, mask) for it in item)
        if is_dataclass(item):
            # Rebuild the dataclass with each field masked.
            return type(item)(**{f.name: Recursevly.apply_mask(getattr(item, f.name), mask)
                                 for f in fields(item)})
        return item

    @staticmethod
    def requires_grad(item: Any, requires: bool = True) -> Any:
        return Recursevly._apply(lambda x: x.requires_grad_(requires), item)

    @staticmethod
    def grad(item: Any) -> Any:
        return Recursevly._apply(lambda x: x.grad, item)

    @staticmethod
    def multiply(item: Any, number: Union[int, float]) -> Any:
        return Recursevly._apply(lambda x: x * number, item)

    @staticmethod
    def backward(item: Any, grad: Any, retain_graph: bool = True) -> None:
        if item is None or grad is None:
            return
        if isinstance(item, t.Tensor):
            item.backward(grad, retain_graph=retain_graph)
        elif isinstance(item, (list, tuple)):
            assert len(item) == len(grad), "Item and grad must have the same length"
            for it, gt in zip(item, grad):
                Recursevly.backward(it, gt, retain_graph=retain_graph)
        elif is_dataclass(item):
            for f in fields(item):
                Recursevly.backward(getattr(item, f.name), getattr(grad, f.name), retain_graph=retain_graph)
        else:
            # Fallback: if the item has a backward method, call it
            try:
                item.backward(grad, retain_graph=retain_graph)
            except Exception:
                raise NotImplementedError(
                    f"Backward operation is not implemented for type {type(item)}. "
                    "Ensure the item is a torch.Tensor, a dataclass with CollectionMeta, or a list/tuple of these."
                )

    @staticmethod
    def contiguous(item: Any) -> Any:
        return Recursevly._apply(lambda x: x.contiguous(), item)

    @staticmethod
    def _gen_tensors_recursive(obj: Any) -> Generator[t.Tensor, None, None]:
        if isinstance(obj, (t.Tensor, LazyDataset)) or obj is None:
            if isinstance(obj, LazyDataset):
                obj = obj._get_full()
            yield obj
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                yield from Recursevly._gen_tensors_recursive(item)
        elif is_dataclass(obj):
            for f_info in fields(obj):
                field_value = getattr(obj, f_info.name)
                yield from Recursevly._gen_tensors_recursive(field_value)
        else:
            raise TypeError(f"Unsupported type for tensor generation: {type(obj)}. Only torch.Tensor, LazyDataset, lists, tuples, and dataclasses are supported.")

    @staticmethod
    def _permute_branch(obj: Any, perm_idx: Union[t.Tensor, None] = None) -> Any:
        if isinstance(obj, t.Tensor):
            if obj is None or obj.dim() == 0:
                return obj
            local_idx = perm_idx
            if local_idx is None or local_idx.shape[0] != obj.shape[0]:
                local_idx = t.randperm(obj.shape[0], device=obj.device)
            return obj[local_idx]
        elif isinstance(obj, LazyDataset):
            obj = obj._get_full()
            if obj is None or obj.dim() == 0:
                return obj
            local_idx = perm_idx
            if local_idx is None or local_idx.shape[0] != obj.shape[0]:
                local_idx = t.randperm(obj.shape[0], device=obj.device)
            return obj[local_idx]
        elif isinstance(obj, (list, tuple)):
            return type(obj)(Recursevly._permute_branch(item, perm_idx) for item in obj)
        elif isinstance(obj, dict):
            return {k: Recursevly._permute_branch(v, perm_idx) for k, v in obj.items()}
        elif is_dataclass(obj):
            return type(obj)(
                **{
                    f.name: Recursevly._permute_branch(getattr(obj, f.name), perm_idx)
                    for f in fields(obj)
                }
            )
        elif obj is None:
            return obj
        else:
            raise TypeError(f"Unsupported type for permutation: {type(obj)}. Only torch.Tensor, LazyDataset, lists, tuples, dicts, and dataclasses are supported.")

    @staticmethod
    def permute(obj: Any, seed: Union[int, None] = None) -> Any:
        with rng_seed(seed) if seed is not None else contextlib.nullcontext():
            if isinstance(obj, LazyDataset):
                obj = obj._get_full()
            if is_dataclass(obj):
                new_fields = {}
                for field in fields(obj):
                    field_value = getattr(obj, field.name)
                    first_tensor, perm_idx = next(Recursevly._gen_tensors_recursive(field_value), None), None
                    if first_tensor is not None and first_tensor.dim() > 0:
                        perm_idx = t.randperm(first_tensor.shape[0], device=first_tensor.device)
                    new_fields[field.name] = Recursevly._permute_branch(field_value, perm_idx)
                return type(obj)(**new_fields)
            elif isinstance(obj, (list, tuple)):
                first_tensor, perm_idx = next(Recursevly._gen_tensors_recursive(obj), None), None
                if first_tensor is not None and first_tensor.dim() > 0:
                    perm_idx = t.randperm(first_tensor.shape[0], device=first_tensor.device)
                return Recursevly._permute_branch(obj, perm_idx)
            elif isinstance(obj, t.Tensor):
                if obj.dim() > 0:
                    return obj[t.randperm(obj.shape[0], device=obj.device)]
                return obj
            elif obj is None:
                return obj
            else:
                raise TypeError(f"Unsupported type for permutation: {type(obj)}. Only torch.Tensor, lists, tuples, and dataclasses are supported.")

class BatchLoader:
    def __init__(
        self,
        collection_class,
        storage_path: str,
        metadata: Dict[str, Any],
        skeleton: Any,
        free_after_use: bool = False,
        device: str = "cpu",
        path_prefix: Optional[str] = None,
        fs: Any = None,
        storage_options: Dict[str, Any] = None,
        n_prefetch: int = 0,
        num_workers: int = 1,
        pin_memory: bool = True,
        cache_device: Union[str, None] = None,
    ):
        self.collection_class = collection_class
        self.storage_path = storage_path
        self.metadata = metadata
        self.skeleton = skeleton
        self.free_after_use = free_after_use
        self.device = device
        self.path_prefix = path_prefix
        self.num_batches = metadata['num_batches']
        self.lengths = metadata['lengths']
        self.compression = self.metadata.get('compression', None)
        self.fs = fs
        self.storage_options = storage_options
        self.n_prefetch = n_prefetch
        self.num_workers = max(1, num_workers)
        self.pin_memory = pin_memory
        self.cache_device = cache_device

        self._path_map = {path: path.split('.') for path in self.lengths.keys()}
        self._cache = {}

    def __len__(self) -> int:
        return self.num_batches

    def _join(self, *paths: str) -> str:
        if self.fs is None:
            return os.path.join(*paths)

        base, rest = paths[0].rstrip("/"), [p.strip("/") for p in paths[1:]]
        return "/".join([base] + rest)

    def _load_batch(self, batch_index: int, device: str) -> Any:
        batch_instance = self.skeleton.get_skeleton()

        for leaf_path, leaf_length in self._path_map.items():
            if self.path_prefix and not leaf_path.startswith(self.path_prefix):
                continue

            leaf_dir = self._join(self.storage_path, leaf_path.replace('.', '/'))
            chunk_file_base = self._join(leaf_dir, f'chunk_{batch_index}.pt')

            chunk = self._load_tensor_file(chunk_file_base)

            if chunk is None:
                continue

            chunk = chunk.to(device)

            if self.pin_memory and chunk.device.type == "cpu":
                chunk = chunk.pin_memory()

            path_parts = leaf_path.split('.')
            current = batch_instance
            for part in path_parts[:-1]:
                current = getattr(current, part)
            setattr(current, path_parts[-1], chunk)

        return batch_instance

    def _load_tensor_file(self, path: str) -> t.Tensor:
        compression_lib, compression_suffix = get_compression_lib_and_suffix(self.compression)

        path = f"{path}{compression_suffix}"

        if not (self.fs if self.fs is not None else os.path).exists(path):
            return None

        with (self.fs.open if self.fs is not None else open)(path, "rb") as raw_stream:
            if compression_lib is not None:
                with compression_lib.open(raw_stream, "rb") as f_in:
                    return t.load(f_in, weights_only=True)
            else:
                return t.load(raw_stream, weights_only=True)

    def __getitem__(self, batch_index: int) -> Any:
        if not 0 <= batch_index < self.num_batches:
            raise IndexError(f"Batch index {batch_index} out of range [0, {self.num_batches})")

        if not self.free_after_use and batch_index in self._cache:
            return Recursevly.to_device(self._cache[batch_index], self.device)

        target_device = self.cache_device if self.cache_device is not None else self.device
        batch = self._load_batch(batch_index, target_device)

        if not self.free_after_use:
            self._cache[batch_index] = batch

        return Recursevly.to_device(batch, self.device)

    def _generate_with_prefetch(self, load_fn: Callable[[int], Any]) -> Generator[Any, None, None]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures, idx_to_submit = deque(), 0
            while idx_to_submit < min(self.n_prefetch, self.num_batches):
                futures.append(executor.submit(load_fn, idx_to_submit))
                idx_to_submit += 1
            for _ in range(self.num_batches):
                if not futures:
                    break
                item = futures.popleft().result()
                if idx_to_submit < self.num_batches:
                    futures.append(executor.submit(load_fn, idx_to_submit))
                    idx_to_submit += 1
                yield item

    def __iter__(self):
        if self.n_prefetch <= 0:
            for i in range(self.num_batches):
                yield self[i]
            return

        target_device = self.cache_device if self.cache_device is not None else self.device

        def _load_fn(idx):
            if not self.free_after_use and idx in self._cache:
                return self._cache[idx]
            return self._load_batch(idx, target_device)

        for i, batch in enumerate(self._generate_with_prefetch(_load_fn)):
            if not self.free_after_use and i not in self._cache:
                self._cache[i] = batch

            yield Recursevly.to_device(batch, self.device)

class LazyBatchLoader(BatchLoader):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        field_names = [f.name for f in fields(self.collection_class)]
        if name in field_names:
            return LazyDataset(self, name)
        if hasattr(self.collection_class, name):
            attr = getattr(self.collection_class, name)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    warnings.warn(
                        f"Calling method '{name}' requires loading the full dataset, which may be memory-intensive",
                        stacklevel=2
                    )
                    full_dataset = self.collection_class.load_chunked(
                        self.storage_path,
                        lazy=False,
                        free_after_use=self.free_after_use,
                        device=self.device,
                        storage_options=self.storage_options,
                        n_prefetch=self.n_prefetch,
                        num_workers=self.num_workers,
                        pin_memory=self.pin_memory,
                        cache_device=self.cache_device,
                    )
                    return attr(full_dataset, *args, **kwargs)
                return wrapper
            return attr
        raise AttributeError(f"No such attribute '{name}' in {self.collection_class.__name__}")

class CollectionMeta(type):
    @staticmethod
    def _recursive_info_printer(value: Any, indent_level: int, indent_char: str, field_prefix: str = ""):
        current_indent_str = indent_char * indent_level
        print(f"{current_indent_str}{field_prefix}", end="")

        if value is None:
            print("None")
            return

        is_torch_tensor = isinstance(value, t.Tensor)
        is_numpy_array = isinstance(value, np.ndarray)

        if is_torch_tensor:
            print(f"torch.Tensor with shape {value.shape}")
        elif is_numpy_array:
            print(f"numpy.ndarray with shape {value.shape}")
        elif is_dataclass(value) and isinstance(type(value), CollectionMeta):
            print(f"{type(value).__name__}:")
            for field_obj in fields(type(value)):
                CollectionMeta._recursive_info_printer(
                    getattr(value, field_obj.name),
                    indent_level + 1,
                    indent_char,
                    field_prefix=f"{field_obj.name}: "
                )
        elif isinstance(value, (list, tuple)):
            print(f"{type(value).__name__} of length {len(value)}:")
            for i, item in enumerate(value):
                CollectionMeta._recursive_info_printer(
                    item,
                    indent_level + 1,
                    indent_char,
                    field_prefix=f"[{i}]: "
                )
        elif isinstance(value, dict):
            print(f"dict of size {len(value)}:")
            for k, dict_item in value.items():
                key_repr = repr(k) if not isinstance(k, str) else f"'{k}'"
                CollectionMeta._recursive_info_printer(
                    dict_item,
                    indent_level + 1,
                    indent_char,
                    field_prefix=f"{key_repr}: "
                )
        elif isinstance(value, (int, float, str, bool)):
            value_repr = repr(value)
            if len(value_repr) > 60:
                value_repr = value_repr[:57] + "..."
            print(f"{type(value).__name__} = {value_repr}")
        else:
            print(f"{type(value).__name__}")

    def __new__(mcs, name, bases, namespace):
        def make_transform(cls):
            class _transform:
                def __init__(self, func: Callable[[Any], Any]):
                    self.func = func

                def __call__(self, *args, **kwargs):
                    if args and not kwargs:
                        return cls(**{f.name: self.func(arg) for f, arg in zip(fields(cls), args)})
                    else:
                        return cls(**{f.name: self.func(kwargs[f.name]) for f in fields(cls)})

                def to_collection(self, other):
                    return cls(**{f.name: self.func(getattr(other, f.name)) for f in fields(cls)})
            return _transform

        cls_annotations = namespace.get('__annotations__', {})

        def safe_inject(method_name, method_func):
            if method_name not in namespace and method_name not in cls_annotations:
                namespace[method_name] = method_func
            else:
                warnings.warn(
                    f"CollectionMeta: The method '{method_name}' could not be injected into class '{name}' "
                    f"because a field or method with this name already exists. "
                    f"This overrides the default CollectionMeta behavior for '{method_name}'.",
                    category=UserWarning,
                    stacklevel=2
                )
                return None

        def unrolled(self) -> Tuple:
            return tuple(getattr(self, f.name) for f in fields(self))
        safe_inject('unrolled', property(unrolled))

        @classmethod
        def to_collection(cls, other):
            if not is_dataclass(other):
                return other
            field_dict = {}
            for f in fields(cls):
                value = getattr(other, f.name, None)
                if value is None:
                    field_dict[f.name] = None
                else:
                    expected_type = f.type
                    origin = get_origin(expected_type)
                    if origin is not None:
                        expected_types = get_args(expected_type)
                        expected_type = next((t for t in expected_types if t is not type(None)), None)
                    if expected_type and is_dataclass(expected_type) and isinstance(type(expected_type), CollectionMeta):
                        field_dict[f.name] = expected_type.to_collection(value)
                    else:
                        field_dict[f.name] = value
            return cls(**field_dict)
        safe_inject('to_collection', to_collection)

        @classmethod
        def apply(cls, func: Callable[[Any], Any]):
            return make_transform(cls)(func)
        safe_inject('apply', apply)

        @classmethod
        def cat(cls, *instances, axis=0):
            if not instances:
                raise ValueError("No instances provided for concatenation.")
            for inst in instances:
                if not isinstance(inst, cls):
                    raise TypeError(f"All instances must be of the same type. Got {type(inst).__name__} instead of {cls.__name__}")
            field_dict = {}
            for f in fields(cls):
                values = [getattr(inst, f.name) for inst in instances]
                if all(v is None for v in values):
                    continue
                non_none_values = [v for v in values if v is not None]
                if not non_none_values:
                    field_dict[f.name] = None
                    continue
                first_valid = non_none_values[0]
                if isinstance(first_valid, t.Tensor):
                    field_dict[f.name] = t.cat(non_none_values, dim=axis)
                elif isinstance(first_valid, np.ndarray):
                    field_dict[f.name] = np.concatenate(non_none_values, axis=axis)
                elif is_dataclass(first_valid) and isinstance(type(first_valid), CollectionMeta):
                    field_dict[f.name] = type(first_valid).cat(*non_none_values, axis=axis)
                else:
                    raise TypeError(
                        f"Field '{f.name}' of type {type(first_valid)} cannot be concatenated."
                    )
            if not field_dict:
                return cls(**{f.name: None for f in fields(cls)})
            return cls(**field_dict)
        safe_inject("cat", cat)

        def keys(self):
            return [f.name for f in fields(self)]
        safe_inject('keys', keys)

        def values(self):
            return [getattr(self, f.name) for f in fields(self)]
        safe_inject('values', values)

        def items(self):
            return [(f.name, getattr(self, f.name)) for f in fields(self)]
        safe_inject('items', items)

        def to(self, device: str):
            return Recursevly.to_device(self, device)
        safe_inject('to', to)

        def to_numpy(self):
            return Recursevly.to_numpy(self)
        safe_inject('to_numpy', to_numpy)

        def to_torch(self, dtype: Union[t.dtype, None] = None, device: Union[str, None] = None):
            return Recursevly.to_torch(self, dtype=dtype, device=device)
        safe_inject('to_torch', to_torch)

        def apply_func(self, func: Callable, *args, **kwargs):
            return Recursevly.apply_func(func, self, *args, **kwargs)
        safe_inject('apply_func', apply_func)

        def __contains__(self, key):
            return key in self.keys()
        safe_inject('__contains__', __contains__)

        def __getitem__(self, key):
            if key not in (f.name for f in fields(self)):
                raise KeyError(key)
            return getattr(self, key)
        safe_inject('__getitem__', __getitem__)

        @classmethod
        def __class_getitem__(cls, key):
            for field_info in fields(cls):
                if field_info.name == key:
                    if field_info.default is not MISSING:
                        return field_info.default
                    elif field_info.default_factory is not MISSING:
                        return field_info.default_factory()
                    else:
                        raise AttributeError(f"Field '{key}' in class '{cls.__name__}' has no default value or default_factory.")
            raise KeyError(f"Field '{key}' not found in class '{cls.__name__}'.")
        safe_inject('__class_getitem__', __class_getitem__)

        def extract_attribute(self, attribute_name: str, target_type: type):
            if not isinstance(attribute_name, str):
                raise TypeError("attribute_name must be a string.")
            if not isinstance(target_type, type):
                raise TypeError("target_type must be a type.")

            current_class, new_fields_values = type(self), {}

            for field_info in fields(current_class):
                field_value = getattr(self, field_info.name)

                if isinstance(field_value, target_type):
                    if hasattr(field_value, attribute_name):
                        new_fields_values[field_info.name] = getattr(field_value, attribute_name)
                    else:
                        raise AttributeError(
                            f"Instance of {target_type.__name__} (field: {field_info.name}) "
                            f"does not have attribute '{attribute_name}'."
                        )
                elif is_dataclass(field_value) and isinstance(type(field_value), CollectionMeta):
                    new_fields_values[field_info.name] = field_value.extract_attribute(attribute_name, target_type)
                else:
                    new_fields_values[field_info.name] = field_value

            return current_class(**new_fields_values)
        safe_inject('extract_attribute', extract_attribute)

        def info(self, indent_char: str = "  "):
            """Prints a structured representation of the dataclass instance."""
            print(f"{type(self).__name__}:")
            for field_obj in fields(type(self)):
                field_name = field_obj.name
                field_value = getattr(self, field_name)
                mcs._recursive_info_printer(  # mcs = CollectionMeta
                    field_value,
                    1,
                    indent_char,
                    field_prefix=f"{field_name}: "
                )
        safe_inject('info', info)

        def sparse(self, step: int, class_based: bool = False, field: str = "Y"):
            if step < 1:
                raise ValueError("step must be >= 1")

            def _resolve_length(obj) -> Union[int, None]:
                if obj is None:
                    return None
                if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
                    return len(obj)
                if isinstance(obj, torch.Tensor):
                    return obj.shape[0]
                if isinstance(obj, (list, tuple)):
                    return _resolve_length(obj[0]) if obj else None
                if dc_is_dataclass(obj):
                    for f in dc_fields(obj):
                        length = _resolve_length(getattr(obj, f.name))
                        if length is not None:
                            return length
                return None

            def _leaf_entries(obj, path: str) -> list[tuple[str, Any, int]]:
                entries: list[tuple[str, Any, int]] = []
                if obj is None:
                    return entries
                if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series, torch.Tensor, list, tuple)):
                    length = _resolve_length(obj)
                    if length is not None:
                        entries.append((path, obj, length))
                    return entries
                if dc_is_dataclass(obj):
                    for f in dc_fields(obj):
                        entries.extend(_leaf_entries(getattr(obj, f.name), f"{path}.{f.name}" if path else f.name))
                return entries

            LabelEntry = tuple[str, Any, int]

            def _collect_y(obj, path: str = "") -> list[LabelEntry]:
                entries: list[LabelEntry] = []
                if obj is None or not dc_is_dataclass(obj):
                    return entries
                for field_obj in dc_fields(obj):
                    value = getattr(obj, field_obj.name)
                    sub_path = f"{path}.{field_obj.name}" if path else field_obj.name
                    if field_obj.name == field and value is not None:
                        entries.extend(_leaf_entries(value, sub_path))
                        continue
                    if dc_is_dataclass(value):
                        entries.extend(_collect_y(value, sub_path))
                return entries

            def _as_numpy(y):
                if y is None:
                    return None
                if isinstance(y, torch.Tensor):
                    y = y.detach().cpu().numpy()
                elif isinstance(y, pd.DataFrame):
                    y = y.to_numpy()
                elif isinstance(y, pd.Series):
                    y = y.to_numpy()
                elif isinstance(y, np.ndarray):
                    pass
                else:
                    return None
                if y.ndim == 2 and y.shape[1] > 1:
                    return y.argmax(axis=1)
                if y.ndim == 2 and y.shape[1] == 1:
                    return y.reshape(-1)
                if y.ndim == 1:
                    return y
                raise ValueError(f"Unsupported label shape {y.shape}")

            all_y_entries = _collect_y(self, "") if class_based else []
            y_by_length: dict[int, list[LabelEntry]] = defaultdict(list)
            for entry in all_y_entries:
                y_by_length[entry[2]].append(entry)

            if class_based and not y_by_length:
                raise ValueError(f"class_based=True but no '{field}' field found anywhere in the structure")

            def _select_labels(length: int, origin_path: str):
                matches = y_by_length.get(length, [])
                if not matches:
                    return None, ""
                for y_path, y_val, _ in matches:
                    if y_path == origin_path or y_path.startswith(f"{origin_path}."):
                        return y_val, y_path
                origin_suffix = origin_path.split(".")[-1] if origin_path else ""
                if origin_suffix:
                    for y_path, y_val, _ in matches:
                        if y_path.split(".")[-1] == origin_suffix:
                            return y_val, y_path
                chosen = matches[0]
                warnings.warn(
                    f"Using '{field}' from '{chosen[0]}' for component '{origin_path}' (matched by length={length})",
                    stacklevel=3,
                )
                return chosen[1], chosen[0]

            def _mask_for(length: int, labels, origin_path: str):
                length = int(length)
                if not class_based:
                    mask = np.zeros(length, dtype=bool)
                    mask[::step] = True
                    return mask
                if labels is None:
                    raise ValueError(
                        f"class_based=True but no '{field}' with matching length for component '{origin_path}'"
                    )
                labels = _as_numpy(labels)
                if labels is None:
                    raise ValueError(f"Cannot convert labels for '{origin_path}' to numpy array")
                if labels.shape[0] != length:
                    raise ValueError(
                        f"Label length {labels.shape[0]} does not match component length {length} "
                        f"for '{origin_path}'"
                    )
                mask = np.zeros(length, dtype=bool)
                for klass in np.unique(labels):
                    idxs = np.where(labels == klass)[0]
                    mask[idxs[::step]] = True
                return mask

            def _apply_mask_to_leaf(value, mask: np.ndarray, path: str):
                length = _resolve_length(value)
                if length is None or length != mask.shape[0]:
                    return value
                if isinstance(value, np.ndarray):
                    return value[mask]
                if isinstance(value, pd.DataFrame):
                    return value.iloc[mask].reset_index(drop=True)
                if isinstance(value, pd.Series):
                    return value.iloc[mask].reset_index(drop=True)
                if isinstance(value, torch.Tensor):
                    torch_mask = torch.as_tensor(mask, device=value.device, dtype=torch.bool)
                    keep = torch_mask.nonzero(as_tuple=True)[0]
                    return value.index_select(0, keep)
                if isinstance(value, list):
                    return [_apply_mask_to_leaf(v, mask, f"{path}[{idx}]") for idx, v in enumerate(value)]
                if isinstance(value, tuple):
                    return type(value)(_apply_mask_to_leaf(v, mask, f"{path}[{idx}]") for idx, v in enumerate(value))
                return value

            def _sparsify(obj, path: str = "", label_hint: Any = None):
                if obj is None:
                    return None

                if dc_is_dataclass(obj):
                    local_label_store = getattr(obj, field, None) if hasattr(obj, field) else None
                    kwargs = {}
                    for field_obj in dc_fields(obj):
                        child = getattr(obj, field_obj.name)
                        child_path = f"{path}.{field_obj.name}" if path else field_obj.name

                        child_label_hint = None
                        if field_obj.name == field:
                            child_label_hint = child
                        elif dc_is_dataclass(local_label_store) and hasattr(local_label_store, field_obj.name):
                            child_label_hint = getattr(local_label_store, field_obj.name)
                        elif dc_is_dataclass(label_hint) and hasattr(label_hint, field_obj.name):
                            child_label_hint = getattr(label_hint, field_obj.name)
                        else:
                            child_label_hint = label_hint

                        kwargs[field_obj.name] = _sparsify(child, child_path, child_label_hint)
                    return type(obj)(**kwargs)

                length = _resolve_length(obj)
                if length is None:
                    return obj

                candidate_label = None if not class_based else label_hint
                if class_based and dc_is_dataclass(candidate_label):
                    candidate_label = None
                if class_based and candidate_label is None:
                    candidate_label, _ = _select_labels(length, path)

                mask = _mask_for(length, candidate_label, path)
                return _apply_mask_to_leaf(obj, mask, path)

            return _sparsify(self)
        safe_inject("sparse", sparse)

        def get_leaves(self) -> Dict[str, Any]:
            leaves = {}

            def recurse(obj, current_path: str):
                if obj is None:
                    return
                if is_dataclass(obj) and isinstance(type(obj), CollectionMeta):
                    for field_obj in fields(obj):
                        recurse(getattr(obj, field_obj.name), f"{current_path}.{field_obj.name}" if current_path else field_obj.name)
                else:
                    leaves[current_path] = obj
            recurse(self, "")
            return leaves
        safe_inject('get_leaves', get_leaves)

        def get_skeleton(self):
            def recurse(obj):
                if obj is None:
                    return None
                if is_dataclass(obj) and isinstance(type(obj), CollectionMeta):
                    kwargs = {f.name: recurse(getattr(obj, f.name)) for f in fields(obj)}
                    return type(obj)(**kwargs)
                else:
                    return None
            return recurse(self)
        safe_inject('get_skeleton', get_skeleton)

        def save_chunked(
            self,
            storage_path: str,
            batch_size: Optional[int] = None,
            n_chunks: Optional[int] = None,
            downcast_rules: Optional[Dict[str, t.dtype]] = None,
            compression: Optional[str] = None,
        ):

            if compression not in (None, "gzip", "lzma"):
                raise ValueError(f"Unsupported compression '{compression}'. Only 'gzip' or None are supported.")

            compression_lib, compression_suffix = get_compression_lib_and_suffix(compression)

            DEFAULT_DOWNCAST_RULES = {k: t.float16 for k in ["X", "Y", "weights", "class_weights"]}
            if downcast_rules is not None:
                for k, v in DEFAULT_DOWNCAST_RULES.items():
                    if k not in downcast_rules:
                        downcast_rules[k] = v
            else:
                downcast_rules = DEFAULT_DOWNCAST_RULES

            if (batch_size is None and n_chunks is None) or (batch_size is not None and n_chunks is not None):
                raise ValueError("Exactly one of batch_size or n_chunks must be specified")
            if batch_size is not None and batch_size <= 0:
                raise ValueError("batch_size must be positive")
            if n_chunks is not None and n_chunks <= 0:
                raise ValueError("n_chunks must be positive")

            os.makedirs(storage_path, exist_ok=True)
            skeleton = self.get_skeleton()
            t.save(skeleton, os.path.join(storage_path, 'skeleton.pt'))

            leaves = self.get_leaves()
            leaves_metadata = {}
            lengths = {}
            max_length = 0

            for path, value in leaves.items():
                if value is None:
                    lengths[path] = 0
                    continue

                if not isinstance(value, t.Tensor):
                    raise TypeError(f"Leaf at '{path}' must be a torch.Tensor for chunked saving")

                length = value.shape[0]
                lengths[path] = length
                max_length = max(max_length, length)

                target_dtype, path_parts = None, path.split('.')
                if downcast_rules:
                    if path in downcast_rules:  # exact match
                        target_dtype = downcast_rules[path]

                    elif path_parts[-1] in downcast_rules:  # exact match on leaf name
                        target_dtype = downcast_rules[path_parts[-1]]

                    else:  # any parent in rules? get closest
                        for part in reversed(path_parts[:-1]):
                            if part in downcast_rules:
                                target_dtype = downcast_rules[part]
                                break

                final_dtype = target_dtype if (target_dtype is not None and value.is_floating_point()) else value.dtype
                leaves_metadata[path] = {
                    'shape': list(value.shape),
                    'dtype': str(final_dtype).replace('torch.', ''),
                    'ndim': value.ndim
                }

                if n_chunks is not None:
                    chunk_size = math.ceil(length / n_chunks) if length > 0 else 0
                    num_chunks = n_chunks if length > 0 else 0
                else:
                    chunk_size = batch_size
                    num_chunks = (length + batch_size - 1) // batch_size if length > 0 else 0

                leaf_dir = os.path.join(storage_path, path.replace('.', '/'))
                os.makedirs(leaf_dir, exist_ok=True)

                with LogContext(logger).redirect_tqdm():
                    for chunk_idx in tqdm(range(num_chunks), desc=f"Saving chunks for '{path}'", unit="chunk"):
                        start = chunk_idx * chunk_size
                        if start >= length:
                            continue
                        end = min(start + chunk_size, length)
                        chunk = value[start:end]

                        if target_dtype is not None and chunk.is_floating_point():
                            with LogContext(logger).duplicate_filter():
                                logger.info(f"Downcasting '{path}' to {target_dtype}")
                            chunk = chunk.to(target_dtype)

                        base_filename = f'chunk_{chunk_idx}.pt'

                        if compression_lib is not None:
                            full_path = os.path.join(leaf_dir, f"{base_filename}{compression_suffix}")
                            with compression_lib.open(full_path, 'wb') as f_out:
                                t.save(chunk, f_out)
                        else:
                            full_path = os.path.join(leaf_dir, base_filename)
                            t.save(chunk, full_path)

            num_batches = n_chunks if n_chunks is not None else (max_length + batch_size - 1) // batch_size if max_length > 0 else 0
            metadata = {
                'class': type(self).__name__,
                'lengths': lengths,
                'batch_size': batch_size if n_chunks is None else None,
                'num_batches': num_batches,
                'n_chunks': n_chunks,
                'compression': compression,
                'leaves_metadata': leaves_metadata,
            }
            with open(os.path.join(storage_path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
        safe_inject('save_chunked', save_chunked)

        # Automatically add the new class to torch safe globals
        cls = super().__new__(mcs, name, bases, namespace)
        t.serialization.add_safe_globals([cls])
        namespace['__safe_global_added'] = True  # Optional: flag to confirm addition

        @classmethod
        def load_chunked(
            cls,
            storage_path: str,
            lazy: bool = True,
            free_after_use: bool = False,
            device: str = "cpu",
            storage_options: Optional[Dict[str, Any]] = None,
            n_prefetch: int = 3,
            num_workers: int = 1,
            pin_memory: bool = False,
            cache_device: Optional[str] = None,
        ):
            fs = None
            if "://" in str(storage_path):
                fs, storage_path = fsspec.core.url_to_fs(storage_path, **(storage_options or {}))

            if not (fs if fs is not None else os.path).exists(storage_path):
                raise FileNotFoundError(f"Storage path '{storage_path}' does not exist")

            def _join(*paths: str) -> str:
                if fs is None:
                    return os.path.join(*paths)

                base, rest = paths[0].rstrip("/"), [p.strip("/") for p in paths[1:]]
                return "/".join([base] + rest)

            meta_path = _join(storage_path, 'metadata.json')
            try:
                with (fs.open if fs is not None else open)(meta_path, 'r') as f:
                    metadata = json.load(f)
            except (FileNotFoundError, IndexError):
                raise FileNotFoundError(f"Metadata file not found at '{meta_path}'")

            if metadata['class'] != cls.__name__:
                raise ValueError(f"Mismatch in class: expected '{cls.__name__}', got '{metadata['class']}'")

            compression = metadata.get('compression', None)
            compression_lib, compression_suffix = get_compression_lib_and_suffix(compression)

            skeleton_path = _join(storage_path, 'skeleton.pt')
            with (fs.open if fs is not None else open)(skeleton_path, 'rb') as f:
                loaded_skeleton = t.load(f, weights_only=False)
            skeleton = cls.to_collection(loaded_skeleton)

            if not lazy:
                full_instance = copy.deepcopy(skeleton)
                for path, length in metadata['lengths'].items():
                    if length == 0:
                        continue

                    num_chunks = metadata['num_batches']
                    chunks = []
                    leaf_dir = _join(storage_path, path.replace('.', '/'))
                    for chunk_idx in range(num_chunks):
                        chunk_file = _join(leaf_dir, f'chunk_{chunk_idx}.pt{compression_suffix}')

                        if (fs if fs is not None else os.path).exists(chunk_file):
                            with (fs.open if fs is not None else open)(chunk_file, 'rb') as raw_stream:
                                if compression_lib is not None:
                                    with compression_lib.open(raw_stream, 'rb') as f_in:
                                        chunks.append(t.load(f_in, weights_only=True))
                                else:
                                    chunks.append(t.load(raw_stream, weights_only=True))

                    if chunks:
                        full_tensor = t.cat(chunks, dim=0).to(device)
                        path_parts = path.split('.')
                        current = full_instance
                        for part in path_parts[:-1]:
                            current = getattr(current, part)
                        setattr(current, path_parts[-1], full_tensor)

                return full_instance
            else:
                return LazyBatchLoader(
                    cls,
                    storage_path,
                    metadata,
                    skeleton,
                    free_after_use,
                    device=device,
                    fs=fs,
                    storage_options=storage_options,
                    n_prefetch=n_prefetch,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    cache_device=cache_device,
                )
        safe_inject('load_chunked', load_chunked)

        return super().__new__(mcs, name, bases, namespace)

@dataclass
class _collection:
    values: Any
    weights: Any
    histograms: Any
    
    @property
    def unrolled(self) -> tuple[Any, ...]:
        return (self.values, self.weights, self.histograms)

@dataclass
class _same_sign_opposite_sign_split(metaclass=CollectionMeta):
    ss: Union[t.Tensor, pd.DataFrame, np.ndarray]
    os: Union[t.Tensor, pd.DataFrame, np.ndarray]

@dataclass
class _component_collection(metaclass=CollectionMeta):
    _: KW_ONLY
    X: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    Y: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    class_weights: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    process: Union[t.Tensor, pd.DataFrame, np.ndarray, None] = None
    qcd_weights_os: Union[t.Tensor, None] = None
    SR_like: Union[t.Tensor, pd.DataFrame, np.ndarray, int, None] = None

def get_my_data(df, training_var):
    _df = df  # fold/fold_train/fold_val to load, should contain SS/OS columns
    ss_os_split = _same_sign_opposite_sign_split(
            ss=_df[(_df.SS)],
            os=_df[((_df.OS) & (_df.Label != 2)) | ((_df.SS) & (_df.Label == 2))],
        )

    return _component_collection(
            X=ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype = np.float32)),
            Y=ss_os_split.apply_func(lambda x: x["Label"].to_numpy(dtype = np.float32)),  # or ss_os_split.apply_func(extract_label)
            # instead of _same_sign_opposite_sign_split.apply(lambda x: x["Label"].to_numpy()).to_collection(ss_os_split)
            weights=ss_os_split.apply_func(lambda __df: __df["weight"].to_numpy(dtype = np.float32)),
            class_weights=ss_os_split.apply_func(lambda x: x["class_weights"].to_numpy()),
            process=ss_os_split.apply_func(lambda x: x['process'].to_numpy(dtype = np.float32))
        )

def get_plotting_data(df, training_var):
    _df = df  # fold/fold_train/fold_val to load, should contain SS/OS columns
    ss_os_split = _same_sign_opposite_sign_split(
            ss=_df[((_df.SS) & (_df.Label == 2))],
            os=_df[((_df.OS) & (_df.Label == 2))]
        )

    return _component_collection(
            X=ss_os_split.apply_func(lambda x: x[training_var].to_numpy(dtype = np.float32)),
            weights=ss_os_split.apply_func(lambda __df: __df["weight"].to_numpy(dtype = np.float32)),
        )

def get_class_weights(
    weights: Union[pd.Series, np.ndarray, t.Tensor],
    Y: Union[pd.Series, np.ndarray, t.Tensor],
    classes: tuple = (0, 1),
    class_weighted: bool = True,
) -> Union[pd.Series, np.ndarray, t.Tensor]:
    _weights = np.zeros_like(weights)
    for _class in classes:
        _weights[Y == _class] = weights.sum() / weights[Y == _class].sum()
    return _weights * (weights if class_weighted else 1.0)