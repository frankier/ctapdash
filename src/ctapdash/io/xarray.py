"""Pickled xarray metadata plus raw, memory-mappable data-variable storage.

Requires Python >= 3.10, NumPy, and xarray with built-in DataTree support
(xarray >= 2024.10). Example::

    save_xarray(ds_or_tree, "store")
    save_xarray(ds_or_tree, "packed", consolidated=True)
    restored = load_xarray("store")           # read-only mappings
    editable = load_xarray("store", mode="c") # copy-on-write mappings

The directory contains metadata.pkl and either data/00000000.bin, etc., or
one consolidated.bin for every node's data variables. Consolidated stores
record node byte ranges and absolute byte offsets for each variable. Loading
creates one shared byte memmap and constructs ndarray views over its buffer.
All paths are relative, so the directory can be moved as a unit.

Only load trusted directories: unpickling can execute arbitrary code. This
is a Python persistence format, not a stable cross-version interchange format;
use compatible Python, NumPy, pandas, and xarray versions when loading.

Coordinates (including object coordinates), indexes, attributes, and encodings
are pickled. Data variables must have fixed-size, non-object NumPy dtypes.
Lazy data variables are materialized one variable at a time during saving;
chunking, device placement, and original strides are not preserved. Zero-byte
arrays cannot be mmaped and are reconstructed as ordinary empty arrays.

Loaded nonempty data variables share memory with NumPy memmaps, although
xarray may expose them as ndarray views. Keep the store's files available
while using the result or any views. Mappings live as long as their arrays;
Dataset.close() does not explicitly close these mappings. Default mode='r'
is read-only; mode='c' permits changes without modifying the stored files.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Literal

import numpy as np
import xarray as xr
from ctapdash.io.utils import atomic_write

__all__ = ["save_xarray", "load_xarray"]
_FORMAT = "xarray-pickle-raw"
_VERSION = 2


def save_xarray(
    obj: xr.Dataset | xr.DataTree, directory: str | os.PathLike, *,
    consolidated: bool = False,
) -> None:
    """Save a Dataset or DataTree to a new directory.

    Refuses to overwrite an existing path. Builds in a temporary sibling
    directory and renames only after a successful write, cleaning up on error.
    Concurrent writers targeting the same path are not supported.

    consolidated=False writes one raw file per data variable. True writes all
    data variables to consolidated.bin, with dtype-aligned offsets in the
    pickle. Coordinates and other metadata remain in metadata.pkl in both
    layouts. An entirely empty consolidated payload needs no mmap on load.

    When saving an attached subtree, inherited coordinates are included at its
    new root so it can be loaded independently. Descendants retain local-only
    coordinates and inherit normally when reconstructed.

    Raises TypeError for object/variable-width data-variable dtypes (including
    structured dtypes containing Python objects). Convert these to fixed-size
    numeric or string arrays before saving.
    """
    if isinstance(obj, xr.Dataset):
        kind, name = "Dataset", None
        nodes = [("/", obj)]
    elif isinstance(obj, xr.DataTree):
        kind, name = "DataTree", obj.name
        nodes = []

        def visit(node, path, root=False):
            nodes.append((path, node.to_dataset(inherit=root)))
            for child_name, child in node.children.items():
                visit(child, path.rstrip("/") + "/" + child_name)

        visit(obj, "/", root=True)
    else:
        raise TypeError("Expected xarray.Dataset or xarray.DataTree")

    destination = Path(directory).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(destination, dir=True) as staging, ExitStack() as stack:
        shared_file = (stack.enter_context((staging / "consolidated.bin").open("wb"))
                    if consolidated else None)
        if not consolidated:
            (staging / "data").mkdir()
        metadata = {"format": _FORMAT, "version": _VERSION, "kind": kind,
                    "name": name, "nodes": [], "consolidated": consolidated}
        counter = 0
        for path, ds in nodes:
            # Keeping an actual coordinate-only Dataset preserves custom indexes,
            # MultiIndexes, coordinate encodings, attrs, and Dataset encodings.
            shell = ds.drop_vars(list(ds.data_vars)).compute()
            shell.set_close(None)
            record = {"path": path, "shell": shell, "variables": [],
                    "variable_order": list(ds.variables)}
            if consolidated:
                record["offset"] = shared_file.tell()
            for var_name, da in ds.data_vars.items():
                array = np.asarray(da.data)
                dtype = array.dtype
                if dtype.hasobject or dtype.kind == "T":
                    raise TypeError(
                        f"{path}: variable {var_name!r} has non-mappable dtype {dtype}; "
                        "use a fixed-size dtype without Python objects"
                    )
                filename = "consolidated.bin" if consolidated else f"data/{counter:08d}.bin"
                counter += 1
                # tofile writes in C order, including for strided/Fortran inputs.
                if consolidated:
                    padding = (-shared_file.tell()) % max(1, dtype.alignment)
                    shared_file.write(b"\x00" * padding)
                    offset = shared_file.tell()
                    array.tofile(shared_file)
                else:
                    offset = 0
                    with (staging / filename).open("wb") as f:
                        array.tofile(f)
                record["variables"].append({
                    "name": var_name, "file": filename, "dtype": dtype,
                    "offset": offset,
                    "shape": array.shape, "dims": da.dims, "order": "C",
                    "attrs": dict(da.attrs), "encoding": dict(da.encoding),
                })
            if consolidated:
                record["nbytes"] = shared_file.tell() - record["offset"]
            metadata["nodes"].append(record)
        if consolidated:
            metadata["nbytes"] = shared_file.tell()
        with (staging / "metadata.pkl").open("wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_xarray(
    directory: str | os.PathLike, *, mode: Literal["r", "c"] = "r"
) -> xr.Dataset | xr.DataTree:
    """Restore an object using mmap-backed data variables and pickled metadata.

    Only trusted pickle files may be loaded. Data bytes are not eagerly read;
    coordinates and metadata are loaded into memory. mode='r' is read-only,
    while mode='c' gives private, copy-on-write mappings. In-place persistence
    is deliberately unsupported; save a changed result to a new directory.
    Automatically detects separate and consolidated layouts, including legacy
    version-1 stores. Consolidated arrays share exactly one mmap (unless the
    payload is empty). Checks file sizes and variable/node byte bounds.
    """
    if mode not in ("r", "c"):
        raise ValueError("mode must be 'r' (read-only) or 'c' (copy-on-write)")
    directory = Path(directory).resolve()
    with (directory / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)
    if (metadata.get("format") != _FORMAT
            or metadata.get("version") not in (1, _VERSION)
            or metadata.get("kind") not in ("Dataset", "DataTree")):
        raise ValueError("Unsupported xarray mmap store format/version")

    consolidated = metadata.get("consolidated", False)
    shared = None
    if consolidated:
        total_bytes = metadata["nbytes"]
        file = (directory / "consolidated.bin").resolve()
        if not file.is_relative_to(directory):
            raise ValueError("Data file is outside the store directory")
        if (not isinstance(total_bytes, int) or total_bytes < 0
                or file.stat().st_size != total_bytes):
            raise ValueError("Wrong byte size for consolidated.bin")
        if total_bytes:
            shared = np.memmap(file, dtype=np.uint8, mode=mode, shape=(total_bytes,))

    datasets = {}
    for node in metadata["nodes"]:
        if consolidated:
            node_start, node_size = node["offset"], node["nbytes"]
            if (not isinstance(node_start, int) or not isinstance(node_size, int)
                    or node_start < 0 or node_size < 0
                    or node_start + node_size > total_bytes):
                raise ValueError("Invalid consolidated node byte range")
        ds = node["shell"]
        variables = dict(ds.variables)
        for spec in node["variables"]:
            dtype = np.dtype(spec["dtype"])
            shape = tuple(spec["shape"])
            if dtype.hasobject or dtype.kind == "T":
                raise ValueError("Store contains a non-mappable dtype")
            if (any(not isinstance(n, int) or n < 0 for n in shape)
                    or len(shape) != len(spec["dims"]) or spec["order"] != "C"):
                raise ValueError("Invalid array shape, dimensions, or order")
            expected = math.prod(shape) * dtype.itemsize
            if consolidated:
                offset = spec["offset"]
                if (not isinstance(offset, int) or offset < node_start
                        or offset + expected > node_start + node_size):
                    raise ValueError("Variable byte range is outside its node")
            else:
                file = (directory / spec["file"]).resolve()
                if not file.is_relative_to(directory):
                    raise ValueError("Data file is outside the store directory")
                if file.stat().st_size != expected:
                    raise ValueError(f"Wrong byte size for {file}: expected {expected}")
            if expected:
                if consolidated:
                    array = np.ndarray(shape, dtype=dtype, buffer=shared,
                                       offset=offset, order="C")
                else:
                    array = np.memmap(file, dtype=dtype, mode=mode, shape=shape, order="C")
            else:
                array = np.empty(shape, dtype=dtype)
                if mode == "r":
                    array.flags.writeable = False
            variables[spec["name"]] = xr.Variable(
                spec["dims"], array, attrs=spec["attrs"], encoding=spec["encoding"]
            )
        # Supplying Coordinates preserves existing indexes, including no-index
        # dimension coordinates, instead of asking xarray to infer new indexes.
        data_vars = {name: variables[name] for name in node["variable_order"]
                     if name not in ds.coords}
        restored = xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)
        restored.encoding = ds.encoding
        datasets[node["path"]] = restored

    if metadata["kind"] == "Dataset":
        return datasets["/"]
    return xr.DataTree.from_dict(datasets, name=metadata["name"])
