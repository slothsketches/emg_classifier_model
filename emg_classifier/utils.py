import pathlib
import typing as t

import numpy as np
import torch


def load_from_sample(path: t.Union[pathlib.Path, str]):
    if isinstance(path, str):
        path = pathlib.Path(path)
        path.resolve(strict=True)

    for line in map(str.strip, path.read_text().split("\n")):
        if not line:
            continue
        yield int(line[29:])


def load_from_sample_windowed(
    path: t.Union[pathlib.Path, str], window_size: int, stride: t.Optional[int] = None
):
    if stride is None:
        stride = window_size

    signals = list(load_from_sample(path))

    for i in range(0, len(signals) - window_size + 1, stride):
        yield signals[i : i + window_size]


def normalize_windows(arrays: np.ndarray):
    for i in range(arrays.shape[0]):
        mean = np.mean(arrays[i])
        stddev = np.std(arrays[i])

        arrays[i] -= mean

        if stddev != 0:
            arrays[i] /= stddev

    return torch.tensor(arrays)
