import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import label_from_filename, load_from_sample


class EMGDataset(Dataset):
    def __init__(self, directory: pathlib.Path, *, seq_len: int = 150):
        n_label = 0

        self.__seq_len = seq_len
        self.__label_hash = dict()

        __samples = []
        __labels = []

        for path in directory.glob("*.txt"):
            label = label_from_filename(path.name)

            if label not in self.__label_hash:
                self.__label_hash[label] = n_label
                n_label += 1

            index = self.__label_hash[label]

            samples = np.array(list(load_from_sample(path)), dtype=float)
            sample_size = len(samples)

            offset = -(sample_size % seq_len)
            splitted_samples = np.split(samples[:offset], sample_size // seq_len)

            __samples.extend(splitted_samples)
            __labels.extend((index,) * len(splitted_samples))

        self.samples = torch.tensor(__samples, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(__labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    @property
    def classes(self):
        return self.__label_hash.copy()
