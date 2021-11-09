"""from fairseq"""
import shutil

import numpy as np

from mmap_dataset import MMapIndexedDataset


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


def __best_fitting_dtype(vocab_size=None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def make_builder(out_file, impl, vocab_size=None):
    if impl == "mmap":
        return MMapIndexedDatasetBuilder(
            out_file, dtype=__best_fitting_dtype(vocab_size)
        )


class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes)
