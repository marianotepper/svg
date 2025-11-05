from dataclasses import dataclass, field

import numpy as np
import os

import datasets.downloader as downloader
import datasets.vecs_io as vecs_io


@dataclass
class Dataset:
    dirname: str
    subdirname: str
    db_filename: str
    query_filename: str
    gt_filename: str
    X_db: np.ndarray = field(init=False)
    X_query: np.ndarray = field(init=False)
    gt: np.ndarray = field(init=False)

    def __post_init__(self):
        if not os.path.exists(f'{self.dirname}/{self.subdirname}'):
            os.makedirs(f'{self.dirname}/{self.subdirname}')

        relative_filename =  f'{self.subdirname}/{self.db_filename}.fvec'
        filename = f'{self.dirname}/{relative_filename}'
        if not os.path.exists(filename):
            downloader.get_file(relative_filename, filename)
        self.X_db = vecs_io.fvecs_read(filename)

        relative_filename = f'{self.subdirname}/{self.query_filename}.fvec'
        filename = f'{self.dirname}/{relative_filename}'
        if not os.path.exists(filename):
            downloader.get_file(relative_filename, filename)
        self.X_query = vecs_io.fvecs_read(filename)

        relative_filename = f'{self.subdirname}/{self.gt_filename}.ivec'
        filename = f'{self.dirname}/{relative_filename}'
        if not os.path.exists(filename):
            downloader.get_file(relative_filename, filename)
        self.gt = vecs_io.ivecs_read(filename)

        nonzero_mask = np.linalg.norm(self.X_query, axis=1) > 0
        self.X_query = self.X_query[nonzero_mask]
        self.gt = self.gt[nonzero_mask]


def select_dataset(name, dirname='./wikipedia_squad'):
    if name == 'ada002-100k':
        subdir = '100k'
        db_filename = 'ada_002_100000_base_vectors'
        query_filename = 'ada_002_100000_query_vectors_10000'
        gt_filename = 'ada_002_100000_indices_query_10000'
    elif name == 'cohere-english-v3-100k':
        subdir = '100k'
        db_filename = 'cohere_embed-english-v3.0_1024_base_vectors_100000'
        query_filename = 'cohere_embed-english-v3.0_1024_query_vectors_10000'
        gt_filename = 'cohere_embed-english-v3.0_1024_indices_b100000_q10000_k100'
    elif name == 'openai-v3-small-100k':
        subdir = '100k'
        db_filename = 'text-embedding-3-small_1536_100000_base_vectors'
        query_filename = 'text-embedding-3-small_1536_100000_query_vectors_10000'
        gt_filename = 'text-embedding-3-small_1536_100000_indices_query_10000'
    elif name == 'nv-qa-v4-100k':
        subdir = '100k'
        db_filename = 'nvidia-nemo_1024_base_vectors_100000'
        query_filename = 'nvidia-nemo_1024_query_vectors_10000'
        gt_filename = 'nvidia-nemo_1024_indices_b100000_q10000_k100'
    elif name == 'colbert-1M':
        subdir = '1M'
        db_filename = 'colbertv2.0_128_base_vectors_1000000'
        query_filename = 'colbertv2.0_128_query_vectors_100000'
        gt_filename = 'colbertv2.0_128_indices_b1000000_q100000_k100'
    elif name == 'gecko-100k':
        subdir = '100k'
        db_filename = 'textembedding-gecko_100000_base_vectors'
        query_filename = 'textembedding-gecko_100000_query_vectors_10000'
        gt_filename = 'textembedding-gecko_100000_indices_query_10000'
    else:
        raise ValueError(f'Unknown dataset: {name}')

    return Dataset(
        dirname,
        subdir,
        db_filename,
        query_filename,
        gt_filename
    )

if __name__ == '__main__':
    dataset = select_dataset('ada002-100k')