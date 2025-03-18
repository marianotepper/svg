import os.path
from dataclasses import dataclass
import numpy as np

import vecs_io

@dataclass
class Dataset:
    name: str
    db_filename: str
    query_filename: str
    gt_filename: str
    X_db: np.ndarray
    X_query: np.ndarray
    gt: np.ndarray


def select_dataset(dirname, name):
    if name == "ada002-100k":
        subdir = '100k'
        db_filename = 'ada_002_100000_base_vectors'
        query_filename = 'ada_002_100000_query_vectors_10000'
        gt_filename = 'ada_002_100000_indices_query_10000'
    elif name == "cohere-english-v3-100k":
        subdir = '100k'
        db_filename = 'cohere_embed-english-v3.0_1024_base_vectors_100000'
        query_filename = 'cohere_embed-english-v3.0_1024_query_vectors_10000'
        gt_filename = 'cohere_embed-english-v3.0_1024_indices_b100000_q10000_k100'
    elif name == "openai-v3-small-100k":
        subdir = '100k'
        db_filename = 'text-embedding-3-small_1536_100000_base_vectors'
        query_filename = 'text-embedding-3-small_1536_100000_query_vectors_10000'
        gt_filename = 'text-embedding-3-small_1536_100000_indices_query_10000'
    elif name == "nv-qa-v4-100k":
        subdir = '100k'
        db_filename = 'nvidia-nemo_1024_base_vectors_100000'
        query_filename = 'nvidia-nemo_1024_query_vectors_10000'
        gt_filename = 'nvidia-nemo_1024_indices_b100000_q10000_k100'
    elif name == "colbert-1M":
        subdir = '1M'
        db_filename = 'colbertv2.0_128_base_vectors_1000000'
        query_filename = 'colbertv2.0_128_query_vectors_100000'
        gt_filename = 'colbertv2.0_128_indices_b1000000_q100000_k100'
    elif name == "gecko-100k":
        subdir = '100k'
        db_filename = 'textembedding-gecko_100000_base_vectors'
        query_filename = 'textembedding-gecko_100000_query_vectors_10000'
        gt_filename = 'textembedding-gecko_100000_indices_query_10000'
    else:
        raise ValueError(f"Unknown dataset: {name}")

    dirname = os.path.join(dirname, f'{subdir}')
    db_filename = os.path.join(dirname, db_filename + '.fvec')
    query_filename = os.path.join(dirname, query_filename + '.fvec')
    gt_filename = os.path.join(dirname, gt_filename + '.ivec')

    X_db = vecs_io.fvecs_read(db_filename)
    X_query = vecs_io.fvecs_read(query_filename)
    gt = vecs_io.ivecs_read(gt_filename)

    nonzero_mask = np.linalg.norm(X_query, axis=1) > 0
    X_query = X_query[nonzero_mask]
    gt = gt[nonzero_mask]

    return Dataset(
        name,
        db_filename,
        query_filename,
        gt_filename,
        X_db,
        X_query,
        gt
    )
