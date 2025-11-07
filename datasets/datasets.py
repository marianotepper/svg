from dataclasses import dataclass

import numpy as np
import os
import tarfile
from urllib.request import urlretrieve

import datasets.downloader as downloader
import datasets.vecs_io as vecs_io


@dataclass
class Dataset:
    name: str
    X_db: np.ndarray
    X_query: np.ndarray
    gt: np.ndarray


def _create_dataset(name: str, dirname: str, subdirname: str, db_filename: str,
                    query_filename: str, gt_filename: str) -> Dataset:
    if not os.path.exists(f'{dirname}/{subdirname}'):
        os.makedirs(f'{dirname}/{subdirname}')

    relative_filename = f'{subdirname}/{db_filename}.fvec'
    filename = f'{dirname}/{relative_filename}'
    if not os.path.exists(filename):
        downloader.get_file(relative_filename, filename)
    X_db = vecs_io.fvecs_read(filename)

    relative_filename = f'{subdirname}/{query_filename}.fvec'
    filename = f'{dirname}/{relative_filename}'
    if not os.path.exists(filename):
        downloader.get_file(relative_filename, filename)
    X_query = vecs_io.fvecs_read(filename)

    relative_filename = f'{subdirname}/{gt_filename}.ivec'
    filename = f'{dirname}/{relative_filename}'
    if not os.path.exists(filename):
        downloader.get_file(relative_filename, filename)
    gt = vecs_io.ivecs_read(filename)

    nonzero_mask = np.linalg.norm(X_query, axis=1) > 0
    X_query = X_query[nonzero_mask]
    gt = gt[nonzero_mask]

    return Dataset(name, X_db, X_query, gt)


def _select_dataset_wikipedia_squad(name, dirname='./'):
    if name == 'ada002-100k':
        subdir = 'wikipedia_squad/100k'
        db_filename = 'ada_002_100000_base_vectors'
        query_filename = 'ada_002_100000_query_vectors_10000'
        gt_filename = 'ada_002_100000_indices_query_10000'
    elif name == 'cohere-english-v3-100k':
        subdir = 'wikipedia_squad/100k'
        db_filename = 'cohere_embed-english-v3.0_1024_base_vectors_100000'
        query_filename = 'cohere_embed-english-v3.0_1024_query_vectors_10000'
        gt_filename = 'cohere_embed-english-v3.0_1024_indices_b100000_q10000_k100'
    elif name == 'openai-v3-small-100k':
        subdir = 'wikipedia_squad/100k'
        db_filename = 'text-embedding-3-small_1536_100000_base_vectors'
        query_filename = 'text-embedding-3-small_1536_100000_query_vectors_10000'
        gt_filename = 'text-embedding-3-small_1536_100000_indices_query_10000'
    elif name == 'openai-v3-large-3072-100k':
        subdir = 'wikipedia_squad/100k'
        db_filename = 'text-embedding-3-large_3072_100000_base_vectors'
        query_filename = 'text-embedding-3-large_3072_100000_query_vectors_10000'
        gt_filename = 'text-embedding-3-large_3072_100000_indices_query_10000'
    elif name == 'nv-qa-v4-100k':
        subdir = 'wikipedia_squad/100k'
        db_filename = 'nvidia-nemo_1024_base_vectors_100000'
        query_filename = 'nvidia-nemo_1024_query_vectors_10000'
        gt_filename = 'nvidia-nemo_1024_indices_b100000_q10000_k100'
    elif name == 'colbert-1M':
        subdir = 'wikipedia_squad/1M'
        db_filename = 'colbertv2.0_128_base_vectors_1000000'
        query_filename = 'colbertv2.0_128_query_vectors_100000'
        gt_filename = 'colbertv2.0_128_indices_b1000000_q100000_k100'
    elif name == 'gecko-100k':
        subdir = 'wikipedia_squad/100k'
        db_filename = 'textembedding-gecko_100000_base_vectors'
        query_filename = 'textembedding-gecko_100000_query_vectors_10000'
        gt_filename = 'textembedding-gecko_100000_indices_query_10000'
    elif name == 'e5-large-v2-100k':
        subdir = 'wikipedia_squad/100k'
        db_filename = 'intfloat_e5-large-v2_100000_base_vectors'
        query_filename = 'intfloat_e5-large-v2_100000_query_vectors_10000'
        gt_filename = 'intfloat_e5-large-v2_100000_indices_query_10000'
    else:
        raise ValueError(f'Unknown dataset: {name}')

    return _create_dataset(
        name,
        dirname,
        subdir,
        db_filename,
        query_filename,
        gt_filename
    )


def _create_dataset_from_tar(name: str, full_dirname: str,
                             url: str, tar_filename: str, db_filename: str,
                             query_filename: str, gt_filename: str) -> Dataset:
    if not os.path.exists(full_dirname):
        os.makedirs(full_dirname)

    tar_filename = f'{full_dirname}/{tar_filename}'
    if not os.path.exists(tar_filename):
        try:
            urlretrieve(url, tar_filename)
            print(f"File '{tar_filename}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading file: {e}")

    with tarfile.open(tar_filename, 'r:gz') as tar:
        for member in tar:
            if member.isdir():
                continue
            fname = member.name.rsplit('/', 1)[1]
            tar.makefile(member, f'{full_dirname}/{fname}')

    X_db = vecs_io.fvecs_read(f'{full_dirname}/{db_filename}.fvecs')
    X_query = vecs_io.fvecs_read(f'{full_dirname}/{query_filename}.fvecs')
    gt = vecs_io.ivecs_read(f'{full_dirname}/{gt_filename}.ivecs')

    return Dataset(name, X_db, X_query, gt)


def _select_dataset_tar(name, dirname='./'):
    if name == 'siftsmall':
        subdir = 'siftsmall'
        tar_filename = 'siftsmall_base'
        url = 'ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz'
        db_filename = 'siftsmall_base'
        query_filename = 'siftsmall_query'
        gt_filename = 'siftsmall_groundtruth'
    else:
        raise ValueError(f'Unknown dataset: {name}')

    return _create_dataset_from_tar(
        name,
        f'{dirname}/{subdir}',
        url,
        tar_filename,
        db_filename,
        query_filename,
        gt_filename
    )


def select_dataset(name, dirname='./'):
    for loader_function in [_select_dataset_wikipedia_squad,
                            _select_dataset_tar]:
        try:
            return loader_function(name, dirname=dirname)
        except ValueError as e:
            pass


if __name__ == '__main__':
    dataset = select_dataset('ada002-100k')