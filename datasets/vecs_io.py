import numpy as np
import warnings


def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    fv = _vecs_read(fv, dim, c_contiguous, filename)
    return fv


def ivecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv[0]
    assert dim > 0
    fv = _vecs_read(fv, dim, c_contiguous, filename)
    return fv


def _vecs_read(fv, dim, c_contiguous, filename):
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if np.any(fv.view(np.int32)[:, 0] != dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def ivecs_write(arr, filename):
    if arr.dtype != np.int32:
        raise TypeError("dtype must be np.int32")

    _vecs_write(arr, filename, np.int32)


def fvecs_write(arr, filename):
    if arr.dtype != np.float32:
        warnings.warning("Converting to np.float32")

    _vecs_write(arr, filename, np.float32)


def _vecs_write(arr, filename, dtype):
    length = np.array([arr.shape[1]], dtype=dtype).tobytes()

    with open(filename, 'wb') as f:
        for v in arr:
            f.write(length + v.tobytes())
