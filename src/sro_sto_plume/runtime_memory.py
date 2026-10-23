"""Use disk-backed storage for large plume scratch arrays without changing values.

The original full-resolution label arrays exceed this worker's RAM. Only the
storage of large arrays in three upstream modules changes; shapes, dtypes,
algorithms, thresholds, inputs, and scientific notebook cells stay the same.
"""
import importlib
from pathlib import Path
import tempfile
import threading
import zipfile

import h5py
import numpy as np


def enable_fast_array_capture():
    """Use lossless level-one compression only for provenance-created NPZs."""
    import dataerai_notebook
    original = dataerai_notebook.NotebookProvenance._serialize_assigned_value
    if getattr(original, '_srruo3_fast_array_capture', False):
        return

    def serialize(self, *args, **kwargs):
        module = np.lib.npyio
        factory = module.zipfile_factory
        def fast_factory(*factory_args, **factory_kwargs):
            if factory_kwargs.get('compression') == zipfile.ZIP_DEFLATED:
                factory_kwargs.setdefault('compresslevel', 1)
            return factory(*factory_args, **factory_kwargs)
        module.zipfile_factory = fast_factory
        try:
            return original(self, *args, **kwargs)
        finally:
            module.zipfile_factory = factory

    serialize._srruo3_fast_array_capture = True
    dataerai_notebook.NotebookProvenance._serialize_assigned_value = serialize


def enable_daemon_log_drain():
    """Drain SDK-owned daemon pipes so transfer workers cannot block on logs."""
    import dataerai.client
    original = dataerai.client.spawn_daemon
    if getattr(original, '_srruo3_drains_logs', False):
        return

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        def drain(stream):
            while stream.read(65536):
                pass
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                threading.Thread(target=drain, args=(stream,), daemon=True).start()
        return process
    spawn._srruo3_drains_logs = True
    dataerai.client.spawn_daemon = spawn


class DiskArrayBackend:
    def __init__(self, directory, max_bytes):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def __getattr__(self, name):
        return getattr(np, name)

    def allocate(self, shape, dtype, order='C'):
        # mmap duplicates the file descriptor. Closing this anonymous temporary
        # file leaves the mapping alive; its disk blocks are released with it.
        with tempfile.TemporaryFile(dir=self.directory) as backing:
            return np.memmap(backing, mode='w+', dtype=dtype, shape=shape, order=order)

    def zeros(self, shape, dtype=float, order='C', *, like=None):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if size < self.max_bytes or like is not None:
            return np.zeros(shape, dtype=dtype, order=order, like=like)
        return self.allocate(shape, dtype, order)

    def array(self, value, *args, **kwargs):
        if isinstance(value, h5py.Dataset) and not args and not kwargs and value.size * value.dtype.itemsize >= self.max_bytes:
            result = self.allocate(value.shape, value.dtype)
            value.read_direct(result)
            return result
        return np.array(value, *args, **kwargs)


def enable_disk_arrays(directory, max_bytes=256 * 1024 * 1024):
    backend = DiskArrayBackend(directory, max_bytes)
    names = [
        'plume_learn.plume_analyzer.PlumeDataset',
        'plume_learn.plume_analyzer.PlumeMetrics',
        'plume_learn.plume_utils.AutoAlign',
    ]
    for name in names:
        importlib.import_module(name).np = backend
    return {'storage': 'anonymous disk-backed NumPy arrays',
            'threshold_bytes': max_bytes, 'modules': names,
            'shapes_and_dtypes_unchanged': True}
