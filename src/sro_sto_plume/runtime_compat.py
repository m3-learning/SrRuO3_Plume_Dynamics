"""Adapt the pinned XRD plotting API without changing its calculations."""
import functools
import inspect

import numpy as np


def enable_parent_axes_layout():
    """Honor parent_ax, which the pinned layout helper silently ignores."""
    import m3util.viz.layout as module
    from matplotlib.gridspec import GridSpec
    original = module.layout_fig
    if getattr(original, '_srruo3_parent_compatible', False):
        return

    @functools.wraps(original)
    def layout_fig(graph, mod=None, figsize=None, subplot_style='subplots',
                   spacing=(0.3, 0.3), layout='compressed', parent_ax=None, **kwargs):
        if parent_ax is None:
            return original(graph, mod=mod, figsize=figsize, subplot_style=subplot_style,
                            spacing=spacing, layout=layout, **kwargs)
        if subplot_style not in ('subplots', 'gridspec'):
            raise ValueError("Invalid layout option. Choose 'gridspec' or 'subplots'.")
        if mod is None:
            mod = next((columns for limit, columns in [(3,2),(5,3),(10,4),(17,5),(26,6)] if graph < limit), 7)
        rows = graph // mod + (graph % mod > 0)
        figure = parent_ax.figure
        left, bottom, width, height = parent_ax.get_position().bounds
        grid = GridSpec(rows, mod, figure=figure, left=left, bottom=bottom,
                        right=left+width, top=bottom+height,
                        wspace=spacing[0], hspace=spacing[1],
                        width_ratios=kwargs.get('width_ratios'),
                        height_ratios=kwargs.get('height_ratios'))
        axes = [figure.add_subplot(grid[i//mod,i%mod]) for i in range(graph)]
        if subplot_style == 'gridspec':
            return figure, axes
        return figure, axes[0] if rows*mod == 1 else np.asarray(axes)

    layout_fig._srruo3_parent_compatible = True
    module.layout_fig = layout_fig


def enable_xrd_axes_compatibility():
    import xrd_learn.xrd_utils as module
    original = module.detect_fringes_thickness
    if getattr(original, '_srruo3_axes_compatible', False) or 'ax_peak' in inspect.signature(original).parameters:
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def detect(*args, ax_peak=None, ax_fft=None, **kwargs):
        if ax_peak is None and ax_fft is None:
            return original(*args, **kwargs)
        mode = signature.bind(*args, **kwargs).arguments.get('mode', 'both')
        routes = []
        if mode in ('fft', 'both'):
            routes.append(('fft', ax_fft))
        if mode in ('peak', 'both'):
            routes.append(('peak', ax_peak))
        pyplot = module.plt
        captured = {}

        class PlotRouter:
            def __init__(self):
                self.index = -1
                self.role = None

            def __getattr__(self, name):
                return getattr(pyplot, name)

            def figure(self, *figure_args, **figure_kwargs):
                self.index += 1
                self.role, axes = routes[self.index]
                if axes is None:
                    return pyplot.figure(*figure_args, **figure_kwargs)
                pyplot.figure(axes.figure.number)
                pyplot.sca(axes)
                return axes.figure

            def plot(self, *plot_args, **plot_kwargs):
                if self.role == 'fft':
                    captured['FFT Frequencies'] = np.asarray(plot_args[0]).copy()
                    captured['FFT Amplitudes'] = np.asarray(plot_args[1]).copy()
                return pyplot.plot(*plot_args, **plot_kwargs)

            def show(self, *show_args, **show_kwargs):
                # The caller displays its composed figure after filling all axes.
                return None

        module.plt = PlotRouter()
        try:
            result = original(*args, **kwargs)
        finally:
            module.plt = pyplot
        return {**result, **captured}

    detect._srruo3_axes_compatible = True
    module.detect_fringes_thickness = detect


def configure_notebook_runtime(repo_root, notebook_name):
    """Activate the published storage, rendering and SDK compatibility fixes."""
    import os
    from pathlib import Path
    from IPython import get_ipython
    from .runtime_memory import enable_daemon_log_drain, enable_fast_array_capture, enable_disk_arrays
    root = Path(repo_root)
    enable_daemon_log_drain()
    enable_fast_array_capture()
    enable_parent_axes_layout()
    enable_xrd_axes_compatibility()
    shell = get_ipython()
    if shell is not None:
        shell.run_line_magic('matplotlib', 'inline')
    shared = os.environ.get('SRRUO3_SHARED_CACHE')
    if shared:
        for category in ('project-downloads', 'downloads'):
            source = Path(shared) / category
            source.mkdir(parents=True, exist_ok=True)
            target = root / '.dataerai' / category
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(source.resolve(), target_is_directory=True)
    result = {'inline_output_capture': True, 'daemon_log_drain': True,
              'parent_axes_layout': True, 'xrd_axes_adapter': True,
              'provenance_npz_compression_level': 1}
    if notebook_name.startswith(('4_', '5_')):
        result['array_storage'] = enable_disk_arrays(root / '.dataerai' / 'scratch')
    if notebook_name.startswith('5_'):
        import imageio_ffmpeg
        binary = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        tools = root / '.dataerai' / 'tools'
        tools.mkdir(parents=True, exist_ok=True)
        target = tools / 'ffmpeg'
        if not target.exists():
            target.symlink_to(binary)
        os.environ['PATH'] = str(tools) + os.pathsep + os.environ.get('PATH', '')
        import subprocess
        result['ffmpeg_version'] = subprocess.check_output([str(binary), '-version'], text=True).splitlines()[0]
    return result
