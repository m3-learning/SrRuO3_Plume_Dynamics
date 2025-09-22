# plots_xrd.py
from pathlib import Path
import numpy as np
from matplotlib import colormaps
from m3util.viz.text import labelfigs
from m3util.viz.lines import draw_lines
from m3util.viz.layout import layout_fig
from xrd_learn.xrd_viz import plot_xrd
from xrd_learn.xrd_utils import (
    detect_peaks, calculate_fwhm, load_xrd_scans,
    align_peak_to_value, align_fwhm_center_to_value,
    align_peak_y_to_value, upsample_XY
)

# shared color sequence (matches your original)
colors = colormaps.get_cmap('tab10').colors[:6]


def plot_xrd_figure(files, sample_index, fig, ax, xrange, yrange, title, filename):
    """
    Plot stacked 2θ–ω scans aligned to the STO (002) peak with markers and guides.

    Parameters
    ----------
    files : list[str]
    sample_index : list[str]  # labels in legend
    fig, ax : matplotlib Figure and Axes to draw into
    xrange, yrange : tuple, axis ranges (pass None to auto)
    title, filename : optional
    """
    STO_x_peak = 46.4721
    SRO_bulk_x_peak = 46.2425

    Xs, Ys, length_list = load_xrd_scans(files)
    # detect peaks for info (not strictly used here)
    for (X, Y, _) in zip(Xs, Ys, sample_index):
        _ = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)

    # align by STO peak
    Xs_aligned, Ys_aligned = align_peak_to_value(Xs, Ys, STO_x_peak, viz=False)

    # draw stacked traces
    diff = 5e1
    plot_xrd(
        (Xs_aligned, Ys_aligned, length_list),
        sample_index,
        title=title,
        xrange=xrange,
        yrange=yrange,
        diff=diff,
        fig=fig,
        ax=ax,
        legend_style='label',
        text_offset_ratio=(0.992, 3)
    )

    ax.set_xlabel(r'$2\theta$ [°]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Intensity [a.u.]', fontsize=10, fontweight='bold')

    # reference lines/labels
    draw_lines(ax, x_values=[STO_x_peak, STO_x_peak], y_values=[1e3, 3e15],
               style={'color': 'gray', 'linestyle': 'dashed', 'linewidth': 1})
    ax.text(STO_x_peak, 4e15, 'STO\n(002)', fontsize=10, ha='center')

    ax.text(45.9121, 5e14, 'SRO\n(220)', fontsize=10, ha='center')
    ax.text(SRO_bulk_x_peak, 10, 'SRO\n(bulk)', fontsize=10, ha='center')
    draw_lines(ax, x_values=[SRO_bulk_x_peak, SRO_bulk_x_peak], y_values=[2e3, 2e14],
               style={'color': 'gray', 'linestyle': 'dashdot', 'linewidth': 0.8})

    # small markers on SRO peaks and print FWHM
    for i, (X, Y, sample_name, color) in enumerate(zip(Xs_aligned, Ys_aligned, sample_index, colors)):
        peak_x, peak_y = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)
        fwhm_sto, _, _, _ = calculate_fwhm(X, Y, peak_x[0])
        fwhm_sro, _, _, _ = calculate_fwhm(X, Y, peak_x[1])

        # scale peak markers to match stacked scaling in plot_xrd
        peak_y = np.array(peak_y) * diff ** (len(Ys_aligned) - i - 1)
        ax.plot(peak_x[1], peak_y[1] * 3, '+', color=color)

        print(f'Sample: {sample_name}, STO FWHM: {fwhm_sto:.4f}, '
              f'SRO FWHM: {fwhm_sro:.4f}, peak_x[0]: {peak_x[0]:.4f}, peak_x[1]: {peak_x[1]:.4f}')

    if filename:
        fig.savefig(f'{filename}.png', dpi=600)
        fig.savefig(f'{filename}.svg', dpi=600)


def plot_rocking_curve_figure(sample_index, files, fig, ax, inset_coords):
    """
    Plot rocking curves aligned to ωc=0 with an inset scatter of FWHM.

    Parameters
    ----------
    sample_index : list[str]
    files : list[str]
    fig, ax : target figure/axes
    inset_coords : [left, bottom, width, height] in figure coordinates for the inset axis
    """
    Xs, Ys, length_list = load_xrd_scans(files)

    # upsample + center around ~22.95 then align to 0
    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        _ = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)
        X, Y = upsample_XY(X, Y, num_points=5000)
        Xs[i], Ys[i] = X - 22.95, Y

    Xs, Ys = align_peak_to_value(Xs, Ys, target_x_peak=0, viz=False)
    Xs, Ys, FWHM_list = align_fwhm_center_to_value(Xs, Ys, target_x_peak=0, viz=False)
    Xs, Ys = align_peak_y_to_value(Xs, Ys, target_y_peak=None, use_global_max=True, viz=False)

    xrange = (-0.5, 0.55)
    yrange = (8, 5e4)
    plot_xrd((Xs, Ys, length_list), sample_index, xrange=xrange, yrange=yrange,
             diff=1, fig=fig, ax=ax, title=None, legend_style='legend', colors=colors)
    ax.set_yscale('log')
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_xlabel(r'$\omega-\omega_c$ [°]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Intensity [a.u.]', fontsize=10)

    ax.legend(loc='center left', bbox_to_anchor=(0.385, 0.23), fontsize=9, frameon=False,
              labelspacing=0.3, handlelength=1.2, handletextpad=0.5)

    # inset: FWHM scatter
    ax2 = fig.add_axes(inset_coords)
    ax2.scatter(sample_index, FWHM_list, c=colors)
    # Force a fixed tick set BEFORE setting labels (avoids FixedLocator warning)
    ax2.set_xticks(range(len(sample_index)))
    ax2.set_xticklabels(sample_index, fontsize=8, rotation=60)

    ax2.xaxis.set_tick_params(width=1, direction='in', pad=1)
    ax2.yaxis.set_tick_params(width=0.5, direction='in', labelsize=8, pad=2)
    ax2.set_ylabel('FWHM [°]', fontsize=8.5, labelpad=0)
    ax2.set_xlim(-0.5, len(sample_index) - 0.5)
    # keep your original vertical span
    ax2.set_ylim(0.055, 0.072)


def plot_xrd_multiple(xrd_files, rocking_curve_files, rsm103_files, label=True):
    """
    Compose a 3-panel figure:
      [2_1] XRD (2θ–ω) traces
      [2_2] Rocking curves
      [3]   RSM(103) strip (the actual RSM drawing is delegated to plot_rsm_figure)
    """
    from m3util.viz.layout import layout_subfigures_inches, layout_fig
    from m3util.viz.text import labelfigs
    from sro_sto_plume.plots_rsm import plot_rsm_figure, ideal_q_h0l
    from xrd_learn.rsm_viz import RSMPlotter

    sample_IDs   = ['YG065', 'YG066', 'YG067', 'YG068', 'YG069', 'YG063']
    sample_names = ['G1',    'G2',    'G3',    'G4',    'G5',    'C-G6']

    figsize = (7.5, 6)
    subfigures_dict = {
        '2_1': {"position": [0,    3,   3.55, 2.6], 'skip_margin': False, 'margin_pts': 5},
        '2_2': {"position": [3.85, 3,   3.65, 2.6], 'skip_margin': False, 'margin_pts': 5},
        '3':   {"position": [0,    0,   7.2,  2.8], 'skip_margin': False, 'margin_pts': 5},
    }

    fig_all, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    for ax in axes_dict.values():
        ax.axis('off')

    # XRD block
    fig, ax = layout_fig(graph=1, mod=1, figsize=(2.4, 2.6),
                         parent_ax=axes_dict['2_1'], layout='tight')
    plot_xrd_figure(
        xrd_files, sample_names, fig_all, ax,
        xrange=(44.2, 48.0), yrange=None, title=None, filename=None
    )
    ax.set_xlim(44.1, 48.1)
    ax.set_ylim(5, 1e18)
    if label:
        labelfigs(ax, number=0, style='bw', size=15, inset_fraction=(0.99, 0.1), loc='tl')

    # Rocking curve block
    rocking_curve_files = sorted(
        rocking_curve_files,
        key=lambda x: sample_IDs.index(Path(x).parts[-2])
    )
    fig, ax = layout_fig(graph=1, mod=1, figsize=(2.4, 2.6),
                         parent_ax=axes_dict['2_2'], layout='tight')
    plot_rocking_curve_figure(
        sample_names, rocking_curve_files, fig_all, ax,
        inset_coords=[0.87, 0.8, 0.122, 0.108]
    )
    if label:
        labelfigs(ax, number=1, style='bw', size=15, inset_fraction=(0.6, 0.1), loc='tl')

    # RSM (103) strip
    rsm103_files = sorted(rsm103_files, key=lambda x: sample_IDs.index(Path(x).parts[-2]))
    plot_params_103 = {
        "xlim": (1.582, 1.64),
        "ylim": (4.72, 4.86),
        "vmax": 3000,
        "label_fontsize": 10,
        "tick_fontsize": 8,
    }
    plotter = RSMPlotter(plot_params_103)

    # grid inside subfigure; last axis for colorbar
    graph, mod = 7, 7
    width_ratios = [1, 1, 1, 1, 1, 1, 0.1]
    fig, axes = layout_fig(graph=graph, mod=mod, figsize=(8, 3),
                           parent_ax=axes_dict['3'],
                           subplot_style='gridspec',
                           width_ratios=width_ratios,
                           spacing=(0.2, 0.2), layout='tight')

    ideal_q = ideal_q_h0l("103", a_bulk=3.93)
    plot_rsm_figure(
        plotter, fig_all,
        axes=axes[:-1],
        files=rsm103_files,
        sample_names=sample_names,
        cbar_ax=axes[-1],
        peak_z_range_substrate=(4.82, 4.84),
        plane="103",
        ideal_q=ideal_q,
        label=label,
        label_start=2  # continue labeling from previous panels
    )
    return fig_all, axes_dict
