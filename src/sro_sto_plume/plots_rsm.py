# sro_sto_plume/plots_rsm.py
import numpy as np
from typing import Iterable, Optional, Tuple, List

from matplotlib import colormaps
from matplotlib.ticker import ScalarFormatter

from m3util.viz.text import labelfigs
from m3util.viz.layout import layout_fig
from xrd_learn.rsm_viz import RSMPlotter
from xrd_learn.xrd_utils import detect_peaks, calculate_fwhm  # used by FWHM helpers

from sro_sto_plume.crystallography import parse_hkl, ideal_q_h0l
from sro_sto_plume.rsm_helpers import (
    intensity_weighted_centroid,
    parse_plane_slope,
    clip_line_to_axes,
)

# palette consistent with rest of project
colors = colormaps.get_cmap('tab10').colors[:6]


# ---------------- main RSM functions ----------------

def plot_rsm_figure(
    plotter: RSMPlotter,
    fig,
    axes: Iterable,
    files: List[str],
    sample_names: List[str],
    cbar_ax,
    *,
    peak_z_range_substrate: Optional[Tuple[float, float]] = None,
    plane: Optional[str] = None,
    ideal_q: Optional[Tuple[float, float]] = None,
    label: bool = False
):
    """
    Draw a strip of RSM panels with a shared colorbar.

    • optional ideal_q=(qx*, qz*) diamond marker (fully relaxed reference)
    • substrate origin from intensity-weighted centroid inside Qz window
    • dashed reference line through that origin with slope from `plane` (e.g. '103' -> 3)
    """
    # Grab desired limits if the plotter specifies them
    xlim_pref = plotter.plot_params.get("xlim", None)
    ylim_pref = plotter.plot_params.get("ylim", None)

    for i, (ax, file, title) in enumerate(zip(axes, files, sample_names)):
        Qx, Qz, I = plotter.plot(file=file, ax=ax, figsize=None,
                                 cbar_ax=cbar_ax, ignore_yaxis=(i != 0))

        # Ensure final limits are known before clipping any lines
        if xlim_pref is not None:
            ax.set_xlim(*xlim_pref)
        if ylim_pref is not None:
            ax.set_ylim(*ylim_pref)

        if label:
            labelfigs(ax, i, size=15, inset_fraction=(0.08, 0.15), loc='tr')

        if ideal_q is not None:
            qx_star, qz_star = ideal_q
            ax.scatter([qx_star], [qz_star], marker='D', s=22,
                       facecolors='none', edgecolors='white',
                       linewidths=0.7, zorder=10)

        if peak_z_range_substrate is not None and plane is not None:
            z_lo, z_hi = peak_z_range_substrate
            mask = (Qz >= z_lo) & (Qz <= z_hi)
            if np.any(mask):
                qx0, qz0 = intensity_weighted_centroid(Qx[mask], Qz[mask], I[mask])
                m = parse_plane_slope(plane)
                if m is not None:
                    seg = clip_line_to_axes(ax, qx0, qz0, m)
                    if seg is not None:
                        x1, z1, x2, z2 = seg
                        # draw above the image
                        ax.plot([x1, x2], [z1, z2], '--',
                                lw=0.8, color='gray', alpha=0.95, zorder=11)
                else:
                    # vertical line x=qx0
                    x_min, x_max = ax.get_xlim()
                    z_min, z_max = ax.get_ylim()
                    ax.plot([qx0, qx0], [z_min, z_max], '--',
                            lw=0.8, color='gray', alpha=0.95, zorder=11)
            else:
                print(f"{title}: no pixels within substrate Qz range {peak_z_range_substrate}.")


def plot_rsm002(rsm002_files, label=True):
    """
    Convenience wrapper to produce a 6-panel RSM(002) strip with a shared colorbar.
    Same signature/name as your original.
    """
    from pathlib import Path
    sample_IDs   = ['YG065', 'YG066', 'YG067', 'YG068', 'YG069', 'YG063']
    sample_names = ['G1',    'G2',    'G3',    'G4',    'G5',    'C-G6']

    figsize = (7.5, 3)
    rsm002_files = sorted(rsm002_files, key=lambda x: sample_IDs.index(Path(x).parts[-2]))

    plot_params_002 = {
        "xlim": (-0.014, 0.015),
        "ylim": (3.05, 3.28),
        "vmax": 30000,
        "label_fontsize": 10,
        "tick_fontsize": 8,
    }
    plotter = RSMPlotter(plot_params_002)

    graph, mod = 7, 7
    width_ratios = [1, 1, 1, 1, 1, 1, 0.1]
    fig, axes = layout_fig(graph=graph, mod=mod, figsize=figsize,
                           subplot_style='gridspec', width_ratios=width_ratios,
                           spacing=(0.2, 0.2), layout='tight')

    ideal_q = ideal_q_h0l("002", a_bulk=3.93)
    plot_rsm_figure(
        plotter, fig,
        axes=axes[:-1],
        files=rsm002_files,
        sample_names=sample_names,
        cbar_ax=axes[-1],
        peak_z_range_substrate=(3.2, 3.23),
        plane="002",
        ideal_q=ideal_q,
        label=label
    )
    return fig, axes


# ---------------- FWHM helpers (kept under original names) ----------------

def plot_fwhm_line_profile_figure(plotter, axes, sample_index, Qx_lines, intensity_lines):
    """
    Plot 1D line profiles and annotate FWHM on each axis.
    - plotter.plot_params can set 'lineplot_yscale', 'lineplot_xlim', 'lineplot_ylim', 'fontsize'
    """
    FWHM_list = []
    for i, (ax, title, Qx_line, intensity_line) in enumerate(zip(axes, sample_index, Qx_lines, intensity_lines)):
        ax.scatter(Qx_line, intensity_line, s=1)

        peak_x, peak_y = detect_peaks(Qx_line, intensity_line, num_peaks=1, prominence=0.1, distance=None)
        fwhm, y_fwhm, x_left, x_right = calculate_fwhm(Qx_line, intensity_line, peak_x[0])
        FWHM_list.append(fwhm)

        ax.plot([x_left, x_right], [y_fwhm, y_fwhm], 'r-', lw=0.5)
        ax.annotate('', xy=(x_right, y_fwhm), xytext=(x_left, y_fwhm),
                    arrowprops=dict(arrowstyle='<->', lw=0.8, color='r', shrinkA=0, shrinkB=0, mutation_scale=5))
        ax.text((x_left + x_right) / 2, y_fwhm*1.05, f'FWHM: {fwhm:.4f}', ha='center', va='bottom', color='r', fontsize=8)

        if plotter.plot_params.get("lineplot_yscale", 'linear') == 'log':
            ax.set_yscale('log')
        ax.set_xlim(plotter.plot_params.get("lineplot_xlim", None))
        ax.set_ylim(plotter.plot_params.get("lineplot_ylim", None))

        ax.tick_params(axis="x", direction="in", top=True, labelsize=plotter.plot_params.get("fontsize", 12))
        ax.tick_params(axis="y", direction="in", right=True, labelsize=plotter.plot_params.get("fontsize", 12))
        ax.set_xlabel(r'$Q_x$ [$\AA^{-1}$]', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold')
        ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold')
        labelfigs(ax, i, size=15, inset_fraction=(0.15, 0.15), loc='tr', style='bw')

    axes[-1].set_ylabel(r'$Q_z$ [$\AA^{-1}$]', fontsize=plotter.plot_params.get("fontsize", 12))
    for ax in axes[1:]:
        ax.set_yticklabels([])
        ax.set_ylabel('')

    return FWHM_list


def plot_fwhm_trend_figure(plotter, ax, sample_index, FWHM_list):
    """
    Draw the two-group FWHM trend with two y-axes (left: first 5, right: last 2).
    """
    # left group (first 5)
    left_x = list(range(len(sample_index[:5])))
    left_x[4] -= 0.05
    ax.plot(left_x, FWHM_list[:5], marker='o', color=colors[0])
    ax.set_xticks(range(len(sample_index)))
    ax.set_xticklabels(sample_index)
    ax.set_ylim(2.2e-3, 4.3e-3)

    ax.set_xlabel('Sample Names', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold')
    ax.set_ylabel('FWHM (set 1)', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold', color=colors[0])
    ax.tick_params(axis="x", direction="in", top=True, labelsize=plotter.plot_params.get("fontsize", 12))
    ax.tick_params(axis="y", direction="in", right=True, labelsize=plotter.plot_params.get("fontsize", 12),
                   color=colors[0], labelcolor=colors[0])

    labelfigs(ax, 10, size=15, inset_fraction=(0.2, 0.05), loc='tr', style='bw')

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((1, 10))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_x(-0.2)

    # right group (last 2) on a twin axis
    right_x = list(range(4, 4 + len(sample_index[-2:])))
    right_x[0] += 0.05
    ax_right = ax.twinx()
    ax_right.plot(right_x, FWHM_list[-2:], marker='o', color=colors[1])
    ax_right.set_ylim(2.2e-3, 4.3e-3)
    ax_right.set_ylabel('FWHM (set 2)', fontsize=plotter.plot_params.get("fontsize", 12),
                        fontweight='bold', color=colors[1])
    ax_right.tick_params(axis="y", direction="in", labelsize=plotter.plot_params.get("fontsize", 12),
                         color=colors[1], labelcolor=colors[1])

    ax_right_formatter = ScalarFormatter(useMathText=True)
    ax_right_formatter.set_scientific(True)
    ax_right_formatter.set_powerlimits((1, 10))
    ax_right.yaxis.set_major_formatter(ax_right_formatter)
    ax_right.yaxis.get_offset_text().set_x(1)
