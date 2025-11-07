import numpy as np
from matplotlib.lines import Line2D
from m3util.viz.layout import layout_fig
from m3util.viz.text import labelfigs
from sro_sto_plume.packed_plot_functions import colors

def plot_target_microscopy_std(per_image_df, rq_variation_df, order, HEIGHT_UNIT):

    fig, axes = layout_fig(1, 1, figsize=(6.5, 4.5), subplot_style='gridspec', spacing=(0, 0.2), layout='tight')
    ax_scatter = axes[0]
    ax_std = ax_scatter.twinx()
    palette = dict(zip(order, colors))
    rng = np.random.default_rng(0)

    # scatter of individual Rq values (left axis)
    for idx, folder in enumerate(order):
        folder_data = per_image_df.loc[per_image_df['folder'] == folder, 'Rq']
        if folder_data.empty:
            continue
        jittered_x = rng.normal(loc=idx, scale=0.08, size=len(folder_data))
        ax_scatter.scatter(jittered_x, folder_data, color=palette[folder], edgecolor='black', linewidth=0.3, alpha=0.85, s=25, zorder=2)

    ax_scatter.set_ylabel(f"Rms ({HEIGHT_UNIT})")
    ax_scatter.set_xlabel("")
    ax_scatter.set_xticks(range(len(order)))
    ax_scatter.set_xticklabels(order)
    ax_scatter.tick_params(axis="both", which="both", direction="in", top=True, bottom=True, left=True, right=False, labelbottom=True, labelleft=True)
    ax_scatter.set_ylim(4.9, 9.4)
    legend_handles = [
        Line2D([], [], marker='o', linestyle='None', markersize=6, markerfacecolor='gray', markeredgecolor='black', label='Surface Roughness-RMS', alpha=0.85),
        Line2D([], [], marker='s', linestyle='None', markersize=8, markerfacecolor='gray', markeredgecolor='gray', label='Scan to Scan Variation-Std', alpha=0.35),
    ]
    # labelfigs(ax_scatter, 0, style='bw', loc='tl', inset_fraction=(0.1, 0.25), size=14)
    ax_scatter.legend(handles=legend_handles, frameon=False, loc='upper right')

    # std bars (right axis)
    std_values = rq_variation_df.reindex(order)['std']
    ax_std.bar(range(len(order)), std_values, width=0.45, color=[palette[f] for f in order], alpha=0.35, edgecolor='none', zorder=0)
    ax_std.set_ylabel(f"Rms Std ({HEIGHT_UNIT})")
    ax_std.tick_params(axis="y", which="both", direction="in", labelright=True, labelleft=False)
    ax_std.set_ylim(0, 1.5)
    
    return fig, ax_scatter, ax_std