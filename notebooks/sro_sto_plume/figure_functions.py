# import re
import glob
import numpy as np
from pathlib import Path
from skimage.feature import peak_local_max
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib import rcParams

rcParams["svg.hashsalt"] = None  # Disable hashing for consistent results
rcParams["path.simplify"] = False  # Disable path simplification
rcParams["path.simplify_threshold"] = 0.0  # Ensure no detail is lost

from sro_sto_plume.coordinate_converter import convert_top_left_origin_to_matplotlib
from plume_learn.plume_utils.viz import label_violinplot, set_labels, set_cbar
from m3util.viz.layout import layout_fig, layout_subfigures_inches
from m3util.viz.text import labelfigs, add_text_to_figure
from m3util.viz.lines import draw_lines
from xrd_learn.xrd_viz import plot_xrd
from xrd_learn.xrd_utils import (
    detect_peaks,
    calculate_fwhm,
    load_xrd_scans,
    align_peak_to_value,
    align_fwhm_center_to_value,
    align_peak_y_to_value,
    upsample_XY,
)
from xrd_learn.rsm_viz import RSMPlotter
from afm_learn.afm_viz import AFMVisualizer
from afm_learn.afm_image_analyzer import (
    fft2d,
    afm_RMS_roughness,
    calculate_height_profile,
)
from afm_learn.afm_utils import parse_ibw, format_func

colors = colormaps.get_cmap("tab10").colors[:6]


def plot_temporal_heatmaps(df_sample, sample_names, label=True):
    figsize = (10, 6)  # Adjusted figure size for combined plots
    subfigures_dict = {  # [left, bottom, width, height]
        "1_1": {"position": [0, 3, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "1_2": {"position": [2, 3, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "1_3": {"position": [4, 3, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "1_4": {"position": [6, 3, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "1_5": {"position": [8, 3, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_1": {"position": [0, 0, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_2": {"position": [2, 0, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_3": {"position": [4, 0, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_4": {"position": [6, 0, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_5": {"position": [8, 0, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    axes_1 = [axes_dict[f"1_{i}"] for i in range(1, 6)]
    axes_2 = [axes_dict[f"2_{i}"] for i in range(1, 6)]

    # Plot heatmaps for Area
    for i, ax, sample in zip(list(range(0, 5)), axes_1, sample_names):
        df_pivot = df_sample[df_sample["Sample Name"] == sample].pivot(
            index="Plume Index", columns="Time (µs)", values="Area (a.u.)"
        )
        df_pivot = df_pivot.loc[
            :, (df_pivot != 0).any(axis=0)
        ]  # Remove columns where all values are 0
        sns.heatmap(df_pivot, cmap="viridis", cbar=False, ax=ax, vmin=0, vmax=17152)
        if i == 0:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="Plume Index",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )
        else:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )
        if label:
            labelfigs(
                axes=ax,
                number=i,
                size=15,
                style="wb",
                loc="tr",
                inset_fraction=(0.08, 0.08),
            )
    set_cbar(
        fig,
        axes_1[-1],
        cbar_label="Area\n(a.u.)",
        scientific_notation=True,
        tick_in=True,
        ticklabel_fontsize=8,
        labelpad=0,
        fontsize=8,
    )

    # Plot heatmaps for Velocity
    for i, ax, sample in zip(list(range(5, 10)), axes_2, sample_names):
        df_pivot = df_sample[df_sample["Sample Name"] == sample].pivot(
            index="Plume Index", columns="Time (µs)", values="Velocity (m/s)"
        )
        df_pivot = df_pivot.loc[
            :, (df_pivot != 0).any(axis=0)
        ]  # Remove columns where all values are 0
        df_pivot[df_pivot == 0] = 200

        sns.heatmap(
            df_pivot,
            cmap="viridis",
            cbar=False,
            ax=ax,
            norm=LogNorm(vmin=200, vmax=29257),
        )

        if i == 5:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="Plume Index",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )
        else:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )

        if label:
            labelfigs(
                axes=ax,
                number=i,
                size=15,
                style="wb",
                loc="tr",
                inset_fraction=(0.08, 0.08),
            )

    set_cbar(
        fig,
        axes_2[-1],
        cbar_label="Velocity\n(m/s)",
        scientific_notation=True,
        tick_in=True,
        logscale=True,
        ticklabel_fontsize=8,
        labelpad=0,
        fontsize=8,
    )

    return fig, axes_dict


def plot_temporal_violins(df_plume_metrics, label=True):
    fig, axes = layout_fig(
        2, 1, figsize=(8, 6), subplot_style="gridspec", spacing=(0, 0.3), layout="tight"
    )

    sns.violinplot(
        x="Sample Name",
        y="Max Area (a.u.)",
        data=df_plume_metrics,
        width=0.9,
        ax=axes[0],
        palette="deep",
        hue="Sample Name",
        legend=False,
    )

    mean_max_area = df_plume_metrics.groupby("Sample Name")["Max Area (a.u.)"].mean()
    label_violinplot(
        axes[0],
        mean_max_area,
        label_type="average_value",
        text_pos="center",
        value_format="scientific",
        text_size=10,
        offset_parms={
            "x_type": "fixed",
            "x_value": 0,
            "y_type": "ratio",
            "y_value": -0.05,
        },
    )

    if label:
        labelfigs(
            axes=axes[0],
            number=0,
            size=15,
            style="bw",
            loc="tr",
            inset_fraction=(0.15, 0.05),
        )

    sns.violinplot(
        x="Sample Name",
        y="Incident Velocity (m/s)",
        data=df_plume_metrics,
        width=0.9,
        ax=axes[1],
        palette="deep",
        hue="Sample Name",
        legend=False,
    )

    mean_incident_velocity = df_plume_metrics.groupby("Sample Name")[
        "Incident Velocity (m/s)"
    ].mean()
    label_violinplot(
        axes[1],
        mean_incident_velocity,
        label_type="average_value",
        text_pos="center",
        value_format="scientific",
        text_size=10,
        offset_parms={
            "x_type": "fixed",
            "x_value": 0,
            "y_type": "ratio",
            "y_value": -0.05,
        },
    )

    if label:
        labelfigs(
            axes=axes[1],
            number=1,
            size=15,
            style="bw",
            loc="br",
            inset_fraction=(0.12, 0.05),
        )

    return fig, axes


def plot_combined_temporal_variation(
    df_sample, sample_names, df_plume_metrics, label=True
):
    figsize = (8, 6)  # Adjusted figure size for combined plots
    subfigures_dict = {  # [left, bottom, width, height]
        "1_1": {"position": [0, 3, 1.5, 2.6], "skip_margin": True, "margin_pts": 5},
        "1_2": {"position": [1.8, 3, 1.5, 2.6], "skip_margin": True, "margin_pts": 5},
        "1_3": {"position": [3.6, 3, 1.5, 2.6], "skip_margin": True, "margin_pts": 5},
        "1_4": {"position": [5.4, 3, 1.5, 2.6], "skip_margin": True, "margin_pts": 5},
        "1_5": {"position": [7.2, 3, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_1": {"position": [0, 0, 1.5, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_2": {"position": [1.8, 0, 1.5, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_3": {"position": [3.6, 0, 1.5, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_4": {"position": [5.4, 0, 1.5, 2.6], "skip_margin": True, "margin_pts": 5},
        "2_5": {"position": [7.2, 0, 1.7, 2.6], "skip_margin": True, "margin_pts": 5},
        "3": {
            "position": [0.1, 6.2, 8.5, 6],
            "skip_margin": True,
            "margin_pts": 5,
        },  # Violin plot for area and velocity
    }

    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    axes_1 = [axes_dict[f"1_{i}"] for i in range(1, 6)]
    axes_2 = [axes_dict[f"2_{i}"] for i in range(1, 6)]

    # Plot heatmaps for Area
    for i, ax, sample in zip(list(range(2, 7)), axes_1, sample_names):
        df_pivot = df_sample[df_sample["Sample Name"] == sample].pivot(
            index="Plume Index", columns="Time (µs)", values="Area (a.u.)"
        )
        df_pivot = df_pivot.loc[
            :, (df_pivot != 0).any(axis=0)
        ]  # Remove columns where all values are 0
        sns.heatmap(df_pivot, cmap="viridis", cbar=False, ax=ax, vmin=0, vmax=17152)
        if i == 0:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="Plume Index",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )
        else:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )
        if label:
            labelfigs(
                axes=ax,
                number=i,
                size=15,
                style="wb",
                loc="tr",
                inset_fraction=(0.08, 0.08),
            )
    set_cbar(
        fig,
        axes_1[-1],
        cbar_label="Area\n(a.u.)",
        scientific_notation=True,
        tick_in=True,
        ticklabel_fontsize=8,
        labelpad=0,
        fontsize=8,
    )

    # Plot heatmaps for Velocity
    for i, ax, sample in zip(list(range(7, 12)), axes_2, sample_names):
        df_pivot = df_sample[df_sample["Sample Name"] == sample].pivot(
            index="Plume Index", columns="Time (µs)", values="Velocity (m/s)"
        )
        df_pivot = df_pivot.loc[
            :, (df_pivot != 0).any(axis=0)
        ]  # Remove columns where all values are 0
        df_pivot[df_pivot == 0] = 200

        sns.heatmap(
            df_pivot,
            cmap="viridis",
            cbar=False,
            ax=ax,
            norm=LogNorm(vmin=200, vmax=29257),
        )

        if i == 5:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="Plume Index",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )
        else:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )

        if label:
            labelfigs(
                axes=ax,
                number=i,
                size=15,
                style="wb",
                loc="tr",
                inset_fraction=(0.08, 0.08),
            )

    set_cbar(
        fig,
        axes_2[-1],
        cbar_label="Velocity\n(m/s)",
        scientific_notation=True,
        tick_in=True,
        logscale=True,
        ticklabel_fontsize=8,
        labelpad=0,
        fontsize=8,
    )

    # df_plume_metrics, label=True

    axes_dict["3"].axis("off")  # Create a new subplot for the violin plots
    _, axes = layout_fig(
        2,
        1,
        figsize=(8, 6),
        subplot_style="subplots",
        spacing=(0, 0.2),
        parent_ax=axes_dict["3"],
        layout="tight",
    )

    sns.violinplot(
        x="Sample Name",
        y="Max Area (a.u.)",
        data=df_plume_metrics,
        width=0.9,
        ax=axes[0],
        palette="deep",
        hue="Sample Name",
        legend=False,
    )

    mean_max_area = df_plume_metrics.groupby("Sample Name")["Max Area (a.u.)"].mean()
    label_violinplot(
        axes[0],
        mean_max_area,
        label_type="average_value",
        text_pos="center",
        value_format="scientific",
        text_size=10,
        offset_parms={
            "x_type": "fixed",
            "x_value": 0,
            "y_type": "ratio",
            "y_value": -0.05,
        },
    )

    if label:
        labelfigs(
            axes=axes[0],
            number=0,
            size=15,
            style="bw",
            loc="tr",
            inset_fraction=(0.15, 0.05),
        )

    sns.violinplot(
        x="Sample Name",
        y="Incident Velocity (m/s)",
        data=df_plume_metrics,
        width=0.9,
        ax=axes[1],
        palette="deep",
        hue="Sample Name",
        legend=False,
    )

    mean_incident_velocity = df_plume_metrics.groupby("Sample Name")[
        "Incident Velocity (m/s)"
    ].mean()
    label_violinplot(
        axes[1],
        mean_incident_velocity,
        label_type="average_value",
        text_pos="center",
        value_format="scientific",
        text_size=10,
        offset_parms={
            "x_type": "fixed",
            "x_value": 0,
            "y_type": "ratio",
            "y_value": -0.05,
        },
    )

    if label:
        labelfigs(
            axes=axes[1],
            number=1,
            size=15,
            style="bw",
            loc="tr",
            inset_fraction=(0.12, 0.05),
        )

    return fig, axes_dict


def plot_combined_plume_inhomogeneity(
    df_plume_metrics, df_sample, sample_names, custom_palette, label=True
):
    figsize = (8, 6)  # Adjusted figure size for combined plots
    subfigures_dict = {  # [left, bottom, width, height]
        "1_1": {
            "position": [0, 2.8, 3.8, 1.8],
            "skip_margin": True,
            "margin_pts": 5,
        },  # Violin plot for area,
        "1_2": {
            "position": [4.3, 2.8, 3.8, 1.8],
            "skip_margin": True,
            "margin_pts": 5,
        },  # Violin plot for velocity
        "2_1": {
            "position": [0.05, 0, 1.65, 2.5],
            "skip_margin": True,
            "margin_pts": 5,
        },  # Heatmap 1 for area
        "2_2": {
            "position": [2.0, 0, 1.95, 2.5],
            "skip_margin": True,
            "margin_pts": 5,
        },  # Heatmap 2 for area
        "2_3": {
            "position": [4.35, 0, 1.65, 2.5],
            "skip_margin": True,
            "margin_pts": 5,
        },  # Heatmap 1 for velocity
        "2_4": {
            "position": [6.3, 0, 1.95, 2.5],
            "skip_margin": True,
            "margin_pts": 5,
        },  # Heatmap 2 for velocity
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    ax_area_violin, ax_velocity_violin = axes_dict["1_1"], axes_dict["1_2"]
    ax_area_heatmap1, ax_area_heatmap2 = axes_dict["2_1"], axes_dict["2_2"]
    ax_velocity_heatmap1, ax_velocity_heatmap2 = axes_dict["2_3"], axes_dict["2_4"]

    # Plot violinplot - Area
    sns.violinplot(
        x="Sample Name",
        y="Max Area (a.u.)",
        data=df_plume_metrics,
        width=0.5,
        ax=ax_area_violin,
        palette=custom_palette,
        hue="Sample Name",
        legend=False,
        order=sample_names,
    )
    mean_area = (
        df_plume_metrics.groupby("Sample Name")["Max Area (a.u.)"]
        .mean()
        .reindex(sample_names)
    )
    label_violinplot(
        ax_area_violin,
        mean_area,
        label_type="average_value",
        text_pos="center",
        value_format="scientific",
        text_size=10,
        offset_parms={
            "x_type": "fixed",
            "x_value": 0,
            "y_type": "fixed",
            "y_value": -1000,
        },
    )
    set_labels(
        ax_area_violin,
        xlabel="",
        ylabel="Max Area (a.u.)",
        label_fontsize=10,
        ticklabel_fontsize=8,
        yaxis_style="sci",
        show_ticks=True,
        tick_padding=2,
    )
    if label:
        labelfigs(
            axes=ax_area_violin,
            number=0,
            size=15,
            style="bw",
            loc="tr",
            inset_fraction=(0.12, 0.08),
        )

    # Plot violinplot - Velocity
    sns.violinplot(
        x="Sample Name",
        y="Incident Velocity (m/s)",
        data=df_plume_metrics,
        width=0.5,
        ax=ax_velocity_violin,
        palette=custom_palette,
        hue="Sample Name",
        legend=False,
        order=sample_names,
    )
    mean_velocity = (
        df_plume_metrics.groupby("Sample Name")["Incident Velocity (m/s)"]
        .mean()
        .reindex(sample_names)
    )
    label_violinplot(
        ax_velocity_violin,
        mean_velocity,
        label_type="average_value",
        text_pos="center",
        value_format="scientific",
        text_size=10,
        offset_parms={
            "x_type": "fixed",
            "x_value": 0,
            "y_type": "fixed",
            "y_value": -800,
        },
    )
    set_labels(
        ax_velocity_violin,
        xlabel="",
        ylabel="Incident Velocity (m/s)",
        label_fontsize=10,
        ticklabel_fontsize=8,
        yaxis_style="sci",
        show_ticks=True,
        tick_padding=2,
    )
    if label:
        labelfigs(
            axes=ax_velocity_violin,
            number=1,
            size=15,
            style="bw",
            loc="tr",
            inset_fraction=(0.12, 0.08),
        )

    # Plot heatmaps for Area
    for i, ax, sample in zip(
        [2, 3], [ax_area_heatmap1, ax_area_heatmap2], sample_names
    ):
        df_pivot = df_sample[df_sample["Sample Name"] == sample].pivot(
            index="Plume Index", columns="Time (µs)", values="Area (a.u.)"
        )
        df_pivot = df_pivot.loc[
            :, (df_pivot != 0).any(axis=0)
        ]  # Remove columns where all values are 0
        sns.heatmap(df_pivot, cmap="viridis", cbar=False, ax=ax, vmin=0, vmax=17152)
        if i == 2:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="Plume Index",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )
        else:
            set_labels(
                ax,
                xlabel="Time (µs)",
                ylabel="",
                label_fontsize=10,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=False,
                tick_padding=2,
            )
        if label:
            labelfigs(
                axes=ax,
                number=i,
                size=15,
                style="wb",
                loc="tr",
                inset_fraction=(0.08, 0.08),
            )
    set_cbar(
        fig,
        ax_area_heatmap2,
        cbar_label="Area\n(a.u.)",
        scientific_notation=True,
        tick_in=True,
        ticklabel_fontsize=8,
        labelpad=0,
        fontsize=8,
    )

    # Plot heatmaps for Velocity
    for i, ax, sample in zip(
        [4, 5], [ax_velocity_heatmap1, ax_velocity_heatmap2], sample_names
    ):
        df_pivot = df_sample[df_sample["Sample Name"] == sample].pivot(
            index="Plume Index", columns="Time (µs)", values="Velocity (m/s)"
        )
        df_pivot = df_pivot.loc[
            :, (df_pivot != 0).any(axis=0)
        ]  # Remove columns where all values are 0
        df_pivot[df_pivot == 0] = 200

        sns.heatmap(
            df_pivot,
            cmap="viridis",
            cbar=False,
            ax=ax,
            norm=LogNorm(vmin=200, vmax=29257),
        )
        set_labels(
            ax,
            xlabel="Time (µs)",
            ylabel="",
            label_fontsize=10,
            ticklabel_fontsize=8,
            yaxis_style="float",
            show_ticks=False,
            tick_padding=2,
        )
        if label:
            labelfigs(
                axes=ax,
                number=i,
                size=15,
                style="wb",
                loc="tr",
                inset_fraction=(0.08, 0.08),
            )

    set_cbar(
        fig,
        ax_velocity_heatmap2,
        cbar_label="Velocity\n(m/s)",
        scientific_notation=True,
        tick_in=True,
        logscale=True,
        ticklabel_fontsize=8,
        labelpad=0,
        fontsize=8,
    )

    return fig, axes_dict


# plot the spatial inhomogeneity of the plume
def plot_plume_inhomogeneity_area(
    df_plume_metrics, df_sample, sample_names, custom_palette, label=True
):
    figsize = (8, 4)
    subfigures_dict = {
        "1": {
            "position": [0, 2.5, 6, 1.5],
            "skip_margin": False,
            "margin_pts": 5,
        },  # [left, bottom, width, height]
        "2_1": {"position": [0, 0, 2.9, 2.4], "skip_margin": False, "margin_pts": 5},
        "2_2": {"position": [3.1, 0, 3.5, 2.4], "skip_margin": False, "margin_pts": 5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    ax0, ax1, ax2 = axes_dict["1"], axes_dict["2_1"], axes_dict["2_2"]

    # Plot violinplot - '1'
    sns.violinplot(
        x="Sample Name",
        y="Max Area (a.u.)",
        data=df_plume_metrics,
        width=0.5,
        ax=ax0,
        palette=custom_palette,
        hue="Sample Name",
        legend=False,
        order=sample_names,
    )
    mean_max_area = df_plume_metrics.groupby("Sample Name")["Max Area (a.u.)"].mean()
    mean_max_area = mean_max_area.reindex(sample_names)

    label_violinplot(
        ax0,
        mean_max_area,
        label_type="average_value",
        text_pos="center",
        value_format="scientific",
        text_size=10,
        offset_parms={
            "x_type": "fixed",
            "x_value": 0,
            "y_type": "fixed",
            "y_value": -1000,
        },
    )
    if label:
        set_labels(
            ax0,
            xlabel="",
            ylabel="Max Area (a.u.)",
            label_fontsize=11,
            yaxis_style="sci",
            show_ticks=True,
        )
    # ax0.xaxis.set_ticks([])
    labelfigs(
        axes=ax0, number=0, size=15, style="bw", loc="tr", inset_fraction=(0.15, 0.05)
    )
    ax0.tick_params(axis="x", direction="in", length=5, labelsize=12)

    # Plot heatmap - '2_1'
    df_pivot = df_sample[df_sample["Sample Name"] == "t5/s1"].pivot(
        index="Plume Index", columns="Time (µs)", values="Area (a.u.)"
    )
    df_pivot = df_pivot.loc[
        :, (df_pivot != 0).any(axis=0)
    ]  # remove the columns where all values are 0
    # plot and extract vmin=0, vmax=17152
    sns.heatmap(
        df_pivot, cmap="viridis", cbar=False, ax=ax1, vmin=0, vmax=17152
    )  # Disable seaborn's default colorbar
    set_labels(
        ax1,
        xlabel="Time (µs)",
        ylabel="Plume Index",
        label_fontsize=11,
        yaxis_style="float",
        show_ticks=False,
    )
    # set_cbar(fig, ax1, cbar_label='Intensity (a.u.)', scientific_notation=True)
    if label:
        labelfigs(
            axes=ax1,
            number=1,
            size=15,
            style="wb",
            loc="tr",
            inset_fraction=(0.08, 0.08),
        )

    # Plot heatmap - '2_2'
    df_pivot = df_sample[df_sample["Sample Name"] == "s2"].pivot(
        index="Plume Index", columns="Time (µs)", values="Area (a.u.)"
    )
    df_pivot = df_pivot.loc[
        :, (df_pivot != 0).any(axis=0)
    ]  # remove the columns where all values are 0
    sns.heatmap(
        df_pivot, cmap="viridis", cbar=False, ax=ax2, vmin=0, vmax=17152
    )  # Disable seaborn's default colorbar
    set_labels(
        ax2,
        xlabel="Time (µs)",
        ylabel="",
        label_fontsize=11,
        yaxis_style="float",
        show_ticks=False,
    )
    set_cbar(fig, ax2, cbar_label="Area (a.u.)", scientific_notation=True, tick_in=True)
    if label:
        labelfigs(
            axes=ax2,
            number=2,
            size=15,
            style="wb",
            loc="tr",
            inset_fraction=(0.08, 0.08),
        )

    # Plot heatmap - '2_1'
    # df_pivot = df_sample[df_sample['Sample Name']=='t5/s1'].pivot(index="Time (µs)", columns="Plume Index", values='Area (a.u.)')
    # df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]  # remove the columns where all values are 0
    # # plot and extract vmin=0, vmax=17152
    # sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax1, vmin=0, vmax=17152)  # Disable seaborn's default colorbar
    # set_labels(ax1, xlabel="Plume Index", ylabel="Time (µs)", label_fontsize=11, yaxis_style='float', show_ticks=False)
    # # set_cbar(fig, ax1, cbar_label='Intensity (a.u.)', scientific_notation=True)
    # labelfigs(axes=ax1, number=1, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))

    # # Plot heatmap - '2_2'
    # df_pivot = df_sample[df_sample['Sample Name']=='s2'].pivot(index="Time (µs)", columns="Plume Index", values='Area (a.u.)')
    # df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]  # remove the columns where all values are 0
    # sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax2, vmin=0, vmax=17152)  # Disable seaborn's default colorbar
    # set_labels(ax2, xlabel="Plume Index", ylabel="", label_fontsize=11, yaxis_style='float', show_ticks=False)
    # set_cbar(fig, ax2, cbar_label='Area (a.u.)', scientific_notation=True, tick_in=True)
    # labelfigs(axes=ax2, number=2, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))
    return fig, axes_dict


def plot_plume_inhomogeneity_velocity(
    df_plume_metrics, df_sample, sample_names, custom_palette, label=True
):
    figsize = (8, 6)
    subfigures_dict = {
        "1": {
            "position": [0, 4, 6, 2],
            "skip_margin": False,
            "margin_pts": 5,
        },  # [left, bottom, width, height]
        "2_1": {"position": [0, 0, 2.9, 3.9], "skip_margin": False, "margin_pts": 5},
        "2_2": {"position": [3.1, 0, 3.5, 3.9], "skip_margin": False, "margin_pts": 5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    ax0, ax1, ax2 = axes_dict["1"], axes_dict["2_1"], axes_dict["2_2"]

    # Plot violinplot - '1'
    sns.violinplot(
        x="Sample Name",
        y="Incident Velocity (m/s)",
        data=df_plume_metrics,
        width=0.5,
        ax=ax0,
        palette=custom_palette,
        hue="Sample Name",
        legend=False,
        order=sample_names,
    )
    mean_max_area = df_plume_metrics.groupby("Sample Name")[
        "Incident Velocity (m/s)"
    ].mean()
    mean_max_area = mean_max_area.reindex(sample_names)

    label_violinplot(
        ax0,
        mean_max_area,
        label_type="average_value",
        text_pos="center",
        value_format="scientific",
        text_size=10,
        offset_parms={
            "x_type": "fixed",
            "x_value": 0,
            "y_type": "fixed",
            "y_value": -800,
        },
    )
    set_labels(
        ax0,
        xlabel="",
        ylabel="Area (a.u.)",
        label_fontsize=11,
        yaxis_style="sci",
        show_ticks=True,
    )
    if label:
        labelfigs(
            axes=ax0,
            number=0,
            size=15,
            style="bw",
            loc="tr",
            inset_fraction=(0.15, 0.05),
        )
    ax0.tick_params(axis="x", direction="in", length=5, labelsize=12)

    # Plot heatmap - '2_1'
    df_pivot = df_sample[df_sample["Sample Name"] == "t5/s1"].pivot(
        index="Plume Index", columns="Time (µs)", values="Velocity (m/s)"
    )
    df_pivot = df_pivot.loc[
        :, (df_pivot != 0).any(axis=0)
    ]  # remove the columns where all values are 0
    df_pivot[df_pivot == 0] = 200

    # plot and extract vmin=0, vmax=17152
    heatmap = sns.heatmap(
        df_pivot, cmap="viridis", cbar=False, ax=ax1, norm=LogNorm(vmin=200, vmax=29257)
    )
    set_labels(
        ax1,
        xlabel="Time (µs)",
        ylabel="Plume Index",
        label_fontsize=11,
        yaxis_style="float",
        show_ticks=False,
    )
    # set_cbar(fig, ax1, cbar_label='Intensity (a.u.)', scientific_notation=True)
    if label:
        labelfigs(
            axes=ax1,
            number=1,
            size=15,
            style="wb",
            loc="tr",
            inset_fraction=(0.08, 0.08),
        )

    # Plot heatmap - '2_2'
    df_pivot = df_sample[df_sample["Sample Name"] == "s2"].pivot(
        index="Plume Index", columns="Time (µs)", values="Velocity (m/s)"
    )
    df_pivot = df_pivot.loc[
        :, (df_pivot != 0).any(axis=0)
    ]  # remove the columns where all values are 0
    df_pivot[df_pivot == 0] = 200

    heatmap = sns.heatmap(
        df_pivot, cmap="viridis", cbar=False, ax=ax2, norm=LogNorm(vmin=200, vmax=29257)
    )  # Disable seaborn's default colorbar
    set_labels(
        ax2,
        xlabel="Time (µs)",
        ylabel="",
        label_fontsize=11,
        yaxis_style="float",
        show_ticks=False,
    )
    set_cbar(
        fig,
        ax2,
        cbar_label="Velocity (m/s)",
        scientific_notation=True,
        tick_in=True,
        logscale=True,
    )
    if label:
        labelfigs(
            axes=ax2,
            number=2,
            size=15,
            style="wb",
            loc="tr",
            inset_fraction=(0.08, 0.08),
        )

    return fig, axes_dict


from plume_learn.plume_utils.viz import show_images, set_labels
from plume_learn.plume_analyzer.PlumeDataset import plume_dataset
from plume_learn.plume_analyzer.Velocity import VelocityCalculator
from sro_sto_plume.cmap import define_white_viridis


def plume_metrics_summary(df_frame_metrics, plume_recording_root, label=True):
    figsize = (6.8, 8)
    subfigures_dict = {
        "1": {
            "position": [0, 4, 6.55, 1.82],
            "skip_margin": False,
            "margin_pts": 5,
        },  # [left, bottom, width, height]
        "1_1": {
            "position": [0.03, 4.05, 6.5, 0.6],
            "skip_margin": False,
            "margin_pts": 5,
        },
        "2": {"position": [0, 2, 6.55, 1.82], "skip_margin": False, "margin_pts": 5},
        "2_1": {
            "position": [0.03, 2.05, 6.5, 0.6],
            "skip_margin": False,
            "margin_pts": 5,
        },
        "3": {"position": [0, 0, 6.55, 1.82], "skip_margin": False, "margin_pts": 5},
    }
    fig_all, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)

    # plume area plot
    df_filtered = df_frame_metrics[df_frame_metrics["Time (µs)"].isin(range(0, 9))]
    lineplot = sns.lineplot(
        x="Time (µs)",
        y="Area (a.u.)",
        hue="Sample Name",
        data=df_filtered,
        ax=axes_dict["1"],
    )
    set_labels(
        axes_dict["1"],
        yaxis_style="sci",
        xlim=(0, 8),
        ylim=(-6000, 18000),
        legend=False,
    )
    axes_dict["1"].legend(fontsize=8, frameon=False)
    if label:
        labelfigs(
            axes=axes_dict["1"],
            number=0,
            size=15,
            style="bw",
            inset_fraction=(0.2, 0.05),
        )

    file = f"{plume_recording_root}/YG065_YichenGuo_09102024.h5"
    plume_ds = plume_dataset(file_path=file, group_name="PLD_Plumes")
    keys = plume_ds.dataset_names()
    plumes = plume_ds.load_plumes("1-SrRuO3")
    plumes = plumes[
        plumes.sum(axis=(1, 2, 3))
        >= np.mean(plumes.sum(axis=(1, 2, 3))) - 3 * np.std(plumes.sum(axis=(1, 2, 3)))
    ]  # remove outliers

    sample_frames = plumes[0][0:17]
    labels = np.arange(len(sample_frames)) * 500e-3
    labels = [f"{l:.1f}µs" for l in labels]
    labels[0] = "t=" + labels[0]
    fig, axes_1 = layout_fig(
        17,
        mod=17,
        figsize=(9, 3),
        subplot_style="gridspec",
        spacing=(0.1, 0.3),
        parent_ax=axes_dict["1_1"],
        layout="tight",
    )
    white_viridis = define_white_viridis()
    show_images(
        sample_frames,
        labels=None,
        img_per_row=17,
        title=None,
        fig=fig,
        axes=axes_1,
        label_size=8,
        cmap=white_viridis,
    )
    axes_dict["1_1"].axis("off")

    # plume position plot
    lineplot = sns.lineplot(
        x="Time (µs)",
        y="Distance (m)",
        hue="Sample Name",
        data=df_filtered,
        ax=axes_dict["2"],
    )
    set_labels(
        axes_dict["2"],
        yaxis_style="sci",
        xlim=(0, 8),
        ylim=(-0.01, 0.038),
        legend=False,
    )
    axes_dict["2"].get_legend().remove()
    if label:
        labelfigs(
            axes=axes_dict["2"],
            number=1,
            size=15,
            style="bw",
            inset_fraction=(0.2, 0.05),
        )

    coords_root = "../data/Plumes/frame_normalize_dataset/"
    coords_path = coords_root + "YG065_coords.npy"
    standard_coords_path = coords_root + "standard_coords.npy"
    coords_standard = np.load(standard_coords_path)
    start_position = np.round(np.mean(coords_standard[:2], axis=0)).astype(
        np.int32
    )  # start position of plume  (x, y)
    position_range = (
        np.min(coords_standard[:, 0]),
        np.max(coords_standard[:, 0]),
    )  # x position range
    V = VelocityCalculator(
        start_position, position_range, threshold=200, progress_bar=False
    )
    plume_positions, plume_distances, plume_velocities = (
        V.calculate_distance_area_for_plumes(plumes)
    )

    sample_plume_positions = plume_positions[0, :17]
    fig, axes_2 = layout_fig(
        17,
        mod=17,
        figsize=(9, 3),
        subplot_style="gridspec",
        spacing=(0.1, 0.3),
        parent_ax=axes_dict["2_1"],
        layout="tight",
    )
    show_images(
        sample_frames,
        labels=None,
        img_per_row=17,
        title=None,
        fig=fig,
        axes=axes_2,
        label_size=8,
        cmap=white_viridis,
    )
    for i, ax in enumerate(axes_2):
        if np.sum(sample_plume_positions[i]) == 0:
            continue
        x, y = sample_plume_positions[i]
        ax.vlines(x, 0, y * 1.8, linestyles="dashed", colors="red", linewidth=0.8)
    axes_dict["2_1"].axis("off")

    # plume velocity plot
    lineplot = sns.lineplot(
        x="Distance (m)",
        y="Velocity (m/s)",
        hue="Sample Name",
        data=df_frame_metrics,
        ax=axes_dict["3"],
    )
    if label:
        set_labels(axes_dict["3"], yaxis_style="sci", ylim=(-3000, 30000), legend=False)
    axes_dict["3"].get_legend().remove()
    if label:
        labelfigs(
            axes=axes_dict["3"],
            number=2,
            size=15,
            style="bw",
            inset_fraction=(0.2, 0.05),
        )
    axes_dict["3"].axvline(
        x=0.029, color="red", ymin=0.15, ymax=0.38, linestyle="--", linewidth=0.8
    )
    axes_dict["3"].axvline(
        x=0.030, color="red", ymin=0.15, ymax=0.38, linestyle="--", linewidth=0.8
    )
    axes_dict["3"].text(
        0.0295,
        1e4,
        "Estimated\nIncident Velocity",
        fontsize=8,
        color="red",
        ha="center",
        bbox=dict(facecolor="none", edgecolor="none"),
    )

    return fig_all, axes_dict


# afm section
def plot_afm_figure_lineprofile(
    afm_visualizer,
    files_ibw,
    files_txt,
    sample_names,
    colors,
    line_ax_indexes,
    line_profile_txt,
    line_coords,
    label=True,
):
    figsize = (8, 4)
    width, height = 2, 2
    w_spacing, v_spacing = 0.2, 0.1
    figsize = (width * 3 + w_spacing * 2, height * 3 + v_spacing * 2)

    subfigures_dict = {
        "1_1": {
            "position": [0.4, 1.5 * height + v_spacing * 2, width, height],
            "skip_margin": True,
            "margin_pts": 5,
        },  # [left, bottom, width, height]
        "1_2": {
            "position": [
                0.4 + width + w_spacing,
                1.5 * height + v_spacing * 2,
                width,
                height,
            ],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "1_3": {
            "position": [
                0.4 + width * 2 + w_spacing * 2,
                1.5 * height + v_spacing * 2,
                width,
                height,
            ],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "2_1": {
            "position": [0.4, height * 0.5 + v_spacing, width, height],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "2_2": {
            "position": [
                0.4 + width + w_spacing,
                height * 0.5 + v_spacing,
                width,
                height,
            ],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "2_3": {
            "position": [
                0.4 + width * 2 + w_spacing * 2,
                height * 0.5 + v_spacing,
                width,
                height,
            ],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "3_1": {
            "position": [0.4, 0, width, height / 2],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "3_2": {
            "position": [0.4 + width + w_spacing, 0, width, height / 2],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "3_3": {
            "position": [0.4 + width * 2 + w_spacing * 2, 0, width, height / 2],
            "skip_margin": True,
            "margin_pts": 5,
        },
    }

    marker_pos_list = [(1.01, 0.7)] * 6
    text_pos_list = [
        (1.99, 4.53),
        (4.18, 4.53),
        (6.38, 4.53),
        (1.99, 2.44),
        (4.18, 2.44),
        (6.38, 2.44),
    ]

    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)

    # AFM
    roughness_list = []
    for i, (ax, file_txt, file_ibw) in enumerate(
        zip(axes_dict.values(), files_txt, files_ibw)
    ):
        img = np.loadtxt(file_txt)
        roughness = afm_RMS_roughness(img)
        roughness_list.append(roughness * 1e9)  # convert to nm
        afm_imgs, sample_name, labels_correct, scan_size = parse_ibw(file_ibw)

        afm_visualizer.viz(img=img, scan_size=scan_size, fig=fig, ax=ax, title=None)
        if label:
            labelfigs(
                ax, number=i, style="wb", size=15, loc="tr", inset_fraction=(0.12, 0.15)
            )

    for ax, sample_name, color, text_pos, marker_pos in zip(
        axes_dict.values(), sample_names, colors, text_pos_list, marker_pos_list
    ):
        marker_line_label = Line2D(
            [0],
            [0],
            color=color,
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=5,
            path_effects=[withStroke(linewidth=4, foreground="white")],
        )

        ax.legend(
            handles=[marker_line_label],
            loc="lower right",
            frameon=False,
            bbox_to_anchor=marker_pos,
            fontsize=8,
        )
        add_text_to_figure(
            fig,
            sample_name,
            text_pos,
            fontsize=10,
            color="white",
            ha="center",
            path_effects=[withStroke(linewidth=0.9, foreground="black")],
        )

    axes_mark = [list(axes_dict.values())[i] for i in line_ax_indexes]
    colors_mark = [colors[i] for i in line_ax_indexes]
    for i, (ax_line, txt, coord, ax_mark, color) in enumerate(
        zip(
            list(axes_dict.values())[-3:],
            line_profile_txt,
            line_coords,
            axes_mark,
            colors_mark,
        )
    ):
        p1, p2 = coord  # [x1, y1], [x2, y2]

        img = np.loadtxt(txt)
        # Plot the black outline line
        ax_mark.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], color="w", linestyle="-", linewidth=1.5
        )

        # Plot the actual colored line on top
        ax_mark.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle="-", linewidth=0.8
        )

        # sample t2 line profile
        x, values = calculate_height_profile(img, p1, p2)
        ax_line.plot(x, values * 1e9, color=color, linestyle="-", linewidth=1)

        if i == 0:
            set_labels(
                ax_line,
                xlabel="X (a.u.)",
                ylabel="Height (nm)",
                label_fontsize=9,
                ticklabel_fontsize=8,
                yaxis_style="float",
                show_ticks=True,
            )
        else:
            set_labels(
                ax_line,
                xlabel="X (a.u.)",
                ylabel="",
                label_fontsize=9,
                ticklabel_fontsize=9,
                yaxis_style="float",
                show_ticks=True,
            )
        ax_line.tick_params(pad=1)  # Adjust label distance

        if label:
            labelfigs(
                ax_line,
                number=i + 6,
                style="bw",
                size=15,
                loc="tl",
                inset_fraction=(0.2, 0.1),
            )

    return fig, axes_dict


# afm section
def plot_afm_figure(
    afm_visualizer,
    files_ibw,
    files_txt,
    files_roughness_txt,
    sample_names,
    colors,
    plot_roughness=True,
    roughness_ylim=None,
    roughness_label_loc="tr",
    label=True,
):
    width, height = 2, 2
    lineplot_height = 1.5
    w_spacing, v_spacing = 0.2, 0.1
    figsize = (width * 3 + w_spacing * 2, height * 3 + v_spacing * 2)

    subfigures_dict = {
        "1_1": {
            "position": [0.4, height + lineplot_height + v_spacing * 2, width, height],
            "skip_margin": True,
            "margin_pts": 5,
        },  # [left, bottom, width, height]
        "1_2": {
            "position": [
                0.4 + width + w_spacing,
                height + lineplot_height + v_spacing * 2,
                width,
                height,
            ],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "1_3": {
            "position": [
                0.4 + width * 2 + w_spacing * 2,
                height + lineplot_height + v_spacing * 2,
                width,
                height,
            ],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "2_1": {
            "position": [0.4, lineplot_height + v_spacing, width, height],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "2_2": {
            "position": [
                0.4 + width + w_spacing,
                lineplot_height + v_spacing,
                width,
                height,
            ],
            "skip_margin": True,
            "margin_pts": 5,
        },
        "2_3": {
            "position": [
                0.4 + width * 2 + w_spacing * 2,
                lineplot_height + v_spacing,
                width,
                height,
            ],
            "skip_margin": True,
            "margin_pts": 5,
        },
        # '3_1': {"position": [0.4, 0, width*3+np.sum(w_spacing)+0.2, lineplot_height], 'skip_margin': True, 'margin_pts':5},
    }

    if plot_roughness:
        subfigures_dict["3_1"] = {
            "position": [0.4, 0, width * 3 + np.sum(w_spacing) + 0.2, lineplot_height],
            "skip_margin": True,
            "margin_pts": 5,
        }

    marker_pos_list = [(1.01, 0.7)] * 6
    text_pos_list = [
        (1.99, 5.03),
        (4.18, 5.03),
        (6.38, 5.03),
        (1.99, 2.94),
        (4.18, 2.94),
        (6.38, 2.94),
    ]

    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)

    # AFM
    roughness_list = []
    for i, (ax, file_txt, file_roughness_txt, file_ibw) in enumerate(
        zip(axes_dict.values(), files_txt, files_roughness_txt, files_ibw)
    ):
        img = np.loadtxt(file_txt)
        img_roughness = np.loadtxt(file_roughness_txt)
        roughness = afm_RMS_roughness(img_roughness)
        roughness_list.append(roughness * 1e9)  # convert to nm
        afm_imgs, sample_name, labels_correct, scan_size = parse_ibw(file_ibw)

        afm_visualizer.viz(img=img, scan_size=scan_size, fig=fig, ax=ax, title=None)
        if label:
            labelfigs(
                ax, number=i, style="wb", size=15, loc="tr", inset_fraction=(0.12, 0.15)
            )
        roughness_str = f"{roughness:.2e}"
        # add_text_to_figure(fig, roughness_str, (1.2*i, 0.8), fontsize=8)

    for ax, sample_name, color, text_pos, marker_pos in zip(
        axes_dict.values(), sample_names, colors, text_pos_list, marker_pos_list
    ):
        marker_line_label = Line2D(
            [0],
            [0],
            color=color,
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=5,
            path_effects=[withStroke(linewidth=4, foreground="white")],
        )

        ax.legend(
            handles=[marker_line_label],
            loc="lower right",
            frameon=False,
            bbox_to_anchor=marker_pos,
            fontsize=8,
        )
        add_text_to_figure(
            fig,
            sample_name,
            text_pos,
            fontsize=10,
            color="white",
            ha="center",
            path_effects=[withStroke(linewidth=0.9, foreground="black")],
        )

    if plot_roughness:
        axes_dict["3_1"].scatter(
            sample_names, roughness_list, c=colors
        )  # Use any colormap, e.g., 'viridis', 'plasma', etc.
        set_labels(
            axes_dict["3_1"],
            xlabel="",
            ylabel="Roughness (nm)",
            label_fontsize=10,
            ticklabel_fontsize=10,
            yaxis_style="float",
            show_ticks=True,
        )
        if roughness_ylim:
            axes_dict["3_1"].set_ylim(roughness_ylim)
        else:
            roughness_ylim = axes_dict["3_1"].get_ylim()
        for i, r in enumerate(roughness_list):
            axes_dict["3_1"].text(
                i,
                r - 0.11 * (roughness_ylim[1] - roughness_ylim[0]),
                f"{r:.2f}nm",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        labelfigs(
            axes_dict["3_1"],
            number=6,
            style="bw",
            size=15,
            loc=roughness_label_loc,
            inset_fraction=(0.18, 0.06),
        )
    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25, hspace=0.1
    )  # Further adjustments
    return fig, axes_dict


########################################################################################
# XRD section
########################################################################################

def plot_xrd_multiple(
    xrd_files: list[str],
    rocking_curve_files: list[str],
    rsm002_files: list[str],
    rsm103_files: list[str],
    label: bool = True,
    **kwargs,
):
    """
    Plots multiple XRD-related graphs including XRD, rocking curves, and RSM (Reciprocal Space Mapping) data.

    Parameters:
    xrd_files (list[str]): List of file paths for XRD data.
    rocking_curve_files (list[str]): List of file paths for rocking curve data.
    rsm002_files (list[str]): List of file paths for RSM (002) data.
    rsm103_files (list[str]): List of file paths for RSM (103) data.
    label (bool): Whether to include labels in the plots. Default is True.
    **kwargs: Additional keyword arguments.
    
    Returns:
    fig_all: The main figure.
    axes_dict: A dictionary containing the axes objects.
    """
    
    # Parameters
    ########################################################################################
    
    sample_IDs = kwargs.get("sample_IDs", ["YG065", "YG066", "YG067", "YG068", "YG069", "YG063"])
    sample_names = kwargs.get("sample_names", ["G1", "G2", "G3", "G4", "G5", "C-G6"])

    figsize = kwargs.get("figsize", (7.5, 9))
    
    subfigures_dict = kwargs.get("subfigures_dict", {
        "1": {
            "position": [0, 5.9, 7.2, 2.8],
            "skip_margin": False,
            "margin_pts": 5,
        },  # RSM (002)
        "2_1": {
            "position": [0, 3, 3.55, 2.6],
            "skip_margin": False,
            "margin_pts": 5,
        },  # XRD
        "2_2": {
            "position": [3.85, 3, 3.65, 2.6],
            "skip_margin": False,
            "margin_pts": 5,
        },  # Rocking
        "3": {
            "position": [0, 0, 7.2, 2.8],
            "skip_margin": False,
            "margin_pts": 5,
        },  # RSM (103)
    })
    
    plot_params_002 = kwargs.get("plot_params_002", {
        "xlim": (-0.014, 0.015),
        "ylim": (3.05, 3.28),
        "vmax": 30000,
        "label_fontsize": 10,
        "tick_fontsize": 8,
    })
    
    plot_params_103 = kwargs.get("plot_params_103", {
        "xlim": (1.582, 1.64),
        "ylim": (4.72, 4.86),
        "vmax": 3000,
        "lineplot_xlim": (1.595, 1.621),
        "lineplot_ylim": (-100, 900),
        "label_fontsize": 10,
        "tick_fontsize": 8,
    })
    
    graph = kwargs.get("graph", 7)
    mod = kwargs.get("mod", 7)
    width_ratios = kwargs.get("width_ratios", [1, 1, 1, 1, 1, 1, 0.1])
    text_locs = kwargs.get("text_locs", [(1.1 + 1.138 * i, 6.25) for i in range(6)])

    
    ########################################################################################
    # Layout
    ########################################################################################

    fig_all, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    
    for ax in axes_dict.values():
        ax.axis("off")

    # RSM (002) plot (moved from row 3 to row 1)
    # files = glob.glob('../../data/XRD_RSM/YG06*/*002*.xrdml')
    rsm002_files = sorted(
        rsm002_files, key=lambda x: sample_IDs.index(Path(x).parts[-2])
    )

    plotter = RSMPlotter(plot_params_002)

    fig, axes = layout_fig(
        graph=graph,
        mod=mod,
        figsize=(2.4, 2.6),
        parent_ax=axes_dict["1"],
        subplot_style="gridspec",
        width_ratios=width_ratios,
        spacing=(0.2, 0.2),
        layout="tight",
    )
    
    Qx_lines, intensity_lines = plot_rsm_figure(
        plotter,
        fig_all,
        axes,
        rsm002_files,
        sample_names,
        cbar_ax=axes[-1],
        peak_z_range=(3.15, 3.20),
        draw_peak=True,
        draw_peak_line=False,
        i_start=0,
        text_locs=text_locs,
        label=label,
    )

    fig, ax = layout_fig(
        graph=1, mod=1, figsize=(2.4, 2.6), parent_ax=axes_dict["2_1"], layout="tight"
    )

    plot_xrd_figure(
        xrd_files,
        sample_names,
        fig_all,
        ax,
        xrange=(44.2, 48),
        yrange=None,
        title=None,
        filename=None,
    )

    ax.set_xlim(44.1, 48.1)
    ax.set_ylim(5, 1e18)

    if label:
        labelfigs(
            ax, number=6, style="bw", size=15, inset_fraction=(0.99, 0.1), loc="tl"
        )

    # Rocking curve plot
    rocking_curve_files = sorted(
        rocking_curve_files, key=lambda x: sample_IDs.index(Path(x).parts[-2])
    )

    fig, ax = layout_fig(
        graph=1, mod=1, figsize=(2.4, 2.6), parent_ax=axes_dict["2_2"], layout="tight"
    )

    plot_rocking_curve_figure(
        sample_names,
        rocking_curve_files,
        fig_all,
        ax,
        inset_coords=[0.87, 0.51, 0.122, 0.108],
    )  

    if label:
        labelfigs(
            ax, number=7, style="bw", size=15, inset_fraction=(0.6, 0.1), loc="tl"
        )

    # RSM 103 phi-angle4 (now on second row)
    rsm103_files = sorted(
        rsm103_files, key=lambda x: sample_IDs.index(Path(x).parts[-2])
    )

    
    plotter = RSMPlotter(plot_params_103)
    
    fig, axes = layout_fig(
        graph=graph,
        mod=mod,
        figsize=(8, 3),
        parent_ax=axes_dict["3"],
        subplot_style="gridspec",
        width_ratios=width_ratios,
        spacing=(0.2, 0.2),
        layout="tight",
    )
    
    Qx_lines, intensity_lines = plot_rsm_figure(
        plotter,
        fig_all,
        axes,
        rsm103_files,
        sample_names,
        cbar_ax=axes[-1],
        peak_z_range=(4.78, 4.80),
        draw_peak=True,
        draw_peak_line=False,
        draw_ideal_Qz_line=True,
        ideal_Qz=4.78,
        i_start=8,
        text_locs=text_locs,
        label=label,
    )

    return fig_all, axes_dict


def plot_xrd_figure(files, sample_index, fig, ax, xrange, yrange, title, filename):
    STO_x_peak = 46.4721
    SRO_bulk_x_peak = 46.2425

    Xs, Ys, length_list = load_xrd_scans(files)
    for i, (X, Y, sample_name) in enumerate(zip(Xs, Ys, sample_index)):
        peak_x, peak_y = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)
    Xs_aligned, Ys_aligned = align_peak_to_value(Xs, Ys, STO_x_peak, viz=False)

    # fig, ax = layout_fig(1, 1, figsize=(8, 4), layout='tight')
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
        legend_style="label",
        text_offset_ratio=(0.992, 3),
    )

    ax.set_xlabel(r"$2\theta$ [°]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Intensity [a.u.]", fontsize=10, fontweight="bold")

    line_style = {"color": "gray", "linestyle": "dashed", "linewidth": 1}
    draw_lines(
        ax, x_values=[STO_x_peak, STO_x_peak], y_values=[1e3, 3e15], style=line_style
    )
    ax.text(STO_x_peak, 4e15, "STO\n(002)", fontsize=10, ha="center")

    ax.text(45.9121, 5e14, "SRO\n(220)", fontsize=10, ha="center")
    line_style = {"color": "gray", "linestyle": "dotted", "linewidth": 0.8}
    # draw_lines(ax, x_values=[45.9121, 45.9121], y_values=[5e2, 5e14], style=line_style)

    ax.text(SRO_bulk_x_peak, 10, "SRO\n(bulk)", fontsize=10, ha="center")
    line_style = {"color": "gray", "linestyle": "dashdot", "linewidth": 0.8}
    draw_lines(
        ax,
        x_values=[SRO_bulk_x_peak, SRO_bulk_x_peak],
        y_values=[2e3, 2e14],
        style=line_style,
    )

    legend = []
    for i, (X, Y, sample_name, color) in enumerate(
        zip(Xs_aligned, Ys_aligned, sample_index, colors)
    ):
        peak_x, peak_y = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)

        # Calculate FWHM for the STO peak (peak_x[0]) and SRO peak (peak_x[1])
        fwhm_sto, y_fwhm_sto, x_left_sto, x_right_sto = calculate_fwhm(X, Y, peak_x[0])
        fwhm_sro, y_fwhm_sro, x_left_sro, x_right_sro = calculate_fwhm(X, Y, peak_x[1])

        # Prepare legend item
        # legend_item = f'SRO(+): {peak_x[1]:.4f}°, STO(*): {peak_x[0]:.4}°'
        # legend_item = f'SRO(+): {peak_x[1]:.4f}°, FWHM: {fwhm_sro:.2f}'
        # legend_item = f'SRO(+): {peak_x[1]:.2f}°'
        # legend.append(legend_item)

        peak_y = np.array(peak_y) * diff ** (len(Ys) - i - 1)
        # plt.plot(peak_x[0], peak_y[0]*3, '*', color=color)
        ax.plot(peak_x[1], peak_y[1] * 3, "+", color=color)

        print(
            f"Sample: {sample_name}, STO FWHM: {fwhm_sto:.4f}, SRO FWHM: {fwhm_sro:.4f}, peak_x[0]: {peak_x[0]:.4f}, peak_x[1]: {peak_x[1]:.4f}"
        )

    # plt.legend(legend, fontsize=9, loc='upper right', frameon=False)
    if filename:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.svg", dpi=600)


def plot_rocking_curve_figure(sample_index, files, fig, ax, inset_coords):
    Xs, Ys, length_list = load_xrd_scans(files)

    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        peak_x, peak_y = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)
        # upsample the X, Y
        X, Y = upsample_XY(X, Y, num_points=5000)
        Xs[i], Ys[i] = X - 22.95, Y

    Xs, Ys = align_peak_to_value(Xs, Ys, target_x_peak=0, viz=False)
    Xs, Ys, FWHM_list = align_fwhm_center_to_value(Xs, Ys, target_x_peak=0, viz=False)
    Xs, Ys = align_peak_y_to_value(
        Xs, Ys, target_y_peak=None, use_global_max=True, viz=False
    )
    # fwhm, y_fwhm, x_left, x_right = calculate_fwhm(Qx_line, intensity_line, peak_x[0])

    xrange = (-0.5, 0.55)
    yrange = (8, 5e4)
    plot_xrd(
        (Xs, Ys, length_list),
        sample_index,
        xrange=xrange,
        yrange=yrange,
        diff=1,
        fig=fig,
        ax=ax,
        title=None,
        legend_style="legend",
        colors=colors,
    )
    ax.set_yscale("log")
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.set_xlabel(r"$\omega-\omega_c$ [°]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Intensity [a.u.]", fontsize=10)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.385, 0.23),
        fontsize=9,
        frameon=False,
        labelspacing=0.3,  # Space between labels
        handlelength=1.2,  # Length of the line in the legend
        handletextpad=0.5,  # Space between line and text
        # borderaxespad=0.2      # Padding between legend and axes
    )

    ax2 = fig.add_axes(
        inset_coords
    )  # [left, bottom, width, height] # inset_coords = [0.6, 0.6, 0.25, 0.25]
    ax2.scatter(
        sample_index, FWHM_list, c=colors
    )  # Use any colormap, e.g., 'viridis', 'plasma', etc.
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    ax2.set_xticklabels(sample_index, fontsize=8, rotation=60)
    ax2.xaxis.set_tick_params(width=1, direction="in", pad=1)
    ax2.yaxis.set_tick_params(width=0.5, direction="in", labelsize=8, pad=2)

    ax2.set_ylabel("FWHM [°]", fontsize=8.5, labelpad=0)

    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(0.055, 0.072)


def set_fig_axes():
    width_margin, height_margin = 0.12, 0.6
    y_start, row_heights = 0, [2.3, 1.4, 1.3]
    first_row_y, first_row_width, first_row_height = y_start, 0.9, row_heights[0]
    second_row_y, second_row_width, second_row_height = (
        y_start + height_margin + row_heights[0],
        0.9,
        row_heights[1],
    )
    third_row_y, third_row_width, third_row_height = (
        y_start + height_margin * 2 + row_heights[0] + row_heights[1],
        6,
        row_heights[2],
    )
    margin_pts = 5

    subfigures_dict = {
        "1_1": {
            "position": [0, first_row_y, first_row_width, first_row_height],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },  # [left, bottom, width, height]
        "1_2": {
            "position": [
                (first_row_width + width_margin),
                first_row_y,
                first_row_width,
                first_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "1_3": {
            "position": [
                2 * (first_row_width + width_margin),
                first_row_y,
                first_row_width,
                first_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "1_4": {
            "position": [
                3 * (first_row_width + width_margin),
                first_row_y,
                first_row_width,
                first_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "1_5": {
            "position": [
                4 * (first_row_width + width_margin),
                first_row_y,
                first_row_width,
                first_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "1_6": {
            "position": [
                5 * (first_row_width + width_margin),
                first_row_y,
                first_row_width,
                first_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "1_7": {
            "position": [
                6 * (first_row_width + width_margin),
                first_row_y,
                0.12,
                first_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "2_1": {
            "position": [0, second_row_y, second_row_width, second_row_height],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "2_2": {
            "position": [
                (second_row_width + width_margin),
                second_row_y,
                second_row_width,
                second_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "2_3": {
            "position": [
                2 * (second_row_width + width_margin),
                second_row_y,
                second_row_width,
                second_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "2_4": {
            "position": [
                3 * (second_row_width + width_margin),
                second_row_y,
                second_row_width,
                second_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "2_5": {
            "position": [
                4 * (second_row_width + width_margin),
                second_row_y,
                second_row_width,
                second_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "2_6": {
            "position": [
                5 * (second_row_width + width_margin),
                second_row_y,
                second_row_width,
                second_row_height,
            ],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
        "3_1": {
            "position": [0, third_row_y, third_row_width, third_row_height],
            "skip_margin": True,
            "margin_pts": margin_pts,
        },
    }
    for key, value in subfigures_dict.items():
        subfigures_dict[key]["position"] = convert_top_left_origin_to_matplotlib(
            value["position"],
            fig_height=y_start + height_margin * 2 + np.sum(row_heights),
        )
    fig, axes_dict = layout_subfigures_inches((8, 6), subfigures_dict)
    return fig, axes_dict


def plot_rsm_figure(
    plotter,
    fig,
    axes,
    files,
    sample_names,
    cbar_ax,
    peak_z_range=None,
    draw_peak=True,
    draw_peak_line=True,
    draw_ideal_Qz_line=False,
    ideal_Qz=4.79,
    i_start=0,
    text_locs=None,
    label=True,
):
    # n_plot_fisrt_row = 7
    # n_plot_second_row = 6
    Qx_lines, Qz_lines, intensity_lines = [], [], []
    for i, (ax, file, title) in enumerate(zip(axes, files, sample_names)):
        # Draw RSMs
        if i == 0:
            Qx, Qz, intensity = plotter.plot(
                file, ax, figsize=None, cbar_ax=cbar_ax, ignore_yaxis=False
            )
        else:
            Qx, Qz, intensity = plotter.plot(
                file, ax, figsize=None, cbar_ax=cbar_ax, ignore_yaxis=True
            )

        if label:
            labelfigs(ax, i_start + i, size=15, inset_fraction=(0.08, 0.15), loc="tr")

        # Mark peaks with red '+' and draw a horizontal line
        # Mark peaks with red '+'
        coordinates = peak_local_max(
            intensity, min_distance=20, threshold_abs=80, num_peaks=10
        )
        coordinates_target = []
        # filter to target range of Qz
        for j, z in enumerate(Qz[coordinates[:, 0], coordinates[:, 1]]):
            if peak_z_range != None:
                if z < peak_z_range[1] and z > peak_z_range[0]:
                    coordinates_target.append(coordinates[j])
            else:  # if no range is provided, take the first peak
                coordinates_target.append(coordinates[j])
        coordinates_target = np.array(coordinates_target)[:1]
        Qx_target, Qz_target = (
            Qx[coordinates_target[:, 0], coordinates_target[:, 1]],
            Qz[coordinates_target[:, 0], coordinates_target[:, 1]],
        )

        if draw_peak:
            ax.scatter(
                Qx_target, Qz_target, marker="+", color="red", s=14, linewidth=0.8
            )  # Mark peaks with red '+'

        # extract line profile at Qz_target
        mask = np.isclose(Qz, Qz_target, atol=1e-3)  # Boolean mask
        Qx_line = Qx[mask]
        intensity_line = intensity[mask]
        # sort the Qx_line and intensity_line based on Qx values
        Qx_index = np.argsort(Qx_line)
        Qx_line = Qx_line[Qx_index]
        intensity_line = intensity_line[Qx_index]

        #  draw the horizontal line on the RSM plot
        Qz_line = np.ones_like(Qx_line) * Qz_target[0]
        if draw_peak_line:
            ax.plot(Qx_line, Qz_line, "r--", lw=1)

        Qx_lines.append(Qx_line)
        Qz_lines.append(Qz_line)
        intensity_lines.append(intensity_line)

        if draw_ideal_Qz_line:
            ideal_line = np.ones_like(Qx_line) * ideal_Qz
            ax.plot(Qx_line, ideal_line, "b--", lw=1)

        print(
            f"{title}: Qx_target={Qx_target[0]:.4f}nm, Qz_target={Qz_target[0]:.4f}nm"
        )

    if text_locs:
        for i, (ax, sample_name, color, text_loc) in enumerate(
            zip(axes, sample_names, colors, text_locs)
        ):
            marker_line_label = Line2D(
                [0], [0], color=color, linestyle="-", marker="o", markersize=6
            )
            ax.legend(
                handles=[marker_line_label],
                loc="lower right",
                frameon=False,
                bbox_to_anchor=(1.15, 0.04),
                fontsize=8,
            )
            add_text_to_figure(
                fig, sample_name, text_loc, fontsize=8, color="white", ha="center"
            )

    return Qx_lines, intensity_lines


def plot_fwhm_line_profile_figure(
    plotter, axes, sample_index, Qx_lines, intensity_lines
):
    # n_plot_fisrt_row = 7
    # n_plot_second_row = 6
    FWHM_list = []
    for i, (ax, title, Qx_line, intensity_line) in enumerate(
        zip(axes, sample_index, Qx_lines, intensity_lines)
    ):
        # plot the line profile and mark the peak, FWHM
        # ax = axes[i+n_plot_fisrt_row] # change to second row
        ax.scatter(Qx_line, intensity_line, s=1)

        # Calculate FWHM for the STO peak (peak_x[0]) and SRO peak (peak_x[1])
        peak_x, peak_y = detect_peaks(
            Qx_line, intensity_line, num_peaks=1, prominence=0.1, distance=None
        )
        fwhm, y_fwhm, x_left, x_right = calculate_fwhm(
            Qx_line, intensity_line, peak_x[0]
        )
        FWHM_list.append(fwhm)

        # Draw FWHM arrows for both peaks
        ax.plot(
            [x_left, x_right], [y_fwhm, y_fwhm], "r-", lw=0.5
        )  # 'k-' is for black line
        # No shrink at the starting point and ending point and Controls arrowhead size
        ax.annotate(
            "",
            xy=(x_right, y_fwhm),
            xytext=(x_left, y_fwhm),
            arrowprops=dict(
                arrowstyle="<->",
                lw=0.8,
                color="r",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=5,
            ),
        )
        ax.text(
            (x_left + x_right) / 2,
            y_fwhm * 1.05,
            f"FWHM: {fwhm:.4f}",
            ha="center",
            va="bottom",
            color="r",
            fontsize=8,
        )
        if plotter.plot_params.get("lineplot_yscale", "linear") == "log":
            ax.set_yscale("log")
        ax.set_xlim(plotter.plot_params.get("lineplot_xlim", None))
        ax.set_ylim(plotter.plot_params.get("lineplot_ylim", None))

        ax.tick_params(
            axis="x",
            direction="in",
            top=True,
            labelsize=plotter.plot_params.get("fontsize", 12),
        )
        ax.tick_params(
            axis="y",
            direction="in",
            right=True,
            labelsize=plotter.plot_params.get("fontsize", 12),
        )
        ax.set_xlabel(
            r"$Q_x$ [$\AA^{-1}$]",
            fontsize=plotter.plot_params.get("fontsize", 12),
            fontweight="bold",
        )
        ax.set_ylabel(
            r"$Q_z$ [$\AA^{-1}$]",
            fontsize=plotter.plot_params.get("fontsize", 12),
            fontweight="bold",
        )
        labelfigs(ax, i, size=15, inset_fraction=(0.15, 0.15), loc="tr", style="bw")

        # adjust the yticks and ylabel for the line profile plots
    axes[-1].set_ylabel(
        r"$Q_z$ [$\AA^{-1}$]", fontsize=plotter.plot_params.get("fontsize", 12)
    )
    for ax in axes[1:]:  # start from second ax in the second row
        ax.set_yticklabels([])
        ax.set_ylabel("")

    return FWHM_list


def plot_fwhm_trend_figure(plotter, ax, sample_index, FWHM_list):
    # Plot the trend of FWHM on the left y-axis with a small offset for the 5th data point
    left_x = list(range(len(sample_index[:5])))
    left_x[4] -= 0.05  # Shift the 5th data point slightly to the left

    ax.plot(
        left_x, FWHM_list[:5], marker="o", color=colors[0]
    )  # Set line color to blue
    ax.set_xticks(range(len(sample_index)))  # Set x-ticks at integer positions
    ax.set_xticklabels(
        sample_index
    )  # Replace x-tick integers with `sample_index` labels
    ax.set_ylim(2.2e-3, 4.3e-3)

    ax.set_xlabel(
        "Sample Names",
        fontsize=plotter.plot_params.get("fontsize", 12),
        fontweight="bold",
    )
    ax.set_ylabel(
        "FWHM (set 1)",
        fontsize=plotter.plot_params.get("fontsize", 12),
        fontweight="bold",
        color=colors[0],
    )
    ax.tick_params(
        axis="x",
        direction="in",
        top=True,
        labelsize=plotter.plot_params.get("fontsize", 12),
    )
    ax.tick_params(
        axis="y",
        direction="in",
        right=True,
        labelsize=plotter.plot_params.get("fontsize", 12),
        color=colors[0],
        labelcolor=colors[0],
    )

    # Label the figure as needed
    labelfigs(ax, 10, size=15, inset_fraction=(0.2, 0.05), loc="tr", style="bw")

    # Set the left y-axis to scientific notation
    formatter = ScalarFormatter(useMathText=True)  # Use MathText for cleaner output
    formatter.set_scientific(True)  # Enable scientific notation
    formatter.set_powerlimits((1, 10))  # Set when to switch to scientific notation
    ax.yaxis.set_major_formatter(formatter)  # Apply formatter to y-axis
    ax.yaxis.get_offset_text().set_x(
        -0.2
    )  # Set horizontal position (relative to the axis)

    # Create a secondary y-axis on the right in red
    right_x = list(range(4, 4 + len(sample_index[-2:])))
    right_x[0] += 0.05  # Shift the 5th data point slightly to the right
    ax_right = ax.twinx()  # Create a twin y-axis
    ax_right.plot(
        right_x, FWHM_list[-2:], marker="o", color=colors[1]
    )  # Set line color to red

    ax_right.set_ylim(2.2e-3, 4.3e-3)
    ax_right.set_ylabel(
        "FWHM (set 2)",
        fontsize=plotter.plot_params.get("fontsize", 12),
        fontweight="bold",
        color=colors[1],
    )  # Customize label for the secondary y-axis
    ax_right.tick_params(
        axis="y",
        direction="in",
        labelsize=plotter.plot_params.get("fontsize", 12),
        color=colors[1],
        labelcolor=colors[1],
    )

    # Optional: Format the secondary y-axis if needed, e.g., scientific notation
    ax_right_formatter = ScalarFormatter(useMathText=True)
    ax_right_formatter.set_scientific(True)
    ax_right_formatter.set_powerlimits((1, 10))
    ax_right.yaxis.set_major_formatter(ax_right_formatter)
    ax_right.yaxis.get_offset_text().set_x(1)  # Adjust offset for clarity


def show_sample_frames(plumes, n_plumes=40, n_frames=25, figsize=(8, 12)):
    plume_index_list = np.round(np.linspace(0, len(plumes) - 1, n_plumes)).astype(int)
    sample_frames = plumes[plume_index_list, :n_frames]
    sample_frames = np.array(sample_frames).reshape(
        n_plumes * n_frames, plumes.shape[2], plumes.shape[3]
    )
    labels = np.arange(n_frames) * 500e-3
    labels = [f"{l:.0f}µs" for l in labels]
    labels = labels * n_plumes
    fig, axes = layout_fig(
        n_plumes * n_frames,
        mod=n_frames,
        figsize=figsize,
        subplot_style="gridspec",
        spacing=(0.02, 0.02),
        layout="tight",
    )
    for i, (ax, img, label) in enumerate(zip(axes, sample_frames, labels)):
        ax.imshow(img)
        ax.axis("off")
        if i < n_frames and i % 2 == 0:
            ax.set_title(label, fontsize=8)

    return fig, axes
