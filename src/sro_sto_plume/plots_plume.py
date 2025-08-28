# sro_sto_plume/plots_plume.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke

from m3util.viz.layout import layout_fig, layout_subfigures_inches
from m3util.viz.text import labelfigs, add_text_to_figure
from plume_learn.plume_utils.viz import label_violinplot, set_labels, set_cbar, show_images
from plume_learn.plume_analyzer.PlumeDataset import plume_dataset
from plume_learn.plume_analyzer.Velocity import VelocityCalculator
from sro_sto_plume.cmap import define_white_viridis

# ---------- Heatmaps ----------
def plot_temporal_heatmaps(df_sample, sample_names, label=True):
    figsize = (10, 6)
    subfigures_dict = {
        '1_1': {"position": [0, 3, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '1_2': {"position": [2, 3, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '1_3': {"position": [4, 3, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '1_4': {"position": [6, 3, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '1_5': {"position": [8, 3, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_1': {"position": [0, 0, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_2': {"position": [2, 0, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_3': {"position": [4, 0, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_4': {"position": [6, 0, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_5': {"position": [8, 0, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    axes_1 = [axes_dict[f'1_{i}'] for i in range(1, 5+1)]
    axes_2 = [axes_dict[f'2_{i}'] for i in range(1, 5+1)]

    for i, ax, sample in zip(range(0,5), axes_1, sample_names):
        df_pivot = df_sample[df_sample['Sample Name']==sample].pivot(
            index="Plume Index", columns="Time (µs)", values='Area (a.u.)'
        )
        df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
        sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax, vmin=0, vmax=17152)
        set_labels(ax, xlabel="Time (µs)", ylabel=("Plume Index" if i==0 else ""),
                   label_fontsize=10, ticklabel_fontsize=8, yaxis_style='float', show_ticks=False, tick_padding=2)
        if label:
            labelfigs(ax, number=i, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))
    set_cbar(fig, axes_1[-1], cbar_label='Area\n(a.u.)', scientific_notation=True, tick_in=True,
             ticklabel_fontsize=8, labelpad=0, fontsize=8)

    for i, ax, sample in zip(range(5,10), axes_2, sample_names):
        df_pivot = df_sample[df_sample['Sample Name']==sample].pivot(
            index="Plume Index", columns="Time (µs)", values='Velocity (m/s)'
        )
        df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
        df_pivot[df_pivot==0] = 200
        sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax, norm=LogNorm(vmin=200, vmax=29257))
        set_labels(ax, xlabel="Time (µs)", ylabel=("Plume Index" if i==5 else ""),
                   label_fontsize=10, ticklabel_fontsize=8, yaxis_style='float', show_ticks=False, tick_padding=2)
        if label:
            labelfigs(ax, number=i, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))
    set_cbar(fig, axes_2[-1], cbar_label='Velocity\n(m/s)', scientific_notation=True, tick_in=True,
             logscale=True, ticklabel_fontsize=8, labelpad=0, fontsize=8)
    return fig, axes_dict

# ---------- Violins ----------
def plot_temporal_violins(df_plume_metrics, label=True):
    fig, axes = layout_fig(2, 1, figsize=(8, 6), subplot_style='gridspec', spacing=(0, 0.3), layout='tight')

    sns.violinplot(x='Sample Name', y='Max Area (a.u.)', data=df_plume_metrics, width=0.9,
                   ax=axes[0], palette='deep', hue='Sample Name', legend=False)
    mean_max_area = df_plume_metrics.groupby('Sample Name')['Max Area (a.u.)'].mean()
    label_violinplot(axes[0], mean_max_area, label_type='average_value', text_pos='center',
                     value_format='scientific', text_size=10,
                     offset_parms={'x_type':'fixed','x_value':0,'y_type':'ratio','y_value':-0.05})
    if label:
        labelfigs(axes=axes[0], number=0, size=15, style='bw', loc='tr', inset_fraction=(0.15, 0.05))

    sns.violinplot(x='Sample Name', y='Incident Velocity (m/s)', data=df_plume_metrics, width=0.9,
                   ax=axes[1], palette='deep', hue='Sample Name', legend=False)
    mean_incident_velocity = df_plume_metrics.groupby('Sample Name')['Incident Velocity (m/s)'].mean()
    label_violinplot(axes[1], mean_incident_velocity, label_type='average_value', text_pos='center',
                     value_format='scientific', text_size=10,
                     offset_parms={'x_type':'fixed','x_value':0,'y_type':'ratio','y_value':-0.05})
    if label:
        labelfigs(axes=axes[1], number=1, size=15, style='bw', loc='br', inset_fraction=(0.12, 0.05))
    return fig, axes

# ---------- Combined temporal variation ----------
def plot_combined_temporal_variation(df_sample, sample_names, df_plume_metrics, label=True):
    figsize = (8, 6)
    subfigures_dict = {
        '1_1': {"position": [0, 3, 1.5, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '1_2': {"position": [1.8, 3, 1.5, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '1_3': {"position": [3.6, 3, 1.5, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '1_4': {"position": [5.4, 3, 1.5, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '1_5': {"position": [7.2, 3, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_1': {"position": [0, 0, 1.5, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_2': {"position": [1.8, 0, 1.5, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_3': {"position": [3.6, 0, 1.5, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_4': {"position": [5.4, 0, 1.5, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '2_5': {"position": [7.2, 0, 1.7, 2.6], 'skip_margin': True, 'margin_pts': 5},
        '3':   {"position": [0.1, 6.2, 8.5, 6], 'skip_margin': True, 'margin_pts': 5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    axes_1 = [axes_dict[f'1_{i}'] for i in range(1, 6)]
    axes_2 = [axes_dict[f'2_{i}'] for i in range(1, 6)]

    for i, ax, sample in zip(range(2,7), axes_1, sample_names):
        df_pivot = df_sample[df_sample['Sample Name']==sample].pivot(index="Plume Index", columns="Time (µs)", values='Area (a.u.)')
        df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
        sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax, vmin=0, vmax=17152)
        set_labels(ax, xlabel="Time (µs)", ylabel=("Plume Index" if i==0 else ""),
                   label_fontsize=10, ticklabel_fontsize=8, yaxis_style='float', show_ticks=False, tick_padding=2)
        if label:
            labelfigs(ax, number=i, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))
    set_cbar(fig, axes_1[-1], cbar_label='Area\n(a.u.)', scientific_notation=True, tick_in=True,
             ticklabel_fontsize=8, labelpad=0, fontsize=8)

    for i, ax, sample in zip(range(7,12), axes_2, sample_names):
        df_pivot = df_sample[df_sample['Sample Name']==sample].pivot(index="Plume Index", columns="Time (µs)", values='Velocity (m/s)')
        df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
        df_pivot[df_pivot==0] = 200
        sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax, norm=LogNorm(vmin=200, vmax=29257))
        set_labels(ax, xlabel="Time (µs)", ylabel=("Plume Index" if i==5 else ""),
                   label_fontsize=10, ticklabel_fontsize=8, yaxis_style='float', show_ticks=False, tick_padding=2)
        if label:
            labelfigs(ax, number=i, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))
    set_cbar(fig, axes_2[-1], cbar_label='Velocity\n(m/s)', scientific_notation=True, tick_in=True,
             logscale=True, ticklabel_fontsize=8, labelpad=0, fontsize=8)

    axes_dict['3'].axis('off')
    _, axes = layout_fig(2, 1, figsize=(8, 6), subplot_style='subplots', spacing=(0, 0.2),
                         parent_ax=axes_dict['3'], layout='tight')

    sns.violinplot(x='Sample Name', y='Max Area (a.u.)', data=df_plume_metrics, width=0.9,
                   ax=axes[0], palette='deep', hue='Sample Name', legend=False)
    mean_max_area = df_plume_metrics.groupby('Sample Name')['Max Area (a.u.)'].mean()
    label_violinplot(axes[0], mean_max_area, label_type='average_value', text_pos='center',
                     value_format='scientific', text_size=10,
                     offset_parms={'x_type':'fixed','x_value':0,'y_type':'ratio','y_value':-0.05})
    if label:
        labelfigs(axes=axes[0], number=0, size=15, style='bw', loc='tr', inset_fraction=(0.15, 0.05))

    sns.violinplot(x='Sample Name', y='Incident Velocity (m/s)', data=df_plume_metrics, width=0.9,
                   ax=axes[1], palette='deep', hue='Sample Name', legend=False)
    mean_incident_velocity = df_plume_metrics.groupby('Sample Name')['Incident Velocity (m/s)'].mean()
    label_violinplot(axes[1], mean_incident_velocity, label_type='average_value', text_pos='center',
                     value_format='scientific', text_size=10,
                     offset_parms={'x_type':'fixed','x_value':0,'y_type':'ratio','y_value':-0.05})
    if label:
        labelfigs(axes=axes[1], number=1, size=15, style='bw', loc='tr', inset_fraction=(0.12, 0.05))
    return fig, axes_dict

# ---------- Plume inhomogeneity (area + velocity) ----------
def plot_combined_plume_inhomogeneity(df_plume_metrics, df_sample, sample_names, custom_palette, label=True):
    figsize = (8, 6)
    subfigures_dict = {
        '1_1': {"position": [0, 2.8, 3.8, 1.8], 'skip_margin': True, 'margin_pts': 5},
        '1_2': {"position": [4.3, 2.8, 3.8, 1.8], 'skip_margin': True, 'margin_pts': 5},
        '2_1': {"position": [0.05, 0, 1.65, 2.5], 'skip_margin': True, 'margin_pts': 5},
        '2_2': {"position": [2.0, 0, 1.95, 2.5], 'skip_margin': True, 'margin_pts': 5},
        '2_3': {"position": [4.35, 0, 1.65, 2.5], 'skip_margin': True, 'margin_pts': 5},
        '2_4': {"position": [6.3, 0, 1.95, 2.5], 'skip_margin': True, 'margin_pts': 5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    ax_area_violin, ax_velocity_violin = axes_dict['1_1'], axes_dict['1_2']
    ax_area_heatmap1, ax_area_heatmap2 = axes_dict['2_1'], axes_dict['2_2']
    ax_velocity_heatmap1, ax_velocity_heatmap2 = axes_dict['2_3'], axes_dict['2_4']

    sns.violinplot(x='Sample Name', y='Max Area (a.u.)', data=df_plume_metrics, width=0.5,
                   ax=ax_area_violin, palette=custom_palette, hue='Sample Name', legend=False, order=sample_names)
    mean_area = df_plume_metrics.groupby('Sample Name')['Max Area (a.u.)'].mean().reindex(sample_names)
    label_violinplot(ax_area_violin, mean_area, label_type='average_value', text_pos='center', value_format='scientific',
                     text_size=10, offset_parms={'x_type':'fixed','x_value':0,'y_type':'fixed','y_value':-1000})
    set_labels(ax_area_violin, xlabel='', ylabel='Max Area (a.u.)', label_fontsize=10, ticklabel_fontsize=8,
               yaxis_style='sci', show_ticks=True, tick_padding=2)
    if label:
        labelfigs(ax_area_violin, number=0, size=15, style='bw', loc='tr', inset_fraction=(0.12, 0.08))

    sns.violinplot(x='Sample Name', y='Incident Velocity (m/s)', data=df_plume_metrics, width=0.5,
                   ax=ax_velocity_violin, palette=custom_palette, hue='Sample Name', legend=False, order=sample_names)
    mean_velocity = df_plume_metrics.groupby('Sample Name')['Incident Velocity (m/s)'].mean().reindex(sample_names)
    label_violinplot(ax_velocity_violin, mean_velocity, label_type='average_value', text_pos='center', value_format='scientific',
                     text_size=10, offset_parms={'x_type':'fixed','x_value':0,'y_type':'fixed','y_value':-800})
    set_labels(ax_velocity_violin, xlabel='', ylabel='Incident Velocity (m/s)', label_fontsize=10, ticklabel_fontsize=8,
               yaxis_style='sci', show_ticks=True, tick_padding=2)
    if label:
        labelfigs(ax_velocity_violin, number=1, size=15, style='bw', loc='tr', inset_fraction=(0.12, 0.08))

    for i, ax, sample in zip([2,3], [ax_area_heatmap1, ax_area_heatmap2], sample_names):
        df_pivot = df_sample[df_sample['Sample Name']==sample].pivot(index="Plume Index", columns="Time (µs)", values='Area (a.u.)')
        df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
        sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax, vmin=0, vmax=17152)
        set_labels(ax, xlabel="Time (µs)", ylabel=("Plume Index" if i==2 else ""),
                   label_fontsize=10, ticklabel_fontsize=8, yaxis_style='float', show_ticks=False, tick_padding=2)
        if label:
            labelfigs(ax, number=i, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))
    set_cbar(fig, ax_area_heatmap2, cbar_label='Area\n(a.u.)', scientific_notation=True, tick_in=True,
             ticklabel_fontsize=8, labelpad=0, fontsize=8)

    for i, ax, sample in zip([4,5], [ax_velocity_heatmap1, ax_velocity_heatmap2], sample_names):
        df_pivot = df_sample[df_sample['Sample Name']==sample].pivot(index="Plume Index", columns="Time (µs)", values='Velocity (m/s)')
        df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
        df_pivot[df_pivot==0] = 200
        sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax, norm=LogNorm(vmin=200, vmax=29257))
        set_labels(ax, xlabel="Time (µs)", ylabel="", label_fontsize=10, ticklabel_fontsize=8,
                   yaxis_style='float', show_ticks=False, tick_padding=2)
        if label:
            labelfigs(ax, number=i, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))
    set_cbar(fig, ax_velocity_heatmap2, cbar_label='Velocity\n(m/s)', scientific_notation=True, tick_in=True,
             logscale=True, ticklabel_fontsize=8, labelpad=0, fontsize=8)
    return fig, axes_dict

def plot_plume_inhomogeneity_area(df_plume_metrics, df_sample, sample_names, custom_palette, label=True):
    figsize = (8, 4)
    subfigures_dict = {
        '1':   {"position": [0, 2.5, 6, 1.5], 'skip_margin': False, 'margin_pts':5},
        '2_1': {"position": [0, 0, 2.9, 2.4], 'skip_margin': False, 'margin_pts':5},
        '2_2': {"position": [3.1, 0, 3.5, 2.4], 'skip_margin': False, 'margin_pts':5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    ax0, ax1, ax2 = axes_dict['1'], axes_dict['2_1'], axes_dict['2_2']

    sns.violinplot(x='Sample Name', y='Max Area (a.u.)', data=df_plume_metrics, width=0.5,
                   ax=ax0, palette=custom_palette, hue='Sample Name', legend=False, order=sample_names)
    mean_max_area = df_plume_metrics.groupby('Sample Name')['Max Area (a.u.)'].mean().reindex(sample_names)
    label_violinplot(ax0, mean_max_area, label_type='average_value', text_pos='center', value_format='scientific',
                     text_size=10, offset_parms={'x_type':'fixed','x_value':0,'y_type':'fixed','y_value':-1000})
    if label:
        set_labels(ax0, xlabel='', ylabel='Max Area (a.u.)', label_fontsize=11, yaxis_style='sci', show_ticks=True)
    labelfigs(axes=ax0, number=0, size=15, style='bw', loc='tr', inset_fraction=(0.15, 0.05))
    ax0.tick_params(axis="x", direction="in", length=5, labelsize=12)

    df_pivot = df_sample[df_sample['Sample Name']=='t5/s1'].pivot(index="Plume Index", columns="Time (µs)", values='Area (a.u.)')
    df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
    sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax1, vmin=0, vmax=17152)
    set_labels(ax1, xlabel="Time (µs)", ylabel="Plume Index", label_fontsize=11, yaxis_style='float', show_ticks=False)
    if label:
        labelfigs(axes=ax1, number=1, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))

    df_pivot = df_sample[df_sample['Sample Name']=='s2'].pivot(index="Plume Index", columns="Time (µs)", values='Area (a.u.)')
    df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
    sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax2, vmin=0, vmax=17152)
    set_labels(ax2, xlabel="Time (µs)", ylabel="", label_fontsize=11, yaxis_style='float', show_ticks=False)
    set_cbar(fig, ax2, cbar_label='Area (a.u.)', scientific_notation=True, tick_in=True)
    if label:
        labelfigs(axes=ax2, number=2, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))
    return fig, axes_dict

def plot_plume_inhomogeneity_velocity(df_plume_metrics, df_sample, sample_names, custom_palette, label=True):
    figsize = (8, 6)
    subfigures_dict = {
        '1':   {"position": [0, 4, 6, 2], 'skip_margin': False, 'margin_pts':5},
        '2_1': {"position": [0, 0, 2.9, 3.9], 'skip_margin': False, 'margin_pts':5},
        '2_2': {"position": [3.1, 0, 3.5, 3.9], 'skip_margin': False, 'margin_pts':5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    ax0, ax1, ax2 = axes_dict['1'], axes_dict['2_1'], axes_dict['2_2']

    sns.violinplot(x='Sample Name', y='Incident Velocity (m/s)', data=df_plume_metrics, width=0.5,
                   ax=ax0, palette=custom_palette, hue='Sample Name', legend=False, order=sample_names)
    mean_v = df_plume_metrics.groupby('Sample Name')['Incident Velocity (m/s)'].mean().reindex(sample_names)
    label_violinplot(ax0, mean_v, label_type='average_value', text_pos='center', value_format='scientific',
                     text_size=10, offset_parms={'x_type':'fixed','x_value':0,'y_type':'fixed','y_value':-800})
    set_labels(ax0, xlabel='', ylabel='Area (a.u.)', label_fontsize=11, yaxis_style='sci', show_ticks=True)
    if label:
        labelfigs(axes=ax0, number=0, size=15, style='bw', loc='tr', inset_fraction=(0.15, 0.05))
    ax0.tick_params(axis="x", direction="in", length=5, labelsize=12)

    df_pivot = df_sample[df_sample['Sample Name']=='t5/s1'].pivot(index="Plume Index", columns="Time (µs)", values='Velocity (m/s)')
    df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
    df_pivot[df_pivot==0] = 200
    sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax1, norm=LogNorm(vmin=200, vmax=29257))
    set_labels(ax1, xlabel="Time (µs)", ylabel="Plume Index", label_fontsize=11, yaxis_style='float', show_ticks=False)
    if label:
        labelfigs(axes=ax1, number=1, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))

    df_pivot = df_sample[df_sample['Sample Name']=='s2'].pivot(index="Plume Index", columns="Time (µs)", values='Velocity (m/s)')
    df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]
    df_pivot[df_pivot==0] = 200
    sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax2, norm=LogNorm(vmin=200, vmax=29257))
    set_labels(ax2, xlabel="Time (µs)", ylabel="", label_fontsize=11, yaxis_style='float', show_ticks=False)
    set_cbar(fig, ax2, cbar_label='Velocity (m/s)', scientific_notation=True, tick_in=True, logscale=True)
    if label:
        labelfigs(axes=ax2, number=2, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))
    return fig, axes_dict

# ---------- Plume summary page ----------
def plume_metrics_summary(df_frame_metrics, plume_recording_root, label=True):
    figsize = (6.8, 8)
    subfigures_dict = {
        '1': {"position": [0, 4, 6.55, 1.82], 'skip_margin': False, 'margin_pts':5},
        '1_1': {"position": [0.03, 4.05, 6.5, 0.6], 'skip_margin': False, 'margin_pts':5},
        '2': {"position": [0, 2, 6.55, 1.82], 'skip_margin': False, 'margin_pts':5},
        '2_1': {"position": [0.03, 2.05, 6.5, 0.6], 'skip_margin': False, 'margin_pts':5},
        '3': {"position": [0, 0, 6.55, 1.82], 'skip_margin': False, 'margin_pts':5},
    }
    fig_all, axes_dict
