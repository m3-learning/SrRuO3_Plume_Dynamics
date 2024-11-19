# import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from skimage.feature import peak_local_max
import seaborn as sns

from sro_sto_plume.coordinate_converter import convert_top_left_origin_to_matplotlib
from m3util.viz.layout import layout_fig, layout_subfigures_inches
from m3util.viz.text import labelfigs, add_text_to_figure
from m3util.viz.lines import draw_lines
from xrd_learn.xrd_viz import plot_xrd
from xrd_learn.xrd_utils import detect_peaks, calculate_fwhm, load_xrd_scans, align_peak_to_value, align_fwhm_center_to_value, align_peak_y_to_value, upsample_XY
from plume_learn.plume_utils.viz import label_violinplot, set_labels, set_cbar
colors = colormaps.get_cmap('tab10').colors[:6]


# plot the spatial inhomogeneity of the plume
def plot_plume_inhomogeneity_area(df_plume_metrics, df_sample, sample_names, custom_palette):
    figsize = (8, 6)
    subfigures_dict = {
        '1': {"position": [0, 4, 6, 2], 'skip_margin': False, 'margin_pts':5}, # [left, bottom, width, height]
        '2_1': {"position": [0, 0, 2.9, 3.9], 'skip_margin': False, 'margin_pts':5},
        '2_2': {"position": [3.1, 0, 3.5, 3.9], 'skip_margin': False, 'margin_pts':5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    ax0, ax1, ax2 = axes_dict['1'], axes_dict['2_1'], axes_dict['2_2']

    # Plot violinplot - '1'
    sns.violinplot(x='Sample Name', y='Max Area (a.u.)', data=df_plume_metrics, width=0.5, ax=ax0, palette=custom_palette, hue='Sample Name', legend=False, order=sample_names)
    mean_max_area = df_plume_metrics.groupby('Sample Name')['Max Area (a.u.)'].mean()
    mean_max_area = mean_max_area.reindex(sample_names)

    label_violinplot(ax0, mean_max_area, label_type='average_value', text_pos='center', value_format='scientific', text_size=10,
                    offset_parms={'x_type': 'fixed', 'x_value': 0, 'y_type': 'fixed', 'y_value': -1000})
    set_labels(ax0, xlabel='', ylabel='Area (a.u.)', label_fontsize=11, yaxis_style='sci', show_ticks=True)
    # ax0.xaxis.set_ticks([])
    labelfigs(axes=ax0, number=0, size=15, style='bw', loc='tr', inset_fraction=(0.15, 0.05))
    ax0.tick_params(axis="x", direction="in", length=5, labelsize=12)


    # Plot heatmap - '2_1'
    df_pivot = df_sample[df_sample['Sample Name']=='t5/s1'].pivot(index="Plume Index", columns="Time (µs)", values='Area (a.u.)')
    df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]  # remove the columns where all values are 0
    # plot and extract vmin=0, vmax=17152
    sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax1, vmin=0, vmax=17152)  # Disable seaborn's default colorbar 
    set_labels(ax1, xlabel="Time (µs)", ylabel="Plume Index", label_fontsize=11, yaxis_style='float', show_ticks=False)
    # set_cbar(fig, ax1, cbar_label='Intensity (a.u.)', scientific_notation=True)
    labelfigs(axes=ax1, number=1, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))

    # Plot heatmap - '2_2'
    df_pivot = df_sample[df_sample['Sample Name']=='s2'].pivot(index="Plume Index", columns="Time (µs)", values='Area (a.u.)')
    df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]  # remove the columns where all values are 0
    sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax2, vmin=0, vmax=17152)  # Disable seaborn's default colorbar
    set_labels(ax2, xlabel="Time (µs)", ylabel="", label_fontsize=11, yaxis_style='float', show_ticks=False)
    set_cbar(fig, ax2, cbar_label='Area (a.u.)', scientific_notation=True, tick_in=True)
    labelfigs(axes=ax2, number=2, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))

    
def plot_plume_inhomogeneity_velocity(df_plume_metrics, df_sample, sample_names, custom_palette):
    figsize = (8, 6)
    subfigures_dict = {
        '1': {"position": [0, 4, 6, 2], 'skip_margin': False, 'margin_pts':5}, # [left, bottom, width, height]
        '2_1': {"position": [0, 0, 2.9, 3.9], 'skip_margin': False, 'margin_pts':5},
        '2_2': {"position": [3.1, 0, 3.5, 3.9], 'skip_margin': False, 'margin_pts':5},
    }
    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)
    ax0, ax1, ax2 = axes_dict['1'], axes_dict['2_1'], axes_dict['2_2']

    # Plot violinplot - '1'
    sns.violinplot(x='Sample Name', y='Incident Velocity (m/s)', data=df_plume_metrics, width=0.5, ax=ax0, palette=custom_palette, hue='Sample Name', legend=False, order=sample_names)
    mean_max_area = df_plume_metrics.groupby('Sample Name')['Incident Velocity (m/s)'].mean()
    mean_max_area = mean_max_area.reindex(sample_names)

    label_violinplot(ax0, mean_max_area, label_type='average_value', text_pos='center', value_format='scientific', text_size=10, offset_parms={'x_type': 'fixed', 'x_value': 0, 'y_type': 'fixed', 'y_value': -800})
    set_labels(ax0, xlabel='', ylabel='Area (a.u.)', label_fontsize=11, yaxis_style='sci', show_ticks=True)
    labelfigs(axes=ax0, number=0, size=15, style='bw', loc='tr', inset_fraction=(0.15, 0.05))
    ax0.tick_params(axis="x", direction="in", length=5, labelsize=12)


    # Plot heatmap - '2_1'
    df_pivot = df_sample[df_sample['Sample Name']=='t5/s1'].pivot(index="Plume Index", columns="Time (µs)", values='Velocity (m/s)')
    df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]  # remove the columns where all values are 0
    df_pivot[df_pivot==0] = 200

    # plot and extract vmin=0, vmax=17152
    heatmap = sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax1, norm=LogNorm(vmin=200, vmax=29257))
    set_labels(ax1, xlabel="Time (µs)", ylabel="Plume Index", label_fontsize=11, yaxis_style='float', show_ticks=False)
    # set_cbar(fig, ax1, cbar_label='Intensity (a.u.)', scientific_notation=True)
    labelfigs(axes=ax1, number=1, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))


    # Plot heatmap - '2_2'
    df_pivot = df_sample[df_sample['Sample Name']=='s2'].pivot(index="Plume Index", columns="Time (µs)", values='Velocity (m/s)')
    df_pivot = df_pivot.loc[:, (df_pivot != 0).any(axis=0)]  # remove the columns where all values are 0
    df_pivot[df_pivot==0] = 200

    heatmap = sns.heatmap(df_pivot, cmap='viridis', cbar=False, ax=ax2, norm=LogNorm(vmin=200, vmax=29257))  # Disable seaborn's default colorbar
    set_labels(ax2, xlabel="Time (µs)", ylabel="", label_fontsize=11, yaxis_style='float', show_ticks=False)
    set_cbar(fig, ax2, cbar_label='Velocity (m/s)', scientific_notation=True, tick_in=True, logscale=True)
    labelfigs(axes=ax2, number=2, size=15, style='wb', loc='tr', inset_fraction=(0.08, 0.08))


# xrd section
def plot_xrd_figure(files, sample_index, fig, ax, xrange, yrange, title, filename):
            
    STO_x_peak = 46.4721
    SRO_bulk_x_peak = 46.2425

    Xs, Ys, length_list = load_xrd_scans(files)
    for i, (X, Y, sample_name) in enumerate(zip(Xs, Ys, sample_index)):
        peak_x, peak_y = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)
    Xs_aligned, Ys_aligned = align_peak_to_value(Xs, Ys, STO_x_peak, viz=False)

    # fig, ax = layout_fig(1, 1, figsize=(8, 4), layout='tight')
    diff = 5e1
    plot_xrd((Xs_aligned, Ys_aligned, length_list), sample_index, title=title, xrange=xrange, yrange=yrange, diff=diff, fig=fig, ax=ax, legend_style='label', text_offset_ratio=(0.992, 3))

    line_style = {'color': 'gray', 'linestyle': 'dashed', 'linewidth': 1}
    draw_lines(ax, x_values=[STO_x_peak, STO_x_peak], y_values=[1e3, 3e15], style=line_style)
    ax.text(STO_x_peak, 4e15, 'STO\n(002)', fontsize=10, ha='center')

    ax.text(45.9121, 5e14, 'SRO\n(220)', fontsize=10, ha='center')
    line_style = {'color': 'gray', 'linestyle': 'dotted', 'linewidth': 0.8}
    # draw_lines(ax, x_values=[45.9121, 45.9121], y_values=[5e2, 5e14], style=line_style)

    ax.text(SRO_bulk_x_peak, 10, 'SRO\n(bulk)', fontsize=10, ha='center')
    line_style = {'color': 'gray', 'linestyle': 'dashdot', 'linewidth': 0.8}
    draw_lines(ax, x_values=[SRO_bulk_x_peak, SRO_bulk_x_peak], y_values=[2e3, 2e14], style=line_style)

    legend = []
    for i, (X, Y, sample_name, color) in enumerate(zip(Xs_aligned, Ys_aligned, sample_index, colors)):
        peak_x, peak_y = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)

        # Calculate FWHM for the STO peak (peak_x[0]) and SRO peak (peak_x[1])
        fwhm_sto, y_fwhm_sto, x_left_sto, x_right_sto = calculate_fwhm(X, Y, peak_x[0])
        fwhm_sro, y_fwhm_sro, x_left_sro, x_right_sro = calculate_fwhm(X, Y, peak_x[1])

        # Prepare legend item
        # legend_item = f'SRO(+): {peak_x[1]:.4f}°, STO(*): {peak_x[0]:.4}°'
        # legend_item = f'SRO(+): {peak_x[1]:.4f}°, FWHM: {fwhm_sro:.2f}'
        # legend_item = f'SRO(+): {peak_x[1]:.2f}°'
        # legend.append(legend_item)
        
        peak_y = np.array(peak_y)*diff**(len(Ys)-i-1)
        # plt.plot(peak_x[0], peak_y[0]*3, '*', color=color)
        ax.plot(peak_x[1], peak_y[1]*3, '+', color=color)
        
    # plt.legend(legend, fontsize=9, loc='upper right', frameon=False)
    if filename:
        plt.savefig(f'{filename}.png', dpi=600)
        plt.savefig(f'{filename}.svg', dpi=600)



def plot_rocking_curve_figure(sample_index, files, fig, ax):
    Xs, Ys, length_list = load_xrd_scans(files)

    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        peak_x, peak_y = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)
        # upsample the X, Y
        X, Y = upsample_XY(X, Y, num_points=5000)
        Xs[i], Ys[i] = X-22.95, Y
        
    Xs, Ys = align_peak_to_value(Xs, Ys, target_x_peak=0, viz=False)
    Xs, Ys, FWHM_list = align_fwhm_center_to_value(Xs, Ys, target_x_peak=0, viz=False)
    Xs, Ys = align_peak_y_to_value(Xs, Ys, target_y_peak=None, use_global_max=True, viz=False)
    # fwhm, y_fwhm, x_left, x_right = calculate_fwhm(Qx_line, intensity_line, peak_x[0])

    xrange = (-0.5, 0.55)
    yrange = (8, 5e4)
    plot_xrd((Xs, Ys, length_list), sample_index, xrange=xrange, yrange=yrange, 
            diff=1, fig=fig, ax=ax, xlabel=r'$\omega-\omega_c$ [°]', title=None, legend_style='legend', colors=colors)
    ax.set_yscale('log')
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)

    plt.legend(loc='center left', bbox_to_anchor=(0.385, 0.23), fontsize=9, frameon=False,
                labelspacing=0.3,      # Space between labels
                handlelength=1.2,        # Length of the line in the legend
                handletextpad=0.5,       # Space between line and text
                # borderaxespad=0.2      # Padding between legend and axes
                )

    ax2 = fig.add_axes([0.73, 0.65, 0.22, 0.29]) # [left, bottom, width, height]
    ax2.scatter(sample_index, FWHM_list, c=colors)  # Use any colormap, e.g., 'viridis', 'plasma', etc.
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    ax2.set_xticklabels(sample_index, fontsize=8, rotation=60)
    ax2.xaxis.set_tick_params(width=1, direction='in', pad=1)
    ax2.yaxis.set_tick_params(width=0.5, direction='in', labelsize=8, pad=2)

    ax2.set_ylabel('FWHM [°]', fontsize=8.5, labelpad=0)

    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(0.055, 0.072)



def set_fig_axes():
    width_margin, height_margin = 0.12, 0.6
    y_start, row_heights = 0, [2.3, 1.4, 1.3]
    first_row_y, first_row_width, first_row_height = y_start, 0.9, row_heights[0]
    second_row_y, second_row_width, second_row_height = y_start+height_margin+row_heights[0], 0.9, row_heights[1]
    third_row_y, third_row_width, third_row_height = y_start+height_margin*2+row_heights[0]+row_heights[1], 6, row_heights[2]
    margin_pts = 5

    subfigures_dict = {
                        '1_1': {"position": [0, first_row_y, first_row_width, first_row_height], 'skip_margin': True, 'margin_pts':margin_pts}, # [left, bottom, width, height]
                        '1_2': {"position": [(first_row_width+width_margin), first_row_y, first_row_width, first_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '1_3': {"position": [2*(first_row_width+width_margin), first_row_y, first_row_width, first_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '1_4': {"position": [3*(first_row_width+width_margin), first_row_y, first_row_width, first_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '1_5': {"position": [4*(first_row_width+width_margin), first_row_y, first_row_width, first_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '1_6': {"position": [5*(first_row_width+width_margin), first_row_y, first_row_width, first_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '1_7': {"position": [6*(first_row_width+width_margin), first_row_y, 0.12, first_row_height], 'skip_margin': True, 'margin_pts':margin_pts},

                        '2_1': {"position": [0, second_row_y, second_row_width, second_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '2_2': {"position": [(second_row_width+width_margin), second_row_y, second_row_width, second_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '2_3': {"position": [2*(second_row_width+width_margin), second_row_y, second_row_width, second_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '2_4': {"position": [3*(second_row_width+width_margin), second_row_y, second_row_width, second_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '2_5': {"position": [4*(second_row_width+width_margin), second_row_y, second_row_width, second_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        '2_6': {"position": [5*(second_row_width+width_margin), second_row_y, second_row_width, second_row_height], 'skip_margin': True, 'margin_pts':margin_pts},

                        '3_1': {"position": [0, third_row_y, third_row_width, third_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
                        }
    for key, value in subfigures_dict.items():
        subfigures_dict[key]["position"] = convert_top_left_origin_to_matplotlib(value["position"], fig_height=y_start+height_margin*2+np.sum(row_heights))
    fig, axes_dict = layout_subfigures_inches((8,6), subfigures_dict)
    return fig, axes_dict

def plot_rsm_figure(plotter, fig, axes, files, sample_index, peak_z_range=None, draw_peak=True, draw_peak_line=True, i_start=0):
    
    # n_plot_fisrt_row = 7
    # n_plot_second_row = 6
    Qx_lines, Qz_lines, intensity_lines = [], [], []
    for i, (ax, file, title) in enumerate(zip(axes, files, sample_index)):
        
    # Draw RSMs
        Qx, Qz, intensity = plotter.plot(file, fig, axes, ax, figsize=None)
        labelfigs(ax, i_start+i, size=15, inset_fraction=(0.08, 0.15), loc='tr')

    # Mark peaks with red '+' and draw a horizontal line
        # Mark peaks with red '+'
        coordinates = peak_local_max(intensity, min_distance=20, threshold_abs=80, num_peaks=10)
        coordinates_target = []
        # filter to target range of Qz
        for j, z in enumerate(Qz[coordinates[:, 0], coordinates[:, 1]]):
            if peak_z_range != None:
                if z < peak_z_range[1] and z > peak_z_range[0]: 
                    coordinates_target.append(coordinates[j])
            else: # if no range is provided, take the first peak
                coordinates_target.append(coordinates[j])
        coordinates_target = np.array(coordinates_target)[:1]
        Qx_target, Qz_target = Qx[coordinates_target[:, 0], coordinates_target[:, 1]], Qz[coordinates_target[:, 0], coordinates_target[:, 1]]
        
        if draw_peak:
            ax.scatter(Qx_target, Qz_target, marker='+', color='red', s=14, linewidth=0.8)  # Mark peaks with red '+'
        
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
            ax.plot(Qx_line, Qz_line, 'r--', lw=1)
        
        Qx_lines.append(Qx_line)
        Qz_lines.append(Qz_line)
        intensity_lines.append(intensity_line)
    return Qx_lines, intensity_lines
        

def plot_fwhm_line_profile_figure(plotter, axes, sample_index, Qx_lines, intensity_lines):
    
    # n_plot_fisrt_row = 7
    # n_plot_second_row = 6
    FWHM_list = []
    for i, (ax, title, Qx_line, intensity_line) in enumerate(zip(axes, sample_index, Qx_lines, intensity_lines)):
    # plot the line profile and mark the peak, FWHM
        # ax = axes[i+n_plot_fisrt_row] # change to second row
        ax.scatter(Qx_line, intensity_line, s=1)

        # Calculate FWHM for the STO peak (peak_x[0]) and SRO peak (peak_x[1])
        peak_x, peak_y = detect_peaks(Qx_line, intensity_line, num_peaks=1, prominence=0.1, distance=None)
        fwhm, y_fwhm, x_left, x_right = calculate_fwhm(Qx_line, intensity_line, peak_x[0])
        FWHM_list.append(fwhm)

    # Draw FWHM arrows for both peaks
        ax.plot([x_left, x_right], [y_fwhm, y_fwhm], 'r-', lw=0.5)  # 'k-' is for black line
        # No shrink at the starting point and ending point and Controls arrowhead size
        ax.annotate('', xy=(x_right, y_fwhm), xytext=(x_left, y_fwhm),
                    arrowprops=dict(arrowstyle='<->', lw=0.8, color='r', shrinkA=0,  shrinkB=0, mutation_scale=5))
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
        

        # adjust the yticks and ylabel for the line profile plots
    axes[-1].set_ylabel(r'$Q_z$ [$\AA^{-1}$]', fontsize=plotter.plot_params.get("fontsize", 12))
    for ax in axes[1:]: # start from second ax in the second row
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
    return FWHM_list

    
def plot_fwhm_trend_figure(plotter, ax, sample_index, FWHM_list):

    # Plot the trend of FWHM on the left y-axis with a small offset for the 5th data point
    left_x = list(range(len(sample_index[:5])))
    left_x[4] -= 0.05  # Shift the 5th data point slightly to the left

    ax.plot(left_x, FWHM_list[:5], marker='o', color=colors[0])  # Set line color to blue
    ax.set_xticks(range(len(sample_index)))  # Set x-ticks at integer positions
    ax.set_xticklabels(sample_index)  # Replace x-tick integers with `sample_index` labels
    ax.set_ylim(2.2e-3, 4.3e-3)

    ax.set_xlabel('Sample Names', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold')
    ax.set_ylabel('FWHM (set 1)', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold', color=colors[0])
    ax.tick_params(axis="x", direction="in", top=True, labelsize=plotter.plot_params.get("fontsize", 12))
    ax.tick_params(axis="y", direction="in", right=True, labelsize=plotter.plot_params.get("fontsize", 12), color=colors[0], labelcolor=colors[0])

    # Label the figure as needed
    labelfigs(ax, 10, size=15, inset_fraction=(0.2, 0.05), loc='tr', style='bw')

    # Set the left y-axis to scientific notation
    formatter = ScalarFormatter(useMathText=True)  # Use MathText for cleaner output
    formatter.set_scientific(True)  # Enable scientific notation
    formatter.set_powerlimits((1, 10))  # Set when to switch to scientific notation
    ax.yaxis.set_major_formatter(formatter)  # Apply formatter to y-axis
    ax.yaxis.get_offset_text().set_x(-0.2)  # Set horizontal position (relative to the axis)


    # Create a secondary y-axis on the right in red
    right_x = list(range(4, 4+len(sample_index[-2:])))
    right_x[0] += 0.05  # Shift the 5th data point slightly to the right
    ax_right = ax.twinx()  # Create a twin y-axis
    ax_right.plot(right_x, FWHM_list[-2:], marker='o', color=colors[1])  # Set line color to red

    ax_right.set_ylim(2.2e-3, 4.3e-3)
    ax_right.set_ylabel('FWHM (set 2)', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold', color=colors[1])  # Customize label for the secondary y-axis
    ax_right.tick_params(axis="y", direction="in", labelsize=plotter.plot_params.get("fontsize", 12), color=colors[1], labelcolor=colors[1])

    # Optional: Format the secondary y-axis if needed, e.g., scientific notation
    ax_right_formatter = ScalarFormatter(useMathText=True)
    ax_right_formatter.set_scientific(True)
    ax_right_formatter.set_powerlimits((1, 10))
    ax_right.yaxis.set_major_formatter(ax_right_formatter)
    ax_right.yaxis.get_offset_text().set_x(1)  # Adjust offset for clarity
    
    
def show_sample_frames(plumes, n_plumes=40, n_frames=25):
        
    plume_index_list = np.round(np.linspace(0, len(plumes)-1, n_plumes)).astype(int)
    sample_frames = plumes[plume_index_list, :n_frames]
    sample_frames = np.array(sample_frames).reshape(n_plumes*n_frames, plumes.shape[2], plumes.shape[3])
    labels = np.arange(n_frames)*500e-3
    labels = [f'{l:.0f}µs' for l in labels]
    labels = labels*n_plumes
    fig, axes = layout_fig(n_plumes*n_frames, mod=n_frames, figsize=(8, 12), subplot_style='gridspec', spacing=(0.02, 0.02), layout='tight')
    for i, (ax, img, label) in enumerate(zip(axes, sample_frames, labels)):
        ax.imshow(img)
        ax.axis('off')
        if i < n_frames and i % 2 == 0:
            ax.set_title(label, fontsize=8)
        