# import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
# import glob
from matplotlib.ticker import ScalarFormatter
from skimage.feature import peak_local_max

from sro_sto_plume.coordinate_converter import convert_top_left_origin_to_matplotlib
# from sro_sto_plume.xrd_ref import calculate_2theta, plot_ref_xrd
from m3util.viz.layout import layout_fig, layout_subfigures_inches
from m3util.viz.text import labelfigs, add_text_to_figure
from m3util.viz.lines import draw_lines
from xrd_learn.xrd_viz import plot_xrd
from xrd_learn.xrd_utils import detect_peaks, calculate_fwhm, load_xrd_scans, align_peak_to_value
# from xrd_learn.rsm_viz import RSMPlotter
# from afm_learn.afm_viz import visualize_afm_image
# from afm_learn.afm_utils import parse_ibw, format_func
colors = colormaps.get_cmap('tab10').colors[:6]


# xrd section
def plot_xrd_figure(fig, ax, filename):
            
    files = ['../data/XRD_RSM/YG065/YG065_2theta-Omega_Path1_42-49 degree_slow_2.xrdml',
            '../data/XRD_RSM/YG066/YG066 2theta-Omega_Path1_42-49 degree_slow_2.xrdml',
            '../data/XRD_RSM/YG067/YG067 2theta-Omega_Path1_42-49 degree_slow_2 1.xrdml',
            '../data/XRD_RSM/YG068/YG068 2theta-Omega_Path1_42-49 degree_slow_2.xrdml',
            '../data/XRD_RSM/YG069/YG069 2theta-Omega_Path1_42-49 degree_slow_2.xrdml',
            '../data/XRD_RSM/YG063/YG063 2theta-Omega_Path1_42-49 degree_slow_2.xrdml',]
    sample_names = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample X']
    STO_x_peak = 46.4721
    SRO_bulk_x_peak = 46.2425

    Xs, Ys, length_list = load_xrd_scans(files)
    for i, (X, Y, sample_name) in enumerate(zip(Xs, Ys, sample_names)):
        peak_x, peak_y = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)
    Xs_aligned, Ys_aligned = align_peak_to_value(Xs, Ys, STO_x_peak, viz=False)

    # fig, ax = layout_fig(1, 1, figsize=(8, 4), layout='tight')
    diff = 5e1
    plot_xrd((Xs_aligned, Ys_aligned, length_list), sample_names, title='XRD Scans', xrange=(42.8, 49.8), yrange=(1, 1e18), diff=diff, fig=fig, ax=ax, legend_style='label', text_offset_ratio=(1.001, 0.45))

    line_style = {'color': 'gray', 'linestyle': 'dashed', 'linewidth': 1}
    draw_lines(ax, x_values=[STO_x_peak, STO_x_peak], y_values=[1e1, 5e15], style=line_style)
    ax.text(STO_x_peak, 8e15, 'STO\n(002)', fontsize=10, ha='center')

    ax.text(45.9121, 8e14, 'SRO\n(220)', fontsize=10, ha='center')
    line_style = {'color': 'gray', 'linestyle': 'dotted', 'linewidth': 1}
    draw_lines(ax, x_values=[45.9121, 45.9121], y_values=[5e2, 5e14], style=line_style)

    ax.text(SRO_bulk_x_peak, 10, 'SRO\n(bulk)', fontsize=10, ha='center')
    line_style = {'color': 'gray', 'linestyle': 'dashdot', 'linewidth': 1}
    draw_lines(ax, x_values=[SRO_bulk_x_peak, SRO_bulk_x_peak], y_values=[8e2, 5e14], style=line_style)

    legend = []
    for i, (X, Y, sample_name, color) in enumerate(zip(Xs_aligned, Ys_aligned, sample_names, colors)):
        peak_x, peak_y = detect_peaks(X, Y, num_peaks=2, prominence=0.1, distance=10)

        # Calculate FWHM for the STO peak (peak_x[0]) and SRO peak (peak_x[1])
        fwhm_sto, y_fwhm_sto, x_left_sto, x_right_sto = calculate_fwhm(X, Y, peak_x[0])
        fwhm_sro, y_fwhm_sro, x_left_sro, x_right_sro = calculate_fwhm(X, Y, peak_x[1])

        # Prepare legend item
        legend_item = f'SRO(+): {peak_x[1]:.4f}°, STO(*): {peak_x[0]:.4}°'
        legend_item = f'SRO(+): {peak_x[1]:.4f}°, FWHM: {fwhm_sro:.2f}'
        legend.append(legend_item)
        
        peak_y = np.array(peak_y)*diff**(len(Ys)-i-1)
        # plt.plot(peak_x[0], peak_y[0]*3, '*', color=color)
        ax.plot(peak_x[1]-0.05, peak_y[1]*3, '+', color=color)
        
    plt.ylim(5, 1e18)
    plt.legend(legend, fontsize=9, loc='upper right', frameon=False)
    if filename:
        plt.savefig(f'{filename}.png', dpi=600)
        plt.savefig(f'{filename}.svg', dpi=600)

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

def plot_rsm_figure(plotter, fig, axes_dict, files, sample_names, peak_z_range, draw_peak=True, draw_peak_line=True):
    axes = list(axes_dict.values())
    
    n_plot_fisrt_row = 7
    n_plot_second_row = 6
    Qx_lines, Qz_lines, intensity_lines = [], [], []
    for i, (ax, file, title) in enumerate(zip(axes[:len(files)], files, sample_names)):
        
    # Draw RSMs
        Qx, Qz, intensity = plotter.plot(file, fig, axes[:n_plot_fisrt_row], ax, figsize=None)
        labelfigs(ax, i, size=15, inset_fraction=(0.08, 0.15), loc='tr')

    # Mark peaks with red '+' and draw a horizontal line
        # Mark peaks with red '+'
        coordinates = peak_local_max(intensity, min_distance=20, threshold_abs=80, num_peaks=10)
        coordinates_target = []
        # filter to target range of Qz
        for j, z in enumerate(Qz[coordinates[:, 0], coordinates[:, 1]]):
            if z < peak_z_range[1] and z > peak_z_range[0]: 
                coordinates_target.append(coordinates[j])
        coordinates_target = np.array(coordinates_target)[:1]
        Qx_target, Qz_target = Qx[coordinates_target[:, 0], coordinates_target[:, 1]], Qz[coordinates_target[:, 0], coordinates_target[:, 1]]
        
        if draw_peak:
            ax.scatter(Qx_target, Qz_target, marker='+', color='red')  # Mark peaks with red '+'
        
        # extract line profile at Qz_target
        mask = np.isclose(Qz, Qz_target, atol=1e-3)  # Boolean mask
        Qx_line = Qx[mask]
        intensity_line = intensity[mask]
        # sort the Qx_line and intensity_line based on Qx values
        Qx_index = np.argsort(Qx_line)
        Qx_line = Qx_line[Qx_index]
        intensity_line = intensity_line[Qx_index]
        
    #  draw the horizontal line on the RSM plot
        if draw_peak_line:
            Qz_line = np.ones_like(Qx_line) * Qz_target[0]
            ax.plot(Qx_line, Qz_line, 'r--', lw=1)
        
        Qx_lines.append(Qx_line)
        Qz_lines.append(Qz_line)
        intensity_lines.append(intensity_line)
    return Qx_lines, intensity_lines
        

def plot_fwhm_line_profile_figure(plotter, axes_dict, sample_names, Qx_lines, intensity_lines):
    
    axes = list(axes_dict.values())
    
    n_plot_fisrt_row = 7
    n_plot_second_row = 6
    FWHM_list = []
    for i, (ax, title, Qx_line, intensity_line) in enumerate(zip(axes[:n_plot_fisrt_row-1], sample_names, Qx_lines, intensity_lines)):

    # plot the line profile and mark the peak, FWHM
        ax = axes[i+n_plot_fisrt_row] # change to second row
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
        ax.set_ylim(-100, 900)
        ax.set_xlim(1.595, 1.621)
        
        ax.tick_params(axis="x", direction="in", top=True, labelsize=plotter.plot_params.get("fontsize", 12))
        ax.tick_params(axis="y", direction="in", right=True, labelsize=plotter.plot_params.get("fontsize", 12))       
        ax.set_xlabel(r'$Q_x$ [$\AA^{-1}$]', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold')
        ax.set_ylabel(r'$Q_z$ [$\AA^{-1}$]', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold')
        labelfigs(ax, i+n_plot_fisrt_row-1, size=15, inset_fraction=(0.15, 0.15), loc='tr', style='bw')
        
        # adjust the yticks and ylabel for the line profile plots
    axes[n_plot_fisrt_row].set_ylabel(r'$Q_z$ [$\AA^{-1}$]', fontsize=plotter.plot_params.get("fontsize", 12))
    for ax in axes[7:-1]: # start from second ax in the second row
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
    return FWHM_list

    
def plot_fwhm_trend_figure(plotter, axes_dict, sample_names, FWHM_list):
    axes = list(axes_dict.values())

    # Plot the trend of FWHM on the left y-axis with a small offset for the 5th data point
    left_x = list(range(len(sample_names[:5])))
    left_x[4] -= 0.05  # Shift the 5th data point slightly to the left

    axes[-1].plot(left_x, FWHM_list[:5], marker='o', color=colors[0])  # Set line color to blue
    axes[-1].set_xticks(range(len(sample_names)))  # Set x-ticks at integer positions
    axes[-1].set_xticklabels(sample_names)  # Replace x-tick integers with `sample_names` labels
    axes[-1].set_ylim(2.2e-3, 4.3e-3)

    axes[-1].set_xlabel('Sample Names', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold')
    axes[-1].set_ylabel('FWHM (set 1)', fontsize=plotter.plot_params.get("fontsize", 12), fontweight='bold', color=colors[0])
    axes[-1].tick_params(axis="x", direction="in", top=True, labelsize=plotter.plot_params.get("fontsize", 12))
    axes[-1].tick_params(axis="y", direction="in", right=True, labelsize=plotter.plot_params.get("fontsize", 12), color=colors[0], labelcolor=colors[0])

    # Label the figure as needed
    labelfigs(axes[-1], 10, size=15, inset_fraction=(0.2, 0.05), loc='tr', style='bw')

    # Set the left y-axis to scientific notation
    formatter = ScalarFormatter(useMathText=True)  # Use MathText for cleaner output
    formatter.set_scientific(True)  # Enable scientific notation
    formatter.set_powerlimits((1, 10))  # Set when to switch to scientific notation
    axes[-1].yaxis.set_major_formatter(formatter)  # Apply formatter to y-axis
    axes[-1].yaxis.get_offset_text().set_x(-0.2)  # Set horizontal position (relative to the axis)


    # Create a secondary y-axis on the right in red
    right_x = list(range(4, 4+len(sample_names[-2:])))
    right_x[0] += 0.05  # Shift the 5th data point slightly to the right
    ax_right = axes[-1].twinx()  # Create a twin y-axis
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