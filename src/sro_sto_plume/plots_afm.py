# plots_afm.py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from m3util.viz.text import labelfigs, add_text_to_figure
from m3util.viz.layout import layout_fig, layout_subfigures_inches
from plume_learn.plume_utils.viz import set_labels
from afm_learn.afm_utils import parse_ibw
from afm_learn.afm_image_analyzer import (
    calculate_height_profile, afm_RMS_roughness
)


def plot_afm_figure_lineprofile(afm_visualizer, files_ibw, files_txt, sample_names,
                                colors, line_ax_indexes, line_profile_txt, line_coords, label=True):
    """
    AFM montage + three line-profile panels with lines drawn on selected AFM images.
    """
    width, height = 2, 2
    w_spacing, v_spacing = 0.2, 0.1
    figsize = (width*3+w_spacing*2, height*3+v_spacing*2)

    subfigures_dict = {
        '1_1': {"position": [0.4, 1.5*height+v_spacing*2, width, height], 'skip_margin': True, 'margin_pts': 5},
        '1_2': {"position": [0.4+width+w_spacing, 1.5*height+v_spacing*2, width, height], 'skip_margin': True, 'margin_pts': 5},
        '1_3': {"position": [0.4+width*2+w_spacing*2, 1.5*height+v_spacing*2, width, height], 'skip_margin': True, 'margin_pts': 5},
        '2_1': {"position": [0.4, height*0.5+v_spacing, width, height], 'skip_margin': True, 'margin_pts': 5},
        '2_2': {"position": [0.4+width+w_spacing, height*0.5+v_spacing, width, height], 'skip_margin': True, 'margin_pts': 5},
        '2_3': {"position": [0.4+width*2+w_spacing*2, height*0.5+v_spacing, width, height], 'skip_margin': True, 'margin_pts': 5},
        '3_1': {"position": [0.4, 0, width, height/2], 'skip_margin': True, 'margin_pts': 5},
        '3_2': {"position": [0.4+width+w_spacing, 0, width, height/2], 'skip_margin': True, 'margin_pts': 5},
        '3_3': {"position": [0.4+width*2+w_spacing*2, 0, width, height/2], 'skip_margin': True, 'margin_pts': 5},
    }

    marker_pos_list = [(1.01, 0.7)] * 6
    text_pos_list = [(1.99, 4.53), (4.18, 4.53), (6.38, 4.53),
                     (1.99, 2.44), (4.18, 2.44), (6.38, 2.44)]

    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)

    # AFM tiles
    for i, (ax, file_txt, file_ibw) in enumerate(zip(axes_dict.values(), files_txt, files_ibw)):
        img = np.loadtxt(file_txt)
        afm_imgs, sample_name, labels_correct, scan_size = parse_ibw(file_ibw)
        afm_visualizer.viz(img=img, scan_size=scan_size, fig=fig, ax=ax, title=None)
        if label:
            labelfigs(ax, number=i, style='wb', size=15, loc='tr', inset_fraction=(0.12, 0.15))

    # legends + sample name text
    for ax, sample_name, color, text_pos, marker_pos in zip(axes_dict.values(), sample_names, colors, text_pos_list, marker_pos_list):
        marker_line_label = Line2D([0], [0], color=color, linestyle='-', linewidth=2, marker='o', markersize=5,
                                   path_effects=[withStroke(linewidth=4, foreground="white")])
        ax.legend(handles=[marker_line_label], loc='lower right', frameon=False, bbox_to_anchor=marker_pos, fontsize=8)
        add_text_to_figure(fig, sample_name, text_pos, fontsize=10, color='white', ha='center',
                           path_effects=[withStroke(linewidth=0.9, foreground='black')])

    # line profile panels (the last 3 entries of axes_dict)
    axes_mark = [list(axes_dict.values())[i] for i in line_ax_indexes]
    colors_mark = [colors[i] for i in line_ax_indexes]
    for i, (ax_line, txt, coord, ax_mark, color) in enumerate(
        zip(list(axes_dict.values())[-3:], line_profile_txt, line_coords, axes_mark, colors_mark)
    ):
        p1, p2 = coord
        img = np.loadtxt(txt)

        # overlay the line on the AFM tile
        ax_mark.plot([p1[0], p2[0]], [p1[1], p2[1]], color='w', linestyle='-', linewidth=1.5)
        ax_mark.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle='-', linewidth=0.8)

        # compute and draw line profile
        x, values = calculate_height_profile(img, p1, p2)
        ax_line.plot(x, values*1e9, color=color, linestyle='-', linewidth=1)
        if i == 0:
            set_labels(ax_line, xlabel='X (a.u.)', ylabel='Height (nm)', label_fontsize=9, ticklabel_fontsize=8,
                       yaxis_style='float', show_ticks=True)
        else:
            set_labels(ax_line, xlabel='X (a.u.)', ylabel='', label_fontsize=9, ticklabel_fontsize=9,
                       yaxis_style='float', show_ticks=True)
        ax_line.tick_params(pad=1)

        if label:
            labelfigs(ax_line, number=i+6, style='bw', size=15, loc='tl', inset_fraction=(0.2, 0.1))

    return fig, axes_dict


def plot_afm_figure(afm_visualizer, files_ibw, files_txt, files_roughness_txt,
                    sample_names, colors, plot_roughness=True, roughness_ylim=None,
                    roughness_label_loc='tr', label=True):
    """
    AFM montage with optional bottom-row roughness scatter.
    """
    import matplotlib.pyplot as plt

    width, height = 2, 2
    lineplot_height = 1.5
    w_spacing, v_spacing = 0.2, 0.1
    figsize = (width*3+w_spacing*2, height*3+v_spacing*2)

    subfigures_dict = {
        '1_1': {"position": [0.4, height+lineplot_height+v_spacing*2, width, height], 'skip_margin': True, 'margin_pts': 5},
        '1_2': {"position": [0.4+width+w_spacing, height+lineplot_height+v_spacing*2, width, height], 'skip_margin': True, 'margin_pts': 5},
        '1_3': {"position": [0.4+width*2+w_spacing*2, height+lineplot_height+v_spacing*2, width, height], 'skip_margin': True, 'margin_pts': 5},
        '2_1': {"position": [0.4, lineplot_height+v_spacing, width, height], 'skip_margin': True, 'margin_pts': 5},
        '2_2': {"position": [0.4+width+w_spacing, lineplot_height+v_spacing, width, height], 'skip_margin': True, 'margin_pts': 5},
        '2_3': {"position": [0.4+width*2+w_spacing*2, lineplot_height+v_spacing, width, height], 'skip_margin': True, 'margin_pts': 5},
    }
    if plot_roughness:
        subfigures_dict['3_1'] = {"position": [0.4, 0, width*3+np.sum(w_spacing)+0.2, lineplot_height], 'skip_margin': True, 'margin_pts': 5}

    marker_pos_list = [(1.01, 0.7)] * 6
    text_pos_list = [(1.99, 5.03), (4.18, 5.03), (6.38, 5.03),
                     (1.99, 2.94), (4.18, 2.94), (6.38, 2.94)]

    fig, axes_dict = layout_subfigures_inches(figsize, subfigures_dict)

    # tiles + roughness
    roughness_list = []
    for i, (ax, file_txt, file_roughness_txt, file_ibw) in enumerate(zip(axes_dict.values(), files_txt, files_roughness_txt, files_ibw)):
        img = np.loadtxt(file_txt)
        img_roughness = np.loadtxt(file_roughness_txt)
        roughness = afm_RMS_roughness(img_roughness)
        roughness_list.append(roughness*1e9)  # nm
        afm_imgs, sample_name, labels_correct, scan_size = parse_ibw(file_ibw)

        afm_visualizer.viz(img=img, scan_size=scan_size, fig=fig, ax=ax, title=None)
        if label:
            labelfigs(ax, number=i, style='wb', size=15, loc='tr', inset_fraction=(0.12, 0.15))

    # legends + names
    for ax, sample_name, color, text_pos, marker_pos in zip(axes_dict.values(), sample_names, colors, text_pos_list, marker_pos_list):
        marker_line_label = Line2D([0], [0], color=color, linestyle='-', linewidth=2, marker='o', markersize=5,
                                   path_effects=[withStroke(linewidth=4, foreground="white")])
        ax.legend(handles=[marker_line_label], loc='lower right', frameon=False, bbox_to_anchor=marker_pos, fontsize=8)
        add_text_to_figure(fig, sample_name, text_pos, fontsize=10, color='white', ha='center',
                           path_effects=[withStroke(linewidth=0.9, foreground='black')])

    # bottom roughness scatter
    if plot_roughness:
        axr = axes_dict['3_1']
        axr.scatter(sample_names, roughness_list, c=colors)
        set_labels(axr, xlabel="", ylabel="Roughness (nm)", label_fontsize=10, ticklabel_fontsize=10,
                   yaxis_style='float', show_ticks=True)
        if roughness_ylim:
            axr.set_ylim(roughness_ylim)
        else:
            roughness_ylim = axr.get_ylim()
        # annotate values a bit above points
        span = (roughness_ylim[1] - roughness_ylim[0])
        for i, r in enumerate(roughness_list):
            axr.text(i, r - 0.11*span, f'{r:.2f}nm', ha='center', va='bottom', fontsize=8)
        labelfigs(axr, number=6, style='bw', size=15, loc=roughness_label_loc, inset_fraction=(0.18, 0.06))

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.25, hspace=0.1)
    return fig, axes_dict
