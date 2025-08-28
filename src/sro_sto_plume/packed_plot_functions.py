# sro_sto_plume/packed_plot_functions.py
from matplotlib import colormaps
import numpy as np

from m3util.viz.layout import layout_subfigures_inches
from sro_sto_plume.coordinate_converter import convert_top_left_origin_to_matplotlib

# --------- public constants kept for back-compat ----------
colors = colormaps.get_cmap('tab10').colors[:6]

def set_fig_axes():
    """
    Back-compat helper returning (fig, axes_dict) for your 3-row composite layout.
    """
    width_margin, height_margin = 0.12, 0.6
    y_start, row_heights = 0, [2.3, 1.4, 1.3]
    first_row_y, first_row_width, first_row_height   = y_start, 0.9, row_heights[0]
    second_row_y, second_row_width, second_row_height = y_start+height_margin+row_heights[0], 0.9, row_heights[1]
    third_row_y, third_row_width, third_row_height    = y_start+height_margin*2+row_heights[0]+row_heights[1], 6, row_heights[2]
    margin_pts = 5

    subfigures_dict = {
        '1_1': {"position": [0, first_row_y, first_row_width, first_row_height], 'skip_margin': True, 'margin_pts':margin_pts},
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
        subfigures_dict[key]["position"] = convert_top_left_origin_to_matplotlib(
            value["position"], fig_height=y_start+height_margin*2+np.sum(row_heights)
        )
    fig, axes_dict = layout_subfigures_inches((8, 6), subfigures_dict)
    return fig, axes_dict

# --------- re-exports to keep old import lines working ----------
from sro_sto_plume.plots_xrd import (
    plot_xrd_figure,
    plot_rocking_curve_figure,
    plot_xrd_multiple,
)

from sro_sto_plume.plots_rsm import (
    plot_rsm_figure,
    plot_rsm002,
    plot_fwhm_line_profile_figure,  # <-- now exported
    plot_fwhm_trend_figure,         # <-- now exported
)

from sro_sto_plume.plots_afm import (
    plot_afm_figure,
    plot_afm_figure_lineprofile,
)

__all__ = [
    "colors",
    "set_fig_axes",
    # xrd
    "plot_xrd_figure", "plot_rocking_curve_figure", "plot_xrd_multiple",
    # rsm
    "plot_rsm_figure", "plot_rsm002", "plot_fwhm_line_profile_figure", "plot_fwhm_trend_figure",
    # afm
    "plot_afm_figure", "plot_afm_figure_lineprofile",
]
