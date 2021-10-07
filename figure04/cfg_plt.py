import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

fontname = 'Arial'
fontsize = 6.0
linewidth = 1.25

def plot_coord_ax(ax,
                  label_x, label_y,
                  len_x, len_y,
                  offset_x_fac=0.05, offset_y_fac=0.05):
    ax_xlim = ax.get_xlim()
    ax_ylim = ax.get_ylim()
    ax_xlen = abs(np.diff(ax_xlim)[0])
    ax_ylen = abs(np.diff(ax_ylim)[0])

    offset_x = ax_xlim[0] + ax_xlen * 0.01
    offset_y = ax_ylim[0] + ax_ylen * 0.01
    offset_x_text = -ax_ylen * offset_x_fac
    offset_y_text = -ax_xlen * offset_y_fac

    ax.plot(np.array([offset_x, offset_x + len_x], dtype=np.float64),
            np.array([offset_y, offset_y], dtype=np.float64),
            linestyle='-', marker='', color='black',
            linewidth=linewidth)
    ax.plot(np.array([offset_x, offset_x], dtype=np.float64),
            np.array([offset_y, offset_y + len_y], dtype=np.float64),
            linestyle='-', marker='', color='black',
            linewidth=linewidth)
    ax.text(offset_x + len_x/2.0, offset_y+offset_x_text, '{:s}'.format(label_x),
            fontsize=fontsize, fontname=fontname, ha='center', va='top', rotation='horizontal')
    ax.text(offset_x+offset_y_text, offset_y + len_y/2.0, '{:s}'.format(label_y),
            fontsize=fontsize, fontname=fontname, ha='right', va='center', rotation='vertical')
    ax.set_axis_off()
    return