#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats
import sys

save = False
show = True

label_type = 'full'

folder_save = os.path.abspath('panels')

mm_in_inch = 5.0 / 127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

bottom_margin_x = 0.05  # inch
left_margin_x = 0.05  # inch
left_margin = 0.4  # inch
right_margin = 0.05  # inch
bottom_margin = 0.375  # inch
top_margin = 0.05  # inch

fontsize = 6
linewidth_hist = 0.5  # 0.25
fontname = "Arial"
markersize = 2

cmap = plt.cm.tab10
color_mode1 = cmap(5 / 9)
color_mode2 = cmap(1 / 9)
color_mode3 = cmap(0 / 9)
color_mode4 = cmap(2 / 9)
colors = list([color_mode1, color_mode2, color_mode3, color_mode4])

if __name__ == '__main__':
    fig_w = np.round(mm_in_inch * (88.0 * 1 / 2), decimals=2)
    fig_h = np.round(mm_in_inch * 88.0 * 1 / 2, decimals=2)
    # acceleration
    fig_acc_all_w = fig_w
    fig_acc_all_h = fig_h
    fig_acc_all = plt.figure(1, figsize=(fig_acc_all_w, fig_acc_all_h))
    fig_acc_all.canvas.manager.window.move(0, 0)
    fig_acc_all.clear()
    ax_acc_all_x = left_margin / fig_acc_all_w
    ax_acc_all_y = bottom_margin / fig_acc_all_h
    ax_acc_all_w = 1.0 - (left_margin / fig_acc_all_w + right_margin / fig_acc_all_w)
    ax_acc_all_h = 1.0 - (bottom_margin / fig_acc_all_h + top_margin / fig_acc_all_h)
    ax_acc_all = fig_acc_all.add_axes([ax_acc_all_x, ax_acc_all_y, ax_acc_all_w, ax_acc_all_h])
    ax_acc_all.clear()
    ax_acc_all.spines["top"].set_visible(False)
    ax_acc_all.spines["right"].set_visible(False)
    # velocity
    fig_velo_all_w = fig_w
    fig_velo_all_h = fig_h
    fig_velo_all = plt.figure(2, figsize=(fig_velo_all_w, fig_velo_all_h))
    fig_velo_all.canvas.manager.window.move(0, 0)
    fig_velo_all.clear()
    ax_velo_all_x = left_margin / fig_velo_all_w
    ax_velo_all_y = bottom_margin / fig_velo_all_h
    ax_velo_all_w = 1.0 - (left_margin / fig_velo_all_w + right_margin / fig_velo_all_w)
    ax_velo_all_h = 1.0 - (bottom_margin / fig_velo_all_h + top_margin / fig_velo_all_h)
    ax_velo_all = fig_velo_all.add_axes([ax_velo_all_x, ax_velo_all_y, ax_velo_all_w, ax_acc_all_h])
    ax_velo_all.clear()
    ax_velo_all.spines["top"].set_visible(False)
    ax_velo_all.spines["right"].set_visible(False)
    # position
    fig_pos_all_w = fig_w
    fig_pos_all_h = fig_h
    fig_pos_all = plt.figure(3, figsize=(fig_pos_all_w, fig_pos_all_h))
    fig_pos_all.canvas.manager.window.move(0, 0)
    fig_pos_all.clear()
    ax_pos_all_x = left_margin / fig_pos_all_w
    ax_pos_all_y = bottom_margin / fig_pos_all_h
    ax_pos_all_w = 1.0 - (left_margin / fig_pos_all_w + right_margin / fig_pos_all_w)
    ax_pos_all_h = 1.0 - (bottom_margin / fig_pos_all_h + top_margin / fig_pos_all_h)
    ax_pos_all = fig_pos_all.add_axes([ax_pos_all_x, ax_pos_all_y, ax_pos_all_w, ax_pos_all_h])
    ax_pos_all.clear()
    ax_pos_all.spines["top"].set_visible(False)
    ax_pos_all.spines["right"].set_visible(False)
    # angle
    fig_ang_all_w = fig_w
    fig_ang_all_h = fig_h
    fig_ang_all = plt.figure(31, figsize=(fig_ang_all_w, fig_ang_all_h))
    fig_ang_all.canvas.manager.window.move(0, 0)
    fig_ang_all.clear()
    ax_ang_all_x = left_margin / fig_ang_all_w
    ax_ang_all_y = bottom_margin / fig_ang_all_h
    ax_ang_all_w = 1.0 - (left_margin / fig_ang_all_w + right_margin / fig_ang_all_w)
    ax_ang_all_h = 1.0 - (bottom_margin / fig_ang_all_h + top_margin / fig_ang_all_h)
    ax_ang_all = fig_ang_all.add_axes([ax_ang_all_x, ax_ang_all_y, ax_ang_all_w, ax_ang_all_h])
    ax_ang_all.clear()
    ax_ang_all.spines["top"].set_visible(False)
    ax_ang_all.spines["right"].set_visible(False)

    position0_all = np.load(folder_save + '/position0_all_{:s}.npy'.format(label_type), allow_pickle=True)
    angle0_all = np.load(folder_save + '/angle0_all_{:s}.npy'.format(label_type), allow_pickle=True)
    position0 = list()
    angle0 = list()
    nT_all = list()
    nSeq = np.size(position0_all, 0)
    for i_seq in range(nSeq):
        pos = position0_all[i_seq]
        ang = angle0_all[i_seq]
        pos_shape = np.shape(pos)
        position0 = position0 + list(pos)
        angle0 = angle0 + list(ang)
        nT_all.append(np.size(ang, 0))
    position0 = np.array(position0, dtype=np.float64)
    angle0 = np.array(angle0, dtype=np.float64)
    #
    position_all = np.load(folder_save + '/position_all.npy', allow_pickle=True).item()
    angle_all = np.load(folder_save + '/angle_all.npy', allow_pickle=True).item()
    position = dict()
    angle = dict()
    velocity = dict()
    acceleration = dict()
    for i_mode in range(4):
        position[i_mode] = list()
        angle[i_mode] = list()
        velocity[i_mode] = list()
        acceleration[i_mode] = list()
        for i_seq in range(nSeq):
            pos = position_all[i_mode + 1][i_seq]
            ang = angle_all[i_mode + 1][i_seq]
            pos_shape = np.shape(pos)
            velo = np.full(pos_shape, np.nan, dtype=np.float64)
            acc = np.full(pos_shape, np.nan, dtype=np.float64)
            #         velo_fit[1:-1] = (0.5 * pos[2:, :, :, :2] - 0.5 * pos[:-2, :, :, :2]) / (1.0/200.0)
            #         acc_fit[1:-1] = (pos[2:, :, :, :2] - 2.0 * pos[1:-1, :, :, :2] + pos[:-2, :, :, :2]) / (1.0/200.0)**2
            velo[4:-4] = \
                (+1.0 / 280.0 * pos[:-8] + \
                 -4.0 / 105.0 * pos[1:-7] + \
                 +1.0 / 5.0 * pos[2:-6] + \
                 -4.0 / 5.0 * pos[3:-5] + \
                 #                      0.0 * pos[4:-4] + \
                 +4.0 / 5.0 * pos[5:-3] + \
                 -1.0 / 5.0 * pos[6:-2] + \
                 +4.0 / 105.0 * pos[7:-1] + \
                 -1.0 / 280.0 * pos[8:]) / (1.0 / 200.0)
            acc[4:-4] = \
                (-1.0 / 560.0 * pos[:-8] + \
                 +8.0 / 315.0 * pos[1:-7] + \
                 -1.0 / 5.0 * pos[2:-6] + \
                 +8.0 / 5.0 * pos[3:-5] + \
                 -205.0 / 72.0 * pos[4:-4] + \
                 +8.0 / 5.0 * pos[5:-3] + \
                 -1.0 / 5.0 * pos[6:-2] + \
                 +8.0 / 315.0 * pos[7:-1] + \
                 -1.0 / 560.0 * pos[8:]) / (1.0 / 200.0) ** 2
            position[i_mode] = position[i_mode] + list(pos)
            angle[i_mode] = angle[i_mode] + list(ang)
            velocity[i_mode] = velocity[i_mode] + list(velo)
            acceleration[i_mode] = acceleration[i_mode] + list(acc)
        position[i_mode] = np.array(position[i_mode], dtype=np.float64)
        angle[i_mode] = np.array(angle[i_mode], dtype=np.float64)
        velocity[i_mode] = np.array(velocity[i_mode], dtype=np.float64)
        acceleration[i_mode] = np.array(acceleration[i_mode], dtype=np.float64)

    inf_fac = 0.25
    nBins = 25
    # acceleration
    acc_unit = 'cm/ms²'  # use unicord here for the superscript, otherwise alignment of x-axis labels is messed up [https://stackoverflow.com/questions/21226868/superscript-in-python-plots]
    conversion_fac_acc = 1.0 / (1e3) ** 2  # cm/s^2 -> acc_unit
    maxBin_acc0 = 2e4  # cm/s**2
    dBin_acc = maxBin_acc0 / nBins  # 5e2 # cm/s**2
    maxBin_acc = maxBin_acc0 * (1.0 + inf_fac)
    # velocity
    velo_unit = 'cm/ms'
    conversion_fac_velo = 1.0 / 1e3  # cm/s -> velo_unit
    maxBin_velo0 = 8e1  # cm/s
    dBin_velo = maxBin_velo0 / nBins  # 2e0 # cm/s
    maxBin_velo = maxBin_velo0 * (1.0 + inf_fac)
    # position
    pos_unit = 'cm'
    conversion_fac_pos = 1.0  # cm -> pos_unit
    maxBin_pos0 = 4e0  # cm
    dBin_pos = maxBin_pos0 / nBins  # 1e-1 # cm
    maxBin_pos = maxBin_pos0 * (1.0 + inf_fac)
    #  angle
    ang_unit = 'deg'
    conversion_fac_ang = 1.0  # deg -> ang_unit
    maxBin_ang0 = 60.0  # deg
    dBin_ang = maxBin_ang0 / nBins  # 2.25 # deg
    maxBin_ang = maxBin_ang0 * (1.0 + inf_fac)
    #
    bin_range_acc = np.arange(0.0, maxBin_acc + dBin_acc, dBin_acc, dtype=np.float64)
    nBin_acc = len(bin_range_acc) - 1
    bin_range_velo = np.arange(0.0, maxBin_velo + dBin_velo, dBin_velo, dtype=np.float64)
    nBin_velo = len(bin_range_velo) - 1
    bin_range_pos = np.arange(0.0, maxBin_pos + dBin_pos, dBin_pos, dtype=np.float64)
    nBin_pos = len(bin_range_pos) - 1
    bin_range_ang = np.arange(0.0, maxBin_ang + dBin_ang, dBin_ang, dtype=np.float64)
    nBin_ang = len(bin_range_ang) - 1
    #
    mode_pairs = list([[4, 3],
                       [4, 2],
                       [4, 1]])
    nPairs = np.size(mode_pairs, 0)
    modes_ploted = np.zeros(4, dtype=bool)

    print('acceleration bin size:\t\t{:0.2e} {:s}'.format(dBin_acc * conversion_fac_acc, acc_unit))
    print('velocity bin size:\t\t{:0.2e} {:s}'.format(dBin_velo * conversion_fac_velo, velo_unit))
    print('position bin size:\t\t{:0.2e} {:s}'.format(dBin_pos * conversion_fac_pos, pos_unit))
    print('angle bin size:\t\t\t{:0.2e} {:s}'.format(dBin_ang * conversion_fac_ang, ang_unit))
    #
    print('acceleration bin (max):\t\t{:0.2e} {:s}'.format(maxBin_acc * conversion_fac_acc, acc_unit))
    print('velocity bin (max):\t\t{:0.2e} {:s}'.format(maxBin_velo * conversion_fac_velo, velo_unit))
    print('position bin (max):\t\t{:0.2e} {:s}'.format(maxBin_pos * conversion_fac_pos, pos_unit))
    print('angle bin (max):\t\t{:0.2e} {:s}'.format(maxBin_ang * conversion_fac_ang, ang_unit))
    #
    print('acceleration bin number:\t{:d}'.format(nBin_acc))
    print('velocity bin number:\t\t{:d}'.format(nBin_velo))
    print('position bin number:\t\t{:d}'.format(nBin_pos))
    print('angle bin number:\t\t{:d}'.format(nBin_ang))
    print()
    for i_mode_pair in range(nPairs):
        mode_pair = mode_pairs[i_mode_pair]
        i_mode1 = mode_pair[0] - 1
        i_mode2 = mode_pair[1] - 1

        pos0 = np.copy(position0)
        ang0 = np.copy(angle0)
        #
        acc1 = acceleration[i_mode1]
        acc2 = acceleration[i_mode2]
        velo1 = velocity[i_mode1]
        velo2 = velocity[i_mode2]
        pos1 = position[i_mode1]
        pos2 = position[i_mode2]
        ang1 = angle[i_mode1]
        ang2 = angle[i_mode2]

        mask1 = ~np.logical_or(np.isnan(pos0), np.isnan(pos1))
        mask2 = ~np.logical_or(np.isnan(pos0), np.isnan(pos2))
        wasserstein1 = scipy.stats.wasserstein_distance(pos0[:, :, :2][mask1[:, :, :2]].ravel(),
                                                        pos1[:, :, :2][mask1[:, :, :2]].ravel())
        wasserstein2 = scipy.stats.wasserstein_distance(pos0[:, :, :2][mask2[:, :, :2]].ravel(),
                                                        pos2[:, :, :2][mask2[:, :, :2]].ravel())
        print('position')
        print('mode{:01d}:\t{:0.8}'.format(i_mode1 + 1, wasserstein1))
        print('mode{:01d}:\t{:0.8}'.format(i_mode2 + 1, wasserstein2))
        print(np.max(ang0[~np.isnan(ang0)]), np.max(ang1[~np.isnan(ang1)]), np.max(ang2[~np.isnan(ang2)]))
        mask1 = ~np.logical_or(np.isnan(ang0), np.isnan(ang1))
        mask2 = ~np.logical_or(np.isnan(ang0), np.isnan(ang2))
        wasserstein1 = scipy.stats.wasserstein_distance(ang0[mask1].ravel(), ang1[mask1].ravel())
        wasserstein2 = scipy.stats.wasserstein_distance(ang0[mask2].ravel(), ang2[mask2].ravel())
        print('angle')
        print('mode{:01d}:\t{:0.8}'.format(i_mode1 + 1, wasserstein1))
        print('mode{:01d}:\t{:0.8}'.format(i_mode2 + 1, wasserstein2))
        print(np.max(ang0[~np.isnan(ang0)]), np.max(ang1[~np.isnan(ang1)]), np.max(ang2[~np.isnan(ang2)]))

        # get correct metric
        acc1 = np.sqrt(np.sum(acc1 ** 2, 2))
        acc2 = np.sqrt(np.sum(acc2 ** 2, 2))
        velo1 = np.sqrt(np.sum(velo1 ** 2, 2))
        velo2 = np.sqrt(np.sum(velo2 ** 2, 2))
        pos1 = np.sqrt(np.sum((pos1 - pos0)[:, :, :2] ** 2, 2))  # should use only xy-positions here
        pos2 = np.sqrt(np.sum((pos2 - pos0)[:, :, :2] ** 2, 2))  # should use only xy-positions here
        ang1 = np.min(np.stack([abs(ang1 - ang0),
                                360.0 - abs(ang1) - abs(ang0)], 0), 0)
        ang2 = np.min(np.stack([abs(ang2 - ang0),
                                360.0 - abs(ang2) - abs(ang0)], 0), 0)

        acc1 = acc1.ravel()
        acc2 = acc2.ravel()
        velo1 = velo1.ravel()
        velo2 = velo2.ravel()
        pos1 = pos1.ravel()
        pos2 = pos2.ravel()
        ang1 = ang1.ravel()
        ang2 = ang2.ravel()

        stat_acc_wsr, p_acc_wsr = scipy.stats.wilcoxon(acc1[np.logical_and(~np.isnan(acc1), ~np.isnan(acc2))],
                                                       acc2[np.logical_and(~np.isnan(acc1), ~np.isnan(acc2))],
                                                       alternative='less', mode='auto')
        n_acc_wsr = np.sum(np.logical_and(~np.isnan(acc1), ~np.isnan(acc2)))
        stat_velo_wsr, p_velo_wsr = scipy.stats.wilcoxon(velo1[np.logical_and(~np.isnan(velo1), ~np.isnan(velo2))],
                                                         velo2[np.logical_and(~np.isnan(velo1), ~np.isnan(velo2))],
                                                         alternative='less', mode='auto')
        n_velo_wsr = np.sum(np.logical_and(~np.isnan(velo1), ~np.isnan(velo2)))
        stat_pos_wsr, p_pos_wsr = scipy.stats.wilcoxon(pos1[np.logical_and(~np.isnan(pos1), ~np.isnan(pos2))],
                                                       pos2[np.logical_and(~np.isnan(pos1), ~np.isnan(pos2))],
                                                       alternative='less', mode='auto')
        n_pos_wsr = np.sum(np.logical_and(~np.isnan(pos1), ~np.isnan(pos2)))
        stat_ang_wsr, p_ang_wsr = scipy.stats.wilcoxon(ang1[np.logical_and(~np.isnan(ang1), ~np.isnan(ang2))],
                                                       ang2[np.logical_and(~np.isnan(ang1), ~np.isnan(ang2))],
                                                       alternative='less', mode='auto')
        n_ang_wsr = np.sum(np.logical_and(~np.isnan(ang1), ~np.isnan(ang2)))

        acc1 = acc1[~np.isnan(acc1)]
        velo1 = velo1[~np.isnan(velo1)]
        pos1 = pos1[~np.isnan(pos1)]
        ang1 = ang1[~np.isnan(ang1)]
        acc2 = acc2[~np.isnan(acc2)]
        velo2 = velo2[~np.isnan(velo2)]
        pos2 = pos2[~np.isnan(pos2)]
        ang2 = ang2[~np.isnan(ang2)]
        #
        n_acc1 = np.float64(len(acc1))
        n_velo1 = np.float64(len(velo1))
        n_pos1 = np.float64(len(pos1))
        n_ang1 = np.float64(len(ang1))
        n_acc2 = np.float64(len(acc2))
        n_velo2 = np.float64(len(velo2))
        n_pos2 = np.float64(len(pos2))
        n_ang2 = np.float64(len(ang2))

        hist_acc1 = np.histogram(acc1, bins=nBin_acc, range=[0.0, maxBin_acc0], normed=None, weights=None,
                                 density=False)
        hist_velo1 = np.histogram(velo1, bins=nBin_velo, range=[0.0, maxBin_velo0], normed=None, weights=None,
                                  density=False)
        hist_pos1 = np.histogram(pos1, bins=nBin_pos, range=[0.0, maxBin_pos0], normed=None, weights=None,
                                 density=False)
        hist_ang1 = np.histogram(ang1, bins=nBin_ang, range=[0.0, maxBin_ang0], normed=None, weights=None,
                                 density=False)
        hist_acc2 = np.histogram(acc2, bins=nBin_acc, range=[0.0, maxBin_acc0], normed=None, weights=None,
                                 density=False)
        hist_velo2 = np.histogram(velo2, bins=nBin_velo, range=[0.0, maxBin_velo0], normed=None, weights=None,
                                  density=False)
        hist_pos2 = np.histogram(pos2, bins=nBin_pos, range=[0.0, maxBin_pos0], normed=None, weights=None,
                                 density=False)
        hist_ang2 = np.histogram(ang2, bins=nBin_ang, range=[0.0, maxBin_ang0], normed=None, weights=None,
                                 density=False)

        cumfun_acc1 = np.cumsum(hist_acc1[0] / n_acc1)
        cumfun_acc2 = np.cumsum(hist_acc2[0] / n_acc2)
        cumfun_velo1 = np.cumsum(hist_velo1[0] / n_velo1)
        cumfun_velo2 = np.cumsum(hist_velo2[0] / n_velo2)
        cumfun_pos1 = np.cumsum(hist_pos1[0] / n_pos1)
        cumfun_pos2 = np.cumsum(hist_pos2[0] / n_pos2)
        cumfun_ang1 = np.cumsum(hist_ang1[0] / n_ang1)
        cumfun_ang2 = np.cumsum(hist_ang2[0] / n_ang2)

        #         cumfun_acc1_index = np.argwhere(cumfun_acc1 <= 0.5)[-1]
        #         cumfun_acc2_index = np.argwhere(cumfun_acc2 <= 0.5)[-1]
        #         cumfun_velo1_index = np.argwhere(cumfun_velo1 <= 0.5)[-1]
        #         cumfun_velo2_index = np.argwhere(cumfun_velo2 <= 0.5)[-1]
        #         cumfun_pos1_index = np.argwhere(cumfun_pos1 <= 0.5)[-1]
        #         cumfun_pos2_index = np.argwhere(cumfun_pos2 <= 0.5)[-1]
        #         cumfun_ang1_index = np.argwhere(cumfun_ang1 <= 0.5)[-1]
        #         cumfun_ang2_index = np.argwhere(cumfun_ang2 <= 0.5)[-1]
        #         median_acc1 = (hist_acc1[1][:-1] + 0.5 * dBin_acc)[cumfun_acc1_index][0] * conversion_fac_acc
        #         median_acc2 = (hist_acc2[1][:-1] + 0.5 * dBin_acc)[cumfun_acc2_index][0] * conversion_fac_acc
        #         median_velo1 = (hist_velo1[1][:-1] + 0.5 * dBin_velo)[cumfun_velo1_index][0] * conversion_fac_velo
        #         median_velo2 = (hist_velo2[1][:-1] + 0.5 * dBin_velo)[cumfun_velo2_index][0] * conversion_fac_velo
        #         median_pos1 = (hist_pos1[1][:-1] + 0.5 * dBin_pos)[cumfun_pos1_index][0] * conversion_fac_pos
        #         median_pos2 = (hist_pos2[1][:-1] + 0.5 * dBin_pos)[cumfun_pos2_index][0] * conversion_fac_pos
        #         median_ang1 = (hist_ang1[1][:-1] + 0.5 * dBin_ang)[cumfun_ang1_index][0] * conversion_fac_ang
        #         median_ang2 = (hist_ang2[1][:-1] + 0.5 * dBin_ang)[cumfun_ang2_index][0] * conversion_fac_ang

        alternative = 'greater'
        mode = 'auto'
        #         mode = 'asymp'
        stat_acc, p_acc = scipy.stats.ks_2samp(acc1, acc2, alternative=alternative, mode=mode)
        stat_velo, p_velo = scipy.stats.ks_2samp(velo1, velo2, alternative=alternative, mode=mode)
        stat_pos, p_pos = scipy.stats.ks_2samp(pos1, pos2, alternative=alternative, mode=mode)
        stat_ang, p_ang = scipy.stats.ks_2samp(ang1, ang2, alternative=alternative, mode=mode)

        print('mode pair:')
        print('mode{:01d} / mode{:01d}'.format(i_mode1 + 1, i_mode2 + 1))
        print('nAcceleration:')
        print(int(n_acc1), int(n_acc2))
        print('nVelocity:')
        print(int(n_velo1), int(n_velo2))
        print('nPosition:')
        print(int(n_pos1), int(n_pos2))
        print('nAngle:')
        print(int(n_ang1), int(n_ang2))
        #
        print('acceleration ({:s}):'.format(acc_unit))
        print(
            'avg.:\t\t{:0.8f}\t{:0.8f}'.format(np.mean(acc1 * conversion_fac_acc), np.mean(acc2 * conversion_fac_acc)))
        print('sd.:\t\t{:0.8f}\t{:0.8f}'.format(np.std(acc1 * conversion_fac_acc), np.std(acc2 * conversion_fac_acc)))
        #         print('median:\t\t{:0.8f}\t{:0.8f}'.format(median_acc1, median_acc2))
        print('median:\t\t{:0.8f}\t{:0.8f}'.format(np.median(acc1 * conversion_fac_acc),
                                                   np.median(acc2 * conversion_fac_acc)))
        print('max.:\t\t{:0.8f}\t{:0.8f}'.format(np.max(acc1 * conversion_fac_acc), np.max(acc2 * conversion_fac_acc)))
        print('min.:\t\t{:0.8f}\t{:0.8f}'.format(np.min(acc1 * conversion_fac_acc), np.min(acc2 * conversion_fac_acc)))
        print('cumfun:\t\t{:0.8f}\t{:0.8f}'.format(cumfun_acc1[-1], cumfun_acc2[-1]))
        print('1-cumfun:\t{:0.8f}\t{:0.8f}'.format(1.0 - cumfun_acc1[-1], 1.0 - cumfun_acc2[-1]))
        print('stat={:0.8f}, p={:0.15e}'.format(stat_acc, p_acc))
        print('stat_wsr={:0.8f}, p_wsr={:0.15e} ({:d})'.format(stat_acc_wsr, p_acc_wsr, n_acc_wsr))
        if p_acc > 0.05:
            print('Probably the same distribution')
        else:
            print('Probably different distributions')
        print('velocity ({:s}):'.format(velo_unit))
        print('avg.:\t\t{:0.8f}\t{:0.8f}'.format(np.mean(velo1 * conversion_fac_velo),
                                                 np.mean(velo2 * conversion_fac_velo)))
        print(
            'sd.:\t\t{:0.8f}\t{:0.8f}'.format(np.std(velo1 * conversion_fac_velo), np.std(velo2 * conversion_fac_velo)))
        #         print('median:\t\t{:0.8f}\t{:0.8f}'.format(median_velo1, median_velo2))
        print('median:\t\t{:0.8f}\t{:0.8f}'.format(np.median(velo1 * conversion_fac_velo),
                                                   np.median(velo2 * conversion_fac_velo)))
        print('max.:\t\t{:0.8f}\t{:0.8f}'.format(np.max(velo1 * conversion_fac_velo),
                                                 np.max(velo2 * conversion_fac_velo)))
        print('min.:\t\t{:0.8f}\t{:0.8f}'.format(np.min(velo1 * conversion_fac_velo),
                                                 np.min(velo2 * conversion_fac_velo)))
        print('cumfun:\t\t{:0.8f}\t{:0.8f}'.format(cumfun_velo1[-1], cumfun_velo2[-1]))
        print('1-cumfun:\t{:0.8f}\t{:0.8f}'.format(1.0 - cumfun_velo1[-1], 1.0 - cumfun_velo2[-1]))
        print('stat={:0.8f}, p={:0.15e}'.format(stat_velo, p_velo))
        print('stat_wsr={:0.8f}, p_wsr={:0.15e} ({:d})'.format(stat_velo_wsr, p_velo_wsr, n_velo_wsr))
        if p_velo > 0.05:
            print('Probably the same distribution')
        else:
            print('Probably different distributions')
        print('position ({:s}):'.format(pos_unit))
        print(
            'avg.:\t\t{:0.8f}\t{:0.8f}'.format(np.mean(pos1 * conversion_fac_pos), np.mean(pos2 * conversion_fac_pos)))
        print('sd.:\t\t{:0.8f}\t{:0.8f}'.format(np.std(pos1 * conversion_fac_pos), np.std(pos2 * conversion_fac_pos)))
        #         print('median:\t\t{:0.8f}\t{:0.8f}'.format(median_pos1, median_pos2))
        print('median:\t\t{:0.8f}\t{:0.8f}'.format(np.median(pos1 * conversion_fac_pos),
                                                   np.median(pos2 * conversion_fac_pos)))
        print('max.:\t\t{:0.8f}\t{:0.8f}'.format(np.max(pos1 * conversion_fac_pos), np.max(pos2 * conversion_fac_pos)))
        print('min.:\t\t{:0.8f}\t{:0.8f}'.format(np.min(pos1 * conversion_fac_pos), np.min(pos2 * conversion_fac_pos)))
        print('cumfun:\t\t{:0.8f}\t{:0.8f}'.format(cumfun_pos1[-1], cumfun_pos2[-1]))
        print('1-cumfun:\t{:0.8f}\t{:0.8f}'.format(1.0 - cumfun_pos1[-1], 1.0 - cumfun_pos2[-1]))
        print('stat={:0.8f}, p={:0.15e}'.format(stat_pos, p_pos))
        print('stat_wsr={:0.8f}, p_wsr={:0.15e} ({:d})'.format(stat_pos_wsr, p_pos_wsr, n_pos_wsr))
        if p_pos > 0.05:
            print('Probably the same distribution')
        else:
            print('Probably different distributions')
        print('angle ({:s}):'.format(ang_unit))
        print(
            'avg.:\t\t{:0.8f}\t{:0.8f}'.format(np.mean(ang1 * conversion_fac_ang), np.mean(ang2 * conversion_fac_ang)))
        print('sd.:\t\t{:0.8f}\t{:0.8f}'.format(np.std(ang1 * conversion_fac_ang), np.std(ang2 * conversion_fac_ang)))
        #         print('median:\t\t{:0.8f}\t{:0.8f}'.format(median_ang1, median_ang2))
        print('median:\t\t{:0.8f}\t{:0.8f}'.format(np.median(ang1 * conversion_fac_ang),
                                                   np.median(ang2 * conversion_fac_ang)))
        print('max.:\t\t{:0.8f}\t{:0.8f}'.format(np.max(ang1 * conversion_fac_ang), np.max(ang2 * conversion_fac_ang)))
        print('min.:\t\t{:0.8f}\t{:0.8f}'.format(np.min(ang1 * conversion_fac_ang), np.min(ang2 * conversion_fac_ang)))
        print('cumfun:\t\t{:0.8f}\t{:0.8f}'.format(cumfun_ang1[-1], cumfun_ang2[-1]))
        print('1-cumfun:\t{:0.8f}\t{:0.8f}'.format(1.0 - cumfun_ang1[-1], 1.0 - cumfun_ang2[-1]))
        print('stat={:0.8f}, p={:0.15e}'.format(stat_ang, p_ang))
        print('stat_wsr={:0.8f}, p_wsr={:0.15e} ({:d})'.format(stat_ang_wsr, p_ang_wsr, n_ang_wsr))
        if p_ang > 0.05:
            print('Probably the same distribution')
        else:
            print('Probably different distributions')
        print()

        # plot
        acc1[acc1 > maxBin_acc0] = maxBin_acc
        velo1[velo1 > maxBin_velo0] = maxBin_velo
        pos1[pos1 > maxBin_pos0] = maxBin_pos
        ang1[ang1 > maxBin_ang0] = maxBin_ang
        acc2[acc2 > maxBin_acc0] = maxBin_acc
        velo2[velo2 > maxBin_velo0] = maxBin_velo
        pos2[pos2 > maxBin_pos0] = maxBin_pos
        ang2[ang2 > maxBin_ang0] = maxBin_ang

        hist_acc1_use = np.histogram(acc1, bins=nBin_acc, range=[bin_range_acc[0], bin_range_acc[-1]], normed=None,
                                     weights=None, density=False)
        hist_velo1_use = np.histogram(velo1, bins=nBin_velo, range=[bin_range_velo[0], bin_range_velo[-1]], normed=None,
                                      weights=None, density=False)
        hist_pos1_use = np.histogram(pos1, bins=nBin_pos, range=[bin_range_pos[0], bin_range_pos[-1]], normed=None,
                                     weights=None, density=False)
        hist_ang1_use = np.histogram(ang1, bins=nBin_ang, range=[bin_range_ang[0], bin_range_ang[-1]], normed=None,
                                     weights=None, density=False)
        hist_acc2_use = np.histogram(acc2, bins=nBin_acc, range=[bin_range_acc[0], bin_range_acc[-1]], normed=None,
                                     weights=None, density=False)
        hist_velo2_use = np.histogram(velo2, bins=nBin_velo, range=[bin_range_velo[0], bin_range_velo[-1]], normed=None,
                                      weights=None, density=False)
        hist_pos2_use = np.histogram(pos2, bins=nBin_pos, range=[bin_range_pos[0], bin_range_pos[-1]], normed=None,
                                     weights=None, density=False)
        hist_ang2_use = np.histogram(ang2, bins=nBin_ang, range=[bin_range_ang[0], bin_range_ang[-1]], normed=None,
                                     weights=None, density=False)

        hist_acc1_use[0][bin_range_acc[1:] > maxBin_acc0] = hist_acc1_use[0][-1]
        hist_velo1_use[0][bin_range_velo[1:] > maxBin_velo0] = hist_velo1_use[0][-1]
        hist_pos1_use[0][bin_range_pos[1:] > maxBin_pos0] = hist_pos1_use[0][-1]
        hist_ang1_use[0][bin_range_ang[1:] > maxBin_ang0] = hist_ang1_use[0][-1]
        hist_acc2_use[0][bin_range_acc[1:] > maxBin_acc0] = hist_acc2_use[0][-1]
        hist_velo2_use[0][bin_range_velo[1:] > maxBin_velo0] = hist_velo2_use[0][-1]
        hist_pos2_use[0][bin_range_pos[1:] > maxBin_pos0] = hist_pos2_use[0][-1]
        hist_ang2_use[0][bin_range_ang[1:] > maxBin_ang0] = hist_ang2_use[0][-1]

        #         y_acc1 = np.zeros(1+2*len(hist_acc1[0]), dtype=np.float64)
        #         y_acc1[1::2] = np.copy(hist_acc1[0] / n_acc1)
        #         y_acc1[2::2] = np.copy(hist_acc1[0] / n_acc1)
        #         x_acc1 = np.zeros(1+2*(len(hist_acc1[1])-1), dtype=np.float64)
        #         x_acc1[0] = np.copy(hist_acc1[1][0])
        #         x_acc1[1::2] = np.copy(hist_acc1[1][:-1])
        #         x_acc1[2::2] = np.copy(hist_acc1[1][1:])
        #         y_acc2 = np.zeros(1+2*len(hist_acc2[0]), dtype=np.float64)
        #         y_acc2[1::2] = np.copy(hist_acc2[0] / n_acc2)
        #         y_acc2[2::2] = np.copy(hist_acc2[0] / n_acc2)
        #         x_acc2 = np.zeros(1+2*(len(hist_acc2[1])-1), dtype=np.float64)
        #         x_acc2[0] = np.copy(hist_acc2[1][0])
        #         x_acc2[1::2] = np.copy(hist_acc2[1][:-1])
        #         x_acc2[2::2] = np.copy(hist_acc2[1][1:])
        #         #
        #         y_pos1 = np.zeros(1+2*len(hist_pos1[0]), dtype=np.float64)
        #         y_pos1[1::2] = np.copy(hist_pos1[0] / n_pos1)
        #         y_pos1[2::2] = np.copy(hist_pos1[0] / n_pos1)
        #         x_pos1 = np.zeros(1+2*(len(hist_pos1[1])-1), dtype=np.float64)
        #         x_pos1[0] = np.copy(hist_pos1[1][0])
        #         x_pos1[1::2] = np.copy(hist_pos1[1][:-1])
        #         x_pos1[2::2] = np.copy(hist_pos1[1][1:])
        #         y_pos2 = np.zeros(1+2*len(hist_pos2[0]), dtype=np.float64)
        #         y_pos2[1::2] = np.copy(hist_pos2[0] / n_pos2)
        #         y_pos2[2::2] = np.copy(hist_pos2[0] / n_pos2)
        #         x_pos2 = np.zeros(1+2*(len(hist_pos2[1])-1), dtype=np.float64)
        #         x_pos2[0] = np.copy(hist_pos2[1][0])
        #         x_pos2[1::2] = np.copy(hist_pos2[1][:-1])
        #         x_pos2[2::2] = np.copy(hist_pos2[1][1:])
        #         #
        #         y_ang1 = np.zeros(1+2*len(hist_ang1[0]), dtype=np.float64)
        #         y_ang1[1::2] = np.copy(hist_ang1[0] / n_ang1)
        #         y_ang1[2::2] = np.copy(hist_ang1[0] / n_ang1)
        #         x_ang1 = np.zeros(1+2*(len(hist_ang1[1])-1), dtype=np.float64)
        #         x_ang1[0] = np.copy(hist_ang1[1][0])
        #         x_ang1[1::2] = np.copy(hist_ang1[1][:-1])
        #         x_ang1[2::2] = np.copy(hist_ang1[1][1:])
        #         y_ang2 = np.zeros(1+2*len(hist_ang2[0]), dtype=np.float64)
        #         y_ang2[1::2] = np.copy(hist_ang2[0] / n_ang2)
        #         y_ang2[2::2] = np.copy(hist_ang2[0] / n_ang2)
        #         x_ang2 = np.zeros(1+2*(len(hist_ang2[1])-1), dtype=np.float64)
        #         x_ang2[0] = np.copy(hist_ang2[1][0])
        #         x_ang2[1::2] = np.copy(hist_ang2[1][:-1])
        #         x_ang2[2::2] = np.copy(hist_ang2[1][1:])
        y_acc1 = np.zeros(2 + 2 * len(hist_acc1_use[0]), dtype=np.float64)
        y_acc1[0] = 0.0
        y_acc1[1:-1:2] = np.copy(hist_acc1_use[0] / n_acc1)
        y_acc1[2:-1:2] = np.copy(hist_acc1_use[0] / n_acc1)
        y_acc1[-1] = 0.0
        x_acc1 = np.zeros(2 + 2 * (len(hist_acc1_use[1]) - 1), dtype=np.float64)
        x_acc1[0] = np.copy(hist_acc1_use[1][0])
        x_acc1[1:-1:2] = np.copy(hist_acc1_use[1][:-1])
        x_acc1[2:-1:2] = np.copy(hist_acc1_use[1][1:])
        x_acc1[-2] = maxBin_acc
        x_acc1[-1] = maxBin_acc
        y_acc2 = np.zeros(2 + 2 * len(hist_acc2_use[0]), dtype=np.float64)
        y_acc2[0] = 0.0
        y_acc2[1:-1:2] = np.copy(hist_acc2_use[0] / n_acc2)
        y_acc2[2:-1:2] = np.copy(hist_acc2_use[0] / n_acc2)
        y_acc2[-1] = 0.0
        x_acc2 = np.zeros(2 + 2 * (len(hist_acc2_use[1]) - 1), dtype=np.float64)
        x_acc2[0] = np.copy(hist_acc2_use[1][0])
        x_acc2[1:-1:2] = np.copy(hist_acc2_use[1][:-1])
        x_acc2[2:-1:2] = np.copy(hist_acc2_use[1][1:])
        x_acc2[-2] = maxBin_acc
        x_acc2[-1] = maxBin_acc
        #
        y_velo1 = np.zeros(2 + 2 * len(hist_velo1_use[0]), dtype=np.float64)
        y_velo1[0] = 0.0
        y_velo1[1:-1:2] = np.copy(hist_velo1_use[0] / n_velo1)
        y_velo1[2:-1:2] = np.copy(hist_velo1_use[0] / n_velo1)
        y_velo1[-1] = 0.0
        x_velo1 = np.zeros(2 + 2 * (len(hist_velo1_use[1]) - 1), dtype=np.float64)
        x_velo1[0] = np.copy(hist_velo1_use[1][0])
        x_velo1[1:-1:2] = np.copy(hist_velo1_use[1][:-1])
        x_velo1[2:-1:2] = np.copy(hist_velo1_use[1][1:])
        x_velo1[-2] = maxBin_velo
        x_velo1[-1] = maxBin_velo
        y_velo2 = np.zeros(2 + 2 * len(hist_velo2_use[0]), dtype=np.float64)
        y_velo2[0] = 0.0
        y_velo2[1:-1:2] = np.copy(hist_velo2_use[0] / n_velo2)
        y_velo2[2:-1:2] = np.copy(hist_velo2_use[0] / n_velo2)
        y_velo2[-1] = 0.0
        x_velo2 = np.zeros(2 + 2 * (len(hist_velo2_use[1]) - 1), dtype=np.float64)
        x_velo2[0] = np.copy(hist_velo2_use[1][0])
        x_velo2[1:-1:2] = np.copy(hist_velo2_use[1][:-1])
        x_velo2[2:-1:2] = np.copy(hist_velo2_use[1][1:])
        x_velo2[-2] = maxBin_velo
        x_velo2[-1] = maxBin_velo
        #
        y_pos1 = np.zeros(2 + 2 * len(hist_pos1_use[0]), dtype=np.float64)
        y_pos1[0] = 0.0
        y_pos1[1:-1:2] = np.copy(hist_pos1_use[0] / n_pos1)
        y_pos1[2:-1:2] = np.copy(hist_pos1_use[0] / n_pos1)
        y_pos1[-1] = 0.0
        x_pos1 = np.zeros(2 + 2 * (len(hist_pos1_use[1]) - 1), dtype=np.float64)
        x_pos1[0] = np.copy(hist_pos1_use[1][0])
        x_pos1[1:-1:2] = np.copy(hist_pos1_use[1][:-1])
        x_pos1[2:-1:2] = np.copy(hist_pos1_use[1][1:])
        x_pos1[-2] = maxBin_pos
        x_pos1[-1] = maxBin_pos
        y_pos2 = np.zeros(2 + 2 * len(hist_pos2_use[0]), dtype=np.float64)
        y_pos2[0] = 0.0
        y_pos2[1:-1:2] = np.copy(hist_pos2_use[0] / n_pos2)
        y_pos2[2:-1:2] = np.copy(hist_pos2_use[0] / n_pos2)
        y_pos2[-1] = 0.0
        x_pos2 = np.zeros(2 + 2 * (len(hist_pos2_use[1]) - 1), dtype=np.float64)
        x_pos2[0] = np.copy(hist_pos2_use[1][0])
        x_pos2[1:-1:2] = np.copy(hist_pos2_use[1][:-1])
        x_pos2[2:-1:2] = np.copy(hist_pos2_use[1][1:])
        x_pos2[-2] = maxBin_pos
        x_pos2[-1] = maxBin_pos
        #
        y_ang1 = np.zeros(2 + 2 * len(hist_ang1_use[0]), dtype=np.float64)
        y_ang1[0] = 0.0
        y_ang1[1:-1:2] = np.copy(hist_ang1_use[0] / n_ang1)
        y_ang1[2:-1:2] = np.copy(hist_ang1_use[0] / n_ang1)
        y_ang1[-1] = 0.0
        x_ang1 = np.zeros(2 + 2 * (len(hist_ang1_use[1]) - 1), dtype=np.float64)
        x_ang1[0] = np.copy(hist_ang1_use[1][0])
        x_ang1[1:-1:2] = np.copy(hist_ang1_use[1][:-1])
        x_ang1[2:-1:2] = np.copy(hist_ang1_use[1][1:])
        x_ang1[-2] = maxBin_ang
        x_ang1[-1] = maxBin_ang
        y_ang2 = np.zeros(2 + 2 * len(hist_ang2_use[0]), dtype=np.float64)
        y_ang2[0] = 0.0
        y_ang2[1:-1:2] = np.copy(hist_ang2_use[0] / n_ang2)
        y_ang2[2:-1:2] = np.copy(hist_ang2_use[0] / n_ang2)
        y_ang2[-1] = 0.0
        x_ang2 = np.zeros(2 + 2 * (len(hist_ang2_use[1]) - 1), dtype=np.float64)
        x_ang2[0] = np.copy(hist_ang2_use[1][0])
        x_ang2[1:-1:2] = np.copy(hist_ang2_use[1][:-1])
        x_ang2[2:-1:2] = np.copy(hist_ang2_use[1][1:])
        x_ang2[-2] = maxBin_ang
        x_ang2[-1] = maxBin_ang

        if (show or save):
            if not (modes_ploted[i_mode1]):
                modes_ploted[i_mode1] = True
                ax_acc_all.plot(x_acc1 * conversion_fac_acc, y_acc1, color=colors[i_mode1], linestyle='-', marker='',
                                zorder=-4 + mode_pair[0],
                                linewidth=linewidth_hist, clip_on=False)
                ax_velo_all.plot(x_velo1 * conversion_fac_velo, y_velo1, color=colors[i_mode1], linestyle='-',
                                 marker='', zorder=-4 + mode_pair[0],
                                 linewidth=linewidth_hist, clip_on=False)
                ax_pos_all.plot(x_pos1 * conversion_fac_pos, y_pos1, color=colors[i_mode1], linestyle='-', marker='',
                                zorder=-4 + mode_pair[0],
                                linewidth=linewidth_hist, clip_on=False)
                ax_ang_all.plot(x_ang1 * conversion_fac_ang, y_ang1, color=colors[i_mode1], linestyle='-', marker='',
                                zorder=-4 + mode_pair[0],
                                linewidth=linewidth_hist, clip_on=False)
                pos1
            if not (modes_ploted[i_mode2]):
                modes_ploted[i_mode2] = True
                ax_acc_all.plot(x_acc2 * conversion_fac_acc, y_acc2, color=colors[i_mode2], linestyle='-', marker='',
                                zorder=-4 + mode_pair[1],
                                linewidth=linewidth_hist, clip_on=False)
                ax_velo_all.plot(x_velo2 * conversion_fac_velo, y_velo2, color=colors[i_mode2], linestyle='-',
                                 marker='', zorder=-4 + mode_pair[1],
                                 linewidth=linewidth_hist, clip_on=False)
                ax_pos_all.plot(x_pos2 * conversion_fac_pos, y_pos2, color=colors[i_mode2], linestyle='-', marker='',
                                zorder=-4 + mode_pair[1],
                                linewidth=linewidth_hist, clip_on=False)
                ax_ang_all.plot(x_ang2 * conversion_fac_ang, y_ang2, color=colors[i_mode2], linestyle='-', marker='',
                                zorder=-4 + mode_pair[1],
                                linewidth=linewidth_hist, clip_on=False)

    # all
    offset_x_fac = 0.0  # 0.5
    line_fac = 2.5e-2
    box_h_fac = 1.0 + 0.01
    #
    #     h_legend = ax_acc_all.legend(list(['anatomical & temporal', 'temporal', 'anatomical', 'none']),
    #                                  loc='upper right', frameon=False, fontsize=fontsize)
    #     h_legend.set_zorder(200)
    #     for text in h_legend.get_texts():
    #         text.set_fontname(fontname)
    ax_acc_all.set_xlabel('acceleration ({:s})'.format(acc_unit), va='bottom', ha='center', fontsize=fontsize,
                          fontname=fontname)
    ax_acc_all.set_ylabel('probability', va='top', ha='center', fontsize=fontsize, fontname=fontname)
    ax_acc_all.set_xlim([(bin_range_acc[0] - offset_x_fac * dBin_acc) * conversion_fac_acc,
                         (bin_range_acc[-1] + offset_x_fac * dBin_acc) * conversion_fac_acc])
    ax_acc_all.set_xticks([0.0, maxBin_acc0 * conversion_fac_acc * 0.5, maxBin_acc0 * conversion_fac_acc,
                           maxBin_acc * conversion_fac_acc])
    ax_acc_all.set_ylim([0.0, 0.3])
    ax_acc_all.set_yticks([0.0, 0.16, 0.32])
    ax_acc_all.set_yticklabels([0, 0.16, 0.32])
    #     ax_acc_all.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
    labels = list(['{:0.2f}'.format(i) for i in ax_acc_all.get_xticks()])
    labels[-1] = 'inf'
    labels[0] = 0
    ax_acc_all.set_xticklabels(labels)
    ax_acc_all.xaxis.get_offset_text().set_fontsize(fontsize)
    ax_acc_all.yaxis.get_offset_text().set_fontsize(fontsize)
    for tick in ax_acc_all.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax_acc_all.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    ax_acc_all.xaxis.set_label_coords(x=ax_acc_all_x + 0.5 * ax_acc_all_w, y=bottom_margin_x / fig_acc_all_h,
                                      transform=fig_acc_all.transFigure)
    ax_acc_all.yaxis.set_label_coords(x=left_margin_x / fig_acc_all_w, y=ax_acc_all_y + 0.5 * ax_acc_all_h,
                                      transform=fig_acc_all.transFigure)
    #
    line_w = np.diff(ax_acc_all.get_xlim()) * line_fac * 1 / 3
    line_h = np.diff(ax_acc_all.get_ylim()) * line_fac
    box_w = (maxBin_acc - maxBin_acc0) * conversion_fac_acc * 0.2
    box_h = abs(np.diff(ax_acc_all.get_ylim())) * box_h_fac
    ax_acc_all.plot(np.array([-line_w * 0.5, +line_w * 0.5], dtype=np.float64) + maxBin_acc0 * conversion_fac_acc * (
                1.0 + inf_fac * 0.5) - box_w * 0.5,
                    np.array([-line_h * 0.5, +line_h * 0.5], dtype=np.float64) + ax_acc_all.get_ylim()[0],
                    linewidth=ax_acc_all.spines['left'].get_linewidth(), linestyle='-', color='black', zorder=101,
                    clip_on=False)
    ax_acc_all.plot(np.array([-line_w * 0.5, +line_w * 0.5], dtype=np.float64) + maxBin_acc0 * conversion_fac_acc * (
                1.0 + inf_fac * 0.5) + box_w * 0.5,
                    np.array([-line_h * 0.5, +line_h * 0.5], dtype=np.float64) + ax_acc_all.get_ylim()[0],
                    linewidth=ax_acc_all.spines['bottom'].get_linewidth(), linestyle='-', color='black', zorder=101,
                    clip_on=False)
    box = plt.Rectangle((maxBin_acc0 * conversion_fac_acc * (1.0 + inf_fac * 0.5) - box_w * 0.5,
                         ax_acc_all.get_ylim()[0] + abs(np.diff(ax_acc_all.get_ylim())) * 0.5 - box_h * 0.5),
                        box_w, box_h,
                        color='white', zorder=100, clip_on=False)
    ax_acc_all.add_patch(box)
    #
    fig_acc_all.canvas.draw()
    #
    #     h_legend = ax_velo_all.legend(list(['anatomical & temporal', 'temporal', 'anatomical', 'none']),
    #                        loc='upper right', frameon=False, fontsize=fontsize)
    #     h_legend.set_zorder(200)
    #     for text in h_legend.get_texts():
    #         text.set_fontname(fontname)
    ax_velo_all.set_xlabel('velocity ({:s})'.format(velo_unit), va='bottom', ha='center', fontsize=fontsize,
                           fontname=fontname)
    ax_velo_all.set_ylabel('probability', va='top', ha='center', fontsize=fontsize, fontname=fontname)
    ax_velo_all.set_xlim([(bin_range_velo[0] - offset_x_fac * dBin_velo) * conversion_fac_velo,
                          (bin_range_velo[-1] + offset_x_fac * dBin_velo) * conversion_fac_velo])
    ax_velo_all.set_xticks([0.0, maxBin_velo0 * conversion_fac_velo * 0.5, maxBin_velo0 * conversion_fac_velo,
                            maxBin_velo * conversion_fac_velo])
    ax_velo_all.set_ylim([0.0, 0.32])
    ax_velo_all.set_yticks([0.0, 0.16, 0.32])
    ax_velo_all.set_yticklabels([0, 0.16, 0.32])
    #     ax_velo_all.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
    labels = list(['{:0.2f}'.format(i) for i in ax_velo_all.get_xticks()])
    labels[-1] = 'inf'
    labels[0] = 0
    ax_velo_all.set_xticklabels(labels)
    ax_velo_all.xaxis.get_offset_text().set_fontsize(fontsize)
    ax_velo_all.yaxis.get_offset_text().set_fontsize(fontsize)
    for tick in ax_velo_all.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax_velo_all.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    ax_velo_all.xaxis.set_label_coords(x=ax_velo_all_x + 0.5 * ax_velo_all_w, y=bottom_margin_x / fig_velo_all_h,
                                       transform=fig_velo_all.transFigure)
    ax_velo_all.yaxis.set_label_coords(x=left_margin_x / fig_velo_all_w, y=ax_velo_all_y + 0.5 * ax_velo_all_h,
                                       transform=fig_velo_all.transFigure)
    #
    line_w = np.diff(ax_velo_all.get_xlim()) * line_fac * 1 / 3
    line_h = np.diff(ax_velo_all.get_ylim()) * line_fac
    box_w = (maxBin_velo - maxBin_velo0) * conversion_fac_velo * 0.2
    box_h = abs(np.diff(ax_velo_all.get_ylim())) * box_h_fac
    ax_velo_all.plot(np.array([-line_w * 0.5, +line_w * 0.5], dtype=np.float64) + maxBin_velo0 * conversion_fac_velo * (
                1.0 + inf_fac * 0.5) - box_w * 0.5,
                     np.array([-line_h * 0.5, +line_h * 0.5], dtype=np.float64) + ax_velo_all.get_ylim()[0],
                     linewidth=ax_velo_all.spines['left'].get_linewidth(), linestyle='-', color='black', zorder=101,
                     clip_on=False)
    ax_velo_all.plot(np.array([-line_w * 0.5, +line_w * 0.5], dtype=np.float64) + maxBin_velo0 * conversion_fac_velo * (
                1.0 + inf_fac * 0.5) + box_w * 0.5,
                     np.array([-line_h * 0.5, +line_h * 0.5], dtype=np.float64) + ax_velo_all.get_ylim()[0],
                     linewidth=ax_velo_all.spines['bottom'].get_linewidth(), linestyle='-', color='black', zorder=101,
                     clip_on=False)
    box = plt.Rectangle((maxBin_velo0 * conversion_fac_velo * (1.0 + inf_fac * 0.5) - box_w * 0.5,
                         ax_velo_all.get_ylim()[0] + abs(np.diff(ax_velo_all.get_ylim())) * 0.5 - box_h * 0.5),
                        box_w, box_h,
                        color='white', zorder=100, clip_on=False)
    ax_velo_all.add_patch(box)
    #
    fig_velo_all.canvas.draw()
    #
    #     h_legend = ax_pos_all.legend(list(['anatomical & temporal', 'temporal', 'anatomical', 'none']),
    #                        loc='upper right', frameon=False, fontsize=fontsize)
    #     h_legend.set_zorder(200)
    #     for text in h_legend.get_texts():
    #         text.set_fontname(fontname)
    ax_pos_all.set_xlabel('position error ({:s})'.format(pos_unit), va='bottom', ha='center', fontsize=fontsize,
                          fontname=fontname)
    ax_pos_all.set_ylabel('probability', va='top', ha='center', fontsize=fontsize, fontname=fontname)
    ax_pos_all.set_xlim([(bin_range_pos[0] - offset_x_fac * dBin_pos) * conversion_fac_pos,
                         (bin_range_pos[-1] + offset_x_fac * dBin_pos) * conversion_fac_pos])
    ax_pos_all.set_xticks([0.0, maxBin_pos0 * conversion_fac_pos * 0.5, maxBin_pos0 * conversion_fac_pos,
                           maxBin_pos * conversion_fac_pos])
    ax_pos_all.set_ylim([0.0, 0.24])
    ax_pos_all.set_yticks([0.0, 0.12, 0.24])
    ax_pos_all.set_yticklabels([0, 0.12, 0.24])
    #     ax_pos_all.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    labels = list(['{:0.0f}'.format(i) for i in ax_pos_all.get_xticks()])
    labels[-1] = 'inf'
    ax_pos_all.set_xticklabels(labels)
    ax_pos_all.xaxis.get_offset_text().set_fontsize(fontsize)
    ax_pos_all.yaxis.get_offset_text().set_fontsize(fontsize)
    for tick in ax_pos_all.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax_pos_all.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    ax_pos_all.xaxis.set_label_coords(x=ax_pos_all_x + 0.5 * ax_pos_all_w, y=bottom_margin_x / fig_pos_all_h,
                                      transform=fig_pos_all.transFigure)
    ax_pos_all.yaxis.set_label_coords(x=left_margin_x / fig_pos_all_w, y=ax_pos_all_y + 0.5 * ax_pos_all_h,
                                      transform=fig_pos_all.transFigure)
    #
    line_w = np.diff(ax_pos_all.get_xlim()) * line_fac * 1 / 3
    line_h = np.diff(ax_pos_all.get_ylim()) * line_fac
    box_w = (maxBin_pos - maxBin_pos0) * conversion_fac_pos * 0.2
    box_h = abs(np.diff(ax_pos_all.get_ylim())) * box_h_fac
    ax_pos_all.plot(np.array([-line_w * 0.5, +line_w * 0.5], dtype=np.float64) + maxBin_pos0 * conversion_fac_pos * (
                1.0 + inf_fac * 0.5) - box_w * 0.5,
                    np.array([-line_h * 0.5, +line_h * 0.5], dtype=np.float64) + ax_pos_all.get_ylim()[0],
                    linewidth=ax_pos_all.spines['left'].get_linewidth(), linestyle='-', color='black', zorder=101,
                    clip_on=False)
    ax_pos_all.plot(np.array([-line_w * 0.5, +line_w * 0.5], dtype=np.float64) + maxBin_pos0 * conversion_fac_pos * (
                1.0 + inf_fac * 0.5) + box_w * 0.5,
                    np.array([-line_h * 0.5, +line_h * 0.5], dtype=np.float64) + ax_pos_all.get_ylim()[0],
                    linewidth=ax_pos_all.spines['bottom'].get_linewidth(), linestyle='-', color='black', zorder=101,
                    clip_on=False)
    box = plt.Rectangle((maxBin_pos0 * conversion_fac_pos * (1.0 + inf_fac * 0.5) - box_w * 0.5,
                         ax_pos_all.get_ylim()[0] + abs(np.diff(ax_pos_all.get_ylim())) * 0.5 - box_h * 0.5),
                        box_w, box_h,
                        color='white', zorder=100, clip_on=False)
    ax_pos_all.add_patch(box)
    #
    fig_pos_all.canvas.draw()
    #
    #     h_legend = ax_ang_all.legend(list(['anatomical & temporal', 'temporal', 'anatomical', 'none']),
    #                                  loc='upper right', frameon=False, fontsize=fontsize)
    #     h_legend.set_zorder(200)
    #     for text in h_legend.get_texts():
    #         text.set_fontname(fontname)
    ax_ang_all.set_xlabel('angle error ({:s})'.format(ang_unit), va='bottom', ha='center', fontsize=fontsize,
                          fontname=fontname)
    ax_ang_all.set_ylabel('probability', va='top', ha='center', fontsize=fontsize, fontname=fontname)
    ax_ang_all.set_xlim([(bin_range_ang[0] - offset_x_fac * dBin_ang) * conversion_fac_ang,
                         (bin_range_ang[-1] + offset_x_fac * dBin_ang) * conversion_fac_ang])
    ax_ang_all.set_xticks([0.0, maxBin_ang0 * conversion_fac_ang * 0.5, maxBin_ang0 * conversion_fac_ang,
                           maxBin_ang * conversion_fac_ang])
    ax_ang_all.set_ylim([0.0, 0.24])
    ax_ang_all.set_yticks([0.0, 0.12, 0.24])
    ax_ang_all.set_yticklabels([0, 0.12, 0.24])
    #     ax_ang_all.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    labels = list(['{:0.0f}'.format(i) for i in ax_ang_all.get_xticks()])
    labels[-1] = 'inf'
    ax_ang_all.set_xticklabels(labels)
    ax_ang_all.xaxis.get_offset_text().set_fontsize(fontsize)
    ax_ang_all.yaxis.get_offset_text().set_fontsize(fontsize)
    for tick in ax_ang_all.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax_ang_all.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    ax_ang_all.xaxis.set_label_coords(x=ax_ang_all_x + 0.5 * ax_ang_all_w, y=bottom_margin_x / fig_ang_all_h,
                                      transform=fig_ang_all.transFigure)
    ax_ang_all.yaxis.set_label_coords(x=left_margin_x / fig_ang_all_w, y=ax_ang_all_y + 0.5 * ax_ang_all_h,
                                      transform=fig_ang_all.transFigure)
    #
    line_w = np.diff(ax_ang_all.get_xlim()) * line_fac * 1 / 3
    line_h = np.diff(ax_ang_all.get_ylim()) * line_fac
    box_w = (maxBin_ang - maxBin_ang0) * conversion_fac_ang * 0.2
    box_h = abs(np.diff(ax_ang_all.get_ylim())) * box_h_fac
    ax_ang_all.plot(np.array([-line_w * 0.5, +line_w * 0.5], dtype=np.float64) + maxBin_ang0 * conversion_fac_ang * (
                1.0 + inf_fac * 0.5) - box_w * 0.5,
                    np.array([-line_h * 0.5, +line_h * 0.5], dtype=np.float64) + ax_ang_all.get_ylim()[0],
                    linewidth=ax_pos_all.spines['left'].get_linewidth(), linestyle='-', color='black', zorder=101,
                    clip_on=False)
    ax_ang_all.plot(np.array([-line_w * 0.5, +line_w * 0.5], dtype=np.float64) + maxBin_ang0 * conversion_fac_ang * (
                1.0 + inf_fac * 0.5) + box_w * 0.5,
                    np.array([-line_h * 0.5, +line_h * 0.5], dtype=np.float64) + ax_ang_all.get_ylim()[0],
                    linewidth=ax_pos_all.spines['bottom'].get_linewidth(), linestyle='-', color='black', zorder=101,
                    clip_on=False)
    box = plt.Rectangle((maxBin_ang0 * conversion_fac_ang * (1.0 + inf_fac * 0.5) - box_w * 0.5,
                         ax_ang_all.get_ylim()[0] + abs(np.diff(ax_ang_all.get_ylim())) * 0.5 - box_h * 0.5),
                        box_w, box_h,
                        color='white', zorder=100, clip_on=False)
    ax_ang_all.add_patch(box)
    #
    fig_ang_all.canvas.draw()
    #
    plt.pause(2 ** -10)
    fig_acc_all.canvas.draw()
    fig_velo_all.canvas.draw()
    fig_pos_all.canvas.draw()
    fig_ang_all.canvas.draw()
    if save:
        fig_acc_all.savefig(folder_save + '/acceleration_hist.svg',
                            #                          bbox_inches="tight",
                            dpi=300,
                            transparent=True,
                            format='svg',
                            pad_inches=0)
        fig_velo_all.savefig(folder_save + '/velocity_hist.svg',
                             #                          bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)
        fig_pos_all.savefig(folder_save + '/position_hist.svg',
                            #                          bbox_inches="tight",
                            dpi=300,
                            transparent=True,
                            format='svg',
                            pad_inches=0)
        fig_ang_all.savefig(folder_save + '/angle_hist.svg',
                            #                         bbox_inches="tight",
                            dpi=300,
                            transparent=True,
                            format='svg',
                            pad_inches=0)

    if show:
        plt.show()
