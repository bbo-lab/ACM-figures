#!/usr/bin/env python3

import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

import cfg_plt

sys.path.append(os.path.abspath('../ACM/config'))
import configuration as cfg
sys.path.pop(sys.path.index(os.path.abspath('../ACM/config')))
#
sys.path.append(os.path.abspath('../ACM'))
import anatomy
import data
import helper
import model
sys_path0 = np.copy(sys.path)

save = True
verbose = False

mode = 'mode1'

species = 'mouse'

if species=='rat':
    folder_recon = data.path + '/datasets_figures/reconstruction'
    folder_list = list(['/20200205/arena_20200205_033300_033500', # 0 # 20200205
                        '/20200205/arena_20200205_034400_034650', # 1
                        '/20200205/arena_20200205_038400_039000', # 2
                        '/20200205/arena_20200205_039200_039500', # 3
                        '/20200205/arena_20200205_039800_040000', # 4
                        '/20200205/arena_20200205_040500_040700', # 5
                        '/20200205/arena_20200205_041500_042000', # 6
                        '/20200205/arena_20200205_045250_045500', # 7
                        '/20200205/arena_20200205_046850_047050', # 8
                        '/20200205/arena_20200205_048500_048700', # 9
                        '/20200205/arena_20200205_049200_050300', # 10
                        '/20200205/arena_20200205_050900_051300', # 11
                        '/20200205/arena_20200205_052050_052200', # 12
                        '/20200205/arena_20200205_055200_056000', # 13
                        '/20200207/arena_20200207_032200_032550', # 14 # 20200207
                        '/20200207/arena_20200207_044500_045400', # 15
                        '/20200207/arena_20200207_046300_046900', # 16
                        '/20200207/arena_20200207_048000_049800', # 17
                        '/20200207/arena_20200207_050000_050400', # 18
                        '/20200207/arena_20200207_050800_051300', # 19
                        '/20200207/arena_20200207_052250_052650', # 20
                        '/20200207/arena_20200207_053050_053500', # 21
                        '/20200207/arena_20200207_055100_056200', # 22
                        '/20200207/arena_20200207_058400_058900', # 23
                        '/20200207/arena_20200207_059450_059950', # 24
                        '/20200207/arena_20200207_060400_060800', # 25
                        '/20200207/arena_20200207_061000_062100', # 26
                        '/20200207/arena_20200207_064100_064400', # 27
                        ])
    folder_list_use = list([i.split('/')[2] for i in folder_list])
    print(folder_list_use)
    folder_list_indices = list([[[int(i.split('_')[-2]), int(i.split('_')[-1])]] for i in folder_list_use])
    print(folder_list_indices)
    frame_rate = 100.0

    add2cfg = ''

    if (mode == 'mode4'):
        add2folder = '__pcutoff9e-01'
    elif (mode == 'mode1'):
        add2folder = '__mode1__pcutoff9e-01'
    else:
        raise

elif species=='mouse':
    folder_recon = data.path + '/dataset_analysis/'
    folder_list = list([
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_003412_004000',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_006254_006598',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_008018_008508',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_016543_017033',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_019581_020169',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_021051_021639',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_032712_033398',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_042610_043198',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_053291_054271',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_059955_060641',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_074948_075634',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_078377_078965',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_086609_087687',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_109050_109441',
                        #'/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_111401_112185',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_000300_000900',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_001500_001900',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_002900_003400',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_004600_005100',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_006100_006500',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_009800_010700',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_011600_012100',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_013200_014000',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_015600_016100',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_016700_017400',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_021600_022300',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_044600_045300',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_050700_051200',
                        #'/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_060700_061800',
                        ])
    folder_list_use = list([i.split('/')[6] for i in folder_list])
    print(folder_list_use)
    folder_list_indices = list([[[int(i.split('_')[-2]), int(i.split('_')[-1])]] for i in folder_list_use])
    print(folder_list_indices)
    frame_rate = 196.0

    add2cfg = '/configuration/'

    if (mode == 'mode4'):
        add2folder = '__mode4__pcutoff9e-01'
    elif (mode == 'mode1'):
        add2folder = '__mode1__pcutoff9e-01'
    else:
        raise

#
list_is_large_animal = list([0 for i in folder_list])

threshold_velocity = 25.0 # cm/s
dTime = 0.2 # s
axis = 0
if (axis == 0):
    axis_s = 'x'
elif (axis == 1):
    axis_s = 'y'
elif (axis == 2):
    axis_s = 'z'

folder_save = os.path.abspath('panels') + '/dlc/' + mode + '/' + species
os.makedirs(folder_save,exist_ok=True)
#
paws_joint_names = list(['marker_paw_front_left_start', 'marker_paw_front_right_start', 'marker_ankle_left_start', 'marker_ankle_right_start',])
paws_joint_names_legend = list(['left wrist marker', 'right wrist marker', 'left ankle marker', 'right ankle marker',])
#
angle_joint_list = list([['marker_paw_front_left_start', 'marker_paw_front_right_start', 'marker_ankle_left_start', 'marker_ankle_right_start',]])
angle_joint_legend_list = list([['left wrist marker', 'right wrist marker', 'left ankle marker', 'right ankle marker',]])
#
nAngles_pairs = np.size(angle_joint_list, 0)
if not(nAngles_pairs == 1):
    raise # change naming of figures for nAngles_pairs > 1
#

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

bottom_margin_x = 0.05 # inch
left_margin_x = 0.05 # inch
left_margin  = 0.12 # inch
right_margin = 0.0 # inch
bottom_margin = 0.11 # inch
top_margin = 0.01 # inch

fontsize = 6
linewidth = 0.5
fontname = "Arial"
markersize = 2

cmap = plt.cm.tab10
color_left_front = cmap(4/9)
color_right_front = cmap(3/9)
color_left_hind = cmap(9/9)
color_right_hind = cmap(8/9)
colors_paws = list([color_left_front, color_right_front, color_left_hind, color_right_hind])
color_limb1 = cmap(5/9)
color_limb2 = cmap(1/9)
color_limb3 = cmap(0/9)
color_limb4 = cmap(2/9)
color_limb5 = cmap(7/9)
colors_limb = list([color_limb1, color_limb2, color_limb3, color_limb4, color_limb5])

if __name__ == '__main__':
    fig_w = np.round(mm_in_inch * (90.0*1/3), decimals=2)
    fig_h = np.round(mm_in_inch * (90.0*1/2), decimals=2)
    # position
    fig1_w = fig_w
    fig1_h = fig_h
    fig1 = plt.figure(1, figsize=(fig1_w, fig1_h))
    fig1.canvas.manager.window.move(0, 0)
    fig1.clear()
    ax1_x = left_margin/fig1_w
    ax1_y = bottom_margin/fig1_h
    ax1_w = 1.0 - (left_margin/fig1_w + right_margin/fig1_w)
    ax1_h = 1.0 - (bottom_margin/fig1_h + top_margin/fig1_h)
    ax1 = fig1.add_axes([ax1_x, ax1_y, ax1_w, ax1_h])
    ax1.clear()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1_min = -7.55 # cm
    ax1_max = 6.58 # cm
    # velocity
    fig2_w = fig_w
    fig2_h = fig_h
    fig2 = plt.figure(2, figsize=(fig2_w, fig2_h))
    fig2.canvas.manager.window.move(0, 0)
    fig2.clear()
    ax2_x = left_margin/fig2_w
    ax2_y = bottom_margin/fig2_h
    ax2_w = 1.0 - (left_margin/fig2_w + right_margin/fig2_w)
    ax2_h = 1.0 - (bottom_margin/fig2_h + top_margin/fig2_h)
    ax2 = fig2.add_axes([ax2_x, ax2_y, ax2_w, ax2_h])
    ax2.clear()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2_min = -0.485 # cm/ms
    ax2_max = 0.639 # cm/ms
#     # acceleration
#     fig3_w = fig_w#np.round(mm_in_inch * 86.0*1/2, decimals=2)
#     fig3_h = fig_h#np.round(mm_in_inch * 86.0*1/2, decimals=2)
#     fig3 = plt.figure(3, figsize=(fig3_w, fig3_h))
#     fig3.canvas.manager.window.move(0, 0)
#     fig3.clear()
#     ax3_x = left_margin/fig3_w
#     ax3_y = bottom_margin/fig3_h
#     ax3_w = 1.0 - (left_margin/fig3_w + right_margin/fig3_w)
#     ax3_h = 1.0 - (bottom_margin/fig3_h + top_margin/fig3_h)
#     ax3 = fig3.add_axes([ax3_x, ax3_y, ax3_w, ax3_h])
#     ax3.clear()
#     ax3.spines["top"].set_visible(False)
#     ax3.spines["right"].set_visible(False)
    # angle
    fig4 = list()
    ax4 = list()
    for i_ang_pair in range(nAngles_pairs):
        fig4_w = fig_w
        fig4_h = fig_h
        fig4_single = plt.figure(int('4{:d}'.format(i_ang_pair)), figsize=(fig4_w, fig4_h))
        fig4_single.canvas.manager.window.move(0, 0)
        fig4_single.clear()
        ax4_x = left_margin/fig4_w
        ax4_y = bottom_margin/fig4_h
        ax4_w = 1.0 - (left_margin/fig4_w + right_margin/fig4_w)
        ax4_h = 1.0 - (bottom_margin/fig4_h + top_margin/fig4_h)
        ax4_single = fig4_single.add_axes([ax4_x, ax4_y, ax4_w, ax4_h])
        ax4_single.clear()
        ax4_single.spines["top"].set_visible(False)
        ax4_single.spines["right"].set_visible(False)
        fig4.append(fig4_single)
        ax4.append(ax4_single)
    ax4_min = -54.5 # deg
    ax4_max = 55.0 # deg
    # angle velocity
    fig41 = list()
    ax41 = list()
    for i_ang_pair in range(nAngles_pairs):
        fig41_w = fig_w
        fig41_h = fig_h
        fig41_single = plt.figure(int('41{:d}'.format(i_ang_pair)), figsize=(fig41_w, fig41_h))
        fig41_single.canvas.manager.window.move(0, 0)
        fig41_single.clear()
        ax41_x = left_margin/fig41_w
        ax41_y = bottom_margin/fig41_h
        ax41_w = 1.0 - (left_margin/fig41_w + right_margin/fig41_w)
        ax41_h = 1.0 - (bottom_margin/fig41_h + top_margin/fig41_h)
        ax41_single = fig41_single.add_axes([ax41_x, ax41_y, ax41_w, ax41_h])
        ax41_single.clear()
        ax41_single.spines["top"].set_visible(False)
        ax41_single.spines["right"].set_visible(False)
        fig41.append(fig41_single)
        ax41.append(ax41_single)
    ax41_min = -4.2 # deg/ms
    ax41_max = 3.28 # deg/ms
    # position limb
    fig5_w = fig_w
    fig5_h = fig_h
    fig5 = plt.figure(5, figsize=(fig5_w, fig5_h))
    fig5.canvas.manager.window.move(0, 0)
    fig5.clear()
    ax5_x = left_margin/fig5_w
    ax5_y = bottom_margin/fig5_h
    ax5_w = 1.0 - (left_margin/fig5_w + right_margin/fig5_w)
    ax5_h = 1.0 - (bottom_margin/fig5_h + top_margin/fig5_h)
    ax5 = fig5.add_axes([ax5_x, ax5_y, ax5_w, ax5_h])
    ax5.clear()
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5_min = -7.55 # cm
    ax5_max = 6.58 # cm
    # velocity limb
    fig6_w = fig_w
    fig6_h = fig_h
    fig6 = plt.figure(6, figsize=(fig6_w, fig6_h))
    fig6.canvas.manager.window.move(0, 0)
    fig6.clear()
    ax6_x = left_margin/fig6_w
    ax6_y = bottom_margin/fig6_h
    ax6_w = 1.0 - (left_margin/fig6_w + right_margin/fig6_w)
    ax6_h = 1.0 - (bottom_margin/fig6_h + top_margin/fig6_h)
    ax6 = fig6.add_axes([ax6_x, ax6_y, ax6_w, ax6_h])
    ax6.clear()
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6_min = -0.485 # cm/ms
    ax6_max = 0.639 # cm/ms
#     # acceleration limb
#     fig7_w = fig_w#np.round(mm_in_inch * 86.0*1/2, decimals=2)
#     fig7_h = fig_h#np.round(mm_in_inch * 86.0*1/2, decimals=2)
#     fig7 = plt.figure(7, figsize=(fig7_w, fig7_h))
#     fig7.canvas.manager.window.move(0, 0)
#     fig7.clear()
#     ax7_x = left_margin/fig7_w
#     ax7_y = bottom_margin/fig7_h
#     ax7_w = 1.0 - (left_margin/fig7_w + right_margin/fig7_w)
#     ax7_h = 1.0 - (bottom_margin/fig7_h + top_margin/fig7_h)
#     ax7 = fig7.add_axes([ax7_x, ax7_y, ax7_w, ax7_h])
#     ax7.clear()
#     ax7.spines["top"].set_visible(False)
#     ax7.spines["right"].set_visible(False)
    # angle limb
    fig8_w = fig_w
    fig8_h = fig_h
    fig8 = plt.figure(8, figsize=(fig8_w, fig8_h))
    fig8.canvas.manager.window.move(0, 0)
    fig8.clear()
    ax8_x = left_margin/fig8_w
    ax8_y = bottom_margin/fig8_h
    ax8_w = 1.0 - (left_margin/fig8_w + right_margin/fig8_w)
    ax8_h = 1.0 - (bottom_margin/fig8_h + top_margin/fig8_h)
    ax8 = fig8.add_axes([ax8_x, ax8_y, ax8_w, ax8_h])
    ax8.clear()
    ax8.spines["top"].set_visible(False)
    ax8.spines["right"].set_visible(False)
    ax8_min = -54.5 # deg
    ax8_max = 56.0 # deg
    # angle velocity limb
    fig9_w = fig_w
    fig9_h = fig_h
    fig9 = plt.figure(9, figsize=(fig9_w, fig9_h))
    fig9.canvas.manager.window.move(0, 0)
    fig9.clear()
    ax9_x = left_margin/fig9_w
    ax9_y = bottom_margin/fig9_h
    ax9_w = 1.0 - (left_margin/fig9_w + right_margin/fig9_w)
    ax9_h = 1.0 - (bottom_margin/fig9_h + top_margin/fig9_h)
    ax9 = fig9.add_axes([ax9_x, ax9_y, ax9_w, ax9_h])
    ax9.clear()
    ax9.spines["top"].set_visible(False)
    ax9.spines["right"].set_visible(False)
    ax9_min = -4.16 # deg/ms
    ax9_max = 3.28 # deg/ms
        
        
    
    nFolders = len(folder_list)
    #
    dIndex = int(np.ceil(dTime * frame_rate))
    nAll = int(1+2*dIndex)
    #
    pos_peak_time_diff_max = list()
    velo_peak_time_diff_max = list()
    ang_peak_time_diff_max = list()
    ang_velo_peak_time_diff_max = list()
    pos_peak_time_diff_min = list()
    velo_peak_time_diff_min = list()
    ang_peak_time_diff_min = list()
    ang_velo_peak_time_diff_min = list()
    for i_paw_timing in range(4):
        ax1.clear()
        ax2.clear()
        for i_ang_pair in range(nAngles_pairs):
            ax4[i_ang_pair].clear()
            ax41[i_ang_pair].clear()
        ax5.clear()
        ax6.clear()
        ax8.clear()
        ax9.clear()
        #
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for i_ang_pair in range(nAngles_pairs):
            ax4[i_ang_pair].spines["top"].set_visible(False)
            ax4[i_ang_pair].spines["right"].set_visible(False)
            ax41[i_ang_pair].spines["top"].set_visible(False)
            ax41[i_ang_pair].spines["right"].set_visible(False)
        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)
        ax6.spines["top"].set_visible(False)
        ax6.spines["right"].set_visible(False)
        ax8.spines["top"].set_visible(False)
        ax8.spines["right"].set_visible(False)
        ax9.spines["top"].set_visible(False)
        ax9.spines["right"].set_visible(False)
        #
        if (i_paw_timing == 0):
            joints_list_limb = list(['marker_shoulder_left_start', 'marker_elbow_left_start', 'marker_paw_front_left_start', 'marker_finger_left_002_start'])
            joints_list_legend_limb = list(['left shoulder marker', 'left elbow marker', 'left wrist marker', 'left finger marker'])
        elif (i_paw_timing == 1):
            joints_list_limb = list(['marker_shoulder_right_start', 'marker_elbow_right_start', 'marker_paw_front_right_start', 'marker_finger_right_002_start'])
            joints_list_legend_limb = list(['right shoulder marker', 'right elbow marker', 'right wrist marker', 'right finger marker'])
        elif (i_paw_timing == 2):
            joints_list_limb = list(['marker_hip_left_start', 'marker_knee_left_start', 'marker_ankle_left_start', 'marker_paw_hind_left_start', 'marker_toe_left_002_start'])
            joints_list_legend_limb = list(['left hip marker', 'left knee marker', 'left ankle marker', 'left hind paw marker', 'left toe marker'])
        elif (i_paw_timing == 3):
            joints_list_limb = list(['marker_hip_right_start', 'marker_knee_right_start', 'marker_ankle_right_start', 'marker_paw_hind_right_start', 'marker_toe_right_002_start'])
            joints_list_legend_limb = list(['right hip marker', 'right knee marker', 'right ankle marker', 'right hind paw marker', 'right toe marker'])
        nJointsLimb = len(joints_list_limb)
        #
        pos_power_1 = np.zeros((4, nAll), dtype=np.float64)
        pos_power_2 = np.zeros((4, nAll), dtype=np.float64)
        velo_power_1 = np.zeros((4, nAll), dtype=np.float64)
        velo_power_2 = np.zeros((4, nAll), dtype=np.float64)
        acc_power_1 = np.zeros((4, nAll), dtype=np.float64)
        acc_power_2 = np.zeros((4, nAll), dtype=np.float64)
        ang_power_1 = np.zeros((nAngles_pairs, 4, nAll), dtype=np.float64)
        ang_power_2 = np.zeros((nAngles_pairs, 4, nAll), dtype=np.float64)
        ang_velo_power_1 = np.zeros((nAngles_pairs, 4, nAll), dtype=np.float64)
        ang_velo_power_2 = np.zeros((nAngles_pairs, 4, nAll), dtype=np.float64)
        #
        pos_power_1_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
        pos_power_2_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
        velo_power_1_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
        velo_power_2_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
        acc_power_1_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
        acc_power_2_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
        ang_power_1_limb = np.zeros((nJointsLimb-1, nAll), dtype=np.float64)
        ang_power_2_limb = np.zeros((nJointsLimb-1, nAll), dtype=np.float64)
        ang_velo_power_1_limb = np.zeros((nJointsLimb-1, nAll), dtype=np.float64)
        ang_velo_power_2_limb = np.zeros((nJointsLimb-1, nAll), dtype=np.float64)
        #
        nPeaks_all = np.zeros(nAll, dtype=np.float64)
        nPeaks_all2 = np.zeros(nAll, dtype=np.float64)
        #
        for i_folder in range(nFolders):
            
            folder = folder_recon+folder_list[i_folder]+add2folder
            sys.path = list(np.copy(sys_path0))
            sys.path.append(folder+add2cfg)
            importlib.reload(cfg)
            cfg.animal_is_large = list_is_large_animal[i_folder]
            importlib.reload(anatomy)
            
            if os.path.isfile(folder+'/save_dict.npy'):
                print(folder)

                # get arguments
                folder_reqFiles = data.path + '/datasets_figures/required_files'

                file_origin_coord = folder+'/configuration/file_origin_coord.npy'
                if not os.path.isfile(file_origin_coord):
                    file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
                file_calibration = folder+'/configuration/file_calibration.npy'
                if not os.path.isfile(file_calibration):
                    file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
                file_model = folder+'/configuration/file_model.npy'
                if not os.path.isfile(file_model):
                    file_model = folder_reqFiles + '/model.npy'
                file_labelsDLC = folder+'/configuration/file_labelsDLC.npy'
                if not os.path.isfile(file_labelsDLC):
                    file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/' + cfg.file_labelsDLC.split('/')[-1]

                file_labelsDLC_3d = file_labelsDLC[:-4]+'__3d.npy'
                if not os.path.isfile(file_labelsDLC):
                    file_labelsDLC_split = file_labelsDLC.split('/')
                    file_labelsDLC_3d = '/'.join(file_labelsDLC_split[:-1]) + '/' + file_labelsDLC_split[-1].split('.')[0] + '__3d.npy'
                labels_dlc_3d = np.load(file_labelsDLC_3d, allow_pickle=True)
                
                args_model = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                                  cfg.scale_factor, cfg.pcutoff)
                if ((cfg.mode == 1) or (cfg.mode == 2)):
                    args_model['use_custom_clip'] = False
                elif ((cfg.mode == 3) or (cfg.mode == 4)):
                    args_model['use_custom_clip'] = True
                args_model['plot'] = True
                nBones = args_model['numbers']['nBones']
                nMarkers = args_model['numbers']['nMarkers']
                skeleton_edges = args_model['model']['skeleton_edges'].cpu().numpy()
                joint_order = args_model['model']['joint_order']
                joint_marker_order = args_model['model']['joint_marker_order']    
                bone_lengths_index = args_model['model']['bone_lengths_index'].cpu().numpy()
                joint_marker_index = args_model['model']['joint_marker_index'].cpu().numpy()
                labels_mask = args_model['labels_mask'].cpu().numpy().astype(np.float64)
                # get save_dict
                save_dict = np.load(folder+'/save_dict.npy', allow_pickle=True).item()
                if ('mu_uks' in save_dict):
                    mu_uks_norm_all = np.copy(save_dict['mu_uks'][1:])
                else:
                    mu_uks_norm_all = np.copy(save_dict['mu_fit'][1:])
                #
                # get joint connections for angle calculation
                nJoints = args_model['numbers']['nBones']+1
                angle_joint_connections_names = list([])
                angle_joint_connections_names_middle = list([])
                for i_joint in range(nJoints):
                    mask_from = (skeleton_edges[:, 0] == i_joint)
                    mask_to = (skeleton_edges[:, 1] == i_joint)
                    for i_joint_from in skeleton_edges[:, 0][mask_to]:
                        for i_joint_to in skeleton_edges[:, 1][mask_from]:
                            angle_joint_connections_names.append(list([joint_order[i_joint_from],
                                                                       joint_order[i_joint],
                                                                       joint_order[i_joint_to]]))
                            angle_joint_connections_names_middle.append(joint_order[i_joint])
                nAngles = np.size(angle_joint_connections_names, 0)
                
                # get corresponing marker list [WARNING: THERE ARE MORE MARKERS THEN JOINTS -> SHOULD NOT MATTER FOR LIMBS THOUGH]
                for i_ang in range(nAngles):
                    for i in range(3):
                        name_joint = angle_joint_connections_names[i_ang][i]
                        name_joint_split = name_joint.split('_')
                        if ('wrist' in name_joint_split):
                            if ('left' in name_joint_split):
                                name_maker = 'marker_paw_front_left_start'
                            elif ('right' in name_joint_split):
                                name_maker = 'marker_paw_front_right_start'
                        else:
                            name_maker = 'marker_' + '_'.join(name_joint_split[1:]) + '_start'
                        angle_joint_connections_names[i_ang][i] = name_maker
                        if (i == 1):
                            angle_joint_connections_names_middle[i_ang] = name_maker

                nSeq = np.size(folder_list_indices[i_folder], 0)
                for i_seq in range(nSeq):
                    frame_start = folder_list_indices[i_folder][i_seq][0]
                    frame_end = folder_list_indices[i_folder][i_seq][1]

                    mu_uks_norm = mu_uks_norm_all[frame_start-cfg.index_frame_ini:frame_end-cfg.index_frame_ini]
                    mu_uks = model.undo_normalization(torch.from_numpy(mu_uks_norm), args_model).numpy() # reverse normalization
                    labels_mask_sum = np.sum(labels_mask[frame_start-cfg.index_frame_ini:frame_end-cfg.index_frame_ini], 1)
                    # get x_ini
                    x_ini = np.load(folder+'/x_ini.npy', allow_pickle=True)
                    
                    # free parameters
                    free_para_pose = args_model['free_para_pose'].cpu().numpy()
                    free_para_bones = np.zeros(nBones, dtype=bool)
                    free_para_markers = np.zeros(nMarkers*3, dtype=bool)    
                    free_para = np.concatenate([free_para_bones,
                                                free_para_markers,
                                                free_para_pose], 0)
                    # update args
                    args_model['x_torch'] = torch.from_numpy(x_ini).type(model.float_type)
                    args_model['x_free_torch'] = torch.from_numpy(x_ini[free_para]).type(model.float_type)
                    # full x
                    nT_use = np.size(mu_uks_norm, 0)
                    x = np.tile(x_ini, nT_use).reshape(nT_use, len(x_ini))
                    x[:, free_para] = mu_uks

                    # get poses
                    _, _, skeleton_pos = model.fcn_emission(torch.from_numpy(x), args_model)
                    skeleton_pos = skeleton_pos.cpu().numpy()
                    print(labels_dlc_3d)
                    dlc_mask = np.logical_and(labels_dlc_3d['frame_list']>=frame_start, labels_dlc_3d['frame_list']<frame_end)
                    print(f'{np.sum(dlc_mask)} - {dlc_mask.shape} - {labels_dlc_3d["labels_all"].shape}')
                    skeleton_pos = labels_dlc_3d['labels_all'][dlc_mask]
                    
                    # do coordinate transformation
                    joint_index1_1 = joint_marker_order.index('marker_hip_left_start')
                    joint_index1_2 = joint_marker_order.index('marker_hip_right_start')
                    joint_index2_1 = joint_marker_order.index('marker_shoulder_left_start')
                    joint_index2_2 = joint_marker_order.index('marker_shoulder_right_start')
                    origin = (skeleton_pos[:, joint_index1_1] + skeleton_pos[:, joint_index1_2]) * 0.5
                    x_direc = (skeleton_pos[:, joint_index2_1] + skeleton_pos[:, joint_index2_2]) * 0.5 - origin
                    x_direc = x_direc[:, :2]
                    x_direc = x_direc / np.sqrt(np.sum(x_direc**2, 1))[:, None]
                    alpha = np.arctan2(x_direc[:, 1], x_direc[:, 0])
                    cos_m_alpha = np.cos(-alpha)
                    sin_m_alpha = np.sin(-alpha)
                    R = np.stack([np.stack([cos_m_alpha, -sin_m_alpha], 1),
                                  np.stack([sin_m_alpha, cos_m_alpha], 1)], 1)
                    skeleton_pos = skeleton_pos - origin[:, None, :]
                    skeleton_pos_xy = np.einsum('nij,nmj->nmi', R, skeleton_pos[:, :, :2])
                    skeleton_pos[:, :, :2] = np.copy(skeleton_pos_xy)
                    
                    position_single = np.full((nT_use, nMarkers, 3), np.nan, dtype=np.float64)
                    velocity_single = np.full((nT_use, nMarkers, 3), np.nan, dtype=np.float64)
                    acceleration_single = np.full((nT_use, nMarkers, 3), np.nan, dtype=np.float64)
                    angle_single = np.full((nT_use, nAngles), np.nan, dtype=np.float64)
                    angle_velocity_single = np.full((nT_use, nAngles), np.nan, dtype=np.float64)
                    joints_ang_3d = np.zeros((nAngles_pairs, 4, nT_use), dtype=np.float64)
                    joints_ang_3d_limb = np.zeros((nJointsLimb-1, nT_use), dtype=np.float64)
                    joints_ang_velo_3d = np.zeros((nAngles_pairs, 4, nT_use), dtype=np.float64)
                    joints_ang_velo_3d_limb = np.zeros((nJointsLimb-1, nT_use), dtype=np.float64)
                    #
                    derivative_accuracy = 8
                    derivative_index0 = int(derivative_accuracy/2)
                    derivative_index1 = nT_use - int(derivative_accuracy/2)
                    #
                    position_single = np.copy(skeleton_pos)
                    velocity_single[4:-4] = \
                        (+1.0/280.0 * position_single[:-8] + \
                         -4.0/105.0 * position_single[1:-7] + \
                         +1.0/5.0 * position_single[2:-6] + \
                         -4.0/5.0 * position_single[3:-5] + \
#                          0.0 * position_single[4:-4] + \
                         +4.0/5.0 * position_single[5:-3] + \
                         -1.0/5.0 * position_single[6:-2] + \
                         +4.0/105.0 * position_single[7:-1] + \
                         -1.0/280.0 * position_single[8:]) / (1.0/cfg.frame_rate)
                    acceleration_single[4:-4] = \
                        (-1.0/560.0 * position_single[:-8] + \
                         +8.0/315.0 * position_single[1:-7] + \
                         -1.0/5.0 * position_single[2:-6] + \
                         +8.0/5.0 * position_single[3:-5] + \
                         -205.0/72.0 * position_single[4:-4] + \
                         +8.0/5.0 * position_single[5:-3] + \
                         -1.0/5.0 * position_single[6:-2] + \
                         +8.0/315.0 * position_single[7:-1] + \
                         -1.0/560.0 * position_single[8:]) / (1.0/cfg.frame_rate)**2
                    for i_ang in range(nAngles):
                        index_joint1 = joint_marker_order.index(angle_joint_connections_names[i_ang][0])
                        index_joint2 = joint_marker_order.index(angle_joint_connections_names[i_ang][1])
                        vec1 = position_single[:, index_joint2] - position_single[:, index_joint1]
                        vec1 = vec1 / np.sqrt(np.sum(vec1**2, 1))[:, None]
                        # angle between bone and walking direction
                        vec0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                        vec_use = np.copy(vec1)
                        ang = np.arccos(np.einsum('i,ni->n', vec0, vec_use)) * 180.0/np.pi
                        # source: https://math.stackexchange.com/questions/2140504/how-to-calculate-signed-angle-between-two-vectors-in-3d
                        n_cross = np.cross(vec0, vec_use)
                        n_cross = n_cross / np.sqrt(np.sum(n_cross**2, 1))[:, None]
                        sin = np.einsum('ni,ni->n', np.cross(n_cross, vec0), vec_use)
                        mask = (sin < 0.0)
                        ang[mask] = 360.0 - ang[mask]
                        if np.any(mask):
                            print('WARNING: {:07d}/{:07d} signed angles'.format(np.sum(mask), len(mask.ravel())))
                        
                        angle_single[:, i_ang] = np.copy(ang)
                        angle_velocity_single[4:-4, i_ang] =  \
                            (+1.0/280.0 * ang[:-8] + \
                             -4.0/105.0 * ang[1:-7] + \
                             +1.0/5.0 * ang[2:-6] + \
                             -4.0/5.0 * ang[3:-5] + \
#                              0.0 * ang[4:-4] + \
                             +4.0/5.0 * ang[5:-3] + \
                             -1.0/5.0 * ang[6:-2] + \
                             +4.0/105.0 * ang[7:-1] + \
                             -1.0/280.0 * ang[8:]) / (1.0/cfg.frame_rate)
                    #
                    joints_pos_1d = np.zeros((4, nT_use), dtype=np.float64)
                    dy_dx = np.full(np.shape(joints_pos_1d), np.nan, dtype=np.float64)
                    dy_dx2 = np.full(np.shape(joints_pos_1d), np.nan, dtype=np.float64)
                    for i_paw in range(4):
                        joint_index = joint_marker_order.index(paws_joint_names[i_paw])
                        joints_pos_1d[i_paw] = position_single[:, joint_index, axis]
                        dy_dx[i_paw] = velocity_single[:, joint_index, axis]
                        dy_dx2[i_paw] = acceleration_single[:, joint_index, axis]
                    joints_pos_1d_limb = np.zeros((nJointsLimb, nT_use), dtype=np.float64)
                    dy_dx_limb = np.full(np.shape(joints_pos_1d_limb), np.nan, dtype=np.float64)
                    dy_dx2_limb = np.full(np.shape(joints_pos_1d_limb), np.nan, dtype=np.float64)
                    for i_joint in range(nJointsLimb):
                        joint_index = joint_marker_order.index(joints_list_limb[i_joint])
                        joints_pos_1d_limb[i_joint] = position_single[:, joint_index, axis]
                        dy_dx_limb[i_joint] = velocity_single[:, joint_index, axis]
                        dy_dx2_limb[i_joint] = acceleration_single[:, joint_index, axis] 
#                     joints_ang_3d = np.zeros((nAngles_pairs, 4, nT_use), dtype=np.float64)
                    for i_ang_pair in range(nAngles_pairs):
                        for i_paw in range(4):
                            joint_index = angle_joint_connections_names_middle.index(angle_joint_list[i_ang_pair][i_paw])
                            joints_ang_3d[i_ang_pair, i_paw] = angle_single[:, joint_index]
                            joints_ang_velo_3d[i_ang_pair, i_paw] = angle_velocity_single[:, joint_index]
#                     joints_ang_3d_limb = np.zeros((nJointsLimb-1, nT_use), dtype=np.float64)
                    for i_joint in range(nJointsLimb-1):
                        joint_index = angle_joint_connections_names_middle.index(joints_list_limb[:-1][i_joint])
                        joints_ang_3d_limb[i_joint] = angle_single[:, joint_index]
                        joints_ang_velo_3d_limb[i_joint] = angle_velocity_single[:, joint_index]

                    # find peak indices
                    derivative_accuracy_index = int(derivative_accuracy/2)
                    dy_dx_use = dy_dx[:, derivative_index0:derivative_index1]
                    mask = (dy_dx_use > threshold_velocity)
                    index = np.arange(nT_use - 2 * derivative_accuracy_index, dtype=np.int64)
                    indices_peaks = list()
                    for i_paw in range(4):
                        indices_peaks.append(list())
                        if np.any(mask[i_paw]):
                            diff_index = np.diff(index[mask[i_paw]])
                            nPeaks_mask = (diff_index > 1)
                            nPeaks = np.sum(nPeaks_mask) + 1
                            if (nPeaks > 1):
                                for i_peak in range(nPeaks):
                                    if (i_peak == 0):
                                        index_start = 0
                                        index_end = index[mask[i_paw]][1:][nPeaks_mask][i_peak]
                                    elif (i_peak == nPeaks-1):
                                        index_start = np.copy(index_end)
                                        index_end = nT_use - 2 * derivative_accuracy_index
                                    else:
                                        index_start = np.copy(index_end)
                                        index_end = index[mask[i_paw]][1:][nPeaks_mask][i_peak]
                                    dy_dx_index = index[index_start:index_end]
                                    if (np.size(dy_dx_index) > 0):
                                        index_peak = np.argmax(dy_dx_use[i_paw][dy_dx_index]) + dy_dx_index[0] + derivative_accuracy_index
                                        indices_peaks[i_paw].append(index_peak)
                            else:
                                index_peak = np.argmax(dy_dx_use[i_paw]) + derivative_accuracy_index
                                indices_peaks[i_paw].append(index_peak)

#                     # plot peaks to check if they are detected correctly
#                     fig1 = plt.figure(1, figsize=(14, 7))
#                     fig1.clear()
#                     ax1 = fig1.add_subplot(2,1,1)
#                     ax1.clear()
#                     ax1.spines["top"].set_visible(False)
#                     ax1.spines["right"].set_visible(False) 
#                     ax2 = fig1.add_subplot(2,1,2)
#                     ax2.clear()
#                     ax2.spines["top"].set_visible(False)
#                     ax2.spines["right"].set_visible(False) 
#                     #
#                     time = np.arange(frame_start, frame_end, 1, dtype=np.int64)
#                     pos = joints_pos_1d[i_paw_timing]
#                     ax1.plot(time, pos, linestyle='-', marker='', color='red', alpha=1.0, zorder=3)
#                     #
#                     time = np.arange(frame_start, frame_end, 1, dtype=np.int64)[derivative_index0:derivative_index1]
#                     velo = dy_dx[i_paw_timing]
#                     ax2.plot(time, velo, linestyle='-', marker='', color='red', alpha=1.0, zorder=3)
#                     #
#                     #
#                     time = np.arange(frame_start, frame_end, 1, dtype=np.int64)[derivative_index0:derivative_index1]
#                     velo_full = dy_dx_full[i_paw_timing]
#                     ax2.plot(time, velo_full, linestyle='-', marker='', color='blue', alpha=1.0, zorder=3)
#                     #
#                     nPeaks = len(indices_peaks[i_paw_timing])
#                     if (nPeaks > 0):
#                         for i_peak in range(nPeaks):
#                             x_peak = np.array([frame_start+indices_peaks[i_paw_timing][i_peak]+1,
#                                                frame_start+indices_peaks[i_paw_timing][i_peak]+1], dtype=np.float64)
#                             y_peak_pos = np.array([min(pos), max(pos)], dtype=np.float64)
#                             y_peak_velo = np.array([min(velo), max(velo)], dtype=np.float64)
#                             ax1.plot(x_peak, y_peak_pos, linestyle='-', marker='', color='black', alpha=0.5, zorder=1)
#                             ax2.plot(x_peak, y_peak_velo, linestyle='-', marker='', color='black', alpha=0.5, zorder=1)
#                     ax2.plot(np.array([frame_start-1, frame_end+1], dtype=np.float64),
#                              np.array([threshold_velocity, threshold_velocity], dtype=np.float64),
#                              linestyle='-', marker='', color='green', alpha=0.5, zorder=2)
#                     ax1.set_xlim([frame_start-1, frame_end+1])
#                     ax2.set_xlim([frame_start-1, frame_end+1])
#                     ax1.set_xlabel('time (s)', fontsize=fontsize)
#                     ax1.set_ylabel('{:s}-position (cm)'.format(axis_s), fontsize=fontsize)
#                     ax2.legend(['z', 'x/y/z'], fontsize=fontsize, frameon=False)
#                     ax2.set_xlabel('time (s)', fontsize=fontsize)
#                     ax2.set_ylabel('velocity (cm/s)'.format(axis_s), fontsize=fontsize)
#                     plt.show()
#                     raise

#                     fig1 = plt.figure(1, figsize=(14, 7))
#                     fig1.clear()
#                     ax1 = fig1.add_subplot(1,1,1)
#                     ax1.clear()
#                     ax1.spines["top"].set_visible(False)
#                     ax1.spines["right"].set_visible(False) 

                    nPeaks = len(indices_peaks[i_paw_timing])
                    for i_peak in range(nPeaks):
                        index_peak = indices_peaks[i_paw_timing][i_peak]
                        # index for position and angle
                        index0 = index_peak - dIndex
                        index1 = index_peak + dIndex + 1
                        index0_use = 0
                        index1_use = nAll
                        if (index0 < 0):
                            index0_use = abs(index0)
                            index0 = 0
                        if (index1 > nT_use):
                            index1_use = nAll - (index1 - nT_use)
                            index1 = nT_use
                        # index for velocity and acceleration
                        index0_2 = np.copy(index0)
                        index1_2 = np.copy(index1)
                        index0_use2 = np.copy(index0_use)
                        index1_use2 = np.copy(index1_use)
                        if (index0_2 < derivative_index0):
                            diff = derivative_index0 - index0_2
                            index0_2 = index0_2 + diff
                            index0_use2 = index0_use2 + diff
                        if (index1_2 > derivative_index1):
                            diff = index1_2 - derivative_index1
                            index1_2 = index1_2 - diff
                            index1_use2 = index1_use2 - diff
#                         # index for velocity and acceleration
#                         index0_2 = (index_peak - dIndex) - 1
#                         index1_2 = (index_peak + dIndex + 1) - 1
#                         index0_use2 = 0
#                         index1_use2 = nAll
#                         if (index0_2 < 0):
#                             index0_use2 = abs(index0_2)
#                             index0_2 = 0
#                         if (index1_2 > (nT_use - 2)):
#                             index1_use2 = nAll - (index1_2 - (nT_use - 2))
#                             index1_2 = (nT_use - 2)
                        #
                        nPeaks_all[index0_use:index1_use] = nPeaks_all[index0_use:index1_use] + 1
                        nPeaks_all2[index0_use2:index1_use2] = nPeaks_all2[index0_use2:index1_use2] + 1
                        # compare all four limbs
                        for i_paw in range(4):
                            pos = joints_pos_1d[i_paw, index0:index1]
                            pos_power_1[i_paw, index0_use:index1_use] = pos_power_1[i_paw, index0_use:index1_use] + pos
                            pos_power_2[i_paw, index0_use:index1_use] = pos_power_2[i_paw, index0_use:index1_use] + pos**2
                            velo = dy_dx[i_paw, index0_2:index1_2]
                            velo_power_1[i_paw, index0_use2:index1_use2] = velo_power_1[i_paw, index0_use2:index1_use2] + velo
                            velo_power_2[i_paw, index0_use2:index1_use2] = velo_power_2[i_paw, index0_use2:index1_use2] + velo**2
                            acc = dy_dx2[i_paw, index0_2:index1_2]
                            acc_power_1[i_paw, index0_use2:index1_use2] = acc_power_1[i_paw, index0_use2:index1_use2] + acc
                            acc_power_2[i_paw, index0_use2:index1_use2] = acc_power_2[i_paw, index0_use2:index1_use2] + acc**2
                            #
                            for i_ang_pairs in range(nAngles_pairs):
                                ang = joints_ang_3d[i_ang_pairs, i_paw, index0:index1]
                                ang_power_1[i_ang_pairs, i_paw, index0_use:index1_use] = ang_power_1[i_ang_pairs, i_paw, index0_use:index1_use] + ang
                                ang_power_2[i_ang_pairs, i_paw, index0_use:index1_use] = ang_power_2[i_ang_pairs, i_paw, index0_use:index1_use] + ang**2
                                ang_velo = joints_ang_velo_3d[i_ang_pairs, i_paw, index0_2:index1_2]
                                ang_velo_power_1[i_ang_pairs, i_paw, index0_use2:index1_use2] = ang_velo_power_1[i_ang_pairs, i_paw, index0_use2:index1_use2] + ang_velo
                                ang_velo_power_2[i_ang_pairs, i_paw, index0_use2:index1_use2] = ang_velo_power_2[i_ang_pairs, i_paw, index0_use2:index1_use2] + ang_velo**2
                        # compare joints of single limb
                        for i_joint in range(nJointsLimb):
                            pos = joints_pos_1d_limb[i_joint, index0:index1]
                            pos_power_1_limb[i_joint, index0_use:index1_use] = pos_power_1_limb[i_joint, index0_use:index1_use] + pos
                            pos_power_2_limb[i_joint, index0_use:index1_use] = pos_power_2_limb[i_joint, index0_use:index1_use] + pos**2
                            velo = dy_dx_limb[i_joint, index0_2:index1_2]
                            velo_power_1_limb[i_joint, index0_use2:index1_use2] = velo_power_1_limb[i_joint, index0_use2:index1_use2] + velo
                            velo_power_2_limb[i_joint, index0_use2:index1_use2] = velo_power_2_limb[i_joint, index0_use2:index1_use2] + velo**2
                            acc = dy_dx2_limb[i_joint, index0_2:index1_2]
                            acc_power_1_limb[i_joint, index0_use2:index1_use2] = acc_power_1_limb[i_joint, index0_use2:index1_use2] + acc
                            acc_power_2_limb[i_joint, index0_use2:index1_use2] = acc_power_2_limb[i_joint, index0_use2:index1_use2] + acc**2
                        for i_joint in range(nJointsLimb-1):
                            ang = joints_ang_3d_limb[i_joint, index0:index1]
                            ang_power_1_limb[i_joint, index0_use:index1_use] = ang_power_1_limb[i_joint, index0_use:index1_use] + ang
                            ang_power_2_limb[i_joint, index0_use:index1_use] = ang_power_2_limb[i_joint, index0_use:index1_use] + ang**2    
                            ang_velo = joints_ang_velo_3d_limb[i_joint, index0_2:index1_2]
                            ang_velo_power_1_limb[i_joint, index0_use2:index1_use2] = ang_velo_power_1_limb[i_joint, index0_use2:index1_use2] + ang_velo
                            ang_velo_power_2_limb[i_joint, index0_use2:index1_use2] = ang_velo_power_2_limb[i_joint, index0_use2:index1_use2] + ang_velo**2 
        print('number of all peaks:')
        print(nPeaks_all)
        print('number of all peaks (derivatives):')
        print(nPeaks_all2)

        time_fac = 1e3
        time = (np.arange(nAll, dtype=np.float64) - float(dIndex)) * (1.0/frame_rate) * time_fac
        velo_power_1_limb = velo_power_1_limb / time_fac
        velo_power_2_limb = velo_power_2_limb / time_fac**2
        acc_power_1_limb = acc_power_1_limb / (time_fac**2)
        acc_power_2_limb = acc_power_2_limb / (time_fac**2)**2
        ang_velo_power_1_limb = ang_velo_power_1_limb / time_fac
        ang_velo_power_2_limb = ang_velo_power_2_limb / time_fac**2
        velo_power_1 = velo_power_1 / time_fac
        velo_power_2 = velo_power_2 / time_fac**2
        acc_power_1 = acc_power_1 / (time_fac**2)
        acc_power_2 = acc_power_2 / (time_fac**2)**2
        ang_velo_power_1 = ang_velo_power_1 / time_fac
        ang_velo_power_2 = ang_velo_power_2 / time_fac**2
        #
        pos_avg = pos_power_1 / nPeaks_all[None, :]
        pos_std = np.sqrt((pos_power_2 - (pos_power_1**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])
        velo_avg = velo_power_1 / nPeaks_all2[None :]
        velo_std = np.sqrt((velo_power_2 - (velo_power_1**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
        acc_avg = acc_power_1 / nPeaks_all2[None, :]
        acc_std = np.sqrt((acc_power_2 - (acc_power_1**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
        ang_avg = ang_power_1 / nPeaks_all[None, :]
        ang_std = np.sqrt((ang_power_2 - (ang_power_1**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])
        ang_velo_avg = ang_velo_power_1 / nPeaks_all2[None, :]
        ang_velo_std = np.sqrt((ang_velo_power_2 - (ang_velo_power_1**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
        #
        pos_avg_limb = pos_power_1_limb / nPeaks_all[None, :]
        pos_std_limb = np.sqrt((pos_power_2_limb - (pos_power_1_limb**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])
        velo_avg_limb = velo_power_1_limb / nPeaks_all2[None, :]
        velo_std_limb = np.sqrt((velo_power_2_limb - (velo_power_1_limb**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
        acc_avg_limb = acc_power_1_limb / nPeaks_all2[None, :]
        acc_std_limb = np.sqrt((acc_power_2_limb - (acc_power_1_limb**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
        ang_avg_limb = ang_power_1_limb / nPeaks_all[None, :]
        ang_std_limb = np.sqrt((ang_power_2_limb - (ang_power_1_limb**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])
        ang_velo_avg_limb = ang_velo_power_1_limb / nPeaks_all2[None, :]
        ang_velo_std_limb = np.sqrt((ang_velo_power_2_limb - (ang_velo_power_1_limb**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])


        
        # calculation
        print('timed paw:')
        print('{:s}'.format(paws_joint_names_legend[i_paw_timing]))
       
        index_use = dIndex # time midpoint of limb swing phase
#         index_use  = np.argmax(velo_avg[i_paw_timing]) # time midpoint of limb swing phase
#         index_use = np.argmax(pos_avg[i_paw_timing]) # end of limb swing phase
        print(index_use, dIndex)
        print()

        print('x-position at time midpoint of limb swing phase (cm):')
        print('\t{:0.8f} +/- {:0.8f}'.format(pos_avg[i_paw_timing, index_use], pos_std[i_paw_timing, index_use]))
        print('x-velocity at time midpoint of limb swing phase (cm/ms):')
        print('\t{:0.8f} +/- {:0.8f}'.format(velo_avg[i_paw_timing, index_use], velo_std[i_paw_timing, index_use]))
        print('angle at time midpoint of limb swing phase (deg):')
        for i_joint in range(np.shape(ang_avg_limb)[0]):
            print('\t{:s}:'.format(joints_list_legend_limb[i_joint]))
            print('\t{:0.8f} +/- {:0.8f}'.format(ang_avg_limb[i_joint, index_use], ang_std_limb[i_joint, index_use]))
        print('angular velocity at time midpoint of limb swing phase (deg/ms):')
        for i_joint in range(np.shape(ang_velo_avg_limb)[0]):
            print('\t{:s}:'.format(joints_list_legend_limb[i_joint]))
            print('\t{:0.8f} +/- {:0.8f}'.format(ang_velo_avg_limb[i_joint, index_use], ang_velo_std_limb[i_joint, index_use]))
        
        
        verbose_test = False
        if verbose_test:
            fig_test = plt.figure(123)
            fig_test.clear()
            ax_test = fig_test.add_subplot(111)
            colors_min_max = list(['blue', 'orange'])
        for i_metric in range(4):
            if (i_metric == 0):
                metric_name = 'x-position'
                avg_use = np.copy(pos_avg)
                peak_time_diff_use_list = list([pos_peak_time_diff_max, pos_peak_time_diff_min])
            elif (i_metric == 1):
                metric_name = 'x-velocity'
                avg_use = np.copy(velo_avg)
                peak_time_diff_use_list = list([velo_peak_time_diff_max, velo_peak_time_diff_min])
            elif (i_metric == 2):
                metric_name = 'angle'
                avg_use = np.copy(ang_avg[0])
                peak_time_diff_use_list = list([ang_peak_time_diff_max, ang_peak_time_diff_min])
            elif (i_metric == 3):
                metric_name = 'angular velocity'
                avg_use = np.copy(ang_velo_avg[0])
                peak_time_diff_use_list = list([ang_velo_peak_time_diff_max, ang_velo_peak_time_diff_min])
            avg_use = avg_use - np.min(avg_use, 1)[:, None]
            avg_use = avg_use / np.max(avg_use, 1)[:, None]
            if verbose_test:
                ax_test.clear()
                for i_paw in range(4):
                    ax_test.plot(range(nAll),
                                 avg_use[i_paw],
                                 linestyle='-', marker='', color=colors_paws[i_paw],
                                 alpha=1.0, zorder=1)
                ylim_min = np.nanmin(avg_use)
                ylim_max = np.nanmax(avg_use)
                offset = (ylim_max - ylim_min) * 0.05
                ax_test.set_ylim([ylim_min - offset, ylim_max + offset])
            for i_sign in range(2):
                if (i_sign == 0):
                    sign_name = 'maximum'
                    time_indices = np.argmax(avg_use, 1).astype(np.float64)
                    peak_time_diff_use = peak_time_diff_use_list[0]
                else:
                    sign_name = 'minimum'
                    time_indices = np.argmin(avg_use, 1).astype(np.float64)
                    peak_time_diff_use = peak_time_diff_use_list[1]
                for i_val in range(3):
                    paw_index = np.argmin(time_indices)
                    time_index0 = time_indices[paw_index]
                    time_indices[paw_index] = np.inf
                    paw_index = np.argmin(time_indices)
                    time_index1 = time_indices[paw_index]
                    #
                    diff = time_index1 - time_index0
                    diff = diff * (1.0/frame_rate) * time_fac # index -> ms
                    peak_time_diff_use.append(diff)
                    #
                    if verbose_test:
                        ax_test.plot(np.array([time_index0, time_index0], dtype=np.float64),
                                     np.array([ylim_min - offset, ylim_max + offset], dtype=np.float64),
                                     linestyle='-', marker='', color=colors_min_max[i_sign],
                                     alpha=0.5, zorder=0)
                        ax_test.plot(np.array([time_index1, time_index1], dtype=np.float64),
                                     np.array([ylim_min - offset, ylim_max + offset], dtype=np.float64),
                                     linestyle='-', marker='', color=colors_min_max[i_sign],
                                     alpha=0.5, zorder=0)
                print('{:s} {:s} peak time differences:'.format(metric_name, sign_name))
                print('peaks:')
                print(peak_time_diff_use)
                print('avg.:\t{:0.8f}'.format(np.mean(np.array(peak_time_diff_use, dtype=np.float64))))
                print('sd:\t{:0.8f}'.format(np.std(np.array(peak_time_diff_use, dtype=np.float64))))
            if verbose_test:
                fig_test.canvas.draw()
                plt.show(block=False)
                input()
                
    
        # PLOT
        offset_x_fac = 2.5
        i_paw_plot_start = 0
        i_paw_plot_end = 3
        # position
    #     range_y_min = np.inf
    #     range_y_max = -np.inf
        for i_paw in range(i_paw_plot_start, i_paw_plot_end+1):
            y = pos_avg[i_paw] - np.mean(pos_avg[i_paw], 0)
            ax1.plot(time, y, linestyle='-', marker='', linewidth=linewidth, color=colors_paws[i_paw], alpha=1.0, zorder=i_paw+1+4)
        for i_paw in range(i_paw_plot_start, i_paw_plot_end+1): # for correct legend order
            y = pos_avg[i_paw] - np.mean(pos_avg[i_paw], 0)
            ax1.fill_between(x=time, y1=y-pos_std[i_paw], y2=y+pos_std[i_paw],
                             color=colors_paws[i_paw], linewidth=linewidth,
                             alpha=0.2, zorder=i_paw+1,
                             where=None, interpolate=False, step=None, data=None)
    #         range_y_min = min(range_y_min, np.min(y-pos_std[i_paw]))
    #         range_y_max = max(range_y_max, np.max(y+pos_std[i_paw]))
    #     range_y = range_y_max - range_y_min
    #     range_diff = ax1_range - range_y
    #     if (range_diff < 0.0):
    #         print('ERROR: Adjust y-range for axis 1')
    #         print(range_diff)
    #         raise
    #     else:
    #         range_y_use = np.array([range_y_min-range_diff*0.5, range_y_max+range_diff*0.5], dtype=np.float64)
        ax1.set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
        ax1.set_ylim([ax1_min, ax1_max])
        h_legend = ax1.legend(paws_joint_names_legend[i_paw_plot_start:i_paw_plot_end+1], loc='upper right', fontsize=fontsize, frameon=False)
        for text in h_legend.get_texts():
            text.set_fontname(fontname)
        ax1.set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
        ax1.set_ylabel('avg. {:s}-position (cm)'.format(axis_s), fontsize=fontsize, fontname=fontname)
        cfg_plt.plot_coord_ax(ax1, '100 ms', '1 cm', 100.0, 1.0)
        fig1.canvas.draw()
        if save:
            for i_folder in range(nFolders):
                fig1.savefig(folder_save+'/gait_position__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
    #                          bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)
        # velocity
    #     range_y_min = np.inf
    #     range_y_max = -np.inf
        for i_paw in range(i_paw_plot_start, i_paw_plot_end+1):
            y = velo_avg[i_paw] - np.mean(velo_avg[i_paw], 0)
            ax2.plot(time, y, linestyle='-', marker='', linewidth=linewidth, color=colors_paws[i_paw], alpha=1.0, zorder=i_paw+1+4)
        for i_paw in range(i_paw_plot_start, i_paw_plot_end+1): # for correct legend order
            y = velo_avg[i_paw] - np.mean(velo_avg[i_paw], 0)
            ax2.fill_between(x=time, y1=y-velo_std[i_paw], y2=y+velo_std[i_paw],
                             color=colors_paws[i_paw], linewidth=linewidth,
                             alpha=0.2, zorder=i_paw+1,
                             where=None, interpolate=False, step=None, data=None)
    #         range_y_min = min(range_y_min, np.min(y-velo_std[i_paw]))
    #         range_y_max = max(range_y_max, np.max(y+velo_std[i_paw]))
    #     range_y = range_y_max - range_y_min
    #     range_diff = ax2_range - range_y
    #     if (range_diff < 0.0):
    #         print('ERROR: Adjust y-range for axis 2')
    #         print(range_diff)
    #         raise
    #     else:
    #         range_y_use = np.array([range_y_min-range_diff*0.5, range_y_max+range_diff*0.5], dtype=np.float64)
    #     ax2.plot(time, np.full(len(time), threshold_velocity, dtype=np.float64), linestyle='-', marker='', linewidth=linewidth color='black', alpha=0.5, zorder=10) # also plot velocity threshold
        ax2.set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
        ax2.set_ylim([ax2_min, ax2_max])
        h_legend = ax2.legend(paws_joint_names_legend[i_paw_plot_start:i_paw_plot_end+1], loc='upper right', fontsize=fontsize, frameon=False)
        for text in h_legend.get_texts():
            text.set_fontname(fontname)
        ax2.set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
        ax2.set_ylabel('avg. {:s}-velocity (cm/ms)'.format(axis_s), fontsize=fontsize, fontname=fontname)
        cfg_plt.plot_coord_ax(ax2, '100 ms', '0.05 cm/ms', 100.0, 0.05)
        fig2.canvas.draw()
        if save:
            for i_folder in range(nFolders):
                fig2.savefig(folder_save+'/gait_velocity__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
    #                          bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)
    #     # acceleration 
    #     for i_paw in range(i_paw_plot_start, i_paw_plot_end+1):
    #         ax3.plot(time, acc_avg[i_paw], linestyle='-', marker='', linewidth=linewidth, color=colors_paws[i_paw], alpha=1.0, zorder=i_paw+1+4)
    #     for i_paw in range(i_paw_plot_start, i_paw_plot_end+1): # for correct legend order
    #         ax3.fill_between(x=time, y1=acc_avg[i_paw]-acc_std[i_paw], y2=acc_avg[i_paw]+acc_std[i_paw],
    #                          color=colors_paws[i_paw], linewidth=linewidth,
    #                          alpha=0.2, zorder=i_paw+1,
    #                          where=None, interpolate=False, step=None, data=None)
    #     ax3.set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
    #     h_legend = ax3.legend(paws_joint_names_legend[i_paw_plot_start:i_paw_plot_end+1], loc='upper right', fontsize=fontsize, frameon=False)
    #     for text in h_legend.get_texts():
    #         text.set_fontname(fontname)
    #     ax3.set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
    #     ax3.set_ylabel('avg. {:s}-acceleration (cm/s^2)'.format(axis_s), fontsize=fontsize, fontname=fontname)
    #     fig3.canvas.draw()
    # #     if save:
    # #         for i_folder in range(nFolders):
    # #             fig3.savefig(folder_save+'/gait_acceleration__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
    # #                          bbox_inches="tight",
    # #                          dpi=300,
    # #                          transparent=True,
    # #                          format='svg',
    # #                          pad_inches=0)
        # angle
        for i_ang_pair in range(nAngles_pairs):
            for i_paw in range(i_paw_plot_start, i_paw_plot_end+1):
                y = ang_avg[i_ang_pair, i_paw] - np.mean(ang_avg[i_ang_pair, i_paw], 0)
                ax4[i_ang_pair].plot(time, y, linestyle='-', marker='', linewidth=linewidth, color=colors_paws[i_paw], alpha=1.0, zorder=i_paw+1+4)
            for i_paw in range(i_paw_plot_start, i_paw_plot_end+1): # for correct legend order
                y = ang_avg[i_ang_pair, i_paw] - np.mean(ang_avg[i_ang_pair, i_paw], 0)
                ax4[i_ang_pair].fill_between(x=time, y1=y-ang_std[i_ang_pair, i_paw], y2=y+ang_std[i_ang_pair, i_paw],
                                 color=colors_paws[i_paw], linewidth=linewidth,
                                 alpha=0.2, zorder=i_paw+1,
                                 where=None, interpolate=False, step=None, data=None)
            ax4[i_ang_pair].set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
            ax4[i_ang_pair].set_ylim([ax4_min, ax4_max])
            ax4[i_ang_pair].set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
            h_legend = ax4[i_ang_pair].legend(angle_joint_legend_list[i_ang_pair][i_paw_plot_start:i_paw_plot_end+1], loc='upper right', fontsize=fontsize, frameon=False)
            for text in h_legend.get_texts():
                text.set_fontname(fontname)
            ax4[i_ang_pair].set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
            ax4[i_ang_pair].set_ylabel('avg. 3D joint angle (deg)'.format(axis_s), fontsize=fontsize, fontname=fontname)
            cfg_plt.plot_coord_ax(ax4[i_ang_pair], '100 ms', '20 deg', 100.0, 20.0)
            fig4[i_ang_pair].canvas.draw()
            if save:
                for i_folder in range(nFolders):
                    fig4[i_ang_pair].savefig(folder_save+'/gait_angle__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
#                                              bbox_inches="tight",
                                             dpi=300,
                                             transparent=True,
                                             format='svg',
                                             pad_inches=0)
        # angle velocity
        for i_ang_pair in range(nAngles_pairs):
            for i_paw in range(i_paw_plot_start, i_paw_plot_end+1):
                y = ang_velo_avg[i_ang_pair, i_paw] - np.mean(ang_velo_avg[i_ang_pair, i_paw], 0)
                ax41[i_ang_pair].plot(time, y, linestyle='-', marker='', linewidth=linewidth, color=colors_paws[i_paw], alpha=1.0, zorder=i_paw+1+4)
            for i_paw in range(i_paw_plot_start, i_paw_plot_end+1): # for correct legend order
                y = ang_velo_avg[i_ang_pair, i_paw] - np.mean(ang_velo_avg[i_ang_pair, i_paw], 0)
                ax41[i_ang_pair].fill_between(x=time, y1=y-ang_velo_std[i_ang_pair, i_paw], y2=y+ang_velo_std[i_ang_pair, i_paw],
                                 color=colors_paws[i_paw], linewidth=linewidth,
                                 alpha=0.2, zorder=i_paw+1,
                                 where=None, interpolate=False, step=None, data=None)
            ax41[i_ang_pair].set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
            ax41[i_ang_pair].set_ylim([ax41_min, ax41_max])
            ax41[i_ang_pair].set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
            h_legend = ax41[i_ang_pair].legend(angle_joint_legend_list[i_ang_pair][i_paw_plot_start:i_paw_plot_end+1], loc='upper right', fontsize=fontsize, frameon=False)
            for text in h_legend.get_texts():
                text.set_fontname(fontname)
            ax41[i_ang_pair].set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
            ax41[i_ang_pair].set_ylabel('avg. 3D joint angular velocity (deg/ms)'.format(axis_s), fontsize=fontsize, fontname=fontname)
            cfg_plt.plot_coord_ax(ax41[i_ang_pair], '100 ms', '1 deg/ms', 100.0, 1)
            fig41[i_ang_pair].canvas.draw()
            if save:
                for i_folder in range(nFolders):
                    fig41[i_ang_pair].savefig(folder_save+'/gait_angle_velocity__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
#                                              bbox_inches="tight",
                                             dpi=300,
                                             transparent=True,
                                             format='svg',
                                             pad_inches=0)


        # position limb
        for i_joint in range(nJointsLimb):
            y = pos_avg_limb[i_joint] - np.mean(pos_avg_limb[i_joint], 0)
            ax5.plot(time, y, linestyle='-', marker='', linewidth=linewidth, color=colors_limb[i_joint], alpha=1.0, zorder=i_joint+1+nJointsLimb)
        for i_joint in range(nJointsLimb): # for correct legend order
            y = pos_avg_limb[i_joint] - np.mean(pos_avg_limb[i_joint], 0)
            ax5.fill_between(x=time, y1=y-pos_std_limb[i_joint], y2=y+pos_std_limb[i_joint],
                             color=colors_limb[i_joint], linewidth=linewidth,
                             alpha=0.2, zorder=i_joint+1,
                             where=None, interpolate=False, step=None, data=None)
        ax5.set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
        ax5.set_ylim([ax5_min, ax5_max])
        h_legend = ax5.legend(joints_list_legend_limb, loc='upper right', fontsize=fontsize, frameon=False)
        for text in h_legend.get_texts():
            text.set_fontname(fontname)
        ax5.set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
        ax5.set_ylabel('avg. {:s}-position (cm)'.format(axis_s), fontsize=fontsize, fontname=fontname)
        cfg_plt.plot_coord_ax(ax5, '100 ms', '1 cm', 100.0, 1.0)
        fig5.canvas.draw()
        if save:
            for i_folder in range(nFolders):
                fig5.savefig(folder_save+'/gait_position_limb__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
                             bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)
        # velocity limb
        for i_joint in range(nJointsLimb):
            y = velo_avg_limb[i_joint] - np.mean(velo_avg_limb[i_joint], 0)
            ax6.plot(time, y, linestyle='-', marker='', linewidth=linewidth, color=colors_limb[i_joint], alpha=1.0, zorder=i_joint+1+nJointsLimb)
        for i_joint in range(nJointsLimb): # for correct legend order
            y = velo_avg_limb[i_joint] - np.mean(velo_avg_limb[i_joint], 0)
            ax6.fill_between(x=time, y1=y-velo_std_limb[i_joint], y2=y+velo_std_limb[i_joint],
                             color=colors_limb[i_joint], linewidth=linewidth,
                             alpha=0.2, zorder=i_joint+1,
                             where=None, interpolate=False, step=None, data=None)
    #     ax6.plot(time, np.full(len(time), threshold_velocity, dtype=np.float64), linestyle='-', marker='', linewidth=linewidth, color='black', alpha=0.5, zorder=10) # also plot velocity threshold
        ax6.set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
        ax6.set_ylim([ax6_min, ax6_max])
        h_legend = ax6.legend(joints_list_legend_limb, loc='upper right', fontsize=fontsize, frameon=False)
        for text in h_legend.get_texts():
            text.set_fontname(fontname)
        ax6.set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
        ax6.set_ylabel('avg. {:s}-velocity (cm/ms)'.format(axis_s), fontsize=fontsize, fontname=fontname)
        cfg_plt.plot_coord_ax(ax6, '100 ms', '0.05 cm/ms', 100.0, 0.05)
        fig6.canvas.draw()
        if save:
            for i_folder in range(nFolders):
                fig6.savefig(folder_save+'/gait_velocity_limb__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
                             bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)
    #     # acceleration limb
    #     for i_joint in range(nJointsLimb):
    #         ax7.plot(time, acc_avg_limb[i_joint], linestyle='-', marker='', linewidth=linewidth, color=colors_limb[i_joint], alpha=1.0, zorder=i_joint+1+nJointsLimb)
    #     for i_joint in range(nJointsLimb): # for correct legend order
    #         ax7.fill_between(x=time, y1=acc_avg_limb[i_joint]-acc_std_limb[i_joint], y2=acc_avg_limb[i_joint]+acc_std_limb[i_joint],
    #                          color=colors_limb[i_joint], linewidth=linewidth,
    #                          alpha=0.2, zorder=i_joint+1,
    #                          where=None, interpolate=False, step=None, data=None)
    #     ax7.set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
    #     h_legend = ax7.legend(joints_list_legend_limb, loc='upper right', fontsize=fontsize, frameon=False)
    #     for text in h_legend.get_texts():
    #         text.set_fontname(fontname)
    #     ax7.set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
    #     ax7.set_ylabel('avg. {:s}-acceleration (cm/ms^2)'.format(axis_s), fontsize=fontsize, fontname=fontname)
    #     fig7.canvas.draw()
    # #     if save:
    # #         for i_folder in range(nFolders):
    # #             fig7.savefig(folder_save+'/gait_acceleration_limb__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
    # #                          bbox_inches="tight",
    # #                          dpi=300,
    # #                          transparent=True,
    # #                          format='svg',
    # #                          pad_inches=0)
        # angle limb
    #     range_y_min = np.inf
    #     range_y_max = -np.inf
        for i_joint in range(nJointsLimb-1):
            y = ang_avg_limb[i_joint] - np.mean(ang_avg_limb[i_joint], 0)
            ax8.plot(time, y, linestyle='-', marker='', linewidth=linewidth, color=colors_limb[i_joint], alpha=1.0, zorder=i_joint+1+nJointsLimb-1)
        for i_joint in range(nJointsLimb-1): # for correct legend order
            y = ang_avg_limb[i_joint] - np.mean(ang_avg_limb[i_joint], 0)
            ax8.fill_between(x=time, y1=y-ang_std_limb[i_joint], y2=y+ang_std_limb[i_joint],
                             color=colors_limb[i_joint], linewidth=linewidth,
                             alpha=0.2, zorder=i_joint+1,
                             where=None, interpolate=False, step=None, data=None)
    #         range_y_min = min(range_y_min, np.min(y-ang_std_limb[i_joint]))
    #         range_y_max = max(range_y_max, np.max(y+ang_std_limb[i_joint]))
    #     range_y = range_y_max - range_y_min
    #     range_diff = ax8_range - range_y
    #     if (range_diff < 0.0):
    #         print('ERROR: Adjust y-range for axis 8')
    #         print(range_diff)
    #         raise
    #     else:
    #         range_y_use = np.array([range_y_min-range_diff*0.5, range_y_max+range_diff*0.5], dtype=np.float64)
        ax8.set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
        ax8.set_ylim([ax8_min, ax8_max])
        h_legend = ax8.legend(joints_list_legend_limb[:-1], loc='upper right', fontsize=fontsize, frameon=False)
        for text in h_legend.get_texts():
            text.set_fontname(fontname)
        ax8.set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
        ax8.set_ylabel('avg. 3D joint angle (deg)'.format(axis_s), fontsize=fontsize, fontname=fontname)
        cfg_plt.plot_coord_ax(ax8, '100 ms', '20 deg', 100.0, 20.0)
        fig8.canvas.draw()
        if save:
            for i_folder in range(nFolders):
                fig8.savefig(folder_save+'/gait_angle_limb__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
    #                          bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)
        # angle velocity limb
        for i_joint in range(nJointsLimb-1):
            y = ang_velo_avg_limb[i_joint] - np.mean(ang_velo_avg_limb[i_joint], 0)
            ax9.plot(time, y, linestyle='-', marker='', linewidth=linewidth, color=colors_limb[i_joint], alpha=1.0, zorder=i_joint+1+nJointsLimb-1)
        for i_joint in range(nJointsLimb-1): # for correct legend order
            y = ang_velo_avg_limb[i_joint] - np.mean(ang_velo_avg_limb[i_joint], 0)
            ax9.fill_between(x=time, y1=y-ang_velo_std_limb[i_joint], y2=y+ang_velo_std_limb[i_joint],
                             color=colors_limb[i_joint], linewidth=linewidth,
                             alpha=0.2, zorder=i_joint+1,
                             where=None, interpolate=False, step=None, data=None)
        ax9.set_xlim([time[0]-(1.0/frame_rate)*offset_x_fac*time_fac, time[-1]+(1.0/frame_rate)*offset_x_fac*time_fac])
        ax9.set_ylim([ax9_min, ax9_max])
        h_legend = ax9.legend(joints_list_legend_limb[:-1], loc='upper right', fontsize=fontsize, frameon=False)
        for text in h_legend.get_texts():
            text.set_fontname(fontname)
        ax9.set_xlabel('time (ms)', fontsize=fontsize, fontname=fontname)
        ax9.set_ylabel('avg. 3D joint angular velocity (deg/ms)'.format(axis_s), fontsize=fontsize, fontname=fontname)
        cfg_plt.plot_coord_ax(ax9, '100 ms', '1 deg/ms', 100.0, 1)
        fig9.canvas.draw()
        if save:
            for i_folder in range(nFolders):
                fig9.savefig(folder_save+'/gait_angle_velocity_limb__population__{:s}__dlc.svg'.format(paws_joint_names[i_paw_timing]),
    #                          bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)

        if verbose:
            plt.show(block=False)
            print('Press any key to continue')
            input()
    if verbose:
        plt.show()
