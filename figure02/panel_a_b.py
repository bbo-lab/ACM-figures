#!/usr/bin/env python3

import importlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import os
import sys
import torch

from jax import jacobian
import jax.numpy as jnp
import math
from scipy.optimize import minimize

import cfg_plt

sys.path.append(os.path.abspath('../ACM/config'))
import configuration as cfg
sys.path.pop(sys.path.index(os.path.abspath('../ACM/config')))
#
sys.path.append(os.path.abspath('../ACM'))
import anatomy
import data
import helper
import kalman
import model
sys_path0 = np.copy(sys.path)

save = False
verbose = True

folder_save = os.path.abspath('panels')

folder_recon = data.path + '/reconstruction' 
folder_list = list(['/20200205/arena_20200205_033300_033500__pcutoff9e-01', # 0 # 20200205
                    '/20200205/arena_20200205_034400_034650__pcutoff9e-01', # 1
                    '/20200205/arena_20200205_038400_039000__pcutoff9e-01', # 2
                    '/20200205/arena_20200205_039200_039500__pcutoff9e-01', # 3
                    '/20200205/arena_20200205_039800_040000__pcutoff9e-01', # 4
                    '/20200205/arena_20200205_040500_040700__pcutoff9e-01', # 5
                    '/20200205/arena_20200205_041500_042000__pcutoff9e-01', # 6
                    '/20200205/arena_20200205_045250_045500__pcutoff9e-01', # 7
                    '/20200205/arena_20200205_046850_047050__pcutoff9e-01', # 8
                    '/20200205/arena_20200205_048500_048700__pcutoff9e-01', # 9
                    '/20200205/arena_20200205_049200_050300__pcutoff9e-01', # 10
                    '/20200205/arena_20200205_050900_051300__pcutoff9e-01', # 11
                    '/20200205/arena_20200205_052050_052200__pcutoff9e-01', # 12
                    '/20200205/arena_20200205_055200_056000__pcutoff9e-01', # 13
                    '/20200207/arena_20200207_032200_032550__pcutoff9e-01', # 14 # 20200207
                    '/20200207/arena_20200207_044500_045400__pcutoff9e-01', # 15
                    '/20200207/arena_20200207_046300_046900__pcutoff9e-01', # 16
                    '/20200207/arena_20200207_048000_049800__pcutoff9e-01', # 17
                    '/20200207/arena_20200207_050000_050400__pcutoff9e-01', # 18
                    '/20200207/arena_20200207_050800_051300__pcutoff9e-01', # 19
                    '/20200207/arena_20200207_052250_052650__pcutoff9e-01', # 20
                    '/20200207/arena_20200207_053050_053500__pcutoff9e-01', # 21
                    '/20200207/arena_20200207_055100_056200__pcutoff9e-01', # 22
                    '/20200207/arena_20200207_058400_058900__pcutoff9e-01', # 23
                    '/20200207/arena_20200207_059450_059950__pcutoff9e-01', # 24
                    '/20200207/arena_20200207_060400_060800__pcutoff9e-01', # 25
                    '/20200207/arena_20200207_061000_062100__pcutoff9e-01', # 26
                    '/20200207/arena_20200207_064100_064400__pcutoff9e-01',]) # 27
folder_list = list([folder_recon + i for i in folder_list])
folder = folder_list[23] # fig02 # arena

list_is_large_animal = list([0])

sys.path = list(np.copy(sys_path0))
sys.path.append(folder)
importlib.reload(cfg)
cfg.animal_is_large = list_is_large_animal[0]
importlib.reload(anatomy)

frame_start = cfg.index_frame_start
frame_end = cfg.index_frame_end
#
frame_start = cfg.index_frame_start + 300 # fig02
frame_end = cfg.index_frame_end - 25 # fig02

axis = 0
threshold = 25.0 # cm/s # fig02
nSamples_use = int(0) # fig02

cmap = plt.cm.viridis
cmap2 = plt.cm.tab10
color_left_front = cmap2(4/9)
color_right_front = cmap2(3/9)
color_left_hind = cmap2(9/9)
color_right_hind = cmap2(8/9)
colors_paws = list([color_left_front, color_right_front, color_left_hind, color_right_hind])

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
    
margin_all = 0.0
#
left_margin_x = 0.01 # inch
left_margin  = 1/3 # inch
right_margin = 0.05 # inch
bottom_margin = 0.25 # inch
top_margin = 0.05 # inch
between_margin_h = 0.1 # inch
between_margin_v = 0.1 # inch

fontsize = 6
linewidth = 0.5
fontname = "Arial"
markersize = 2

def rodrigues2rotMat(r): # nSigmaPoints, 3
    sqrt_arg = np.sum(r**2, 0)
    theta = np.sqrt(sqrt_arg)
    omega = r / theta
    omega_hat = np.array([[0.0, -omega[2], omega[1]],
                          [omega[2], 0.0, -omega[0]],
                          [-omega[1], omega[0], 0.0]], dtype=np.float64)
    rotMat = np.eye(3, dtype=np.float64) + \
             np.sin(theta) * omega_hat + \
             (1.0 - np.cos(theta)) * np.einsum('ij,jk->ik', omega_hat, omega_hat)
    return rotMat # nSigmaPoints, 3, 3

def get_paw_front(scale):
    paw = np.zeros((10, 2), dtype=np.float64)
    paw[0] = np.array([0.0, 0.0], dtype=np.float64)
    #
    paw[1] = np.array([2.0, 3.0], dtype=np.float64)
    paw[2] = np.array([10.0, 4.0], dtype=np.float64)
    paw[3] = np.array([7.0, 2.0], dtype=np.float64)
    paw[4] = np.array([11.0, 1.0], dtype=np.float64)
    #
    paw[5] = np.array([7.0, 0.0], dtype=np.float64)
    #
    #
    paw[6] = np.array([11.0, -1.0], dtype=np.float64)
    paw[7] = np.array([7.0, -2.0], dtype=np.float64)
    paw[8] = np.array([10.0, -4.0], dtype=np.float64)
    paw[9] = np.array([2.0, -3.0], dtype=np.float64)
    #
    paw = paw * (scale / np.max(paw))
    return paw

def get_paw_hind(scale):
    paw = np.zeros((12, 2), dtype=np.float64)
    paw[0] = np.array([0.0, 0.0], dtype=np.float64)
    #
    paw[1] = np.array([2.0, 3.0], dtype=np.float64)
    paw[2] = np.array([13.0, 5.0], dtype=np.float64)
    paw[3] = np.array([10.0, 3.0], dtype=np.float64)
    paw[4] = np.array([15.0, 2.0], dtype=np.float64)
    paw[5] = np.array([10.0, 1.0], dtype=np.float64)
    #
    paw[6] = np.array([15.0, 0.0], dtype=np.float64)
    #
    paw[7] = np.array([10.0, -1.0], dtype=np.float64)
    paw[8] = np.array([15.0, -2.0], dtype=np.float64)
    paw[9] = np.array([10.0, -3.0], dtype=np.float64)
    paw[10] = np.array([13.0, -5.0], dtype=np.float64)
    paw[11] = np.array([2.0, -3.0], dtype=np.float64)
    #
    paw = paw * (scale / np.max(paw))
    return paw

if __name__ == '__main__':
    x_ini = np.load(folder+'/x_ini.npy', allow_pickle=True)
    
    save_dict = np.load(folder+'/save_dict.npy', allow_pickle=True).item()
    if ('mu_uks' in save_dict):
        mu_uks = save_dict['mu_uks'][1:]
        var_uks = save_dict['var_uks'][1:]
        nSamples = np.copy(nSamples_use)
        print(save_dict['message'])
    else:
        mu_uks = save_dict['mu'][1:]
        nT = np.size(mu_uks, 0)
        nPara = np.size(mu_uks, 1)
        var_dummy = np.identity(nPara, dtype=np.float64) * 2**-52
        var_uks = np.tile(var_dummy.ravel(), nT).reshape(nT, nPara, nPara)
        nSamples = int(1)
    #
    mu_uks = mu_uks[frame_start-cfg.index_frame_ini:frame_end-cfg.index_frame_ini]
    var_uks = var_uks[frame_start-cfg.index_frame_ini:frame_end-cfg.index_frame_ini]
    #
    folder_reqFiles = data.path + '/required_files' 
    file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
    file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
    file_model = folder_reqFiles + '/model.npy'
#     file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/' + 'labels_dlc_{:06d}_{:06d}.npy'.format(cfg.index_frame_start, cfg.index_frame_end)
    file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/' + cfg.file_labelsDLC.split('/')[-1]

    args_model = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                      cfg.scale_factor, cfg.pcutoff)
    
    if ((cfg.mode == 1) or (cfg.mode == 2)):
        args_model['use_custom_clip'] = False
    elif ((cfg.mode == 3) or (cfg.mode == 4)):
        args_model['use_custom_clip'] = True
    nBones = args_model['numbers']['nBones']
    nMarkers = args_model['numbers']['nMarkers']
    skeleton_edges = args_model['model']['skeleton_edges'].cpu().numpy()
    joint_order = args_model['model']['joint_order']
    bone_lengths_index = args_model['model']['bone_lengths_index'].cpu().numpy()
    joint_marker_order = args_model['model']['joint_marker_order']
    free_para_bones = args_model['free_para_bones'].cpu().numpy()
    free_para_markers = args_model['free_para_markers'].cpu().numpy()
    free_para_pose = args_model['free_para_pose'].cpu().numpy()
    free_para_bones = np.zeros_like(free_para_bones, dtype=bool)
    free_para_markers = np.zeros_like(free_para_markers, dtype=bool)
    nFree_bones = int(0)
    nFree_markers = int(0)
    free_para = np.concatenate([free_para_bones,
                                free_para_markers,
                                free_para_pose], 0)
    args_model['x_torch'] = torch.from_numpy(x_ini).type(model.float_type)
    args_model['x_free_torch'] = torch.from_numpy(x_ini[free_para]).type(model.float_type)
    args_model['free_para_bones'] = torch.from_numpy(free_para_bones)
    args_model['free_para_markers'] = torch.from_numpy(free_para_markers)
    args_model['nFree_bones'] = nFree_bones
    args_model['nFree_markers'] = nFree_markers    
    args_model['plot'] = True
    del(args_model['model']['surface_vertices'])
    #
    nFree_para = np.sum(free_para, dtype=np.int64)
    nJoints = nBones + 1
    nT_use = np.size(mu_uks, 0)
    mu_t = torch.from_numpy(mu_uks)
    var_t = torch.from_numpy(var_uks)
    
    distribution = torch.distributions.MultivariateNormal(loc=mu_t,
                                                          scale_tril=kalman.cholesky_save(var_t))
    z_samples = distribution.sample((nSamples,))
    z_all = torch.cat([mu_t[None, :], z_samples], 0)
    z_all = z_all.reshape((nSamples+1)*nT_use, nFree_para)
    _, _, skeleton_pos_samples = model.fcn_emission_free(z_all, args_model)
    skeleton_pos_samples = skeleton_pos_samples.reshape(nSamples+1, nT_use, nJoints, 3)
    skeleton_pos_samples = skeleton_pos_samples.cpu().numpy()
    skeleton_pos = skeleton_pos_samples[0]
    skeleton_pos_peak = np.copy(skeleton_pos)
    
    # do coordinate transformation (skeleton_pos_samples)
    joint_index1 = joint_order.index('joint_spine_002') # pelvis
    joint_index2 = joint_order.index('joint_spine_004') # tibia
    origin = skeleton_pos_samples[:, :, joint_index1]
    x_direc = skeleton_pos_samples[:, :, joint_index2] - origin
    x_direc = x_direc[:, :, :2]
    x_direc = x_direc / np.sqrt(np.sum(x_direc**2, 2))[:, :, None]
    alpha = np.arctan2(x_direc[:, :, 1], x_direc[:, :, 0])
    cos_m_alpha = np.cos(-alpha)
    sin_m_alpha = np.sin(-alpha)
    R = np.stack([np.stack([cos_m_alpha, -sin_m_alpha], 2),
                  np.stack([sin_m_alpha, cos_m_alpha], 2)], 2)
    skeleton_pos_samples = skeleton_pos_samples - origin[:, :, None, :]
    skeleton_pos_xy = np.einsum('snij,snmj->snmi', R, skeleton_pos_samples[:, :, :, :2])
    skeleton_pos_samples[:, :, :, :2] = np.copy(skeleton_pos_xy)
    # do coordinate transformation (skeleton_pos_peak)
    joint_index1 = joint_order.index('joint_spine_002') # pelvis
    joint_index2 = joint_order.index('joint_spine_004') # tibia
    origin = skeleton_pos_peak[:, joint_index1]
    x_direc = skeleton_pos_peak[:, joint_index2] - origin
    x_direc = x_direc[:, :2]
    x_direc = x_direc / np.sqrt(np.sum(x_direc**2, 1))[:, None]
    alpha = np.arctan2(x_direc[:, 1], x_direc[:, 0])
    cos_m_alpha = np.cos(-alpha)
    sin_m_alpha = np.sin(-alpha)
    R = np.stack([np.stack([cos_m_alpha, -sin_m_alpha], 1),
                  np.stack([sin_m_alpha, cos_m_alpha], 1)], 1)
    skeleton_pos_peak = skeleton_pos_peak - origin[:, None, :]
    skeleton_pos_xy = np.einsum('nij,nmj->nmi', R, skeleton_pos_peak[:, :, :2])
    skeleton_pos_peak[:, :, :2] = np.copy(skeleton_pos_xy)

    paws_joint_names = list(['joint_wrist_left', 'joint_wrist_right', 'joint_ankle_left', 'joint_ankle_right'])

    paws_joint_names_legend = list([' '.join(i.split('_')[::-1]) for i in paws_joint_names])
    paws_marker_names_legend = list([' '.join(i.split(' ')[:-1])+' marker' for i in paws_joint_names_legend])
    
    angle_joint_list = list([paws_joint_names])
    
    #
    if (axis == 0):
        axis_s = 'x'
    elif (axis == 1):
        axis_s = 'y'
    elif (axis == 2):
        axis_s = 'z'        
    # 1D position
    joints_pos_1d = np.zeros((4, nT_use), dtype=np.float64)
    for i_paw in range(4):
        joint_index = joint_order.index(paws_joint_names[i_paw])
        joints_pos_1d[i_paw] = skeleton_pos_peak[:, joint_index, axis] # cm
    dy_dx_accuracy = 8
    dy_dx = \
            (+1.0/280.0 * joints_pos_1d[:, :-8] + \
             -4.0/105.0 * joints_pos_1d[:, 1:-7] + \
             +1.0/5.0 * joints_pos_1d[:, 2:-6] + \
             -4.0/5.0 * joints_pos_1d[:, 3:-5] + \
#              0.0 * joints_pos_1d[:, 4:-4] + \
             +4.0/5.0 * joints_pos_1d[:, 5:-3] + \
             -1.0/5.0 * joints_pos_1d[:, 6:-2] + \
             +4.0/105.0 * joints_pos_1d[:, 7:-1] + \
             -1.0/280.0 * joints_pos_1d[:, 8:]) / (1.0/cfg.frame_rate)
    # 3D derivative
    velo = np.zeros((4, nT_use-dy_dx_accuracy, 3), dtype=np.float64)
    for i_paw in range(4):
        joint_index = joint_order.index(paws_joint_names[i_paw])
        velo[i_paw] = \
                    (+1.0/280.0 * skeleton_pos_peak[:-8, joint_index] + \
                     -4.0/105.0 * skeleton_pos_peak[1:-7, joint_index] + \
                     +1.0/5.0 * skeleton_pos_peak[2:-6, joint_index] + \
                     -4.0/5.0 * skeleton_pos_peak[3:-5, joint_index] + \
#                      0.0 * skeleton_pos_peak[4:-4, joint_index] + \
                     +4.0/5.0 * skeleton_pos_peak[5:-3, joint_index] + \
                     -1.0/5.0 * skeleton_pos_peak[6:-2, joint_index] + \
                     +4.0/105.0 * skeleton_pos_peak[7:-1, joint_index] + \
                     -1.0/280.0 * skeleton_pos_peak[8:, joint_index]) / (1.0/cfg.frame_rate)
        
    position_mean = np.full((4, nT_use), np.nan, dtype=np.float64)
    velocity_mean = np.full((4, nT_use), np.nan, dtype=np.float64)
    angle_mean =  np.full((4, nT_use), np.nan, dtype=np.float64)   
    angle_velocity_mean =  np.full((4, nT_use), np.nan, dtype=np.float64)   
    position_std = np.full((4, nT_use), np.nan, dtype=np.float64)
    velocity_std = np.full((4, nT_use), np.nan, dtype=np.float64)
    angle_std =  np.full((4, nT_use), np.nan, dtype=np.float64)
    angle_velocity_std =  np.full((4, nT_use), np.nan, dtype=np.float64)
    for i_paw in range(4):
        joint_index = joint_order.index(paws_joint_names[i_paw])
        position = skeleton_pos_samples[:, :, joint_index, axis]
        velocity = \
                (+1.0/280.0 * position[:, :-8] + \
                 -4.0/105.0 * position[:, 1:-7] + \
                 +1.0/5.0 * position[:, 2:-6] + \
                 -4.0/5.0 * position[:, 3:-5] + \
#                  0.0 * position[:, 4:-4] + \
                 +4.0/5.0 * position[:, 5:-3] + \
                 -1.0/5.0 * position[:, 6:-2] + \
                 +4.0/105.0 * position[:, 7:-1] + \
                 -1.0/280.0 * position[:, 8:]) / (1.0/cfg.frame_rate)
        
        index_joint1 = skeleton_edges[np.where(skeleton_edges[:, 1] == joint_index)[0][0], 0]
        index_joint2 = np.copy(joint_index)
        index_joint3 = skeleton_edges[np.where(skeleton_edges[:, 0] == joint_index)[0][0], 1]
        vec1 = skeleton_pos_samples[:, :, index_joint2] - skeleton_pos_samples[:, :, index_joint1]
        vec2 = skeleton_pos_samples[:, :, index_joint3] - skeleton_pos_samples[:, :, index_joint2]
        vec1 = vec1 / np.sqrt(np.sum(vec1**2, 2))[:, :, None]
        vec2 = vec2 / np.sqrt(np.sum(vec2**2, 2))[:, :, None]
        angle = np.arccos(np.einsum('sni,sni->sn', vec1, vec2)) * 180.0/np.pi 
        
        angle_velocity = \
                (+1.0/280.0 * angle[:, :-8] + \
                 -4.0/105.0 * angle[:, 1:-7] + \
                 +1.0/5.0 * angle[:, 2:-6] + \
                 -4.0/5.0 * angle[:, 3:-5] + \
#                  0.0 * angle[:, 4:-4] + \
                 +4.0/5.0 * angle[:, 5:-3] + \
                 -1.0/5.0 * angle[:, 6:-2] + \
                 +4.0/105.0 * angle[:, 7:-1] + \
                 -1.0/280.0 * angle[:, 8:]) / (1.0/cfg.frame_rate)
        position_mean[i_paw] = np.copy(position[0])
        position_std[i_paw] = np.std(position, 0)
        velocity_mean[i_paw, int(dy_dx_accuracy/2):nT_use-int(dy_dx_accuracy/2)] = np.copy(velocity[0])
        velocity_std[i_paw, int(dy_dx_accuracy/2):nT_use-int(dy_dx_accuracy/2)] = np.std(velocity, 0)
        angle_mean[i_paw] = np.copy(angle[0])
        angle_std[i_paw] = np.std(angle, 0)
        angle_velocity_mean[i_paw, int(dy_dx_accuracy/2):nT_use-int(dy_dx_accuracy/2)] = np.copy(angle_velocity[0])
        angle_velocity_std[i_paw, int(dy_dx_accuracy/2):nT_use-int(dy_dx_accuracy/2)] = np.std(angle_velocity, 0)
        
    # find peak indices
    derivative_accuracy_index = int(dy_dx_accuracy/2)
    mask = (abs(dy_dx) < threshold)
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
                        index_peak = np.argmin(abs(dy_dx[i_paw][dy_dx_index])) + dy_dx_index[0] + derivative_accuracy_index
                        indices_peaks[i_paw].append(index_peak)
            else:
                index_peak = np.argmin(abs(dy_dx[i_paw])) + derivative_accuracy_index
                indices_peaks[i_paw].append(index_peak)

    # get paw polygons
    bone_lengths = x_ini[:nBones]
    bone_lengths = bone_lengths[bone_lengths_index]
    bone_ends_name = list(np.array(joint_order)[skeleton_edges[:, 1]])
    #
    paw_front_scale = bone_lengths[bone_ends_name.index('joint_finger_left_002')]
    paw_front = get_paw_front(paw_front_scale)
    paw_hind_scale = bone_lengths[bone_ends_name.index('joint_paw_hind_left')] + bone_lengths[bone_ends_name.index('joint_toe_left_002')]
    paw_hind = get_paw_hind(paw_hind_scale)
    
    time = np.arange(nT_use, dtype=np.float64) * 1.0/cfg.frame_rate

    # PLOT
    # create figures
    fig_w_metric = np.round(mm_in_inch * 88.0*2/3, decimals=2)
    fig_h_metric = np.round(mm_in_inch * 88.0*1/3, decimals=2)
    fig_w_hist = np.round(mm_in_inch * 88.0*2/3, decimals=2)
    fig_h_hist = np.round(mm_in_inch * 88.0*1/3, decimals=2)
    # position
    fig1_w = fig_w_metric
    fig1_h = fig_h_metric
    fig1 = plt.figure(1, figsize=(fig1_w, fig1_h))
    fig1.canvas.manager.window.move(0, 0)
    fig1.clear()
    ax1_x = margin_all/fig1_w
    ax1_y = margin_all/fig1_h
    ax1_w = 1.0 - (margin_all/fig1_w + margin_all/fig1_w)
    ax1_h = 1.0 - (margin_all/fig1_h + margin_all/fig1_h)
    ax1 = fig1.add_axes([ax1_x, ax1_y, ax1_w, ax1_h])
    ax1.clear()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_position([0.1, 0, 0.9, 1])
    # position hist
    fig11_w = fig_w_hist
    fig11_h = fig_h_hist
    fig11 = plt.figure(11, figsize=(fig11_w, fig11_h))
    fig11.canvas.manager.window.move(0, 0)
    fig11.clear()
    ax11_x = left_margin/fig11_w
    ax11_y = bottom_margin/fig11_h
    ax11_w = 1.0 - (left_margin/fig11_w + right_margin/fig11_w)
    ax11_h = 1.0 - (bottom_margin/fig11_h + top_margin/fig11_h)
    ax11 = fig11.add_axes([ax11_x, ax11_y, ax11_w, ax11_h])
    ax11.clear()
    ax11.spines["top"].set_visible(False)
    ax11.spines["right"].set_visible(False)
    # velocity
    fig2_w = fig_w_metric
    fig2_h = fig_h_metric
    fig2 = plt.figure(2, figsize=(fig2_w, fig2_h))
    fig2.canvas.manager.window.move(0, 0)
    fig2.clear()
    ax2_x = margin_all/fig2_w
    ax2_y = margin_all/fig2_h
    ax2_w = 1.0 - (margin_all/fig2_w + margin_all/fig2_w)
    ax2_h = 1.0 - (margin_all/fig2_h + margin_all/fig2_h)
    ax2 = fig2.add_axes([ax2_x, ax2_y, ax2_w, ax2_h])
    ax2.clear()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_position([0.1, 0, 0.9, 1])
    # velocity hist
    fig21_w = fig_w_hist
    fig21_h = fig_h_hist
    fig21 = plt.figure(21, figsize=(fig21_w, fig21_h))
    fig21.canvas.manager.window.move(0, 0)
    fig21.clear()
    ax21_x = left_margin/fig21_w
    ax21_y = bottom_margin/fig21_h
    ax21_w = 1.0 - (left_margin/fig21_w + right_margin/fig21_w)
    ax21_h = 1.0 - (bottom_margin/fig21_h + top_margin/fig21_h)
    ax21 = fig21.add_axes([ax21_x, ax21_y, ax21_w, ax21_h])
    ax21.clear()
    ax21.spines["top"].set_visible(False)
    ax21.spines["right"].set_visible(False)
    # angles
    fig50_w = fig_w_metric
    fig50_h = fig_h_metric
    fig50 = plt.figure(50, figsize=(fig50_w, fig50_h))
    fig50.canvas.manager.window.move(0, 0)
    fig50.clear()    
    ax50_x = margin_all/fig50_w
    ax50_y = margin_all/fig50_h
    ax50_w = 1.0 - (margin_all/fig50_w + margin_all/fig50_w)
    ax50_h = 1.0 - (margin_all/fig50_h + margin_all/fig50_h)
    ax50 = fig50.add_axes([ax50_x, ax50_y, ax50_w, ax50_h])
    ax50.clear()
    ax50.spines["top"].set_visible(False)
    ax50.spines["right"].set_visible(False)
    ax50.set_position([0.1, 0, 0.9, 1])
    # angles hist
    fig501_w = fig_w_hist
    fig501_h = fig_h_hist
    fig501 = plt.figure(501, figsize=(fig501_w, fig501_h))
    fig501.canvas.manager.window.move(0, 0)
    fig501.clear()
    ax501_x = left_margin/fig501_w
    ax501_y = bottom_margin/fig501_h
    ax501_w = 1.0 - (left_margin/fig501_w + right_margin/fig501_w)
    ax501_h = 1.0 - (bottom_margin/fig501_h + top_margin/fig501_h)
    ax501 = fig501.add_axes([ax501_x, ax501_y, ax501_w, ax501_h])
    ax501.clear()
    ax501.spines["top"].set_visible(False)
    ax501.spines["right"].set_visible(False)
    # angular velocities
    fig60_w = fig_w_metric
    fig60_h = fig_h_metric
    fig60 = plt.figure(60, figsize=(fig60_w, fig60_h))
    fig60.canvas.manager.window.move(0, 0)
    fig60.clear()    
    ax60_x = margin_all/fig60_w
    ax60_y = margin_all/fig60_h
    ax60_w = 1.0 - (margin_all/fig60_w + margin_all/fig60_w)
    ax60_h = 1.0 - (margin_all/fig60_h + margin_all/fig60_h)
    ax60 = fig60.add_axes([ax60_x, ax60_y, ax60_w, ax60_h])
    ax60.clear()
    ax60.spines["top"].set_visible(False)
    ax60.spines["right"].set_visible(False)
    ax60.set_position([0.1, 0, 0.9, 1])

    fig3_w = np.round(mm_in_inch * 88.0, decimals=2)
    fig3_h = np.round(mm_in_inch * 88.0*2/3, decimals=2)
    fig3 = plt.figure(3, figsize=(fig3_w, fig3_h))
    fig3.canvas.manager.window.move(0, 0)
    fig3.clear()
    ax3_3_x = left_margin/fig3_w
    ax3_3_y = bottom_margin/fig3_h
    ax3_3_w = 1.0 - (left_margin/fig3_w + right_margin/fig3_w)
    ax3_3_h = 1.0/3.0 * ((1.0-top_margin/fig3_h-bottom_margin/fig3_h) - 2.0*between_margin_h/fig3_h)
    ax3_3 = fig3.add_axes([ax3_3_x, ax3_3_y, ax3_3_w, ax3_3_h])
    ax3_3.clear()
    ax3_3.spines["top"].set_visible(False)
    ax3_3.spines["right"].set_visible(False)
    ax3_2_x = left_margin/fig3_w
    ax3_2_y = ax3_3_y + ax3_3_h + between_margin_h/fig3_h
    ax3_2_w = 1.0 - (left_margin/fig3_w + right_margin/fig3_w)
    ax3_2_h = 1.0/3.0 * ((1.0-top_margin/fig3_h-bottom_margin/fig3_h) - 2.0*between_margin_h/fig3_h)
    ax3_2 = fig3.add_axes([ax3_2_x, ax3_2_y, ax3_2_w, ax3_2_h])
    ax3_2.clear()
    ax3_2.spines["top"].set_visible(False)
    ax3_2.spines["right"].set_visible(False)
    ax3_1_x = left_margin/fig3_w
    ax3_1_y = ax3_2_y + ax3_2_h + between_margin_h/fig3_h
    ax3_1_w = 1.0 - (left_margin/fig3_w + right_margin/fig3_w)
    ax3_1_h = 1.0/3.0 * ((1.0-top_margin/fig3_h-bottom_margin/fig3_h) - 2.0*between_margin_h/fig3_h)
    ax3_1 = fig3.add_axes([ax3_1_x, ax3_1_y, ax3_1_w, ax3_1_h])
    ax3_1.clear()
    ax3_1.spines["top"].set_visible(False)
    ax3_1.spines["right"].set_visible(False)

    # plot 3d
    fig100_w = np.round(mm_in_inch * 88.0, decimals=2)
    fig100_h = fig100_w
    fig100 = plt.figure(100, figsize=(fig100_w, fig100_h))
    fig100.canvas.manager.window.move(0, 0)
    fig100.clear()
    ax100 = fig100.add_subplot(1, 1, 1, projection='3d')
    ax100.clear()
    # plot 3d leg
    fig105_w = np.round(mm_in_inch * 88.0*5/12, decimals=2)
    fig105_h = fig105_w
    fig105 = plt.figure(105, figsize=(fig105_w, fig105_h))
    fig105.canvas.manager.window.move(0, 0)
    fig105.clear()
    ax105 = fig105.add_subplot(1, 1, 1, projection='3d')
    ax105.clear()
    # plot 3d leg (pos, velo, ang)
    fig106_w = np.round(mm_in_inch * 88.0*0.425, decimals=2)
    fig106_h = fig106_w
    fig106 = plt.figure(106, figsize=(fig106_w, fig106_h))
    fig106.canvas.manager.window.move(0, 0)
    fig106.clear()
    ax106 = fig106.add_subplot(1, 1, 1, projection='3d')
    ax106.clear()
    # plot 2d paws
    fig101_w = np.round(mm_in_inch * 88.0, decimals=2)
    fig101_h = np.round(mm_in_inch * 88.0, decimals=2)
    fig101 = plt.figure(101, figsize=(fig101_w, fig101_h))
    fig101.canvas.manager.window.move(0, 0)
    fig101.clear()
    ax101 = fig101.add_subplot(1, 1, 1)
    ax101.clear()
    ax101.set_position([0, 0, 1, 1])
    ax101.set_aspect(1)
    
    def calc_corr_fit(x, args):
        corr_fit = jnp.exp(-x[1]*args['t']) * jnp.cos(x[0]*2.0*math.pi*args['t'])
        return corr_fit
    
    def obj_fcn(x, args):
        corr1 = calc_corr_fit(x, args)
        res = jnp.sum((corr1 - args['corr0'])**2)
        return res

    def obj_fcn_wrap(x, args):
        output = obj_fcn(x, args)
        output_wrap = np.asarray(output, dtype=np.float64)
        return output_wrap

    grad_fcn = jacobian(obj_fcn)
    
    def grad_fcn_wrap(x, args):
        output = grad_fcn(x, args)
        output_wrap = np.asarray(output, dtype=np.float64)
        return output_wrap
    #
    fig111_w = fig_w_metric
    fig111_h = fig_h_metric
    fig111 = plt.figure(111, figsize=(fig111_w, fig111_h))
    fig111.canvas.manager.window.move(0, 0)
    fig111.clear()
    ax111_x = margin_all/fig111_w
    ax111_y = margin_all/fig111_h
    ax111_w = 1.0 - (margin_all/fig111_w + margin_all/fig111_w)
    ax111_h = 1.0 - (margin_all/fig111_h + margin_all/fig111_h)
    ax111 = fig111.add_axes([ax11_x, ax11_y, ax11_w, ax11_h])
    ax111.clear()
    ax111.spines["top"].set_visible(False)
    ax111.spines["right"].set_visible(False)
    for i in range(4):
        y = velocity_mean[i]-np.mean(velocity_mean[i][~np.isnan(velocity_mean[i])]) # velocity
        mask = ~np.isnan(y)
        nValid = np.sum(mask, dtype=np.int64)
        y = y[mask]
        
        corr = np.correlate(y, y, mode='full')
        corr0 = np.copy(corr[int(len(time[:nValid]))-1:])
        corr0 = corr0 / len(time[:nValid])
        corr0 = corr0 / np.std(y)**2
  
        ax111.plot(time[:nValid], corr0, color=colors_paws[i], linewidth=linewidth, zorder=1)
        #
        args_fit = dict()
        args_fit['t'] = np.copy(time[:nValid])
        args_fit['corr0'] = np.copy(corr0)
        #
        x = np.array([1.0/0.3, 1.0], dtype=np.float64)
        #
        tol = 2**-52
        min_result = minimize(obj_fcn_wrap,
                              x,
                              args=args_fit,
                              method='l-bfgs-b',
                              jac=grad_fcn_wrap,
                              hess=None,
                              hessp=None,
                              bounds=None,
                              constraints=(),
                              tol=tol,
                              callback=None,
                              options={'disp': True,
                                       'maxcor': 100,
                                       'ftol': tol,
                                       'gtol': tol,
                                       'eps': None,
                                       'maxfun': np.inf,
                                       'maxiter': np.inf,
                                       'iprint': -1,
                                       'maxls': 200,
                                       'finite_diff_rel_step': None})
        corr_fit = calc_corr_fit(min_result.x, args_fit)
        #
        SS_tot = np.sum((corr0 - np.mean(corr0))**2)
        SS_res = np.sum((corr0 - corr_fit)**2)
        R2 = 1.0 - SS_res/SS_tot
        #
        print(grad_fcn_wrap(x, args_fit))
        print(grad_fcn_wrap(min_result.x, args_fit))
        print(i, 1.0/min_result.x[0], min_result.x[1])
        print('n:\t{:04d}'.format(len(corr0)))
        print('R2:\t{:0.8f}'.format(R2))
        #
        ax111.plot(time[:nValid], corr_fit, color=colors_paws[i], linewidth=linewidth, zorder=2, alpha=0.5)
        #
        dBin = 0.05
        bin_range = np.arange(-1.0, 1.0+dBin, dBin, dtype=np.float64)
        nBin = len(bin_range) - 1
        n_hist = np.float64(len(corr0))
        hist = np.histogram(corr0, bins=nBin, range=[bin_range[0], bin_range[-1]], normed=None, weights=None, density=False)
        y_hist = np.zeros(2+2*len(hist[0]), dtype=np.float64)
        y_hist[0] = 0.0
        y_hist[1:-1:2] = np.copy(hist[0] / n_hist)
        y_hist[2:-1:2] = np.copy(hist[0] / n_hist)
        y_hist[-1] = 0.0
        x_hist = np.zeros(2+2*(len(hist[1])-1), dtype=np.float64)
        x_hist[0] = np.copy(hist[1][0])
        x_hist[1:-1:2] = np.copy(hist[1][:-1])
        x_hist[2:-1:2] = np.copy(hist[1][1:])
        x_hist[-2] = 1.0
        x_hist[-1] = 1.0
        ax111.plot(y_hist - 0.5, x_hist, color=colors_paws[i], linewidth=linewidth, zorder=2, alpha=0.5)
        #
        fig111.canvas.draw()
    ax111.set_xlabel('time (s)')
    ax111.set_ylabel('auto-correlation')
    fig111.canvas.draw()
    
    # position
    for i in range(4):
        x = np.copy(time)
        y = position_mean[i]-np.mean(position_mean[i])
        ax1.plot(x, y, color=colors_paws[i], linewidth=linewidth, zorder=1)
    for i in range(4):
        x = np.copy(time)
        y = position_mean[i]-np.mean(position_mean[i])
        y_std = position_std[i]
        ax1.fill_between(x=x, y1=y-y_std, y2=y+y_std,
                         color=colors_paws[i],
                         alpha=0.2, zorder=0, linewidth=linewidth,
                         where=None, interpolate=False, step=None, data=None)
    h_legend = ax1.legend(paws_joint_names_legend, loc='upper right', fontsize=fontsize, frameon=False)
    for text in h_legend.get_texts():
        text.set_fontname(fontname)
    ax1.set_xlim([time[0]-(1.0/cfg.frame_rate)*5.0, time[-1]+(1.0/cfg.frame_rate)*5.0])
    ax1.set_axis_off()
    cfg_plt.plot_coord_ax(ax1, '0.25 s', '2 cm', 0.25, 2.0)
    fig1.canvas.draw()
    # position hist
    dBin = 1.0
    bin_range = np.arange(np.floor(np.min(position_mean))-dBin,
                          np.ceil(np.max(position_mean))+2*dBin,
                          dBin, dtype=np.float64)
    for i in range(4):
        y = position_mean[i]
        hist = np.histogram(y, bins=len(bin_range)-1, range=[bin_range[0], bin_range[-1]], normed=None, weights=None, density=True)
        x = np.zeros(1+2*len(hist[0]), dtype=np.float64)
        x[1::2] = np.copy(hist[0])
        x[2::2] = np.copy(hist[0])
        y = np.zeros(1+2*(len(hist[1])-1), dtype=np.float64)
        y[0] = np.copy(hist[1][0])
        y[1::2] = np.copy(hist[1][:-1])
        y[2::2] = np.copy(hist[1][1:])
        ax11.plot(x, y, color=colors_paws[i], linestyle='-', marker='', linewidth=linewidth)
    ax11.set_ylim([bin_range[0], bin_range[-1]])
    ax11.set_yticks([-2, 5, 12])
    ax11.set_xlim([0.0-0.05, 0.4+0.05])
    ax11.set_xticks([0.0, 0.2, 0.4])
    ax11.set_xlabel('probability'.format(axis_s), va='center', ha='center', fontsize=fontsize, fontname=fontname)
    ax11.set_ylabel('{:s}-position (cm)'.format(axis_s), va='top', ha='center', fontsize=fontsize, fontname=fontname)
    fig11.canvas.draw()
    
    # velocity  
    for i in range(4):
        x = np.copy(time)
        y = velocity_mean[i]-np.mean(velocity_mean[i][~np.isnan(velocity_mean[i])])
        ax2.plot(x, y, color=colors_paws[i], linewidth=linewidth, zorder=1)
    for i in range(4):
        x = np.copy(time)
        y = velocity_mean[i]-np.mean(velocity_mean[i][~np.isnan(velocity_mean[i])])
        y_std = velocity_std[i]
        ax2.fill_between(x=x, y1=y-y_std, y2=y+y_std,
                         color=colors_paws[i],
                         alpha=0.2, zorder=0, linewidth=linewidth,
                         where=None, interpolate=False, step=None, data=None)
    h_legend = ax2.legend(paws_joint_names_legend, loc='lower right', fontsize=fontsize, frameon=False)
    for text in h_legend.get_texts():
        text.set_fontname(fontname)
    ax2.set_xlim([time[0]-(1.0/cfg.frame_rate)*5.0, time[-1]+(1.0/cfg.frame_rate)*5.0])
    ax2.set_axis_off()
    cfg_plt.plot_coord_ax(ax2, '0.25 s', '50 cm/s', 0.25, 50.0)
    fig2.canvas.draw()
    # velocity hist
    dBin = 5.0 # cm/s
    bin_range = np.arange(np.floor(np.min(velocity_mean[~np.isnan(velocity_mean)]))-dBin,
                          np.ceil(np.max(velocity_mean[~np.isnan(velocity_mean)]))+2*dBin,
                          dBin, dtype=np.float64)
    for i in range(4):
        y = velocity_mean[i][~np.isnan(velocity_mean[i])]
        hist = np.histogram(y, bins=len(bin_range)-1, range=[bin_range[0], bin_range[-1]], normed=None, weights=None, density=True)
        x = np.zeros(1+2*len(hist[0]), dtype=np.float64)
        x[1::2] = np.copy(hist[0])
        x[2::2] = np.copy(hist[0])
        y = np.zeros(1+2*(len(hist[1])-1), dtype=np.float64)
        y[0] = np.copy(hist[1][0])
        y[1::2] = np.copy(hist[1][:-1])
        y[2::2] = np.copy(hist[1][1:])
        ax21.plot(x, y/1e3, color=colors_paws[i], linestyle='-', marker='', linewidth=linewidth)
    ax21.set_ylim([bin_range[0]/1e3, bin_range[-1]/1e3])
    ax21.set_yticks([-60/1e3, 0.0, 60/1e3])
    ax21.set_xlim([0.0-0.005, 0.04+0.005])
    ax21.set_xticks([0.0, 0.02, 0.04])
    for text in h_legend.get_texts():
        text.set_fontname(fontname)
    ax21.set_xlabel('probability'.format(axis_s), va='center', ha='center', fontsize=fontsize, fontname=fontname)
    ax21.set_ylabel('{:s}-velocity (cm/ms)'.format(axis_s), va='top', ha='center', fontsize=fontsize, fontname=fontname)
    fig21.canvas.draw()

    # angle
    for i in range(4):
        x = np.copy(time)
        y = angle_mean[i]-np.mean(angle_mean[i])
        ax50.plot(x, y, color=colors_paws[i], linewidth=linewidth, zorder=1)
    for i in range(4):
        x = np.copy(time)
        y = angle_mean[i]-np.mean(angle_mean[i])
        y_std = angle_std[i]
        ax50.fill_between(x=x, y1=y-y_std, y2=y+y_std,
                         color=colors_paws[i],
                         alpha=0.2, zorder=0, linewidth=linewidth,
                         where=None, interpolate=False, step=None, data=None)
    h_legend = ax50.legend(paws_joint_names_legend, loc='upper right', fontsize=fontsize, frameon=False)
    for text in h_legend.get_texts():
        text.set_fontname(fontname)
    ax50.set_xlim([time[0]-(1.0/cfg.frame_rate)*5.0, time[-1]+(1.0/cfg.frame_rate)*5.0])
    ax50.set_axis_off()
    cfg_plt.plot_coord_ax(ax50, '0.25 s', '40 deg', 0.25, 40.0)
    fig50.canvas.draw()
    # angle hist
    dBin = 5.0 # deg
    bin_range = np.arange(np.floor(np.min(angle_mean[~np.isnan(angle_mean)]))-dBin,
                          np.ceil(np.max(angle_mean[~np.isnan(angle_mean)]))+2*dBin,
                          dBin, dtype=np.float64)
    for i in range(4):
        y = angle_mean[i]
        hist = np.histogram(y, bins=len(bin_range)-1, range=[bin_range[0], bin_range[-1]], normed=None, weights=None, density=True)
        x = np.zeros(1+2*len(hist[0]), dtype=np.float64)
        x[1::2] = np.copy(hist[0])
        x[2::2] = np.copy(hist[0])
        y = np.zeros(1+2*(len(hist[1])-1), dtype=np.float64)
        y[0] = np.copy(hist[1][0])
        y[1::2] = np.copy(hist[1][:-1])
        y[2::2] = np.copy(hist[1][1:])
        ax501.plot(x, y, color=colors_paws[i], linestyle='-', marker='', linewidth=linewidth)
    ax501.set_ylim([bin_range[0], bin_range[-1]])
    ax501.set_yticks([40, 90, 140])
    ax501.set_xlim([0.0-0.01, 0.08+0.01])
    ax501.set_xticks([0.0, 0.04, 0.08])
    ax501.set_xlabel('probability', va='center', ha='center', fontsize=fontsize, fontname=fontname)
    ax501.set_ylabel('angle (deg)', va='top', ha='center', fontsize=fontsize, fontname=fontname)
    fig501.canvas.draw()

    # angular velocity
    for i in range(4):
        x = np.copy(time)
        y = angle_velocity_mean[i]-np.mean(angle_velocity_mean[i][~np.isnan(angle_velocity_mean[i])])
        ax60.plot(x, y, color=colors_paws[i], linewidth=linewidth, zorder=1)
    for i in range(4):
        x = np.copy(time)
        y = angle_velocity_mean[i]-np.mean(angle_velocity_mean[i][~np.isnan(angle_velocity_mean[i])])
        y_std = angle_velocity_std[i]
        ax60.fill_between(x=x, y1=y-y_std, y2=y+y_std,
                         color=colors_paws[i],
                         alpha=0.2, zorder=0, linewidth=linewidth,
                         where=None, interpolate=False, step=None, data=None)
    h_legend = ax60.legend(paws_joint_names_legend, loc='upper right', fontsize=fontsize, frameon=False)
    for text in h_legend.get_texts():
        text.set_fontname(fontname)
    ax60.set_xlim([time[0]-(1.0/cfg.frame_rate)*5.0, time[-1]+(1.0/cfg.frame_rate)*5.0])
    ax60.set_axis_off()
    cfg_plt.plot_coord_ax(ax60, '0.25 s', '1000 deg/s', 0.25, 1000.0)
    fig60.canvas.draw()
    

    # detection
    joint_names_detec = list([['joint_elbow_left', 'joint_elbow_right', 'joint_knee_left', 'joint_knee_right'],
                              ['joint_wrist_left', 'joint_wrist_right', 'joint_ankle_left', 'joint_ankle_right'],
                              ['joint_finger_left_002', 'joint_finger_right_002', 'joint_paw_hind_left', 'joint_paw_hind_right']])
    joint_names_detec_legend = list([['left elbow marker', 'right elbow marker', 'left knee marker', 'right knee marker'],
                                     ['left wrist marker', 'right wrist marker', 'left ankle marker', 'right ankle marker'],
                                     ['left finger marker', 'right finger marker', 'left tarsometatarsal marker', 'right tarsometatarsal marker']])
    cmap3 = plt.cm.tab20b
    joint_names_detec_colors = list([[cmap3(17/19), cmap3(13/19), cmap3(1/19), cmap3(9/19)],
                                     [colors_paws[0], colors_paws[1], colors_paws[2], colors_paws[3]],
                                     [cmap3(18/19), cmap3(14/19), cmap3(2/19), cmap3(10/19)]])
    for j in range(3):
        if (j == 0):
            ax_use = ax3_1
        elif (j == 1):
            ax_use = ax3_2
        elif (j == 2):
            ax_use = ax3_3
        for i in range(4):
            joint_index = args_model['model']['joint_order'].index(joint_names_detec[j][i])
            marker_index = np.where(joint_index == abs(args_model['model']['joint_marker_index']))[0][0]
            x = np.copy(time)
            y = np.sum(args_model['labels_mask'][frame_start-cfg.index_frames_calib[0][0]:frame_end-cfg.index_frames_calib[0][0], :, marker_index].cpu().numpy(), 1)
            ax_use.plot(x, y, color=joint_names_detec_colors[j][i], linewidth=linewidth)
        h_legend = ax_use.legend(joint_names_detec_legend[j], loc='upper right', fontsize=fontsize, frameon=False)
        for text in h_legend.get_texts():
            text.set_fontname(fontname)
        ax_use.set_xlim([time[0]-(1.0/cfg.frame_rate)*2.0, time[-1]+(1.0/cfg.frame_rate)*2.0])
        ax_use.set_ylim([0.0-0.2, 4.0+0.2])
        if (j == 2):
            ax_use.set_xlabel('time (s)', va='center', ha='center', fontsize=fontsize, fontname=fontname)
            ax_use.set_xticks([0, 0.65, 1.3])
        else:
            ax_use.xaxis.set(visible=False)
            ax_use.spines["bottom"].set_visible(False)
        if (j == 1):
            ax_use.set_ylabel('number of detections', va='top', ha='center', fontsize=fontsize, fontname=fontname)
        ax_use.set_yticks([0.0, 2.0, 4.0])
    fig3.canvas.draw()

    # plot bones of last time point
    i_t = 74 # fig02
    color_joint = 'black'
    i_bone = 0
    joint_index_start = skeleton_edges[i_bone, 0]
    joint_start = skeleton_pos[i_t, joint_index_start]
    zorder_joint = 4
    ax100.plot([joint_start[0]], [joint_start[1]], [joint_start[2]],
            linestyle='', marker='.', markersize=2.5, markeredgewidth=0.5, color=color_joint, zorder=zorder_joint)
    for i_bone in range(nBones):
        joint_index_start = skeleton_edges[i_bone, 0]
        joint_index_end = skeleton_edges[i_bone, 1]
        joint_start = skeleton_pos[i_t, joint_index_start]
        joint_end = skeleton_pos[i_t, joint_index_end]
        vec = np.stack([joint_start, joint_end], 0)
        linewidth_bones = 2.0
        # flip colors because image will be mirrored in the paper
        if (joint_order[joint_index_end] == 'joint_finger_left_002'):
            color_bone = color_right_front
        elif (joint_order[joint_index_end] == 'joint_finger_right_002'):
            color_bone = color_left_front
        elif ((joint_order[joint_index_end] == 'joint_toe_left_002') or \
              (joint_order[joint_index_end] == 'joint_paw_hind_left')):
            color_bone = color_right_hind
        elif ((joint_order[joint_index_end] == 'joint_toe_right_002') or \
              (joint_order[joint_index_end] == 'joint_paw_hind_right')):
            color_bone = color_left_hind
        else:
            color_bone = 'darkgray'
            linewidth_bones = 1.5
        joint_name = joint_order[joint_index_end]
        joint_name_split = joint_name.split('_')
        if ('right' in joint_name_split):
            zorder_bone = 1
            zorder_joint = 2
        elif ('left' in joint_name_split):
            zorder_bone = 5
            zorder_joint = 6
        else:
            zorder_bone = 3
            zorder_joint = 4
        ax100.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                linestyle='-', marker=None, linewidth=linewidth_bones, color=color_bone, zorder=zorder_bone)
        ax100.plot([joint_end[0]], [joint_end[1]], [joint_end[2]],
                linestyle='', marker='.', markersize=2.5, markeredgewidth=0.5, color=color_joint, zorder=zorder_joint)
    #
    xyz_mean = np.mean(skeleton_pos[i_t], 0)
    xyzlim_range = 25
    #
    ground_plane_direc = skeleton_pos[i_t, joint_order.index('joint_spine_002'), :2] - \
                         skeleton_pos[i_t, joint_order.index('joint_spine_004'), :2]
    ground_plane_direc = ground_plane_direc / np.sqrt(np.sum(ground_plane_direc**2))
    alpha = np.arctan2(ground_plane_direc[1], ground_plane_direc[0])
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
    ground_plane_len_x = 35
    ground_plane_len_y = 15
    ground_plane = np.array([[-ground_plane_len_x/2, -ground_plane_len_y/2],
                             [ground_plane_len_x/2, -ground_plane_len_y/2],
                             [ground_plane_len_x/2, ground_plane_len_y/2],
                             [-ground_plane_len_x/2, ground_plane_len_y/2],], dtype=np.float64)
    ground_plane = np.einsum('ij,nj->ni', R, ground_plane)
    ground_plane = ground_plane + xyz_mean[None, :2]
    ground_plane = ground_plane + ground_plane_direc * 2.5
    ground_plane_poly = plt.Polygon(ground_plane, color='black', zorder=0, alpha=0.1)   
    ax100.add_patch(ground_plane_poly)
    art3d.pathpatch_2d_to_3d(ground_plane_poly, z=0, zdir='z')
    #
    ax100.set_proj_type('persp')
    ax100.view_init(15, -45+180)
    ax100.dist = 4.5
    ax100.set_axis_off()
    #
    ax100.set_xlim3d([xyz_mean[0]-xyzlim_range, xyz_mean[0]+xyzlim_range])
    ax100.set_ylim3d([xyz_mean[1]-xyzlim_range, xyz_mean[1]+xyzlim_range])
    ax100.set_zlim3d([xyz_mean[2]-xyzlim_range, xyz_mean[2]+xyzlim_range])
    #
    fig100.canvas.draw()
    
    # plot 3d leg
    side_list = list(['right'])
    for side in side_list:
        ax105.clear()
        time_point_index_min = nT_use
        time_point_index_max = 0
        if (side == 'left'):
            i_paw = 2
            i_peak = 0#1
            time_point_indices_list = np.arange(-5, 5+1, 1, dtype=np.int64)
        elif (side == 'right'):
            i_paw = 3
            i_peak = 0#1
            time_point_indices_list = np.arange(-5, 5+1, 1, dtype=np.int64)
        nLegs = len(time_point_indices_list)
        joint_name_origin = 'joint_hip_{:s}'.format(side)
        leg_joints_list = list(['joint_hip_{:s}'.format(side),
                                'joint_knee_{:s}'.format(side),
                                'joint_ankle_{:s}'.format(side),
                                'joint_paw_hind_{:s}'.format(side)])
        xyz_mean = np.zeros(3, dtype=np.float64)
        xyz_mean_counter = 0
        for i_leg in time_point_indices_list:
            time_point_index = indices_peaks[i_paw][i_peak] + i_leg * 1
            joint_name_origin_index = joint_order.index(joint_name_origin)
            joint_origin = skeleton_pos_peak[time_point_index, joint_name_origin_index]
            for i_bone in range(nBones):
                joint_index_start = skeleton_edges[i_bone, 0]
                joint_name = joint_order[joint_index_start]
                if joint_name in leg_joints_list:
                    joint_index_end = skeleton_edges[i_bone, 1]
                    joint_start = skeleton_pos_peak[time_point_index, joint_index_start] - joint_origin
                    joint_end = skeleton_pos_peak[time_point_index, joint_index_end] - joint_origin
                    vec = np.stack([joint_start, joint_end], 0)
                    color_leg = cmap((i_leg+abs(min(time_point_indices_list)))/(nLegs-1))
                    ax105.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                            linestyle='-', marker='', linewidth=1, color=color_leg, zorder=1, alpha=0.5)
                    ax105.plot([vec[1, 0]], [vec[1, 1]], [vec[1, 2]],
                               linestyle='', marker='.', markersize=3, markeredgewidth=0, color='black', zorder=2, alpha=1.0)

                    xyz_mean = xyz_mean + joint_end
                    xyz_mean_counter = xyz_mean_counter + 1
                    if (i_leg == 0): # origin joint is always the same for each leg
                        ax105.plot([vec[0, 0]], [vec[0, 1]], [vec[0, 2]],
                                   linestyle='', marker='.', markersize=3, markeredgewidth=0, color='black', zorder=2, alpha=1.0)
                        if (joint_name == joint_name_origin):
                            xyz_mean = xyz_mean + joint_start
                            xyz_mean_counter = xyz_mean_counter + 1
        xyz_mean = xyz_mean / float(xyz_mean_counter)
        #
        scale_length = 1.0 # cm
        scale_start = xyz_mean + np.array([-3.5, 0.0, 1.75], dtype=np.float64)
        scale_end = scale_start + np.array([scale_length, 0.0, 0.0], dtype=np.float64)
        vec = np.stack([scale_start, scale_end], 0)
        ax105.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                    linestyle='-', marker='', linewidth=1, color='black', zorder=3, alpha=1.0)
        ax105.text(scale_start[0] + scale_length/2.0, scale_start[1], scale_start[2] + 1/2, 
                  '{:0.0f} cm'.format(abs(scale_length)),
                  fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
        #
        ax105.view_init(0, -90)
        ax105.dist = 4.0
        ax105.set_axis_off()
        #
        xyzlim_range = 6.0
        ax105.set_xlim3d([xyz_mean[0]-xyzlim_range, xyz_mean[0]+xyzlim_range])
        ax105.set_ylim3d([xyz_mean[1]-xyzlim_range, xyz_mean[1]+xyzlim_range])
        ax105.set_zlim3d([xyz_mean[2]-xyzlim_range, xyz_mean[2]+xyzlim_range])
        #
        cax = fig105.add_axes([0.25, 0.25, 0.5, 0.05])
        mappable2d = plt.cm.ScalarMappable(cmap=cmap)
        mappable2d.set_clim([-5/cfg.frame_rate * 1e3, 5/cfg.frame_rate * 1e3])
        colorbar2d = fig105.colorbar(mappable=mappable2d, ax=ax105, cax=cax, ticks=[-50, 50], orientation='horizontal')
        colorbar2d.set_label('time (ms)', labelpad=-5, rotation=0, fontsize=fontsize, fontname=fontname)
        colorbar2d.ax.tick_params(labelsize=fontsize)
        #
        fig105.canvas.draw()
        if save:
            fig105.savefig(folder_save+'/gait_3d_sketch_leg_movement_{:s}.svg'.format(side),
                        bbox_inches='tight',
                        dpi=300,
                        transparent=True,
                        format='svg',
                        pad_inches=0)

        # plot 3d leg (pos, velo, ang)
        ax106.clear()
        time_point_index = indices_peaks[i_paw][i_peak] - 4
        skeleton_pos_rest = np.copy(skeleton_pos_peak[time_point_index])
        #
        xyz_mean = np.zeros(3, dtype=np.float64)
        xyz_mean_counter = 0
        joint_name_origin_index = joint_order.index(joint_name_origin)
        joint_origin = skeleton_pos_rest[joint_name_origin_index]
        for i_bone in range(nBones):
            joint_index_start = skeleton_edges[i_bone, 0]
            joint_name = joint_order[joint_index_start]
            if joint_name in leg_joints_list:
                joint_index_end = skeleton_edges[i_bone, 1]
                joint_start = skeleton_pos_rest[joint_index_start] - joint_origin
                joint_end = skeleton_pos_rest[joint_index_end] - joint_origin
                xyz_mean = xyz_mean + joint_end
                xyz_mean_counter = xyz_mean_counter + 1
                if (joint_name == joint_name_origin):
                    xyz_mean = xyz_mean + joint_start
                    xyz_mean_counter = xyz_mean_counter + 1
                vec = np.stack([joint_start, joint_end], 0)
                ax106.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                        linestyle='-', marker='', linewidth=1, color='darkgray', zorder=1, alpha=1.0)
                ax106.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                        linestyle='', marker='.', markersize=3, markeredgewidth=1, color='black', zorder=2, alpha=1.0)
        xyz_mean = xyz_mean / xyz_mean_counter
        #
        ax106.view_init(0, -90)
        ax106.dist = ax105.dist
        ax106.set_axis_off()
        #
        xyzlim_range = 4.0
        ax106.set_xlim3d([xyz_mean[0]-xyzlim_range, xyz_mean[0]+xyzlim_range])
        ax106.set_ylim3d([xyz_mean[1]-xyzlim_range, xyz_mean[1]+xyzlim_range])
        ax106.set_zlim3d([xyz_mean[2]-xyzlim_range, xyz_mean[2]+xyzlim_range])
        #
        vec = np.stack([scale_start, scale_end], 0)
        ax106.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                linestyle='-', marker='', linewidth=1, color='black', zorder=3, alpha=1.0)
        ax106.text(scale_start[0] + scale_length/2.0, scale_start[1], scale_start[2] + 1/2, 
                  '{:0.0f} cm'.format(abs(scale_length)),
                  fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
        fig106.canvas.draw()

        text2_offset_z = -2.5
        # pos drawing
        vec_start = skeleton_pos_rest[joint_name_origin_index] - joint_origin
        vec_start[1] = 0.0
        vec_end = skeleton_pos_rest[joint_order.index('joint_ankle_{:s}'.format(side))] - joint_origin
        vec_end[1] = 0.0
        vec_start[2] = np.copy(vec_end[2])
        vec = np.stack([vec_start, vec_end], 0)
        vec_length = vec_end[0] - vec_start[0]
        pos_plot = ax106.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                linestyle='--', marker='', linewidth=1, color='black', zorder=1, alpha=1.0)
        pos_text = ax106.text(vec_start[0] + vec_length/2.0, vec_start[1], vec_start[2] + 1/3, 
                r'$p_x$',
                fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
        pos_text2 = ax106.text(xyz_mean[0], xyz_mean[1], xyz_mean[2] + text2_offset_z, 
                               r'joint x-position: $p_x$',
                               fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
        fig106.canvas.draw()
        if save:
            fig106.savefig(folder_save+'/gait_3d_sketch_leg_drawing_position_{:s}.svg'.format(side),
                        bbox_inches="tight",
                        dpi=300,
                        transparent=True,
                        format='svg',
                        pad_inches=0)
        # velo drawing
        pos_plot[0].set(visible=False)
        pos_text.set(visible=False)
        pos_text2.set(visible=False)
        vec_start = skeleton_pos_rest[joint_name_origin_index] - joint_origin
        vec_start[1] = 0.0
        vec_end = skeleton_pos_rest[joint_order.index('joint_ankle_{:s}'.format(side))] - joint_origin
        vec_end[1] = 0.0
        vec_start[2] = np.copy(vec_end[2])
        vec_start0 = np.copy(vec_start)
        vec = np.stack([vec_start, vec_end], 0)
        vec_length = vec_end[0] - vec_start[0]
        velo_plot1 = ax106.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                linestyle='-', marker='', linewidth=1, color='black', zorder=1, alpha=1.0)
        vec_start = vec_start0 + np.array([0.05, 0.0, 0.0], dtype=np.float64) # for 'round' arrow tip
        vec_end = vec_start + np.array([-0.1, 0.0, 0.1], dtype=np.float64)
        vec = np.stack([vec_start, vec_end], 0)
        velo_plot2 = ax106.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                linestyle='-', marker='', linewidth=1, color='black', zorder=1, alpha=1.0)
        vec_start = vec_start0 + np.array([0.05, 0.0, 0.0], dtype=np.float64) # for 'round' arrow tip
        vec_end = vec_start + np.array([-0.1, 0.0, -0.1], dtype=np.float64)
        vec = np.stack([vec_start, vec_end], 0)
        velo_plot3 = ax106.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                linestyle='-', marker='', linewidth=1, color='black', zorder=1, alpha=1.0)
        velo_text = ax106.text(vec_start[0], vec_start[1], vec_start[2] + 1/3, 
                r'$v_x$',
                fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
        velo_text2 = ax106.text(xyz_mean[0], xyz_mean[1], xyz_mean[2] + text2_offset_z, 
                                r'joint x-velocity: $|v_x|$',
                               fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
        fig106.canvas.draw()
        if save:
            fig106.savefig(folder_save+'/gait_3d_sketch_leg_drawing_velocity_{:s}.svg'.format(side),
                        bbox_inches="tight",
                        dpi=300,
                        transparent=True,
                        format='svg',
                        pad_inches=0)
        # ang drawing
        velo_plot1[0].set(visible=False)
        velo_plot2[0].set(visible=False)
        velo_plot3[0].set(visible=False)
        velo_text.set(visible=False)
        velo_text2.set(visible=False)
        vec_length_vec = skeleton_pos_rest[joint_order.index('joint_paw_hind_{:s}'.format(side))] - skeleton_pos_rest[joint_order.index('joint_ankle_{:s}'.format(side))]
        vec_length = np.sqrt(np.sum(vec_length_vec**2)) # cm
        vec_direc = skeleton_pos_rest[joint_order.index('joint_ankle_{:s}'.format(side))] - skeleton_pos_rest[joint_order.index('joint_knee_{:s}'.format(side))]
        vec_direc = vec_direc / np.sqrt(np.sum(vec_direc**2))
        vec_start = skeleton_pos_rest[joint_order.index('joint_ankle_{:s}'.format(side))] - joint_origin
        vec_end = vec_start + vec_direc * vec_length    
        vec = np.stack([vec_start, vec_end], 0)
        ax106.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                linestyle='-', marker='', linewidth=1.0, color='black', zorder=1, alpha=1.0)
        nAng = 20
        vec1 = vec_end - vec_start
        vec1 = vec1 / np.sqrt(np.sum(vec1**2))
        vec2 = np.copy(vec_length_vec)
        vec2 = vec2 / np.sqrt(np.sum(vec2**2))
        alpha = np.arccos(np.dot(vec1, vec2))
        vec_ang = vec1[None, :] * vec_length
        rodrigues = np.cross(vec1, vec2)
        rodrigues = rodrigues / np.sqrt(np.sum(rodrigues**2))
        for i in range(1, nAng+1):
            alpha_use = alpha * float(i)/float(nAng)
            rodrigues_use = rodrigues * alpha_use
            R = rodrigues2rotMat(rodrigues_use)
            vec_new = np.einsum('ij,j->i', R, vec_ang[0])     
            vec_ang = np.concatenate([vec_ang, vec_new[None, :]], 0)
        vec_ang = vec_ang + vec_start[None, :]
        ax106.plot(vec_ang[:, 0], vec_ang[:, 1], vec_ang[:, 2],
                linestyle='--', marker='', linewidth=1.0, color='black', zorder=1, alpha=1.0)
        text_ang_loc = (vec_start + vec_ang[0] + vec_ang[-1]) / 3.0
        ax106.text(text_ang_loc[0], text_ang_loc[1], text_ang_loc[2], 
                r'$\alpha$',
                fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
        ax106.text(xyz_mean[0], xyz_mean[1], xyz_mean[2] + text2_offset_z, 
                   r'joint angle: $\alpha$',
                   fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
        fig106.canvas.draw()
        if save:
            fig106.savefig(folder_save+'/gait_3d_sketch_leg_drawing_angle_{:s}.svg'.format(side),
                        bbox_inches="tight",
                        dpi=300,
                        transparent=True,
                        format='svg',
                        pad_inches=0)

    # schematic plot of 2d paws
    ax101.set_aspect(1)
    # plot paws at time points between the peaks
    time_point_index_min = nT_use
    time_point_index_max = 0
    for i in range(4):
        if (len(indices_peaks[i]) > 0):
            nPeaks = len(indices_peaks[i])
            paw_centers = list()
            for j in range(nPeaks):
                time_point_index = indices_peaks[i][j]
                #
                time_point_index_min = min(time_point_index_min, time_point_index)
                time_point_index_max = max(time_point_index_max, time_point_index)
                #
                joint_index = joint_order.index(paws_joint_names[i])
                paw_pos_xy = skeleton_pos[time_point_index, joint_index, :2]
                #
                joint_index = joint_order.index(paws_joint_names[i])
                index = np.where(skeleton_edges[:, 0] == joint_index)[0][0]
                paw_direc = skeleton_pos[time_point_index, skeleton_edges[index, 1], :2] - paw_pos_xy
                paw_direc_use = np.copy(paw_direc)
                alpha = np.arctan2(paw_direc_use[1], paw_direc_use[0])
                R = np.array([[np.cos(alpha), -np.sin(alpha)],
                              [np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
                if (i < 2):
                    paw_plot = np.einsum('ij,nj->ni', R, paw_front) + paw_pos_xy[None, :]
                else:
                    paw_plot = np.einsum('ij,nj->ni', R, paw_hind) + paw_pos_xy[None, :]
                    
                paw_center = np.mean(paw_plot, 0)
                cond = True
                if (np.size(paw_centers, 0) > 0):
                    dist = np.sqrt(np.sum((paw_center[None, :] - np.array(paw_centers, dtype=np.float64))**2, 1))
                    if np.any(dist < 5.0): # to prevent plotting of paw stamps that are too close to each other in the schematic
                        cond = False
                if cond:
                    paw_centers.append(paw_center)
                    paw_poly = plt.Polygon(paw_plot, color=colors_paws[i], zorder=0, alpha=0.5, linewidth=0.)   
                    ax101.add_patch(paw_poly)
    i_t = int(time_point_index_min + (time_point_index_max - time_point_index_min) * 2/4)
    xy_mean = np.mean(skeleton_pos[i_t], 0)[:2]

    offset_x_text = 1.0 # cm
    offset_y_text = -1.0 # cm
    offset_x = -11 # cm
    offset_y = 7 # cm
    len_x = 10.0 # cm
    len_y = -10.0 # cm
    ax101.plot(np.array([xy_mean[0]+offset_x, xy_mean[0]+offset_x + len_x], dtype=np.float64),
            np.array([xy_mean[1]+offset_y, xy_mean[1]+offset_y], dtype=np.float64),
            linestyle='-', marker='', color='black', zorder=2, linewidth=5*linewidth)
    ax101.plot(np.array([xy_mean[0]+offset_x, xy_mean[0]+offset_x], dtype=np.float64),
            np.array([xy_mean[1]+offset_y, xy_mean[1]+offset_y + len_y], dtype=np.float64),
            linestyle='-', marker='', color='black', zorder=2, linewidth=5*linewidth)
    ax101.text(xy_mean[0]+offset_x + len_x/2.0, xy_mean[1]+offset_y+offset_x_text, '{:0.0f} cm'.format(abs(len_x)),
            fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
    ax101.text(xy_mean[0]+offset_x+offset_y_text, xy_mean[1]+offset_y + len_y/2.0, '{:0.0f} cm'.format(abs(len_y)),
            fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='vertical')
    #
    ax101.set_axis_off()
    ax101.set_aspect(1)
    #
    legend_text = list(['left front paw', 'right front paw', 'left hind paw', 'right hind paw'])
    h_legend_list = list()
    for i_paw in range(4):
        h_legend = mlines.Line2D([], [],
                                 color=colors_paws[i_paw],
                                 marker='',
                                 linestyle='-',
                                 linewidth=2.5,
                                 label=legend_text[i_paw])
        h_legend_list.append(h_legend)    
    h_legend = ax101.legend(handles=h_legend_list, fontsize=fontsize, frameon=False)
    for text in h_legend.get_texts():
        text.set_fontname(fontname)
    #
    fig101.canvas.draw()
    
    for tick in ax1.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax1.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    for tick in ax11.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax11.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    for tick in ax2.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax2.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    for tick in ax21.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax21.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    for tick in ax3_1.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax3_1.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    for tick in ax3_2.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax3_2.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    for tick in ax3_3.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax3_3.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
        
    ax11.yaxis.set_label_coords(x=left_margin_x/fig11_w, y=ax11_y+0.5*ax11_h, transform=fig11.transFigure)
    ax21.yaxis.set_label_coords(x=left_margin_x/fig21_w, y=ax21_y+0.5*ax21_h, transform=fig21.transFigure)
    ax3_1.yaxis.set_label_coords(x=left_margin_x/fig3_w, y=ax3_1_y+0.5*ax3_1_h, transform=fig3.transFigure)
    ax3_2.yaxis.set_label_coords(x=left_margin_x/fig3_w, y=ax3_2_y+0.5*ax3_2_h, transform=fig3.transFigure)
    ax3_3.yaxis.set_label_coords(x=left_margin_x/fig3_w, y=ax3_3_y+0.5*ax3_3_h, transform=fig3.transFigure)
        
    fig100.canvas.draw()
    fig101.canvas.draw()
    fig1.canvas.draw()
    fig11.canvas.draw()
    fig2.canvas.draw()
    fig21.canvas.draw()
    fig3.canvas.draw()

    ax11.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax11.xaxis.get_offset_text().set_fontsize(fontsize)
    ax21.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax21.xaxis.get_offset_text().set_fontsize(fontsize)
    ax21.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax21.yaxis.get_offset_text().set_fontsize(fontsize)
    ax501.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax501.xaxis.get_offset_text().set_fontsize(fontsize)   
    for tick in ax11.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax11.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    for tick in ax21.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax21.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    for tick in ax501.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax501.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
        
    plt.pause(2**-10)
    fig100.canvas.draw()
    fig101.canvas.draw()
    fig1.canvas.draw()
    fig11.canvas.draw()
    fig2.canvas.draw()
    fig21.canvas.draw()
    fig50.canvas.draw()
    fig501.canvas.draw()
    fig60.canvas.draw()
    fig3.canvas.draw()
    plt.pause(2**-10)
    # save & verbose
    if save:
        fig100.savefig(folder_save+'/gait_3d_sketch.svg',
                    bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)
        fig101.savefig(folder_save+'/gait_3d_sketch_xy.svg',
                    bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0) 
#         fig1.savefig(folder_save+'/gait_3d_sketch_position.svg',
# #                     bbox_inches="tight",
#                     dpi=300,
#                     transparent=True,
#                     format='svg',
#                    pad_inches=0)
#         fig11.savefig(folder_save+'/gait_3d_sketch_position_hist.svg',
# #                     bbox_inches="tight",
#                     dpi=300,
#                     transparent=True,
#                     format='svg',
#                    pad_inches=0)
#         fig2.savefig(folder_save+'/gait_3d_sketch_velocity.svg',
# #                     bbox_inches="tight",
#                     dpi=300,
#                     transparent=True,
#                     format='svg',
#                    pad_inches=0)
#         fig21.savefig(folder_save+'/gait_3d_sketch_velocity_hist.svg',
# #                     bbox_inches="tight",
#                     dpi=300,
#                     transparent=True,
#                     format='svg',
#                    pad_inches=0)
#         fig501.savefig(folder_save+'/gait_3d_sketch_angle_hist.svg',
# #                         bbox_inches="tight",
#                     dpi=300,
#                     transparent=True,
#                     format='svg',
#                    pad_inches=0)
#         fig50.savefig(folder_save+'/gait_3d_sketch_angle.svg',
# #                         bbox_inches="tight",
#                     dpi=300,
#                     transparent=True,
#                     format='svg',
#                    pad_inches=0)   
#         fig60.savefig(folder_save+'/gait_3d_sketch_angle_velocity.svg',
# #                         bbox_inches="tight",
#                     dpi=300,
#                     transparent=True,
#                     format='svg',
#                    pad_inches=0) 
#         fig3.savefig(folder_save+'/gait_3d_sketch_detection.svg',
#                     bbox_inches="tight",
#                     dpi=300,
#                     transparent=True,
#                     format='svg',
#                    pad_inches=0)
    if verbose:
        plt.show()
    
