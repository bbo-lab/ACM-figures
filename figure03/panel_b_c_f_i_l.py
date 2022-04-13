#!/usr/bin/env python3

import importlib
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys
import torch

import jax.numpy as jnp
from jax import jacobian
from mpl_toolkits.mplot3d import Axes3D
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

save = True
save_leg = False
verbose = True

mode = 'mode1' # 'mode4' or 'mode1' change to the latter to generate plots for naive skeleton model

folder_save = os.path.abspath('panels') + '/trace/' + mode

if (mode == 'mode4'):
    add2folder = '__pcutoff9e-01'
elif (mode == 'mode1'):
    add2folder = '__mode1__pcutoff9e-01'
else:
    raise

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
                    '/dataset_analysis/M220217_DW01/ACM/M220217_DW01_20220401-155200/results/M220217_DW01_20220401-155200_20220401-155411', # 28
                    'dataset_analysis/R220318_DW01/20220318/ACM/R220318_DW01_20220318_01/results/R220318_DW01_20220318_01_20220331-205800_20220403-171117', # 29
                    'dataset_analysis/R220318_DW01/20220318/ACM/R220318_DW01_20220318_01/results/R220318_DW01_20220318_01_083000-083000_mode1_20220408-091122', # 30
                    ])
folder_list[0:28] = list([folder_recon + i + add2folder for i in folder_list[0:28]])
folder_list[28:] = list([data.path + '/' + i for i in folder_list[28:]])

project_folder_list = folder_list.copy()
project_folder_list[28:] = [ i + '/../../' for i in folder_list[28:] ]

frame_range = [None for i in folder_list]
frame_range[28] = [16600, int(16600+15+2*0.6*196)]
#frame_range[29] = [83240-15, int(83540+0.6*196)]
frame_range[29] = [83240, int(83240+15+3*196+0.6*196)]
frame_range[30] = [83240, int(83240+15+3*196+0.6*196)]

dataset_to_use = 30
# 14
folder = folder_list[dataset_to_use]
#
print(project_folder_list[dataset_to_use])
sys.path.append(project_folder_list[dataset_to_use])
importlib.reload(cfg)
cfg.animal_is_large = 0
importlib.reload(anatomy)

if type(cfg.index_frames_calib[0][0])=='int':
    index_frames_calib_start = cfg.index_frames_calib[0][0]
elif cfg.index_frames_calib=='all':
    index_frames_calib_start = sorted(np.load(cfg.file_labelsManual,allow_pickle=True)['arr_0'].item().keys())[0]
else:
    print(f'Error: {cfg.index_frames_calib[0]}')
    exit()

assert cfg.dt, "Frameskip not supported!"
#

if frame_range[dataset_to_use] is None:
    frame_start = cfg.index_frame_start + 130
    frame_end = frame_start + 120
else:
    frame_start = frame_range[dataset_to_use][0]
    frame_end = frame_range[dataset_to_use][1]

#frame_start = cfg.index_frame_start + 130
#frame_end = frame_start + 120
print(f"{frame_start} {frame_end}")

folder_save = f"{os.path.abspath('panels')}/trace/{mode}/{folder.split('/')[-1]}_{frame_start:06d}-{frame_end:06d}"
os.makedirs(folder_save,exist_ok=True)
print(f"folder_save {folder_save}")
axis = 0
threshold = 25.0 # cm/s
nSamples_use = int(3e3)

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
    
margin_all = 0.0 # inch
#
left_margin_x = 0.02 # inch
left_margin  = 0.4#0.4 # inch
right_margin = 0.1#0.025 # inch
bottom_margin_x = 0.015 # inch
bottom_margin = 0.3 # inch
top_margin = 0.025 # inch
between_margin_h = 0.1 # inch
between_margin_v = 0.1 # inch

fontsize = 6
linewidth = 0.5
fontname = "Arial"
markersize = 2

def calc_corr_fit(x, args):
    corr_fit = jnp.exp(-x[1]*args['t']) * jnp.cos(x[0]*2.0*math.pi*args['t'])
    return corr_fit

def obj_fcn(x, args):
    corr1 = calc_corr_fit(x, args)
    res = jnp.sum((corr1[None, :] - args['corr0'])**2)
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

if __name__ == '__main__':
    paws_joint_names = list(['joint_wrist_left', 'joint_wrist_right', 'joint_ankle_left', 'joint_ankle_right'])
    paws_joint_names_legend = list([' '.join(i.split('_')[::-1]) for i in paws_joint_names])
    
    x_ini = np.load(folder+'/x_ini.npy', allow_pickle=True)
    
    save_dict = np.load(folder+'/save_dict.npy', allow_pickle=True).item()
    if ('mu_uks' in save_dict):
        mu_uks = save_dict['mu_uks'][1:]
        var_uks = save_dict['var_uks'][1:]
        nSamples = np.copy(nSamples_use)
        print(save_dict['message'])
    else:
        mu_uks = save_dict['mu_fit'][1:]
        nT = np.size(mu_uks, 0)
        nPara = np.size(mu_uks, 1)
        var_dummy = np.identity(nPara, dtype=np.float64) * 2**-52
        var_uks = np.tile(var_dummy.ravel(), nT).reshape(nT, nPara, nPara)
        nSamples = int(0)
    #
    mu_uks = mu_uks[frame_start-cfg.index_frame_ini:frame_end-cfg.index_frame_ini]
    var_uks = var_uks[frame_start-cfg.index_frame_ini:frame_end-cfg.index_frame_ini]
    #

    folder_reqFiles = data.path + '/datasets_figures/required_files'

    file_origin_coord = cfg.file_origin_coord
    if not os.path.isfile(file_origin_coord):
        file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
    file_calibration = cfg.file_calibration
    if not os.path.isfile(file_calibration):
        file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
    file_model = cfg.file_model
    if not os.path.isfile(file_model):
        file_model = folder_reqFiles + '/model.npy'
    file_labelsDLC = cfg.file_labelsDLC
    if not os.path.isfile(file_labelsDLC):
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
   #
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
        
    #
    time = np.arange(nT_use, dtype=np.float64) * 1.0/cfg.frame_rate
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
    # 1D derivative (i.e. x-, y-, or z-velocity)
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
    #
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
#         # angle between connected bones
#         angle = np.arccos(np.einsum('sni,sni->sn', vec1, vec2)) * 180.0/np.pi
#         # source: https://math.stackexchange.com/questions/2140504/how-to-calculate-signed-angle-between-two-vectors-in-3d
#         n_cross = np.cross(vec1, vec2)
#         n_cross = n_cross / np.sqrt(np.sum(n_cross**2, 2))[:, :, None]
#         sin = np.einsum('sni,sni->sn', np.cross(n_cross, vec1), vec2)
#         mask = (sin < 0.0)
#         angle[mask] = 360.0 - angle[mask]
#         if np.any(mask):
#             print('WARNING: {:07d}/{:07d} signed angles'.format(np.sum(mask), len(mask.ravel())))
#         # angle between plane spanned connected bones and forward direction
#         vec0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
#         vec_n = np.cross(vec1, vec2)
#         vec_n = vec_n1 / np.sqrt(np.sum(vec_n**2, 2))[:, :, None]
#         angle = np.arccos(np.einsum('i,sni->sn', vec0, vec_n)) * 180.0/np.pi 
#         # source: https://math.stackexchange.com/questions/2140504/how-to-calculate-signed-angle-between-two-vectors-in-3d
#         n_cross = np.cross(vec0, vec_n)
#         n_cross = n_cross / np.sqrt(np.sum(n_cross**2, 2))[:, :, None]
#         sin = np.einsum('sni,sni->sn', np.cross(n_cross, vec0), vec_n)
#         mask = (sin < 0.0)
#         angle[mask] = 360.0 - angle[mask]
#         if np.any(mask):
#             print('WARNING: {:07d}/{:07d} signed angles'.format(np.sum(mask), len(mask.ravel())))
        # angle between bone and walking direction
        vec0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        vec_use = np.copy(vec1)
        angle = np.arccos(np.einsum('i,sni->sn', vec0, vec_use)) * 180.0/np.pi
        # source: https://math.stackexchange.com/questions/2140504/how-to-calculate-signed-angle-between-two-vectors-in-3d
        n_cross = np.cross(vec0, vec_use)
        n_cross = n_cross / np.sqrt(np.sum(n_cross**2, 2))[:, :, None]
        sin = np.einsum('sni,sni->sn', np.cross(n_cross, vec0), vec_use)
        mask = (sin < 0.0)
        angle[mask] = 360.0 - angle[mask]
        if np.any(mask):
            print('WARNING: {:07d}/{:07d} signed angles'.format(np.sum(mask), len(mask.ravel())))

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

    
    
    
    
    
    # PLOT
    # create figures
    fig_w_metric = np.round(mm_in_inch * 90.0*2/3, decimals=2)
    fig_h_metric = np.round(mm_in_inch * 90.0*1/4, decimals=2)
    fig_w_auto = np.round(mm_in_inch * 90.0*1/3, decimals=2)
    fig_h_auto = np.round(mm_in_inch * 90.0*1/4, decimals=2)
    fig_w_freq = np.round(mm_in_inch * 90.0*1/3, decimals=2)
    fig_h_freq = np.round(mm_in_inch * 90.0*1/4, decimals=2)
    #
    ax_pos_left = 0.08
    ax_pos_bottom = 0.12
    ax_pos_width = 1.0 - ax_pos_left
    ax_pos_height = 1.0 - ax_pos_bottom
    #
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
    ax1.set_position([ax_pos_left, ax_pos_bottom, ax_pos_width, ax_pos_height])
    # position auto-correlation
    fig10_w = fig_w_auto
    fig10_h = fig_h_auto
    fig10 = plt.figure(10, figsize=(fig10_w, fig10_h))
    fig10.canvas.manager.window.move(0, 0)
    fig10.clear()
    ax10_x = left_margin/fig10_w
    ax10_y = bottom_margin/fig10_h
    ax10_w = 1.0 - (left_margin/fig10_w + right_margin/fig10_w)
    ax10_h = 1.0 - (bottom_margin/fig10_h + top_margin/fig10_h)
    ax10 = fig10.add_axes([ax10_x, ax10_y, ax10_w, ax10_h])
    ax10.clear()
    ax10.spines["top"].set_visible(False)
    ax10.spines["right"].set_visible(False)
    ax10.spines["left"].set_bounds(-1, 1)
    # position auto-correlation frequency
    fig100_w = fig_w_freq
    fig100_h = fig_h_freq
    fig100 = plt.figure(100, figsize=(fig100_w, fig100_h))
    fig100.canvas.manager.window.move(0, 0)
    fig100.clear()
    ax100_x = left_margin/fig100_w
    ax100_y = bottom_margin/fig100_h
    ax100_w = 1.0 - (left_margin/fig100_w + right_margin/fig100_w)
    ax100_h = 1.0 - (bottom_margin/fig100_h + top_margin/fig100_h)
    ax100 = fig100.add_axes([ax100_x, ax100_y, ax100_w, ax100_h])
    ax100.clear()
    ax100.spines["top"].set_visible(False)
    ax100.spines["right"].set_visible(False)
    # velocities
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
    ax2.set_position([ax_pos_left, ax_pos_bottom, ax_pos_width, ax_pos_height])
    # velocities auto-correlation
    fig20_w = fig_w_auto
    fig20_h = fig_h_auto
    fig20 = plt.figure(20, figsize=(fig20_w, fig20_h))
    fig20.canvas.manager.window.move(0, 0)
    fig20.clear()
    ax20_x = left_margin/fig20_w
    ax20_y = bottom_margin/fig20_h
    ax20_w = 1.0 - (left_margin/fig20_w + right_margin/fig20_w)
    ax20_h = 1.0 - (bottom_margin/fig20_h + top_margin/fig20_h)
    ax20 = fig20.add_axes([ax20_x, ax20_y, ax20_w, ax20_h])
    ax20.clear()
    ax20.spines["top"].set_visible(False)
    ax20.spines["right"].set_visible(False)
    ax20.spines["left"].set_bounds(-1, 1)
    # velocity auto-correlation frequency
    fig200_w = fig_w_freq
    fig200_h = fig_h_freq
    fig200 = plt.figure(200, figsize=(fig200_w, fig200_h))
    fig200.canvas.manager.window.move(0, 0)
    fig200.clear()
    ax200_x = left_margin/fig200_w
    ax200_y = bottom_margin/fig200_h
    ax200_w = 1.0 - (left_margin/fig200_w + right_margin/fig200_w)
    ax200_h = 1.0 - (bottom_margin/fig200_h + top_margin/fig200_h)
    ax200 = fig200.add_axes([ax200_x, ax200_y, ax200_w, ax200_h])
    ax200.clear()
    ax200.spines["top"].set_visible(False)
    ax200.spines["right"].set_visible(False)
    # angles
    fig3_w = fig_w_metric
    fig3_h = fig_h_metric
    fig3 = plt.figure(3, figsize=(fig3_w, fig3_h))
    fig3.canvas.manager.window.move(0, 0)
    fig3.clear()    
    ax3_x = margin_all/fig3_w
    ax3_y = margin_all/fig3_h
    ax3_w = 1.0 - (margin_all/fig3_w + margin_all/fig3_w)
    ax3_h = 1.0 - (margin_all/fig3_h + margin_all/fig3_h)
    ax3 = fig3.add_axes([ax3_x, ax3_y, ax3_w, ax3_h])
    ax3.clear()
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set_position([ax_pos_left, ax_pos_bottom, ax_pos_width, ax_pos_height])
    # angles auto-correlation
    fig30_w = fig_w_auto
    fig30_h = fig_h_auto
    fig30 = plt.figure(30, figsize=(fig30_w, fig30_h))
    fig30.canvas.manager.window.move(0, 0)
    fig30.clear()
    ax30_x = left_margin/fig30_w
    ax30_y = bottom_margin/fig30_h
    ax30_w = 1.0 - (left_margin/fig30_w + right_margin/fig30_w)
    ax30_h = 1.0 - (bottom_margin/fig30_h + top_margin/fig30_h)
    ax30 = fig30.add_axes([ax30_x, ax30_y, ax30_w, ax30_h])
    ax30.clear()
    ax30.spines["top"].set_visible(False)
    ax30.spines["right"].set_visible(False)
    ax30.spines["left"].set_bounds(-1, 1)
    # angle auto-correlation frequency
    fig300_w = fig_w_freq
    fig300_h = fig_h_freq
    fig300 = plt.figure(300, figsize=(fig300_w, fig300_h))
    fig300.canvas.manager.window.move(0, 0)
    fig300.clear()
    ax300_x = left_margin/fig300_w
    ax300_y = bottom_margin/fig300_h
    ax300_w = 1.0 - (left_margin/fig300_w + right_margin/fig300_w)
    ax300_h = 1.0 - (bottom_margin/fig300_h + top_margin/fig300_h)
    ax300 = fig300.add_axes([ax300_x, ax300_y, ax300_w, ax300_h])
    ax300.clear()
    ax300.spines["top"].set_visible(False)
    ax300.spines["right"].set_visible(False)
    # angular velocities
    fig4_w = fig_w_metric
    fig4_h = fig_h_metric
    fig4 = plt.figure(4, figsize=(fig4_w, fig4_h))
    fig4.canvas.manager.window.move(0, 0)
    fig4.clear()    
    ax4_x = margin_all/fig4_w
    ax4_y = margin_all/fig4_h
    ax4_w = 1.0 - (margin_all/fig4_w + margin_all/fig4_w)
    ax4_h = 1.0 - (margin_all/fig4_h + margin_all/fig4_h)
    ax4 = fig4.add_axes([ax4_x, ax4_y, ax4_w, ax4_h])
    ax4.clear()
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_position([ax_pos_left, ax_pos_bottom, ax_pos_width, ax_pos_height])
    # angular velocities auto-correlation
    fig40_w = fig_w_auto
    fig40_h = fig_h_auto
    fig40 = plt.figure(40, figsize=(fig40_w, fig40_h))
    fig40.canvas.manager.window.move(0, 0)
    fig40.clear()
    ax40_x = left_margin/fig40_w
    ax40_y = bottom_margin/fig40_h
    ax40_w = 1.0 - (left_margin/fig40_w + right_margin/fig40_w)
    ax40_h = 1.0 - (bottom_margin/fig40_h + top_margin/fig40_h)
    ax40 = fig40.add_axes([ax40_x, ax40_y, ax40_w, ax40_h])
    ax40.clear()
    ax40.spines["top"].set_visible(False)
    ax40.spines["right"].set_visible(False)
    ax40.spines["left"].set_bounds(-1, 1)
    # position auto-correlation frequency
    fig400_w = fig_w_freq
    fig400_h = fig_h_freq
    fig400 = plt.figure(400, figsize=(fig400_w, fig400_h))
    fig400.canvas.manager.window.move(0, 0)
    fig400.clear()
    ax400_x = left_margin/fig400_w
    ax400_y = bottom_margin/fig400_h
    ax400_w = 1.0 - (left_margin/fig400_w + right_margin/fig400_w)
    ax400_h = 1.0 - (bottom_margin/fig400_h + top_margin/fig400_h)
    ax400 = fig400.add_axes([ax400_x, ax400_y, ax400_w, ax400_h])
    ax400.clear()
    ax400.spines["top"].set_visible(False)
    ax400.spines["right"].set_visible(False)
    #
    # detection
    fig123_w = np.round(mm_in_inch * 88.0, decimals=2)
    fig123_h = np.round(mm_in_inch * 88.0*2/3, decimals=2)
    fig123 = plt.figure(123, figsize=(fig123_w, fig123_h))
    fig123.canvas.manager.window.move(0, 0)
    fig123.clear()
    ax123_3_x = left_margin/fig123_w
    ax123_3_y = bottom_margin/fig123_h
    ax123_3_w = 1.0 - (left_margin/fig123_w + right_margin/fig123_w)
    ax123_3_h = 1.0/3.0 * ((1.0-top_margin/fig123_h-bottom_margin/fig123_h) - 2.0*between_margin_h/fig123_h)
    ax123_3 = fig123.add_axes([ax123_3_x, ax123_3_y, ax123_3_w, ax123_3_h])
    ax123_3.clear()
    ax123_3.spines["top"].set_visible(False)
    ax123_3.spines["right"].set_visible(False)
    ax123_2_x = left_margin/fig123_w
    ax123_2_y = ax123_3_y + ax123_3_h + between_margin_h/fig123_h
    ax123_2_w = 1.0 - (left_margin/fig123_w + right_margin/fig123_w)
    ax123_2_h = 1.0/3.0 * ((1.0-top_margin/fig123_h-bottom_margin/fig123_h) - 2.0*between_margin_h/fig123_h)
    ax123_2 = fig123.add_axes([ax123_2_x, ax123_2_y, ax123_2_w, ax123_2_h])
    ax123_2.clear()
    ax123_2.spines["top"].set_visible(False)
    ax123_2.spines["right"].set_visible(False)
    ax123_1_x = left_margin/fig123_w
    ax123_1_y = ax123_2_y + ax123_2_h + between_margin_h/fig123_h
    ax123_1_w = 1.0 - (left_margin/fig123_w + right_margin/fig123_w)
    ax123_1_h = 1.0/3.0 * ((1.0-top_margin/fig123_h-bottom_margin/fig123_h) - 2.0*between_margin_h/fig123_h)
    ax123_1 = fig123.add_axes([ax123_1_x, ax123_1_y, ax123_1_w, ax123_1_h])
    ax123_1.clear()
    ax123_1.spines["top"].set_visible(False)
    ax123_1.spines["right"].set_visible(False)
    #
    # single leg
    fig106_w = fig_w_metric
    fig106_h = fig106_w
    fig106 = plt.figure(106, figsize=(fig106_w, fig106_h))
    fig106.canvas.manager.window.move(0, 0)
    fig106.clear()
    ax106 = fig106.add_subplot(1, 1, 1, projection='3d')
    ax106.clear()
    ax106.set_position([0, 0, 1, 1])


    #
    item_list = list([[ax1, ax10, ax100, position_mean, position_std],
                      [ax2, ax20, ax200, velocity_mean, velocity_std],
                      [ax3, ax30, ax300, angle_mean, angle_std],
                      [ax4, ax40, ax400, angle_velocity_mean, angle_velocity_std],])
    index_start = int(15)
    print(cfg.frame_rate)
    index_end = frame_end-frame_start - int(0.6*cfg.frame_rate)
    for i_item in range(4):
        print('metric: #{:01d}'.format(i_item+1))
        item = item_list[i_item]
        ax_use = item[0]
        ax_use2 = item[1]
        ax_use3 = item[2]
        y = item[3]
        y_std = item[4]
        #
        corr0_all = np.zeros((4, len(time)), dtype=np.float64)
        #
        for i in range(4):
            y_use = y[i] - np.mean(y[i][~np.isnan(y[i])])
            ax_use.plot(time[index_start:index_end], y_use[index_start:index_end],
                        color=colors_paws[i], linewidth=linewidth, zorder=1)

        for i in range(4):
            y_use = y[i] - np.mean(y[i][~np.isnan(y[i])])
            y_std_use = y_std[i]
            ax_use.fill_between(x=time[index_start:index_end],
                                y1=y_use[index_start:index_end]-y_std_use[index_start:index_end],
                                y2=y_use[index_start:index_end]+y_std_use[index_start:index_end],
                                color=colors_paws[i],
                                alpha=0.2, zorder=0, linewidth=linewidth,
                                where=None, interpolate=False, step=None, data=None)
        for i in range(4):
            y_use = y[i] - np.mean(y[i][~np.isnan(y[i])])
            mask = ~np.isnan(y_use)
            nValid = np.sum(mask, dtype=np.int64)
            y_use = y_use[mask]
            corr = np.correlate(y_use, y_use, mode='full')
            corr0 = np.copy(corr[int(len(time[:nValid]))-1:])
            corr0 = corr0 / len(time[:nValid])
            corr0 = corr0 / np.std(y_use)**2
            #
            corr0_all[i][:nValid] = np.copy(corr0) 
            corr0_all[i][nValid:] = 0.0
            #
            ax_use2.plot(time[:nValid], corr0, color=colors_paws[i], linewidth=linewidth, zorder=1)
            #
            N = len(corr0)
            fft = np.fft.fft(corr0)
            timestep = 1.0/cfg.frame_rate # s
            freq = np.fft.fftfreq(N, d=timestep)
            ax_use3.plot(freq[:int(N/2)], abs(fft)[:int(N/2)] * 1.0/N, color=colors_paws[i], linewidth=linewidth)
            #
            index = np.argmax(abs(fft)[:int(N/2)])
            print(freq[:int(N/2)][index-2:index+2+1])
         
        tol = 2**-52
        #
        args_fit = dict()
        args_fit['t'] = np.copy(time[:nValid])
        args_fit['corr0'] = np.copy(corr0_all[:, :nValid])
        #
        x_ini = np.array([1.0, 1.0], dtype=np.float64)
        bounds = np.array([[tol, np.inf],
                           [tol, np.inf]], dtype=np.float64)
        #
        min_result = minimize(obj_fcn_wrap,
                              x_ini,
                              args=args_fit,
                              method='l-bfgs-b',
                              jac=grad_fcn_wrap,
                              hess=None,
                              hessp=None,
                              bounds=bounds,
                              constraints=(),
                              tol=tol,
                              callback=None,
                              options={'disp': False,
                                       'maxcor': 100,
                                       'ftol': tol,
                                       'gtol': tol,
                                       'eps': None,
                                       'maxfun': np.inf,
                                       'maxiter': np.inf,
                                       'iprint': -1,
                                       'maxls': 200})
        corr_fit = calc_corr_fit(min_result.x, args_fit)
        #
        SS_tot = np.sum((corr0 - np.mean(corr0))**2)
        SS_res = np.sum((corr0 - corr_fit)**2)
        R2 = 1.0 - SS_res/SS_tot
        #
        print('period:\t\t{:0.8f} s'.format(1.0/min_result.x[0]))
        print('frequency:\t{:0.8f} Hz'.format(min_result.x[0]))
        print('decay:\t\t{:0.8f} Hz'.format(min_result.x[1]))
        print('n:\t\t{:04d}'.format(len(corr0)))
        print('R2:\t\t{:0.8f}'.format(R2))
        #
        ax_use2.plot(time[:nValid], corr_fit, color='black', linewidth=linewidth, zorder=2, alpha=1.0)
            
            
            
        #
        h_legend = ax_use.legend(paws_joint_names_legend, loc='upper right', fontsize=fontsize, frameon=False)
        for text in h_legend.get_texts():
            text.set_fontname(fontname)
        print(f"{[time[index_start], time[index_end-1]]} - {len(time)} - {index_start} - {index_end}")
        ax_use.set_xlim([time[index_start], time[index_end-1]])
        #ax_use.set_axis_off()
        #
        ax_use2.set_xlabel('time (ms)', va='bottom', ha='center', fontsize=fontsize, fontname=fontname)
        ax_use2.set_ylabel('auto-correlation', va='top', ha='center', fontsize=fontsize, fontname=fontname)
        ax_use2.set_xlim([0.0, 1.2])
        ax_use2.set_xticks([0.0, 0.6, 1.2])
        ax_use2.set_xticklabels([0, 600, 1200])
        ax_use2.set_ylim([-1.0-0.2, 1.0])
        ax_use2.set_yticks([-1.0, 0.0, 1.0])
        ax_use2.set_yticklabels([-1, 0, 1])
        #
        ax_use3.set_xlabel('frequency (Hz)', va='bottom', ha='center', fontsize=fontsize, fontname=fontname)
        ax_use3.set_ylabel('amplitude', va='top', ha='center', fontsize=fontsize, fontname=fontname)
        ax_use3.set_xlim([0.0, 30.0])
        ax_use3.set_xticks([0.0, 15.0, 30.0])
        ax_use3.set_xticklabels([0, 15, 30])
        ax_use3.set_ylim([0.0, 0.24])
        ax_use3.set_yticks([0.0, 0.12, 0.24])
        ax_use3.set_yticklabels([0, 0.12, 0.24])
    #
    ax10.xaxis.set_label_coords(x=ax10_x+0.5*ax10_w, y=bottom_margin_x/fig10_h, transform=fig10.transFigure)
    ax10.yaxis.set_label_coords(x=left_margin_x/fig10_w, y=ax10_y+0.5*ax10_h, transform=fig10.transFigure)
    ax20.xaxis.set_label_coords(x=ax20_x+0.5*ax20_w, y=bottom_margin_x/fig20_h, transform=fig20.transFigure)
    ax20.yaxis.set_label_coords(x=left_margin_x/fig20_w, y=ax20_y+0.5*ax20_h, transform=fig20.transFigure)
    ax30.xaxis.set_label_coords(x=ax30_x+0.5*ax30_w, y=bottom_margin_x/fig30_h, transform=fig30.transFigure)
    ax30.yaxis.set_label_coords(x=left_margin_x/fig30_w, y=ax30_y+0.5*ax30_h, transform=fig30.transFigure)
    ax40.xaxis.set_label_coords(x=ax40_x+0.5*ax40_w, y=bottom_margin_x/fig40_h, transform=fig40.transFigure)
    ax40.yaxis.set_label_coords(x=left_margin_x/fig40_w, y=ax40_y+0.5*ax40_h, transform=fig40.transFigure)
    #
    ax100.xaxis.set_label_coords(x=ax100_x+0.5*ax100_w, y=bottom_margin_x/fig100_h, transform=fig100.transFigure)
    ax100.yaxis.set_label_coords(x=left_margin_x/fig100_w, y=ax100_y+0.5*ax100_h, transform=fig100.transFigure)
    ax200.xaxis.set_label_coords(x=ax200_x+0.5*ax200_w, y=bottom_margin_x/fig200_h, transform=fig200.transFigure)
    ax200.yaxis.set_label_coords(x=left_margin_x/fig200_w, y=ax200_y+0.5*ax200_h, transform=fig200.transFigure)
    ax300.xaxis.set_label_coords(x=ax300_x+0.5*ax300_w, y=bottom_margin_x/fig300_h, transform=fig300.transFigure)
    ax300.yaxis.set_label_coords(x=left_margin_x/fig300_w, y=ax300_y+0.5*ax300_h, transform=fig300.transFigure)
    ax400.xaxis.set_label_coords(x=ax400_x+0.5*ax400_w, y=bottom_margin_x/fig400_h, transform=fig400.transFigure)
    ax400.yaxis.set_label_coords(x=left_margin_x/fig400_w, y=ax400_y+0.5*ax400_h, transform=fig400.transFigure)

    ax1.set_ylim([-2.564, 3.589])
    ax2.set_ylim([-186.4, 263.8])
    ax3.set_ylim([-76.56, 51.62])
    ax4.set_ylim([-6221, 4003])
    cfg_plt.plot_coord_ax(ax1, '100 ms', '2 cm', 0.1, 2.0)
    cfg_plt.plot_coord_ax(ax2, '100 ms', '0.15 cm/ms', 0.1, 150.0)    
    cfg_plt.plot_coord_ax(ax3, '100 ms', '40 deg', 0.1, 40.0)
    cfg_plt.plot_coord_ax(ax4, '100 ms', '3 deg/ms', 0.1, 3000.0)

    # detection
    joint_names_detec = list([['joint_shoulder_left', 'joint_shoulder_right', 'joint_hip_left', 'joint_hip_right'],
                              ['joint_elbow_left', 'joint_elbow_right', 'joint_knee_left', 'joint_knee_right'],
                              ['joint_wrist_left', 'joint_wrist_right', 'joint_ankle_left', 'joint_ankle_right']])
    joint_names_detec_legend = list([['left shoulder marker', 'right shoulder marker', 'left hip marker', 'right hip marker'],
                                     ['left elbow marker', 'right elbow marker', 'left knee marker', 'right knee marker'],
                                     ['left wrist marker', 'right wrist marker', 'left ankle marker', 'right ankle marker']])
    cmap3 = plt.cm.tab20b
    joint_names_detec_colors = list([[cmap3(17/19), cmap3(13/19), cmap3(1/19), cmap3(9/19)],
                                     [colors_paws[0], colors_paws[1], colors_paws[2], colors_paws[3]],
                                     [cmap3(18/19), cmap3(14/19), cmap3(2/19), cmap3(10/19)]])
    for j in range(3):
        if (j == 0):
            ax_use = ax123_1
        elif (j == 1):
            ax_use = ax123_2
        elif (j == 2):
            ax_use = ax123_3
        for i in range(4):
            joint_index = args_model['model']['joint_order'].index(joint_names_detec[j][i])
            marker_index = np.where(joint_index == abs(args_model['model']['joint_marker_index']))[0][0]
            x = np.copy(time)
            print(frame_start)
            print(type(frame_start))
            print(frame_end)
            print(type(frame_end))
            y = np.sum(args_model['labels_mask'][frame_start-index_frames_calib_start:frame_end-index_frames_calib_start, :, marker_index].cpu().numpy(), 1)
            ax_use.plot(x, y, color=joint_names_detec_colors[j][i], linewidth=linewidth, clip_on=False)
        h_legend = ax_use.legend(joint_names_detec_legend[j], loc='upper right', fontsize=fontsize, frameon=False)
        for text in h_legend.get_texts():
            text.set_fontname(fontname)
        ax_use.set_xlim([time[0], time[-1]])
        ax_use.set_ylim([0.0, 4.0])
        if (j == 2):
            ax_use.set_xlabel('time (s)', va='bottom', ha='center', fontsize=fontsize, fontname=fontname)
            ax_use.set_xticks([0, 0.6, 1.2])
            ax_use.set_xticklabels([0, 0.6, 1.2])
        else:
            ax_use.xaxis.set(visible=False)
            ax_use.spines["bottom"].set_visible(False)
        if (j == 1):
            ax_use.set_ylabel('number of detections', va='top', ha='center', fontsize=fontsize, fontname=fontname)
        ax_use.set_yticks([0.0, 2.0, 4.0])
        ax_use.set_yticklabels([0, 2, 4])
    #
    ax123_1.yaxis.set_label_coords(x=left_margin_x/fig123_w, y=ax123_1_y+0.5*ax123_1_h, transform=fig123.transFigure)
    ax123_2.yaxis.set_label_coords(x=left_margin_x/fig123_w, y=ax123_2_y+0.5*ax123_2_h, transform=fig123.transFigure)
    ax123_3.yaxis.set_label_coords(x=left_margin_x/fig123_w, y=ax123_3_y+0.5*ax123_3_h, transform=fig123.transFigure)
    ax123_1.xaxis.set_label_coords(x=ax123_1_x+0.5*ax123_1_w, y=bottom_margin_x/fig123_h, transform=fig123.transFigure)
    ax123_2.xaxis.set_label_coords(x=ax123_2_x+0.5*ax123_2_w, y=bottom_margin_x/fig123_h, transform=fig123.transFigure)  
    ax123_3.xaxis.set_label_coords(x=ax123_3_x+0.5*ax123_3_w, y=bottom_margin_x/fig123_h, transform=fig123.transFigure)
    
    for ax_use in list([ax1, ax10, ax100,
                        ax2, ax20, ax200,
                        ax3, ax30, ax300,
                        ax4, ax40, ax400,
                        ax123_1, ax123_2, ax123_3]):
        for tick in ax_use.get_xticklabels():
            tick.set_fontsize(fontsize)
            tick.set_fontname(fontname)
        for tick in ax_use.get_yticklabels():
            tick.set_fontname(fontname)
            tick.set_fontsize(fontsize)
    plt.pause(2**-10)
    for fig_use in list([fig1, fig10, fig100,
                         fig2, fig20, fig200,
                         fig3, fig30, fig300,
                         fig4, fig40, fig400,
                         fig123]):
        fig_use.canvas.draw()
    plt.pause(2**-10)
    
    # leg
    side_leg = 'left'
    leg_joints_list = list(['joint_hip_{:s}'.format(side_leg),
                            'joint_knee_{:s}'.format(side_leg),
                            'joint_ankle_{:s}'.format(side_leg),
                            'joint_paw_hind_{:s}'.format(side_leg)])
    markeredgewidth = 0
    #
    time_point_indices = np.arange(index_start, index_end, 1, dtype=np.int64)
    colors = np.copy(time_point_indices)
    colors = colors - min(colors)
    colors = colors / max(colors)
    colors = cmap(colors)
    xyz_mean2 = np.zeros(3, dtype=np.float64)
    xyz_mean_count2 = 0
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    for i_t in range(len(time_point_indices)):
        time_point_index = time_point_indices[i_t]
        joint_start = skeleton_pos[time_point_index, 0]
        for i_bone in range(nBones):
            joint_index_start = skeleton_edges[i_bone, 0]
            joint_name = joint_order[joint_index_start]
            if joint_name in leg_joints_list:
                joint_index_end = skeleton_edges[i_bone, 1]
                joint_start = skeleton_pos[time_point_index, joint_index_start]
                joint_end = skeleton_pos[time_point_index, joint_index_end]
                vec = np.stack([joint_start, joint_end], 0)
                ax106.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                        linestyle='-', marker='', linewidth=linewidth, color=colors[i_t], zorder=1, alpha=0.5)
                ax106.plot([vec[1, 0]], [vec[1, 1]], [vec[1, 2]],
                           linestyle='', marker='.', markersize=markersize, markeredgewidth=markeredgewidth, color='black', zorder=2, alpha=1.0)
                xyz_mean2 = xyz_mean2 + joint_end
                xyz_mean_count2 = xyz_mean_count2 + 1
                for i_d in range(3):
                    xyz_min[i_d] = min(xyz_min[i_d], joint_start[i_d])
                    xyz_min[i_d] = min(xyz_min[i_d], joint_end[i_d])
                    xyz_max[i_d] = max(xyz_max[i_d], joint_start[i_d])
                    xyz_max[i_d] = max(xyz_max[i_d], joint_end[i_d])
    xyz_mean2 = xyz_min + 0.5 * (xyz_max - xyz_min)
    #
    ax106.view_init(0, -0)
    ax106.set_axis_off()
    #
    xyzlim_range2 = 8
    ax106.set_xlim3d([xyz_mean2[0]-xyzlim_range2, xyz_mean2[0]+xyzlim_range2])
    ax106.set_ylim3d([xyz_mean2[1]-xyzlim_range2, xyz_mean2[1]+xyzlim_range2])
    ax106.set_zlim3d([xyz_mean2[2]-xyzlim_range2, xyz_mean2[2]+xyzlim_range2])
    #
    cax = fig106.add_axes([1/4+1/8, 1/4+1/8, 1/4, 0.025])
    clim_min = 0
    clim_max = (index_end-index_start)/cfg.frame_rate
    mappable2d = plt.cm.ScalarMappable(cmap=cmap)
    mappable2d.set_clim([clim_min, clim_max])
    colorbar2d = fig106.colorbar(mappable=mappable2d, ax=ax106, cax=cax, ticks=[clim_min, clim_max], orientation='horizontal')
    colorbar2d.outline.set_edgecolor('white')
    colorbar2d.set_label('time (s)', labelpad=-6, rotation=0, fontsize=fontsize, fontname=fontname)
    colorbar2d.ax.tick_params(labelsize=fontsize, color='black')
    #
    fig106.canvas.draw()
    plt.pause(2**-10)
    if save_leg:
        fig106.savefig(folder_save+'/gait_3d_sketch_sequence_leg_{:s}.svg'.format(side_leg),
                    bbox_inches='tight',
                    dpi=300,
                    transparent=True,
                    format='svg',
                    pad_inches=0)

    # save & verbose
    print(save)
    if save:
        fig1.savefig(folder_save+'/gait_position__{:s}.svg'.format(mode),
#                     bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)
        print(folder_save+'/gait_position__{:s}.svg'.format(mode))
        fig2.savefig(folder_save+'/gait_velocity__{:s}.svg'.format(mode),
#                     bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)
        fig3.savefig(folder_save+'/gait_angle__{:s}.svg'.format(mode),
#                         bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)   
        fig4.savefig(folder_save+'/gait_angle_velocity__{:s}.svg'.format(mode),
#                         bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0) 
        #
        fig10.savefig(folder_save+'/gait_position__corr__{:s}.svg'.format(mode),
#                     bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)
        fig20.savefig(folder_save+'/gait_velocity__corr__{:s}.svg'.format(mode),
#                     bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)
        fig30.savefig(folder_save+'/gait_angle__corr__{:s}.svg'.format(mode),
#                         bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)   
        fig40.savefig(folder_save+'/gait_angle_velocity__corr__{:s}.svg'.format(mode),
#                         bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0) 
        #
        fig100.savefig(folder_save+'/gait_position__freq__{:s}.svg'.format(mode),
#                     bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)
        fig200.savefig(folder_save+'/gait_velocity__freq__{:s}.svg'.format(mode),
#                     bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)
        fig300.savefig(folder_save+'/gait_angle__freq__{:s}.svg'.format(mode),
#                         bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)   
        fig400.savefig(folder_save+'/gait_angle_velocity__freq__{:s}.svg'.format(mode),
#                         bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0) 
        #
        fig123.savefig(folder_save+'/gait_detection.svg',
                    bbox_inches="tight",
                    dpi=300,
                    transparent=True,
                    format='svg',
                   pad_inches=0)
    if verbose:
        plt.show()
    
