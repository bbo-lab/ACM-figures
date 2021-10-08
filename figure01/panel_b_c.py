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

folder_save = os.path.abspath('panels')

folder_recon = data.path+'/reconstruction'
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

folder = folder_list[1]

list_is_large_animal = list([0])

sys.path = list(np.copy(sys_path0))
sys.path.append(folder)
importlib.reload(cfg)
cfg.animal_is_large = list_is_large_animal[0]
importlib.reload(anatomy)

frame_start = cfg.index_frame_start
frame_end = cfg.index_frame_end
nSamples_use = int(0e3)

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

fontsize = 6
linewidth = 0.75
fontname = "Arial"
markersize = 1

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

if __name__ == '__main__':
    x_ini = np.load(folder+'/x_ini.npy', allow_pickle=True)
    save_dict = np.load(folder+'/save_dict.npy', allow_pickle=True).item()
    if ('mu_uks' in save_dict):
        mu_uks = save_dict['mu_uks'][1:]
        var_uks = save_dict['var_uks'][1:]
        nSamples = np.copy(nSamples_use)
        print(save_dict['message'])
    else:
        print('ERROR')
        raise
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
    
#     # do coordinate transformation (skeleton_pos_samples)
#     joint_index1 = joint_order.index('joint_spine_002') # pelvis
#     joint_index2 = joint_order.index('joint_spine_004') # tibia
#     origin = skeleton_pos_samples[:, :, joint_index1]
#     x_direc = skeleton_pos_samples[:, :, joint_index2] - origin
#     x_direc = x_direc[:, :, :2]
#     x_direc = x_direc / np.sqrt(np.sum(x_direc**2, 2))[:, :, None]
#     alpha = np.arctan2(x_direc[:, :, 1], x_direc[:, :, 0])
#     cos_m_alpha = np.cos(-alpha)
#     sin_m_alpha = np.sin(-alpha)
#     R = np.stack([np.stack([cos_m_alpha, -sin_m_alpha], 2),
#                   np.stack([sin_m_alpha, cos_m_alpha], 2)], 2)
#     skeleton_pos_samples = skeleton_pos_samples - origin[:, :, None, :]
#     skeleton_pos_xy = np.einsum('snij,snmj->snmi', R, skeleton_pos_samples[:, :, :, :2])
#     skeleton_pos_samples[:, :, :, :2] = np.copy(skeleton_pos_xy)    

#     time = np.arange(nT_use, dtype=np.float64) * 1.0/cfg.frame_rate

    # plot 3d
    fig_w = np.round(mm_in_inch * 110, decimals=2)
    #
    fig105_w = fig_w
    fig105_h = fig105_w
    fig105 = plt.figure(105, figsize=(fig105_w, fig105_h))
    fig105.canvas.manager.window.move(0, 0)
    fig105.clear()
    ax105 = fig105.add_subplot(1, 1, 1, projection='3d')
    ax105.clear()
    #
    fig106_w = fig_w
    fig106_h = fig106_w
    fig106 = plt.figure(106, figsize=(fig106_w, fig106_h))
    fig106.canvas.manager.window.move(0, 0)
    fig106.clear()
    ax106 = fig106.add_subplot(1, 1, 1, projection='3d')
    ax106.clear()
    
    side = 'right'
    leg_joints_list = list(['joint_hip_{:s}'.format(side),
                            'joint_knee_{:s}'.format(side),
                            'joint_ankle_{:s}'.format(side),
                            'joint_paw_hind_{:s}'.format(side)])
    
    markeredgewidth = 0
    # plot 3d 
    ax105.clear()
    time_point_index_min = nT_use
    time_point_index_max = 0
    dt = 35
    t_start = 85
    t_end = t_start + 140
    t_start = max(0, t_start)
    t_end = min(nT_use-1, t_end)

    # all
    time_point_indices = np.arange(t_start, t_end, dt, dtype=np.int64)
    colors = np.copy(time_point_indices)
    colors = colors - min(colors)
    colors = colors / max(colors)
    colors = cmap(colors)
    xyz_mean = np.zeros(3, dtype=np.float64)
    xyz_mean_count = 0
    for i_t in range(len(time_point_indices)):
        time_point_index = time_point_indices[i_t]
        joint_start = skeleton_pos[time_point_index, 0]
        for i_bone in range(nBones):
            joint_index_start = skeleton_edges[i_bone, 0]
            joint_name = joint_order[joint_index_start]
            joint_index_end = skeleton_edges[i_bone, 1]
            joint_start = skeleton_pos[time_point_index, joint_index_start]
            joint_end = skeleton_pos[time_point_index, joint_index_end]
            vec = np.stack([joint_start, joint_end], 0)
            ax105.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                    linestyle='-', marker='', linewidth=linewidth, color=colors[i_t], zorder=1, alpha=0.5)
            ax105.plot([vec[1, 0]], [vec[1, 1]], [vec[1, 2]],
                       linestyle='', marker='.', markersize=markersize, markeredgewidth=markeredgewidth, color='black', zorder=2, alpha=1.0)
            if (i_bone == 0):
                ax105.plot([joint_start[0]], [joint_start[1]], [joint_start[2]],
                           linestyle='', marker='.', markersize=markersize, markeredgewidth=markeredgewidth, color='black', zorder=2, alpha=1.0)
                xyz_mean = xyz_mean + joint_start
                xyz_mean_count = xyz_mean_count + 1
            xyz_mean = xyz_mean + joint_end
            xyz_mean_count = xyz_mean_count + 1
    xyz_mean = np.array([np.min(skeleton_pos[time_point_indices, :, 0]) + 0.5 * (np.max(skeleton_pos[time_point_indices, :, 0]) - np.min(skeleton_pos[time_point_indices, :, 0])),
                         np.min(skeleton_pos[time_point_indices, :, 1]) + 0.5 * (np.max(skeleton_pos[time_point_indices, :, 1]) - np.min(skeleton_pos[time_point_indices, :, 1])),
                         np.min(skeleton_pos[time_point_indices, :, 2]) + 0.5 * (np.max(skeleton_pos[time_point_indices, :, 2]) - np.min(skeleton_pos[time_point_indices, :, 2])),], dtype=np.float64)
    # leg
    time_point_indices = np.arange(t_start, t_end, 1, dtype=np.int64)
    colors = np.copy(time_point_indices)
    colors = colors - min(colors)
    colors = colors / max(colors)
    colors = cmap(colors)
    xyz_mean2 = np.zeros(3, dtype=np.float64)
    xyz_mean_count2 = 0
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
                if (i_bone == 0):
                    ax105.plot([joint_start[0]], [joint_start[1]], [joint_start[2]],
                               linestyle='', marker='.', markersize=markersize, markeredgewidth=markeredgewidth, color='black', zorder=2, alpha=1.0)
                    xyz_mean2 = xyz_mean2 + joint_start
                    xyz_mean_count2 = xyz_mean_count2 + 1
                xyz_mean2 = xyz_mean2 + joint_end
                xyz_mean_count2 = xyz_mean_count2 + 1
    xyz_mean2 = np.array([np.min(skeleton_pos[time_point_indices, :, 0]) + 0.5 * (np.max(skeleton_pos[time_point_indices, :, 0]) - np.min(skeleton_pos[time_point_indices, :, 0])),
                         np.min(skeleton_pos[time_point_indices, :, 1]) + 0.5 * (np.max(skeleton_pos[time_point_indices, :, 1]) - np.min(skeleton_pos[time_point_indices, :, 1])),
                         np.min(skeleton_pos[time_point_indices, :, 2]) + 0.5 * (np.max(skeleton_pos[time_point_indices, :, 2]) - np.min(skeleton_pos[time_point_indices, :, 2])),], dtype=np.float64)
    #
    ax105.view_init(30, 0)
    ax106.view_init(0, -0)
    ax105.set_axis_off()
    ax106.set_axis_off()
    #
    xyzlim_range = 19
    xyzlim_range2 = 14
    ax105.set_xlim3d([xyz_mean[0]-xyzlim_range, xyz_mean[0]+xyzlim_range])
    ax105.set_ylim3d([xyz_mean[1]-xyzlim_range, xyz_mean[1]+xyzlim_range])
    ax105.set_zlim3d([xyz_mean[2]-xyzlim_range, xyz_mean[2]+xyzlim_range])
    ax106.set_xlim3d([xyz_mean2[0]-xyzlim_range2, xyz_mean2[0]+xyzlim_range2])
    ax106.set_ylim3d([xyz_mean2[1]-xyzlim_range2, xyz_mean2[1]+xyzlim_range2])
    ax106.set_zlim3d([xyz_mean2[2]-xyzlim_range2, xyz_mean2[2]+xyzlim_range2])
    #
    cax = fig106.add_axes([3/8, 0.6, 1/4, 0.025])
    clim_min = 0
    clim_max = (t_end-t_start)/cfg.frame_rate
    mappable2d = plt.cm.ScalarMappable(cmap=cmap)
    mappable2d.set_clim([clim_min, clim_max])
    colorbar2d = fig105.colorbar(mappable=mappable2d, ax=ax106, cax=cax, ticks=[clim_min, clim_max], orientation='horizontal')
    colorbar2d.outline.set_edgecolor('white')
    colorbar2d.set_label('time (s)', labelpad=-5, rotation=0, fontsize=fontsize, fontname=fontname)
    colorbar2d.ax.tick_params(labelsize=fontsize, color='white')
    #
    fig105.canvas.draw()
    fig106.canvas.draw()
    if save:
        fig105.savefig(folder_save+'/gait_3d_sketch_sequence.svg',
                    bbox_inches='tight',
                    dpi=300,
                    transparent=True,
                    format='svg',
                    pad_inches=0)
        fig106.savefig(folder_save+'/gait_3d_sketch_sequence_leg.svg',
                    bbox_inches='tight',
                    dpi=300,
                    transparent=True,
                    format='svg',
                    pad_inches=0)
    plt.show()
