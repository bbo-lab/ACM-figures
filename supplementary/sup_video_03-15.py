#!/usr/bin/env python3

import importlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
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

sys.path.append(os.path.abspath('../ccv'))
import ccv

folder_reconstruction = data.path + '/reconstruction'
folder = folder_reconstruction + '/20200205/gap_20200205_128720_128920__pcutoff9e-01' # has to be set manually
recFileNames = sorted(list(['/media/server/bulk/pose_B.EG.1.09/20200205/cam01_20200205_gap.ccv',
                            '/media/server/bulk/pose_B.EG.1.09/20200205/cam02_20200205_gap.ccv',
                            '/media/server/bulk/pose_B.EG.1.09/20200205/cam03_20200205_gap.ccv',
                            '/media/server/bulk/pose_B.EG.1.09/20200205/cam04_20200205_gap.ccv'])) # has to be set manually
is_large_animal = 0 # has to be set manually
invert_xaxis = False # has to be set manually
invert_yaxis = False # has to be set manually

save = False
verbose = True

folder_save = os.path.abspath('videos')

slow_factor = 0.2
cmap = plt.cm.viridis
    
xy_range = 256
        
nSamples = int(1e3) # minimum: 1
max_std = 5.0 # [px] any std > max_std is treated as totally uncertain

sys.path = list(np.copy(sys_path0))
sys.path.append(folder)
importlib.reload(cfg)
cfg.animal_is_large = is_large_animal
importlib.reload(anatomy)
#
folder_reqFiles = data.path + '/required_files' 
file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
file_model = folder_reqFiles + '/model.npy'
file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_dlc_{:06d}_{:06d}.npy'.format(cfg.index_frame_start, cfg.index_frame_end)

# 2d
def get_joint_colors_2d(skel2d_samples):
    std = np.std(skel2d_samples, 0) # nCameras, nJoints, 2
    std = np.max(std, 0) # nJoints, 2
    std = (std[:, 0] * std[:, 1])**(1.0/2.0) # nJoints
    std_norm = std / max_std
    joint_colors = cmap(1.0 - std_norm)
    return joint_colors    
   
def plot_2d_3d(mu, mu_ini,
               var=None):
    var_is_nan = np.all(var == None)
    
    # get arguments
    args = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                cfg.scale_factor, cfg.pcutoff)
    if ((cfg.mode == 1) or (cfg.mode == 2)):
        args['use_custom_clip'] = False
    elif ((cfg.mode == 3) or (cfg.mode == 4)):
        args['use_custom_clip'] = True
        
    # remove all free parameters that do not modify the pose
    free_para_bones = np.zeros_like(args['free_para_bones'], dtype=bool)
    free_para_markers = np.zeros_like(args['free_para_markers'], dtype=bool)
    nFree_bones = int(0)
    nFree_markers = int(0)
    args['free_para_bones'] = torch.from_numpy(free_para_bones)
    args['free_para_markers'] = torch.from_numpy(free_para_markers)
    args['nFree_bones'] = nFree_bones
    args['nFree_markers'] = nFree_markers
    if ((cfg.mode == 1) or (cfg.mode == 2)):
        args['nFrames'] = 1
    
    labelsDLC = np.load(file_labelsDLC, allow_pickle=True).item()
    frame_list = labelsDLC['frame_list']
    i_frame = np.where(frame_list == cfg.index_frame_ini)[0][0]
    frame_list_fit = np.arange(i_frame,
                               int(i_frame + cfg.dt * cfg.nT),
                               cfg.dt,
                               dtype=np.int64)
    #
    nCameras = args['numbers']['nCameras']
    nEdges = args['numbers']['nBones']
    nMarkers = args['numbers']['nMarkers']
    #
    RX1 = args['calibration']['RX1_fit']
    tX1 = args['calibration']['tX1_fit']
    #
    skel_edges = args['model']['skeleton_edges'].cpu().numpy()
    joint_marker_index = args['model']['joint_marker_index'].cpu().numpy()
    #
    measure = args['labels'][frame_list_fit].cpu().numpy()
    measure_mask = args['labels_mask'][frame_list_fit].cpu().numpy()
    measure = measure[:, :, :, :2]
    
    # add necessary arrays to arguments
    if (var_is_nan):
        nSamples_use = 1
    else:
        nSamples_use = nSamples + 1
    args['plot'] = True
    args['x_torch'] = torch.from_numpy(mu_ini)
    # cast tensor to pytorch
    mu_torch = torch.from_numpy(mu)
    if (var_is_nan):
        var_torch = torch.zeros(cfg.nT, dtype=torch.float64)
    else:
        var_torch = torch.from_numpy(var)     
    
    def get_all(mu_t, var_t, t, args):
        nCameras = args['numbers']['nCameras']
        nMarkers = args['numbers']['nMarkers']
        #
        A = args['calibration']['A_fit']
        k = args['calibration']['k_fit']
        RX1 = args['calibration']['RX1_fit']
        tX1 = args['calibration']['tX1_fit']
        
        if (var_is_nan):
            z_all = mu_t[None, :]
        else:
            distribution = torch.distributions.MultivariateNormal(loc=mu_t,
                                                                  scale_tril=kalman.cholesky_save(var_t))
            z_samples = distribution.sample((nSamples,))
            z_all = torch.cat([mu_t[None, :],
                               z_samples], 0)
        x_all, markers_all, skel_all = model.fcn_emission_free(z_all, args)
        skel2d_all = model.map_m(RX1, tX1, A, k,
                                 skel_all)
        #
        x_all = x_all.reshape(nSamples_use, nCameras, nMarkers, 2)
        x_all[:, :, :, 0] = (x_all[:, :, :, 0] + 1.0) * (cfg.normalize_camera_sensor_x * 0.5)
        x_all[:, :, :, 1] = (x_all[:, :, :, 1] + 1.0) * (cfg.normalize_camera_sensor_y * 0.5)
        #
        x_all = x_all.cpu().numpy()
        markers_all = markers_all.cpu().numpy()
        skel_all = skel_all.cpu().numpy()
        skel2d_all = skel2d_all.cpu().numpy()
        #
        x_single = x_all[0]
        markers_single = markers_all[0]
        skel_single = skel_all[0]
        skel2d_single = skel2d_all[0]
        x_samples = x_all[1:]
        markers_samples = markers_all[1:]
        skel_samples = skel_all[1:]
        skel2d_samples = skel2d_all[1:]
        if (var_is_nan):
            grad = torch.zeros((nEdges+1)*3, dtype=torch.float64)
            grad_joints = torch.zeros((nEdges+1), dtype=torch.float64)
            args['labels_single_torch'] = args['labels'][t][None, :].clone()
            args['labels_mask_single_torch'] = args['labels_mask'][t][None, :].clone()
            z_free = mu_t.clone()
            noise = torch.randn(z_free.size()) * 2**-10
            z_free = z_free + noise
            z_free.requires_grad = True
            args['plot'] = False
            res = model.obj_fcn(z_free, args)
            args['plot'] = True
            res.backward()
            grad[args['free_para_pose']] = z_free.grad[0].clone()
            z_free.grad.data.zero_()
            #
            grad_mask = (grad != 0.0)
            grad_mask = grad_mask.reshape((nEdges+1), 3)
            grad_mask = torch.any(grad_mask, 1).type(torch.float64)
            #
            grad_joints[0] = grad_mask[0]      
            grad_joints[args['model']['skeleton_edges'][:, 1]] = grad_mask[1:]
            #
            grad_joints = grad_joints.detach().cpu().numpy()
            joint_colors = cmap(grad_joints)
        else:
            joint_colors = get_joint_colors_2d(skel2d_samples)
        return x_single, markers_single, skel_single, skel2d_single, \
               x_samples, markers_samples, skel_samples, skel2d_samples, \
               joint_colors

    t = 0
    x_single, markers_single, skel_single, skel2d_single, \
    x_samples, markers_samples, skel_samples, skel2d_samples, \
    joint_colors = \
        get_all(mu_torch[t], var_torch[t], t, args)

    ##### 2D #####
    fig2d = plt.figure(figsize=(9, 9))
    h_cam_line = list()
    h_cam_true = list()
    h_cam_single = list()
    h_cam_skel = list()
    h_cam_joint = list()
    h_cam_joint_samples = list()
    h_title = list()
    h_img = list() 
    ax2d_list = list()
    # animal
    for i_cam in range(nCameras):
        ax2d = fig2d.add_subplot(2, 2, i_cam+1)
        img = np.zeros((int(cfg.normalize_camera_sensor_y), int(cfg.normalize_camera_sensor_x)), dtype=np.uint8)
        h_i_img = ax2d.imshow(img, cmap='gray', vmin=0.0, vmax=127)
        h_img.append(h_i_img)
            
        xy_mean = np.mean(x_single[i_cam], 0)
        ax2d.set_xlim([xy_mean[0]-xy_range, xy_mean[0]+xy_range])
        ax2d.set_ylim([xy_mean[1]-xy_range, xy_mean[1]+xy_range])
            
        ax2d.set_aspect(1)
        ax2d.set_facecolor('black')
        ax2d.set_xticks(list([]))
        ax2d.set_yticks(list([]))
        ax2d.set_xticklabels(list([]))
        ax2d.set_yticklabels(list([]))
        if (invert_xaxis):
            ax2d.invert_xaxis()
        if (invert_yaxis):
            invert_yaxis()
        ax2d_list.append(ax2d)
            
        i_title = ax2d.set_title('cam = {:01d}, index = {:06d}'.format(i_cam, cfg.index_frame_ini))
        h_title.append(i_title)

        # skeleton
        h_edge_skel = list()
        h_edge_joint = list()
        index_bone_start = skel_edges[0, 0]
        i_joint = ax2d.plot(skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1],
                            linestyle='', marker='.', color=joint_colors[index_bone_start], alpha=1.0, zorder=6,
                            markersize=9)
        h_edge_joint.append(i_joint[0])
        for i_edge in range(nEdges):
            index_bone_start = skel_edges[i_edge, 0]
            index_bone_end = skel_edges[i_edge, 1]
            vec = np.array([[skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1]],
                            [skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1]]],
                            dtype=np.float64)
            i_skel_vec = ax2d.plot(vec[:, 0], vec[:, 1],
                                    linestyle='-', marker='', color=joint_colors[index_bone_end], zorder=2)
            h_edge_skel.append(i_skel_vec[0])
            i_joint = ax2d.plot(skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1],
                                linestyle='', marker='.', color=joint_colors[index_bone_end], alpha=1.0, zorder=6,
                                markersize=9)
            h_edge_joint.append(i_joint[0])
        if not(var_is_nan):
            i_joint_samples = ax2d.plot(np.ravel(skel2d_samples[:, i_cam, :, 0]), np.ravel(skel2d_samples[:, i_cam, :, 1]),
                                        linestyle='', marker='.', color='red', alpha=nSamples**-0.5, zorder=1,
                                        markersize=6)
        # marker
        h_edge_line = list()
        h_edge_true = list()
        h_edge_single = list()
        for i_edge in range(nMarkers):
            if (measure_mask[t, i_cam, i_edge]):
                alpha = 1.0
            else:
                alpha = 0.0
            vec = np.array([[measure[t, i_cam, i_edge, 0], measure[t, i_cam, i_edge, 1]],
                            [x_single[i_cam, i_edge, 0], x_single[i_cam, i_edge, 1]]],
                            dtype=np.float64)
            i_line = ax2d.plot(vec[:, 0], vec[:, 1],
                               linestyle='-', marker='', color='orange', alpha=alpha/2, zorder=3)
            i_true = ax2d.plot(measure[t, i_cam, i_edge, 0], measure[t, i_cam, i_edge, 1],
                               linestyle='', marker='.', color='green', alpha=alpha/2, zorder=4,
                               markersize=6)
            i_single = ax2d.plot(x_single[i_cam, i_edge, 0], x_single[i_cam, i_edge, 1],
                                 linestyle='', marker='.', color='blue', alpha=alpha/2, zorder=5,
                                 markersize=6)
            h_edge_line.append(i_line[0])
            h_edge_true.append(i_true[0])
            h_edge_single.append(i_single[0])
            #
        h_cam_line.append(h_edge_line)
        h_cam_true.append(h_edge_true)
        h_cam_single.append(h_edge_single)
        h_cam_skel.append(h_edge_skel)
        h_cam_joint.append(h_edge_joint)
        if (var_is_nan):
            h_cam_joint_samples.append(list([])) 
        else:
            h_cam_joint_samples.append(i_joint_samples[0])
    h2d_ax = list([h_img,
                   h_cam_line, h_cam_true, h_cam_single,
                   h_cam_skel, h_cam_joint, h_cam_joint_samples,
                   h_title])
    fig2d.tight_layout()
    fig2d.canvas.draw()
    
    ##### 3D #####
    markers_single_use = np.copy(markers_single)
    skel_single_use = np.copy(skel_single)
    markers_single_use[:, :2] -= skel_single_use[0, :2]
    skel_single_use[:, :2] -= skel_single_use[0, :2]
    #
    fig3d = plt.figure(figsize=(15, 5))
    h3d_ax = list()
    for i_ax in range(3):
        ax3d = fig3d.add_subplot(1, 3, i_ax+1, projection='3d')
        ax3d.xaxis.set_rotate_label(False)
        ax3d.yaxis.set_rotate_label(False)
        ax3d.zaxis.set_rotate_label(False)
        if (i_ax == 0):
            ax3d.view_init(azim=180.0, elev=90.0)
            ax3d.set_xlabel('x (cm)', rotation=90.0)
            ax3d.set_ylabel('y (cm)')
            ax3d.set_zticks(list([]))
            ax3d.set_zticklabels(list([]))
            ax3d.xaxis._axinfo['juggled'] = (1,0,2)
            ax3d.zaxis._axinfo['juggled'] = (1,2,0)
        elif (i_ax == 1):
            ax3d.view_init(azim=90.0, elev=0.0)
            ax3d.set_xlabel('x (cm)')
            ax3d.set_zlabel('z (cm)', rotation=90.0)
            ax3d.set_yticks(list([]))
            ax3d.set_yticklabels(list([]))
            ax3d.xaxis._axinfo['juggled'] = (2,0,1)
            #
            h3d_title = ax3d.set_title('index = {:06d}'.format(cfg.index_frame_ini),
                                       pad=50.0)
        elif (i_ax == 2):
            ax3d.view_init(azim=180.0, elev=0.0)
            ax3d.set_ylabel('y (cm)')
            ax3d.set_zlabel('z (cm)', rotation=90.0)
            ax3d.set_xticks(list([]))
            ax3d.set_xticklabels(list([]))
            ax3d.yaxis._axinfo['juggled'] = (2,1,0)
            ax3d.zaxis._axinfo['juggled'] = (1,2,0)
        xyzlim_range = 30.0
        ax3d.set_xlim3d([-xyzlim_range, xyzlim_range])
        ax3d.set_ylim3d([-xyzlim_range, xyzlim_range])
        ax3d.set_zlim3d([-xyzlim_range, xyzlim_range])
        ax3d.dist = 7.0
        #
        axis_length = 10.0
        # all cameras
        colors = list(['red', 'green', 'blue'])
        #
        RX1_use = RX1.cpu().numpy()
        tX1_use = tX1.cpu().numpy()
        for i_cam in range(nCameras):
            origin_cam = np.dot(RX1_use[i_cam].T, -tX1_use[i_cam])
            for i_d in range(3):
                vec = np.stack([origin_cam, origin_cam + RX1_use[i_cam][:, i_d] * axis_length], 0)
                i_vec = ax3d.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                                  linestyle='-', marker='', color=colors[i_d], zorder=0, alpha=0.5)
        # arena coordinate system
        colors = list(['darkred', 'darkgreen', 'darkblue'])
        origin_arena = np.zeros(3, dtype=np.float64)
        coord_arena = np.identity(3, dtype=np.float64)
        for i_d in range(3):
            vec = np.stack([origin_arena, origin_arena + coord_arena[:, i_d] * axis_length], 0)
            i_vec = ax3d.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                              linestyle='-', marker='', color=colors[i_d], zorder=0, alpha=0.5)
        #
        h_markers3d = list()
        h_markers3d_vec = list()
        for i_edge in range(nMarkers):
            i_marker = ax3d.plot([markers_single_use[i_edge, 0]],
                                 [markers_single_use[i_edge, 1]],
                                 [markers_single_use[i_edge, 2]],
                                 marker='.', color='blue', alpha=0.1)
            h_markers3d.append(i_marker[0])

            index_start = joint_marker_index[i_edge]
            vec = np.array([[markers_single_use[i_edge, 0],
                             markers_single_use[i_edge, 1],
                             markers_single_use[i_edge, 2]],
                            [skel_single_use[index_start, 0],
                             skel_single_use[index_start, 1],
                             skel_single_use[index_start, 2]]],
                            dtype=np.float64)
            i_marker_vec = ax3d.plot(vec[:, 0],
                                     vec[:, 1],
                                     vec[:, 2],
                                     linestyle='-', marker='', color='blue', alpha=0.1, zorder=2)
            h_markers3d_vec.append(i_marker_vec[0])
        #
        h_skel3d = list()
        h_skel3d_vec = list()
        #
        i_edge = 0
        index_bone_start = skel_edges[i_edge, 0]
        i_skel = ax3d.plot([skel_single_use[index_bone_start, 0]],
                           [skel_single_use[index_bone_start, 1]],
                           [skel_single_use[index_bone_start, 2]],
                           marker='o', color=joint_colors[index_bone_start], zorder=1)
        h_skel3d.append(i_skel[0])
        for i_edge in range(nEdges+1):
            if (i_edge != nEdges):
                index_bone_start = skel_edges[i_edge, 0]
                index_bone_end = skel_edges[i_edge, 1]
                #
                i_skel = ax3d.plot([skel_single_use[index_bone_end, 0]],
                                   [skel_single_use[index_bone_end, 1]],
                                   [skel_single_use[index_bone_end, 2]],
                                   marker='o', color=joint_colors[index_bone_end], zorder=1)
                h_skel3d.append(i_skel[0])
                #
                vec = np.array([[skel_single_use[index_bone_end, 0],
                                 skel_single_use[index_bone_end, 1],
                                 skel_single_use[index_bone_end, 2]],
                                [skel_single_use[index_bone_start, 0],
                                 skel_single_use[index_bone_start, 1],
                                 skel_single_use[index_bone_start, 2]]],
                                dtype=np.float64)
                i_skel_vec = ax3d.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                                       linestyle='-', marker='', color=joint_colors[index_bone_end], zorder=1)
                h_skel3d_vec.append(i_skel_vec[0])
        h3d = list([h_markers3d, h_markers3d_vec, h_skel3d, h_skel3d_vec])
        h3d_ax.append(h3d)
    h3d_ax.append(h3d_title)
    fig3d.tight_layout()
    fig3d.canvas.draw()

    def update_2d(t):
        x_single, markers_single, skel_single, skel2d_single, \
        x_samples, markers_samples, skel_samples, skel2d_samples, \
        joint_colors = \
            get_all(mu_torch[t], var_torch[t], t, args)
        #
        for i_cam in range(nCameras):
            xy_mean = np.mean(x_single[i_cam], 0)
            ax2d_list[i_cam].set_xlim([xy_mean[0]-xy_range, xy_mean[0]+xy_range])
            ax2d_list[i_cam].set_ylim([xy_mean[1]-xy_range, xy_mean[1]+xy_range])
            if (invert_xaxis):
                ax2d_list[i_cam].invert_xaxis()
            if (invert_yaxis):
                ax2d_list[i_cam].invert_yaxis()                
            h2d_ax[-1][i_cam].set(text='cam = {:01d}, index = {:06d}'.format(i_cam, int(cfg.index_frame_ini + t * cfg.dt)))
                
            img = ccv.get_frame(recFileNames[i_cam], int(cfg.index_frame_ini + t * cfg.dt) + 1)
            h2d_ax[0][i_cam].set_data(img)
            # markers
            for i_edge in range(nMarkers):
                if (measure_mask[t, i_cam, i_edge]):
                    alpha = 1.0
                else:
                    alpha = 0.0
                vec = np.array([[measure[t, i_cam, i_edge, 0], measure[t, i_cam, i_edge, 1]],
                                [x_single[i_cam, i_edge, 0], x_single[i_cam, i_edge, 1]]],
                                dtype=np.float64)
                h2d_ax[1][i_cam][i_edge].set_data(vec[:, 0], vec[:, 1])
                h2d_ax[2][i_cam][i_edge].set_data(measure[t, i_cam, i_edge, 0], measure[t, i_cam, i_edge, 1])
                h2d_ax[3][i_cam][i_edge].set_data(x_single[i_cam, i_edge, 0], x_single[i_cam, i_edge, 1])
                #
                h2d_ax[1][i_cam][i_edge].set_alpha(alpha/2)
                h2d_ax[2][i_cam][i_edge].set_alpha(alpha/2)
                h2d_ax[3][i_cam][i_edge].set_alpha(alpha/2)
            # skeleton 
            index_bone_start = skel_edges[0, 0]
            h2d_ax[5][i_cam][0].set_data(skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1])
            h2d_ax[5][i_cam][0].set_color(joint_colors[index_bone_start])
            for i_edge in range(nEdges):
                index_bone_start = skel_edges[i_edge, 0]
                index_bone_end = skel_edges[i_edge, 1]
                vec = np.array([[skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1]],
                                [skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1]]],
                                dtype=np.float64)
                h2d_ax[4][i_cam][i_edge].set_data(vec[:, 0], vec[:, 1])
                h2d_ax[4][i_cam][i_edge].set_color(joint_colors[index_bone_end])
                h2d_ax[5][i_cam][i_edge+1].set_data(skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1])
                h2d_ax[5][i_cam][i_edge+1].set_color(joint_colors[index_bone_end])
            if not(var_is_nan):
                h2d_ax[6][i_cam].set_data(np.ravel(skel2d_samples[:, i_cam, :, 0]), np.ravel(skel2d_samples[:, i_cam, :, 1]))
        return h2d_ax
        
    def update_3d(t):
        x_single, markers_single, skel_single, skel2d_single, \
        x_samples, markers_samples, skel_samples, skel2d_samples, \
        joint_colors = \
            get_all(mu_torch[t], var_torch[t], t, args)
        #
        markers_kalman_use = np.copy(markers_single)
        skel_kalman_use = np.copy(skel_single)
        markers_kalman_use[:, :2] -= skel_kalman_use[0, :2]
        skel_kalman_use[:, :2] -= skel_kalman_use[0, :2]
        #
        h3d_ax[-1].set(text='index = {:06d}'.format(int(cfg.index_frame_ini + t * cfg.dt)))
        for i_ax in range(3):
            for i_edge in range(nMarkers):
                h3d_ax[i_ax][0][i_edge].set_data(markers_kalman_use[i_edge, 0],
                                                 markers_kalman_use[i_edge, 1])
                h3d_ax[i_ax][0][i_edge].set_3d_properties(markers_kalman_use[i_edge, 2])

                index_start = joint_marker_index[i_edge]
                vec = np.array([[markers_kalman_use[i_edge, 0],
                                 markers_kalman_use[i_edge, 1],
                                 markers_kalman_use[i_edge, 2]],
                                [skel_kalman_use[index_start, 0],
                                 skel_kalman_use[index_start, 1],
                                 skel_kalman_use[index_start, 2]]],
                                dtype=np.float64)
                h3d_ax[i_ax][1][i_edge].set_data(vec[:, 0], 
                                                 vec[:, 1])
                h3d_ax[i_ax][1][i_edge].set_3d_properties(vec[:, 2])
            #
            i_edge = 0 
            index_bone_start = skel_edges[i_edge, 0]
            h3d_ax[i_ax][2][i_edge].set_data(skel_kalman_use[index_bone_start, 0],
                                             skel_kalman_use[index_bone_start, 1])
            h3d_ax[i_ax][2][i_edge].set_3d_properties(skel_kalman_use[index_bone_start, 2])
            h3d_ax[i_ax][2][i_edge].set_color(joint_colors[index_bone_start])
            for i_edge in range(nEdges):
                index_bone_start = skel_edges[i_edge, 0]
                index_bone_end = skel_edges[i_edge, 1]
                #
                h3d_ax[i_ax][2][i_edge+1].set_data(skel_kalman_use[index_bone_end, 0],
                                                   skel_kalman_use[index_bone_end, 1])
                h3d_ax[i_ax][2][i_edge+1].set_3d_properties(skel_kalman_use[index_bone_end, 2])
                h3d_ax[i_ax][2][i_edge+1].set_color(joint_colors[index_bone_end])
                #
                vec = np.array([[skel_kalman_use[index_bone_end, 0],
                                 skel_kalman_use[index_bone_end, 1],
                                 skel_kalman_use[index_bone_end, 2]],
                                [skel_kalman_use[index_bone_start, 0],
                                 skel_kalman_use[index_bone_start, 1],
                                 skel_kalman_use[index_bone_start, 2]]],
                                dtype=np.float64)
                h3d_ax[i_ax][3][i_edge].set_data(vec[:, 0], vec[:, 1])
                h3d_ax[i_ax][3][i_edge].set_3d_properties(vec[:, 2])
                h3d_ax[i_ax][3][i_edge].set_color(joint_colors[index_bone_end])
        return h3d_ax

    fps = float(slow_factor) * float(cfg.frame_rate) / float(cfg.dt)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=-1)
    #
    ani_3d = animation.FuncAnimation(fig3d, update_3d,
                                     frames=cfg.nT,
                                     interval=1, blit=False)
    if save:
        ani_3d.save(folder_save + '/' + folder.split('/')[-1] + '__ani_3d.mp4', writer=writer)
        print('Finished saving ani_3d')
    ani_2d = animation.FuncAnimation(fig2d, update_2d,
                                     frames=cfg.nT,
                                     interval=1, blit=False)
    if save:
        ani_2d.save(folder_save + '/' + folder.split('/')[-1] + '__ani_2d.mp4', writer=writer)
        print('Finished saving ani_2d')
    if verbose:
        plt.show()
    return ani_2d, ani_3d

if __name__ == '__main__':
    print(folder)
    #
    mu_ini = np.load(folder + '/x_ini.npy', allow_pickle=True)
    save_dict = np.load(folder + '/save_dict.npy', allow_pickle=True).item()
    if ((cfg.mode == 1) | (cfg.mode == 2)):
        mu = save_dict['mu'][1:]
        plot_2d_3d(mu, mu_ini, None)
    elif ((cfg.mode == 3) | (cfg.mode == 4)):
        mu_uks = save_dict['mu_uks'][1:]
        var_uks = save_dict['var_uks'][1:]
        mu0 = save_dict['mu0']
        var0 = save_dict['var0']
        A = save_dict['A']
        var_f = save_dict['var_f']
        var_g = save_dict['var_g']
        plot_2d_3d(mu_uks, mu_ini, var_uks)