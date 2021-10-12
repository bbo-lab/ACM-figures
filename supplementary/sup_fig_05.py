#!/usr/bin/env python3

import importlib
import matplotlib.pyplot as plt
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

save = False
verbose = True

saveFolder = os.path.abspath('figures')

folder_reconstruction = data.path+'/reconstruction'
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
                    '/20200207/arena_20200207_064100_064400__pcutoff9e-01', # 27
                    ])

i_folder = 15
folder_single = folder_list[i_folder]
folder = folder_reconstruction + folder_single
dt = 100
t_start = 0
t_end = int(folder_single.split('_')[3]) - int(folder_single.split('_')[2]) - dt
t_list = np.arange(t_start, t_end + dt, dt, dtype=np.int64)

sys.path = list(np.copy(sys_path0))
sys.path.append(folder)
importlib.reload(cfg)
cfg.animal_is_large = 0
importlib.reload(anatomy)

nSamples = int(0)
xy_range = 256
cmap = plt.cm.viridis

if __name__ == '__main__':
    folder_reqFiles = data.path + '/required_files' 
    file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
    file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
    file_model = folder_reqFiles + '/model.npy'
    file_labelsDLC = folder_reqFiles + '/labels_dlc_dummy.npy' # DLC labels are not used here

    args = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                cfg.scale_factor, cfg.pcutoff)
    if ((cfg.mode == 1) or (cfg.mode == 2)):
        args['use_custom_clip'] = False
    elif ((cfg.mode == 3) or (cfg.mode == 4)):
        args['use_custom_clip'] = True
    
    # free parameters
    free_para_bones = args['free_para_bones'].cpu().numpy()
    free_para_markers = args['free_para_markers'].cpu().numpy()
    free_para_pose = args['free_para_pose'].cpu().numpy()
    # remove all free parameters that do not modify the pose
    free_para_bones = np.zeros_like(free_para_bones, dtype=bool)
    free_para_markers = np.zeros_like(free_para_markers, dtype=bool)
    nFree_bones = int(0)
    nFree_markers = int(0)
    args['free_para_bones'] = torch.from_numpy(free_para_bones)
    args['free_para_markers'] = torch.from_numpy(free_para_markers)
    args['nFree_bones'] = nFree_bones
    args['nFree_markers'] = nFree_markers
    # create correct free_para
    free_para = np.concatenate([free_para_bones,
                                free_para_markers,
                                free_para_pose], 0)

    # mu_ini
    mu_ini = np.load(folder+'/x_ini.npy', allow_pickle=True)
    #
    mu_ini_fac = 0.9
    mu_ini_min = (args['bounds_free_pose_0'] - args['bounds_free_pose_range'] * mu_ini_fac).numpy()
    mu_ini_max = (args['bounds_free_pose_0'] + args['bounds_free_pose_range'] * mu_ini_fac).numpy()
    mu_ini_free_clamped = np.copy(mu_ini[free_para])
    mask_down = (mu_ini_free_clamped < mu_ini_min)
    mask_up = (mu_ini_free_clamped > mu_ini_max)
    mu_ini_free_clamped[mask_down] = mu_ini_min[mask_down]
    mu_ini_free_clamped[mask_up] = mu_ini_max[mask_up]
    mu_ini[free_para] = np.copy(mu_ini_free_clamped)
   
    save_dict = np.load(folder+'/'+'save_dict.npy', allow_pickle=True).item()
    mu0 = torch.from_numpy(save_dict['mu0'])
    var0 = torch.from_numpy(save_dict['var0'])
    A = torch.from_numpy(save_dict['A'])
    var_f = torch.from_numpy(save_dict['var_f'])
    var_g = torch.from_numpy(save_dict['var_g'])
    mu_uks = torch.from_numpy(save_dict['mu_uks'])
    var_uks = torch.from_numpy(save_dict['var_uks'])

    mu = mu_uks[1:].cpu().numpy()
    var = var_uks[1:].cpu().numpy()
    var_is_nan = np.all(var == None)
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

    # add necessary arrays to arguments
    if (var_is_nan):
        nSamples_use = 1
    else:
        nSamples_use = nSamples + 1
    args['plot'] = True
    args['x_torch'] = torch.from_numpy(mu_ini)
    #
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
            if (nSamples > 0):
                distribution = torch.distributions.MultivariateNormal(loc=mu_t,
                                                                      scale_tril=kalman.cholesky_save(var_t))
                z_samples = distribution.sample((nSamples,))
                z_all = torch.cat([mu_t[None, :], z_samples], 0)
            else:
                z_all = mu_t[None, :]
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
        return x_single, markers_single, skel_single, skel2d_single, \
               x_samples, markers_samples, skel_samples, skel2d_samples
    
    ##### 2D #####
    fig2d = plt.figure(figsize=(10, 10))
    fig2d.canvas.manager.window.move(0, 0)
    fig2d.set_facecolor('black')
    h_cam_line = list()
    h_cam_true = list()
    h_cam_single = list()
    h_cam_skel = list()
    h_cam_joint = list()
    h_cam_joint_samples = list()
    h_title = list()
    h_img = list() 
    ax2d_list = list()
    #
    # image
    t = t_list[-1]
    for i_cam in range(nCameras):
        ax2d = fig2d.add_subplot(2, 2, i_cam+1)
        ax2d.set_facecolor('black')
        file_ccv = data.path_ccv_sup_fig5_list[i_cam]
        img = ccv.get_frame(file_ccv, int(cfg.index_frame_ini + t * cfg.dt) + 1)
        h_i_img = ax2d.imshow(img, cmap='gray', vmin=0.0, vmax=127, zorder=0)
        h_img.append(h_i_img)
        #
        ax2d.set_aspect(1)
        ax2d.set_facecolor('black')
        ax2d.set_axis_off()
        #
        i_title = ax2d.set_title('camera: {:01d}'.format(i_cam),
                                 color='gray')
        h_title.append(i_title)
        #
        ax2d_list.append(ax2d)
    # animal
    for i_t in range(len(t_list)):
        t = t_list[i_t]
        #
        x_single, markers_single, skel_single, skel2d_single, \
        x_samples, markers_samples, skel_samples, skel2d_samples, =\
            get_all(mu_torch[t], var_torch[t], t, args)
        #
        if (len(t_list) <= 1):
            joint_colors_all = cmap(0.0)
        else:
            joint_colors_all = cmap(i_t/(len(t_list) - 1))
        #
        for i_cam in range(nCameras):
            ax2d = ax2d_list[i_cam]
            # skeleton
            markersize = 4
            h_edge_skel = list()
            h_edge_joint = list()
            index_bone_start = skel_edges[0, 0]
            color = joint_colors_all
            i_joint = ax2d.plot(skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1],
                                linestyle='', marker='.', color=color, alpha=1.0, zorder=6,
                                markersize=markersize)
            h_edge_joint.append(i_joint[0])
            for i_edge in range(nEdges):
                index_bone_start = skel_edges[i_edge, 0]
                index_bone_end = skel_edges[i_edge, 1]
                vec = np.array([[skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1]],
                                [skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1]]],
                                dtype=np.float64)
                i_skel_vec = ax2d.plot(vec[:, 0], vec[:, 1],
                                        linestyle='-', marker='', color=color, zorder=1+i_t)
                h_edge_skel.append(i_skel_vec[0])
                i_joint = ax2d.plot(skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1],
                                    linestyle='', marker='.', color=color, alpha=1.0, zorder=1+i_t,
                                    markersize=markersize)
                h_edge_joint.append(i_joint[0])
            # marker
            h_edge_line = list()
            h_edge_true = list()
            h_edge_single = list()
            #
            h_cam_line.append(h_edge_line)
            h_cam_true.append(h_edge_true)
            h_cam_single.append(h_edge_single)
            h_cam_skel.append(h_edge_skel)
            h_cam_joint.append(h_edge_joint)
        h2d_ax = list([h_img,
                       h_cam_line, h_cam_true, h_cam_single,
                       h_cam_skel, h_cam_joint, h_cam_joint_samples,
                       h_title])
    
    fig2d.tight_layout()
    fig2d.canvas.draw()
    plt.pause(1e-16)
    if save:
        file_name = 'free_exploring__{:s}__{:06d}_{:06d}_{:06d}.svg'.format(cfg.date, t_start, t_end+dt, dt)
        fig2d.savefig(saveFolder+'/'+file_name,
    #                     bbox_inches="tight",
                      dpi=300,
                      transparent=True,
                      format='svg',
                      pad_inches=0,
                      facecolor=fig2d.get_facecolor(), edgecolor='none')   
        file_name = 'free_exploring__{:s}__{:06d}_{:06d}_{:06d}.tiff'.format(cfg.date, t_start, t_end+dt, dt)
        fig2d.savefig(saveFolder+'/'+file_name,
    #                     bbox_inches="tight",
                      dpi=300,
                      transparent=True,
                      format='tiff',
                      pad_inches=0,
                      facecolor=fig2d.get_facecolor(), edgecolor='none')   
    if verbose:
        plt.show()
