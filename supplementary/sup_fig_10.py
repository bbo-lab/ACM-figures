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
folder_list = list([
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(5550, 5750), # jump/run or small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(10050, 10250), # wait
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(14830, 15030), # hickup or small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(30030, 30230), # wait
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(33530, 33730), # jump/run
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(38570, 38770), # wait
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(41930, 42130), # small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(46420, 46620), # wait
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(48990, 49190), # wait
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(52760, 52960), # wait
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(128720, 128920), # animal reaching
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(137470, 137670), # small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(145240, 145440), # small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(152620, 152820), # small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(155350, 155550), # small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(159080, 159280), # jump/run
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(161790, 161990), # jump/run or hickup
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(173910, 174110), # small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(176820, 177020), # jump/run
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(179060, 179260), # small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(181330, 181530), # small step
                     '/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(188520, 188720), # jump/run
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(2850, 3050), # jump/run or small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(8850, 9050), # wait
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(21920, 22120), # animal reaching
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(27160, 27360), # jump/run
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(31730, 31930), # small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(36280, 36480), # small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(38610, 38810), # small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(42010, 42210), # wait?
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(44610, 44810), # small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(47360, 47560), # small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(51220, 51420), # small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(53640, 53840), # hickup
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(55680, 55880), # small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(58300, 58500), # jump/run or small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(60410, 60610), # small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(62210, 62410), # hickup
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(64080, 64280), # hickup
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(66550, 66750), # hickup
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(68880, 69080), # jump/run or small step
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(71300, 71500), # hickup
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(74250, 74450), # jump/run
                     '/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(84520, 84720), # small step?
                    ])
folder = folder_reconstruction+'/'+folder_list[15]

sys.path = list(np.copy(sys_path0))
sys.path.append(folder)
importlib.reload(cfg)
cfg.animal_is_large = 0
importlib.reload(anatomy)

nSamples = int(200)
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
    # add necessary arrays to arguments
    args['plot'] = True
    args['x_torch'] = torch.from_numpy(mu_ini)

    mu_torch = torch.from_numpy(mu)
    if (var_is_nan):
        var_torch = torch.zeros(cfg.nT, dtype=torch.float64)
    else:
        var_torch = torch.from_numpy(var)     
    
    def get_all(mu_t, var_t, args):
        nCameras = args['numbers']['nCameras']
        nMarkers = args['numbers']['nMarkers']
        #
        A = args['calibration']['A_fit']
        k = args['calibration']['k_fit']
        RX1 = args['calibration']['RX1_fit']
        tX1 = args['calibration']['tX1_fit']

        z_all = mu_t
        x_all, markers_all, skel_all = model.fcn_emission_free(z_all, args)
        skel2d_all = model.map_m(RX1, tX1, A, k, skel_all)
        #
        x_all = x_all.reshape(nSamples, nCameras, nMarkers, 2)
        x_all[:, :, :, 0] = (x_all[:, :, :, 0] + 1.0) * (cfg.normalize_camera_sensor_x * 0.5)
        x_all[:, :, :, 1] = (x_all[:, :, :, 1] + 1.0) * (cfg.normalize_camera_sensor_y * 0.5)
        #
        x_all = x_all.cpu().numpy()
        markers_all = markers_all.cpu().numpy()
        skel_all = skel_all.cpu().numpy()
        skel2d_all = skel2d_all.cpu().numpy()
        #
        return x_all, markers_all, skel_all, skel2d_all
    
    t_frame = 199
    x_all, markers_all, skel_all, skel2d_all = get_all(mu_torch, var_torch, args)
    com = np.mean(skel2d_all, 2)
    
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
    # animal
    for i_cam in range(nCameras):
        ax2d = fig2d.add_subplot(2, 2, i_cam+1)
        ax2d.set_facecolor('black')
        
        file_ccv = data.path_ccv_sup_fig10_list[i_cam]
        img = ccv.get_frame(file_ccv, int(cfg.index_frame_ini + t_frame * cfg.dt) + 1)
        h_i_img = ax2d.imshow(img, cmap='gray', vmin=0.0, vmax=127, zorder=0)
        h_img.append(h_i_img)

        ax2d.set_aspect(1)
        ax2d.set_facecolor('black')
        ax2d.set_xticks(list([]))
        ax2d.set_yticks(list([]))
        ax2d.set_xticklabels(list([]))
        ax2d.set_yticklabels(list([]))
        if (False):
            ax2d.invert_xaxis()
        if (True):
            ax2d.invert_yaxis()
        ax2d_list.append(ax2d)
            
        i_title = ax2d.set_title('camera: {:01d}'.format(i_cam),
                                 color='darkgray')
        h_title.append(i_title)
    fig2d.canvas.draw()
    plt.pause(1e-16)
    fig2d.tight_layout()

    markersize = 4
    for i_mode in range(2):
        for i_cam in range(nCameras):
            ax2d = fig2d.axes[i_cam]
            ax2d.lines = list()
            if (i_mode == 0):
                s = 'com'
                # com
                nPoints = 200
                for i_point in range(nPoints):
                    ax2d.plot([com[i_point, i_cam, 0]], [com[i_point, i_cam, 1]],
                              linestyle='', marker='.', markersize=markersize, color=cmap(i_point/(nPoints-1)), zorder=i_point+1)
            else:
                s = 'all'
                # skeleton
                for t in np.linspace(0, 199, 6).astype(np.int64):
                    color = cmap(t/t_frame)
                    skel2d_single = skel2d_all[t]
                    h_edge_skel = list()
                    h_edge_joint = list()
                    index_bone_start = skel_edges[0, 0]
                    i_joint = ax2d.plot(skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1],
                                        linestyle='', marker='.', color=color, alpha=1.0, zorder=t+3,
                                        markersize=markersize)
                    h_edge_joint.append(i_joint[0])
                    for i_edge in range(nEdges):
                        index_bone_start = skel_edges[i_edge, 0]
                        index_bone_end = skel_edges[i_edge, 1]
                        vec = np.array([[skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1]],
                                        [skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1]]],
                                        dtype=np.float64)
                        i_skel_vec = ax2d.plot(vec[:, 0], vec[:, 1],
                                                linestyle='-', marker='', color=color, zorder=t+2)
                        h_edge_skel.append(i_skel_vec[0])
                        i_joint = ax2d.plot(skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1],
                                            linestyle='', marker='.', color=color, alpha=1.0, zorder=t+3,
                                            markersize=markersize)
                        h_edge_joint.append(i_joint[0])
        fig2d.canvas.draw()
        plt.pause(1e-16)
        if save:
            folder_single = folder.split('/')[-1]
            file_name = 'gap_crossing__{:s}__{:s}.svg'.format(folder_single, s)
            fig2d.savefig(saveFolder+'/'+file_name,
        #                     bbox_inches="tight",
                          dpi=300,
                          transparent=True,
                          format='svg',
                          pad_inches=0,
                          facecolor=fig2d.get_facecolor(), edgecolor='none')
            file_name = 'gap_crossing__{:s}__{:s}.tiff'.format(folder_single, s)
            fig2d.savefig(saveFolder+'/'+file_name,
        #                     bbox_inches="tight",
                          dpi=300,
                          transparent=True,
                          format='tiff',
                          pad_inches=0,
                          facecolor=fig2d.get_facecolor(), edgecolor='none')   
        if verbose:
            plt.show(block=False)
            print('Press any key to continue')
            input()