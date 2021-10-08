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

folder_save = os.path.abspath('panels')

folder_reconstruction = data.path+'/reconstruction'
folder_data_list = list([
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(5550, 5750), # jump/run or small step 0
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(10050, 10250), # wait 1
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(14830, 15030), # hickup or small step 2
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(30030, 30230), # wait 3
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(33530, 33730), # jump/run 4
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(38570, 38770), # wait 5
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(41930, 42130), # small step 6
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(46420, 46620), # wait 7
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(48990, 49190), # wait 8
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(52760, 52960), # wait 9
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(128720, 128920), # animal reaching 10
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(137470, 137670), # small step 11
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(145240, 145440), # small step 12
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(152620, 152820), # small step 13
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(155350, 155550), # small step 14
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(159080, 159280), # jump/run 15
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(161790, 161990), # jump/run or hickup 16
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(173910, 174110), # small step 17
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(176820, 177020), # jump/run 18
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(179060, 179260), # small step 19
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(181330, 181530), # small step 20
                     folder_reconstruction+'/20200205/gap_20200205_{:06d}_{:06d}__pcutoff9e-01'.format(188520, 188720), # jump/run 21
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(2850, 3050), # jump/run or small step 22
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(8850, 9050), # wait 23
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(21920, 22120), # animal reaching 24
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(27160, 27360), # jump/run 25
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(31730, 31930), # small step 26
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(36280, 36480), # small step 27
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(38610, 38810), # small step 28
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(42010, 42210), # wait? 29
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(44610, 44810), # small step 30
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(47360, 47560), # small step 31
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(51220, 51420), # small step 32
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(53640, 53840), # hickup 33
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(55680, 55880), # small step 34
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(58300, 58500), # jump/run or small step 35
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(60410, 60610), # small step 36
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(62210, 62410), # hickup 37
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(64080, 64280), # hickup 38
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(66550, 66750), # hickup 39
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(68880, 69080), # jump/run or small step 40
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(71300, 71500), # hickup 41
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(74250, 74450), # jump/run 42
                     folder_reconstruction+'/20200207/gap_20200207_{:06d}_{:06d}__pcutoff9e-01'.format(84520, 84720), # small step? 43
                    ])
i_img_list_list = list([
                    [88, 128, 144],
                    [71, 120, 140],
                    [94, 124, 147],
                    [72, 118, 137],
                    [89, 120, 144],
                    [60, 113, 136],
                    [93, 124, 150],
                    [98, 140, 162],
                    [87, 125, 148],
                    [73, 127, 150],
                    [93, 156, 174],
                    [90, 120, 136],
                    [102, 130, 147],
                    [100, 123, 141],
                    [101, 123, 146],
                    [96, 124, 144],
                    [100, 125, 144],
                    [102, 126, 149],
                    [96, 118, 139],
                    [88, 124, 148],
                    [92, 120, 143],
                    [86, 117, 136],
                    [90, 132, 151],
                    [102, 141, 163],
                    [88, 137, 154],
                    [100, 132, 152],
                    [98, 131, 150],
                    [107, 142, 162],
                    [104, 138, 156],
                    [98, 136, 148],
                    [92, 135, 152],
                    [98, 139, 157],
                    [104, 135, 152],
                    [113, 145, 164],
                    [108, 139, 157],
                    [99, 135, 153],
                    [99, 129, 148],
                    [98, 128, 145],
                    [101, 148, 166],
                    [102, 149, 166],
                    [101, 136, 155],
                    [102, 135, 154],
                    [101, 143, 160],
                    [105, 134, 153],
                    ])

file_ccv = data.path_ccv_fig4
i_data = 15
modulus = np.mod(i_data, 2)
if (modulus == 0):
    i_cam = 0
else:
    i_cam = 2
folder = folder_data_list[i_data]
i_img_list = i_img_list_list[i_data]

sys.path = list(np.copy(sys_path0))
sys.path.append(folder)
importlib.reload(cfg)
cfg.animal_is_large = 0
importlib.reload(anatomy)

bg_threshold = 16

dx0 = 260
dy0 = 120

if __name__ == '__main__':
    mm_in_inch = 5.0/127.0
    fig_xz_w = np.round(mm_in_inch * 88.0, decimals=2)
    fig_xz_h = fig_xz_w/2
    fig = plt.figure(1, frameon=False, figsize=(fig_xz_w, fig_xz_h))
    fig.canvas.manager.window.move(0, 0)
    fig.clear()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.clear()
    ax.set_axis_off()
    fig.add_axes(ax)
    img_dummy = np.zeros((2*dy0, 2*dx0), dtype=np.float64)
    h_img = ax.imshow(img_dummy,
                      'gray',
                      vmin=0,
                      vmax=64,
                      aspect=1)
    ax.invert_yaxis()
    
    color_list = list(['green', 'orange', 'red'])
    
    nSamples = int(0)
    save_dict = np.load(folder+'/save_dict.npy', allow_pickle=True).item()
    file_mu_ini = folder+'/'+'x_ini.npy'
    mu_ini = np.load(file_mu_ini, allow_pickle=True)
    if ('mu_uks' in save_dict):
        mu_uks = save_dict['mu_uks'][1:]
        var_uks = save_dict['var_uks'][1:]
        nSamples_use= np.copy(nSamples)
        print(save_dict['message'])
    else:
        mu_uks = save_dict['mu'][1:]
        nT = np.size(mu_uks, 0)
        nPara = np.size(mu_uks, 1)
        var_dummy = np.identity(nPara, dtype=np.float64) * 2**-52
        var_uks = np.tile(var_dummy.ravel(), cfg.nT).reshape(cfg.nT, nPara, nPara)
        nSamples_use = int(0)

    folder_reqFiles = data.path + '/required_files' 
    file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
    file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
    file_model = folder_reqFiles + '/model.npy'
    file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_dlc_{:06d}_{:06d}.npy'.format(cfg.index_frame_start, cfg.index_frame_end)
        
    args = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                cfg.scale_factor, cfg.pcutoff)
    if ((cfg.mode == 1) or (cfg.mode == 2)):
        args['use_custom_clip'] = False
    elif ((cfg.mode == 3) or (cfg.mode == 4)):
        args['use_custom_clip'] = True
    args['plot'] = True
    del(args['model']['surface_vertices'])
    joint_order = args['model']['joint_order']
    joint_marker_order = args['model']['joint_marker_order']
    skeleton_edges = args['model']['skeleton_edges'].cpu().numpy()
    nCameras = args['numbers']['nCameras']
    nMarkers = args['numbers']['nMarkers']
    nBones = args['numbers']['nBones']

    free_para_bones = args['free_para_bones'].cpu().numpy()
    free_para_markers = args['free_para_markers'].cpu().numpy()
    free_para_pose = args['free_para_pose'].cpu().numpy()
    free_para_bones = np.zeros_like(free_para_bones, dtype=bool)
    free_para_markers = np.zeros_like(free_para_markers, dtype=bool)
    nFree_bones = int(0)
    nFree_markers = int(0)
    free_para = np.concatenate([free_para_bones,
                                free_para_markers,
                                free_para_pose], 0)    
    args['x_torch'] = torch.from_numpy(mu_ini).type(model.float_type)
    args['x_free_torch'] = torch.from_numpy(mu_ini[free_para]).type(model.float_type)
    args['free_para_bones'] = torch.from_numpy(free_para_bones)
    args['free_para_markers'] = torch.from_numpy(free_para_markers)
    args['nFree_bones'] = nFree_bones
    args['nFree_markers'] = nFree_markers    
    args['x_torch'] = torch.from_numpy(mu_ini).type(model.float_type)
    args['x_free_torch'] = torch.from_numpy(mu_ini[free_para]).type(model.float_type)
    #
    t_start = 0
    t_end = 200
    nT_single = t_end - t_start
    #
    if (nSamples > 0):
        mu_t = torch.from_numpy(mu_uks[t_start:t_end])
        var_t = torch.from_numpy(var_uks[t_start:t_end])
        distribution = torch.distributions.MultivariateNormal(loc=mu_t,
                                                              scale_tril=kalman.cholesky_save(var_t))
        z_samples = distribution.sample((nSamples,))
        z_all = torch.cat([mu_t[None, :], z_samples], 0)
    else:
        z_all = torch.from_numpy(mu_uks[t_start:t_end])
    _, _, skel_all = model.fcn_emission_free(z_all, args)
    A = args['calibration']['A_fit']
    k = args['calibration']['k_fit']
    RX1 = args['calibration']['RX1_fit']
    tX1 = args['calibration']['tX1_fit']
    skel2d_all = model.map_m(RX1, tX1, A, k,
                             skel_all)
    skel2d_all = skel2d_all.cpu().numpy()
    
    # plot
    for i_img in range(len(i_img_list)):
        frame_index = i_img_list[i_img] + cfg.index_frame_start
        if not(file_ccv == ''):
            img0 = ccv.get_frame(file_ccv, frame_index+1-5).astype(np.float64)
            img = ccv.get_frame(file_ccv, frame_index+1).astype(np.float64)
        else:
            img0 = np.full((1024, 1280), 0.0, dtype=np.float64)
            img = np.full((1024, 1280), 32.5, dtype=np.float64)
        
        diff = abs(img - img0)
        diff[diff < bg_threshold] = 0.0

        index_y, index_x = np.where(diff > 0.0)
        center_x = np.mean(index_x).astype(np.int64)
        center_y = np.mean(index_y).astype(np.int64)
        
        yRes, xRes = np.shape(img)
        xlim_min = center_x - dx0
        xlim_max = center_x + dx0
        ylim_min = center_y - dy0
        ylim_max = center_y + dy0
        xlim_min_use = np.max([xlim_min, 0])
        xlim_max_use = np.min([xlim_max, xRes])
        ylim_min_use = np.max([ylim_min, 0])
        ylim_max_use = np.min([ylim_max, yRes])
        dx = np.int64(xlim_max_use - xlim_min_use)
        dy = np.int64(ylim_max_use - ylim_min_use)
        #
        dx_add = dx0 - (center_x - xlim_min_use)
        dy_add = dy0 - (center_y - ylim_min_use)
        dx_add_int = np.int64(dx_add)
        dy_add_int = np.int64(dy_add)
        #
        img_dummy.fill(0.0)
        img_dummy[dy_add_int:dy+dy_add_int, dx_add_int:dx+dx_add_int] = \
            img[ylim_min_use:ylim_max_use, xlim_min_use:xlim_max_use]        
        h_img.set_array(img_dummy)

        # plot skeleton
        ax.lines = list()
        #
        skel2d_single = skel2d_all[frame_index-cfg.index_frame_start]
        skel2d_single[:, :, 0] = skel2d_single[:, :, 0] - float(xlim_min_use) + float(dx_add_int)
        skel2d_single[:, :, 1] = skel2d_single[:, :, 1] - float(ylim_min_use) + float(dy_add_int)
        #
        alpha = 0.5
        markersize = 0#1
        linewidth = 0.5
        #
        index_bone_start = skeleton_edges[0, 0]
        i_joint = ax.plot(skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1],
                            linestyle='', marker='.', color=color_list[i_img], alpha=alpha, zorder=2,
                            markersize=markersize, markeredgewidth=markersize)
        for i_edge in range(nBones):
            index_bone_start = skeleton_edges[i_edge, 0]
            index_bone_end = skeleton_edges[i_edge, 1]
            vec = np.array([[skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1]],
                            [skel2d_single[i_cam, index_bone_start, 0], skel2d_single[i_cam, index_bone_start, 1]]],
                            dtype=np.float64)
            i_skel_vec = ax.plot(vec[:, 0], vec[:, 1],
                                    linestyle='-', marker='', linewidth=linewidth, color=color_list[i_img], alpha=alpha, zorder=1)
            i_joint = ax.plot(skel2d_single[i_cam, index_bone_end, 0], skel2d_single[i_cam, index_bone_end, 1],
                                linestyle='', marker='.', color=color_list[i_img], alpha=alpha, zorder=2,
                                markersize=markersize, markeredgewidth=markersize)
        
        fig.canvas.draw()
        plt.pause(2**-10)
        if verbose:
            plt.show(block=False)
            plt.pause(2**-10)
            fig.canvas.draw()
            print('Press any key to continue')
            input()

        if save:
            fig.savefig(folder_save+'/{:s}_{:s}_cam{:01d}_img{:06d}__overlay.svg'.format(task, date, i_cam, frame_index),
                         dpi=300,
                         transparent=True,
                         format='svg',
                         pad_inches=0)
            fig.savefig(folder_save+'/{:s}_{:s}_cam{:01d}_img{:06d}__overlay.tiff'.format(task, date, i_cam, frame_index),
                         dpi=300,
                         transparent=True,
                         format='tiff',
                         pad_inches=0)
