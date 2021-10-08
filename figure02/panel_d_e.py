#!/usr/bin/env python3

import importlib
import copy
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
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
save_inset = False
verbose = True

folder_save = os.path.abspath('panels')

file_ccv = data.path_ccv_fig2_d_e
date = '20200205'
task = 'table'
folder_reconstruction = data.path+'/reconstruction'
folder = folder_reconstruction+'/'+date+'/'+'{:s}_{:s}_020300_021800__pcutoff9e-01'.format(task, date)

sys.path = list(np.copy(sys_path0))
sys.path.append(folder)
importlib.reload(cfg)
cfg.animal_is_large = 0
importlib.reload(anatomy)

nCameras = 6
clim_min = list([0 for i in range(nCameras)])
clim_max = list([127.0, 127.0, 127.0, 127.0, 16.0, 16.0])

bg_threshold = 8
dxy = 300

i_cam_down = 0
nCameras_up = 4
nSamples_use = int(1e6)

i_img_start = 20750
dFrame = 50
nFrames = 1

# overview
dx = 220
dy = 110 

# inset
dx_inset = 35
dy_inset = 35
i_paw_inset = 0

data_req_folder = 'data'

label_names = list(['spot_finger_left_001',
                    'spot_finger_left_002',
                    'spot_finger_left_003',
                    'spot_finger_right_001',
                    'spot_finger_right_002',
                    'spot_finger_right_003',
                    'spot_toe_left_001',
                    'spot_toe_left_002',
                    'spot_toe_left_003',
                    'spot_toe_right_001',
                    'spot_toe_right_002',
                    'spot_toe_right_003'])
label_names2 = list(['spot_paw_front_left',
                     'spot_paw_front_right',
                     'spot_paw_hind_left',
                     'spot_paw_hind_right'])
nLabels_check = len(label_names)
nLabels_check2 = len(label_names2)

cmap = plt.cm.tab10
color_left_front = cmap(4/9)
color_right_front = cmap(3/9)
color_left_hind = cmap(9/9)
color_right_hind = cmap(8/9)
colors = list([color_left_front, color_right_front, color_left_hind, color_right_hind])

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

bottom_margin_x = 0.05 # inch
left_margin_x = 0.05 # inch
left_margin  = 0.4 # inch
right_margin = 0.01 # inch
bottom_margin = 0.3 # inch
top_margin = 0.05 # inch

fontname = "Arial"
fontsize = 6
linewidth = 0.1
markersize = 2.5
markersize_inset = 6.0
markeredgewidth = 0.1

path_effect_fac = 1.25
path_effect_fac_text = 0.5

def map_m(M,
          RX1, tX1, A, k): # nLabels, 3
    # RX1 * m + tX1
    m_cam = np.einsum('ij,lj->li', RX1, M) + tX1[None, :] # nLabels, 3
    # m / m[2]
    m = m_cam[:, :2] / m_cam[:, 2][:, None] # nLabels, 2
    # distort & A * m
    r_2 = m[:, 0]**2 + m[:, 1]**2 # nLabels
    sum_term = 1.0 + \
               k[0] * r_2 + \
               k[1] * r_2**2 + \
               k[4] * r_2**3 # nLabels
    x_times_y_times_2 = m[:, 0] * m[:, 1] * 2.0  # nLabels 
    m = np.stack([A[1] + A[0] * \
                  (m[:, 0] * sum_term + \
                   k[2] * x_times_y_times_2 + \
                   k[3] * (r_2 + 2.0 * m[:, 0]**2)),
                  A[3] + A[2] * \
                  (m[:, 1] * sum_term + \
                  k[2] * (r_2 + 2.0 * m[:, 1]**2) + \
                  k[3] * x_times_y_times_2)], 1) # nLabels, 2
    return m # nLabels, 2

if __name__ == '__main__':
    i_img_list = np.arange(i_img_start, i_img_start+dFrame*nFrames, dFrame, dtype=np.int64)

    fig0_w = np.round(mm_in_inch * (88.0*1.0), decimals=2)
    fig0_h = fig0_w
    fig0 = plt.figure(1, figsize=(fig0_w, fig0_h))
    fig0.canvas.manager.window.move(0, 0)
    fig0.clear()
    ax0 = plt.Axes(fig0, [0., 0., 1., 1.])
    ax0.clear()
    ax0.xaxis.set_major_locator(plt.NullLocator())
    ax0.yaxis.set_major_locator(plt.NullLocator())
    ax0.set_axis_off()
    ax0.set_aspect(1)
    fig0.add_axes(ax0)
    #
    fig_w = np.round(mm_in_inch * (87.0*1/2), decimals=2)
    fig_h = fig_w
    fig = plt.figure(2, figsize=(fig_w, fig_h))
    fig.canvas.manager.window.move(0, 0)
    fig.clear()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.clear()
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_axis_off()
    ax.set_aspect(1)
    fig.add_axes(ax)
            
    # underneath ccv images + labels and reconstructions
    extent = (0, 2*dx, 2*dy, 0)
    img_dummy = np.zeros((2*dy, 2*dx), dtype=np.float64)
    h_img = ax0.imshow(img_dummy,
                       'gray',
                       interpolation=None,
                       vmin=0.0, vmax=32.0, aspect=1,
                       extent=extent)
    
    folder_reqFiles = data.path + '/required_files' 
    file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
    file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
    file_model = folder_reqFiles + '/model.npy'
    file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_dlc_{:06d}_{:06d}.npy'.format(cfg.index_frame_start, cfg.index_frame_end)
    #
    file_labels_down = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_down_all.npz'
    file_calibration_all = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/calibration_all/multicalibration.npy'

    labels_down = np.load(file_labels_down, allow_pickle=True)['arr_0'].item()
    origin_coord = np.load(file_origin_coord, allow_pickle=True).item()

    counter = 0
    center_x = 0.0
    center_y = 0.0
    for key in labels_down[i_img_start]:
        center_x = center_x + labels_down[i_img_start][key][i_cam_down][0]
        center_y = center_y + labels_down[i_img_start][key][i_cam_down][1]
        counter = counter + 1
    center_x = int(center_x / counter)
    center_y = int(center_y / counter)
    if not(file_ccv == ''):
        img = ccv.get_frame(file_ccv, i_img_start+1).astype(np.float64)
    else:
        img = np.full((1024, 1280), 8.5, dtype=np.float64)
    yRes, xRes = np.shape(img)
    #
    xlim_min = center_x - dx
    xlim_max = center_x + dx
    ylim_min = center_y - dy
    ylim_max = center_y + dy
    xlim_min_use = np.max([xlim_min, 0])
    xlim_max_use = np.min([xlim_max, xRes])
    ylim_min_use = np.max([ylim_min, 0])
    ylim_max_use = np.min([ylim_max, yRes])
    dx_new = np.int64(xlim_max_use - xlim_min_use)
    dy_new = np.int64(ylim_max_use - ylim_min_use)
    #
    center_x_use = dx - (center_x - xlim_min_use)
    center_y_use = dy - (center_y - ylim_min_use)
    dx_add_int = np.int64(center_x_use)
    dy_add_int = np.int64(center_y_use)
    #
    img_dummy.fill(0.0)
    img_dummy[dy_add_int:dy_new+dy_add_int, dx_add_int:dx_new+dx_add_int] = \
        img[ylim_min_use:ylim_max_use, xlim_min_use:xlim_max_use]
    h_img.set_array(img_dummy)
    h_img.set_clim(clim_min[nCameras_up + i_cam_down], clim_max[nCameras_up + i_cam_down])
    #
    for i_img in i_img_list: 
        mu_ini = np.load(folder+'/x_ini.npy', allow_pickle=True)
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
            nSamples = int(0)
        frame_list_fit = np.arange(cfg.index_frame_ini,
                                   cfg.index_frame_ini + cfg.nT * cfg.dt,
                                   cfg.dt, dtype=np.int64)

        origin_coord = np.load(file_origin_coord, allow_pickle=True).item()
        origin = origin_coord['origin']
        coord = origin_coord['coord']
        calib = np.load(file_calibration_all, allow_pickle=True).item()
        A = calib['A_fit'][nCameras_up + i_cam_down]
        k = calib['k_fit'][nCameras_up + i_cam_down]
        tX1 = calib['tX1_fit'][nCameras_up + i_cam_down] * cfg.scale_factor
        RX1 = calib['RX1_fit'][nCameras_up + i_cam_down]
        frame_list_manual = np.array(sorted(list(labels_down.keys())), dtype=np.int64)
        frame_list_manual = frame_list_manual[np.array(list([i in frame_list_fit for i in frame_list_manual]))]
        nT_check = len(frame_list_manual)
        kalman_index = frame_list_fit[np.array(list([i in frame_list_manual for i in frame_list_fit]))] - cfg.index_frame_ini

        args_torch = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                          cfg.scale_factor, cfg.pcutoff)
        if ((cfg.mode == 1) or (cfg.mode == 2)):
            args_torch['use_custom_clip'] = False
        elif ((cfg.mode == 3) or (cfg.mode == 4)):
            args_torch['use_custom_clip'] = True
        joint_marker_order = args_torch['model']['joint_marker_order']
        free_para_bones = args_torch['free_para_bones'].cpu().numpy()
        free_para_markers = args_torch['free_para_markers'].cpu().numpy()
        free_para_pose = args_torch['free_para_pose'].cpu().numpy()
        free_para_bones = np.zeros_like(free_para_bones, dtype=bool)
        free_para_markers = np.zeros_like(free_para_markers, dtype=bool)
        nFree_bones = int(0)
        nFree_markers = int(0)
        free_para = np.concatenate([free_para_bones,
                                    free_para_markers,
                                    free_para_pose], 0)    
        args_torch['x_torch'] = torch.from_numpy(mu_ini).type(model.float_type)
        args_torch['x_free_torch'] = torch.from_numpy(mu_ini[free_para]).type(model.float_type)
        args_torch['free_para_bones'] = torch.from_numpy(free_para_bones)
        args_torch['free_para_markers'] = torch.from_numpy(free_para_markers)
        args_torch['nFree_bones'] = nFree_bones
        args_torch['nFree_markers'] = nFree_markers    
        args_torch['plot'] = True
        del(args_torch['model']['surface_vertices'])
        
        index = list()
        for i_label in range(nLabels_check):
            label_name = label_names[i_label]
            marker_name = 'marker_' + '_'.join(label_name.split('_')[1:]) + '_start'
            index.append(joint_marker_order.index(marker_name))
        index = np.array(index)
        index2 = list()
        for i_label in range(nLabels_check2):
            label_name = label_names2[i_label]
            marker_name = 'marker_' + '_'.join(label_name.split('_')[1:]) + '_start'
            index2.append(joint_marker_order.index(marker_name))
        index2 = np.array(index2)        
        
        # cam
        fingers_cam = np.full((nLabels_check, 2), np.nan, dtype=np.float64)
        for i_label in range(nLabels_check):
            label_name = label_names[i_label]
            if label_name in labels_down[i_img_start]:
                fingers_cam[i_label] = np.array([labels_down[i_img_start][label_name][i_cam_down][0],
                                                 labels_down[i_img_start][label_name][i_cam_down][1]], dtype=np.float64)
        fingers_cam = fingers_cam.reshape(4, 3, 2)
        paws_cam = np.full((nLabels_check2, 2), np.nan, dtype=np.float64)
        for i_label in range(nLabels_check2):
            label_name = label_names2[i_label]
            if label_name in labels_down[i_img_start]:
                paws_cam[i_label] = np.array([labels_down[i_img_start][label_name][i_cam_down][0],
                                              labels_down[i_img_start][label_name][i_cam_down][1]], dtype=np.float64)
        # fit
        kalman_index = i_img_start - cfg.index_frame_ini
        mu_t = torch.from_numpy(mu_uks[kalman_index])
        var_t = torch.from_numpy(var_uks[kalman_index])
        distribution = torch.distributions.MultivariateNormal(loc=mu_t,
                                                              scale_tril=kalman.cholesky_save(var_t))
        z_samples = distribution.sample((nSamples,))
        z_all = torch.cat([mu_t[None, :], z_samples], 0)
        _, marker_pos_torch, _ = model.fcn_emission_free(z_all, args_torch)
        marker_pos_torch = marker_pos_torch.numpy()
        fingers_fit_3d = marker_pos_torch[:, index].reshape(nSamples+1, 4, 3, 3)
        paws_fit_3d = marker_pos_torch[:, index2]
        #
        fingers_fit = np.full((nSamples+1, 4, 3, 2), np.nan, dtype=np.float64)
        paws_fit = np.full((nSamples+1, 4, 2), np.nan, dtype=np.float64)
        for i_paw in range(4):
            paw_is_visible = ~np.any(np.isnan(paws_cam[i_paw]))
            if (paw_is_visible):
                mask = ~np.isnan(fingers_cam[i_paw])
                n_mask = np.sum(mask)
                if (n_mask >= 3):
                    fingers_fit[:, i_paw] = map_m(fingers_fit_3d[:, i_paw, :].reshape((nSamples+1)*3, 3),
                                                  RX1, tX1, A, k)[:, :2].reshape((nSamples+1), 3, 2) # nSamples, nFingers, 2
                    paws_fit[:, i_paw] = map_m(paws_fit_3d[:, i_paw], RX1, tX1, A, k)[:, :2]
        #
        shift_vector = np.array([xlim_min_use, ylim_min_use], dtype=np.float64)
        fingers_cam = fingers_cam - shift_vector
        paws_cam = paws_cam - shift_vector
        fingers_fit = fingers_fit - shift_vector
        paws_fit = paws_fit - shift_vector

        samples_zoom_factor = 4.0
        dBin = 1.0/samples_zoom_factor
        bins = list([np.linspace(0.0, 2.0*dy, int(samples_zoom_factor*2*dy+1), dtype=np.float64),
                     np.linspace(0.0, 2.0*dx, int(samples_zoom_factor*2*dx+1), dtype=np.float64)])
        hist_paws_samples = np.zeros((4, int(samples_zoom_factor*2*dy), int(samples_zoom_factor*2*dx)), dtype=np.float64)
        hist_paws_samples_xedges = np.zeros((4, int(samples_zoom_factor*2*dy+1)), dtype=np.float64)
        hist_paws_samples_yedges = np.zeros((4, int(samples_zoom_factor*2*dx+1)), dtype=np.float64)
        img_paws_samples_density = np.zeros((4, int(samples_zoom_factor*2*dy), int(samples_zoom_factor*2*dx), 4), dtype=np.float64)
        hist_fingers_samples = np.zeros((4, 3, int(samples_zoom_factor*2*dy), int(samples_zoom_factor*2*dx)), dtype=np.float64)
        hist_fingers_samples_xedges = np.zeros((4, 3, int(samples_zoom_factor*2*dy+1)), dtype=np.float64)
        hist_fingers_samples_yedges = np.zeros((4, 3, int(samples_zoom_factor*2*dx+1)), dtype=np.float64)
        img_fingers_samples_density = np.zeros((4, 3, int(samples_zoom_factor*2*dy), int(samples_zoom_factor*2*dx), 4), dtype=np.float64)
        for i_paw in range(4):
            H, samples_xedges, samples_yedges = \
                np.histogram2d(paws_fit[:, i_paw, 1], paws_fit[:, i_paw, 0],
                               bins=bins)
            hist_paws_samples[i_paw] = np.copy(H)
            hist_paws_samples_xedges[i_paw] = np.copy(samples_xedges)
            hist_paws_samples_yedges[i_paw] = np.copy(samples_yedges)
            hist_paws_samples_use = np.copy(H)
            hist_paws_samples_use = hist_paws_samples_use - np.min(hist_paws_samples_use)
            hist_paws_samples_use = hist_paws_samples_use / np.max(hist_paws_samples_use)
            img_paws_samples_density[i_paw, :, :, :] = np.array(colors[i_paw], dtype=np.float64)[None, None, :]
            img_paws_samples_density[i_paw,:, :, 3] = np.copy(hist_paws_samples_use * 1.0)
            for i_finger in range(3):
                H, samples_xedges, samples_yedges = \
                    np.histogram2d(fingers_fit[:, i_paw, i_finger, 1], fingers_fit[:, i_paw, i_finger, 0],
                                   bins=bins)
                hist_fingers_samples[i_paw, i_finger] = np.copy(H)
                hist_fingers_samples_xedges[i_paw, i_finger] = np.copy(samples_xedges)
                hist_fingers_samples_yedges[i_paw, i_finger] = np.copy(samples_yedges)
                hist_fingers_samples_use = np.copy(H)
                hist_fingers_samples_use = hist_fingers_samples_use - np.min(hist_fingers_samples_use)
                hist_fingers_samples_use = hist_fingers_samples_use / np.max(hist_fingers_samples_use)
                img_fingers_samples_density[i_paw, i_finger,  :, :, :] = np.array(colors[i_paw], dtype=np.float64)[None, None, :]
                img_fingers_samples_density[i_paw, i_finger,:, :, 3] = np.copy(hist_fingers_samples_use * 1.0)
        
        # PLOT
        for i_paw in range(4):
            # manually labeled
            ax0.plot(paws_cam[i_paw, 0], paws_cam[i_paw, 1],
                     color=colors[i_paw], alpha=1.0, linestyle='', marker='X', zorder=3,
                     markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth, markeredgecolor='black')
            ax0.plot(fingers_cam[i_paw, :, 0], fingers_cam[i_paw, :, 1],
                     color=colors[i_paw], alpha=1.0, linestyle='', marker='X', zorder=3,
                     markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth, markeredgecolor='black')
            # fitted
            ax0.plot(paws_fit[0, i_paw, 0], paws_fit[0, i_paw, 1],
                     color=colors[i_paw], alpha=1.0, linestyle='', marker='.', zorder=2,
                     markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth, markeredgecolor='black')
            ax0.imshow(img_paws_samples_density[i_paw],
                      interpolation=None, extent=extent, alpha=0.99, vmin=0.0, vmax=1.0, aspect=1)
            ax0.plot(fingers_fit[0, i_paw, :, 0], fingers_fit[0, i_paw, :, 1],
                     color=colors[i_paw], alpha=1.0, linestyle='', marker='.', zorder=2,
                     markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth, markeredgecolor='black')
            for i_finger in range(3):
                ax0.imshow(img_fingers_samples_density[i_paw, i_finger],
                          interpolation=None, extent=extent, alpha=0.99, vmin=0.0, vmax=1.0, aspect=1)
        
        # inset
        extent_inset = (0, 2*dx_inset, 2*dy_inset, 0)
        center_paw = (paws_cam[i_paw_inset, :] + np.sum(fingers_cam[i_paw_inset, :, :], 0)) / 4.0
        center_paw = center_paw.astype(np.int64)
        img_inset = np.copy(img_dummy[center_paw[1]-dy_inset:center_paw[1]+dy_inset,
                                      center_paw[0]-dx_inset:center_paw[0]+dx_inset])
        vec_inset_x = np.array([center_paw[0]-dx_inset,
                                center_paw[0]+dx_inset,
                                center_paw[0]+dx_inset,
                                center_paw[0]-dx_inset,
                                center_paw[0]-dx_inset], dtype=np.float64)
        vec_inset_y = np.array([center_paw[1]-dy_inset,
                                center_paw[1]-dy_inset,
                                center_paw[1]+dy_inset,
                                center_paw[1]+dy_inset,
                                center_paw[1]-dy_inset], dtype=np.float64)
        ax0.plot(vec_inset_x, vec_inset_y,
                 color='white', alpha=1.0, linestyle='-', linewidth=0.5, marker='', zorder=4)
        
        custom_lines = [Line2D([0], [0], label='reconstruction',
                               color='white', linestyle='', marker='.', markersize=markersize*3.0, markeredgewidth=markeredgewidth*1.0, linewidth=linewidth, markeredgecolor='black'),
                        Line2D([0], [0], label='ground truth',
                               color='white', linestyle='', marker='X', markersize=markersize*3.0, markeredgewidth=markeredgewidth*1.0, linewidth=linewidth, markeredgecolor='black')]
        h_legend = ax0.legend(handles=custom_lines,
                              loc='upper right', frameon=False, fontsize=fontsize)
        for text in h_legend.get_texts():
            text.set_color('white')
            text.set_fontname(fontname)
        
        ax0.set_aspect(1)
        fig0.canvas.draw()
        fig0.tight_layout()
        fig0.canvas.draw()
        plt.pause(2**-10)
        fig0.canvas.draw()
        plt.pause(2**-10)
        if save:
            fig0.savefig(folder_save+'/underneath.svg',
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True,
                        format='svg',
                        pad_inches=0)
            fig0.savefig(folder_save+'/underneath.tiff',
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True,
                        format='tiff',
                        pad_inches=0)

        h_img = ax.imshow(img_inset,
                          'gray',
                          interpolation=None,
                          vmin=0.0, vmax=16.0, aspect=1,
                          extent=extent_inset)
        #
        shift_vector = np.array([center_paw[0]-dx_inset, center_paw[1]-dy_inset], dtype=np.float64)
        fingers_cam = fingers_cam - shift_vector
        paws_cam = paws_cam - shift_vector
        fingers_fit = fingers_fit - shift_vector
        paws_fit = paws_fit - shift_vector
        #
        samples_zoom_factor = 4.0
        dBin = 1.0/samples_zoom_factor
        bins = list([np.linspace(0.0, 2.0*dy_inset, int(samples_zoom_factor*2*dy_inset+1), dtype=np.float64),
                     np.linspace(0.0, 2.0*dx_inset, int(samples_zoom_factor*2*dx_inset+1), dtype=np.float64)])
        hist_paws_samples = np.zeros((4, int(samples_zoom_factor*2*dy_inset), int(samples_zoom_factor*2*dx_inset)), dtype=np.float64)
        hist_paws_samples_xedges = np.zeros((4, int(samples_zoom_factor*2*dy_inset+1)), dtype=np.float64)
        hist_paws_samples_yedges = np.zeros((4, int(samples_zoom_factor*2*dx_inset+1)), dtype=np.float64)
        img_paws_samples_density = np.zeros((4, int(samples_zoom_factor*2*dy_inset), int(samples_zoom_factor*2*dx_inset), 4), dtype=np.float64)
        hist_fingers_samples = np.zeros((4, 3, int(samples_zoom_factor*2*dy_inset), int(samples_zoom_factor*2*dx_inset)), dtype=np.float64)
        hist_fingers_samples_xedges = np.zeros((4, 3, int(samples_zoom_factor*2*dy_inset+1)), dtype=np.float64)
        hist_fingers_samples_yedges = np.zeros((4, 3, int(samples_zoom_factor*2*dx_inset+1)), dtype=np.float64)
        img_fingers_samples_density = np.zeros((4, 3, int(samples_zoom_factor*2*dy_inset), int(samples_zoom_factor*2*dx_inset), 4), dtype=np.float64)
        for i_paw in range(4):
            H, samples_xedges, samples_yedges = \
                np.histogram2d(paws_fit[:, i_paw, 1], paws_fit[:, i_paw, 0],
                               bins=bins)
            hist_paws_samples[i_paw] = np.copy(H)
            hist_paws_samples_xedges[i_paw] = np.copy(samples_xedges)
            hist_paws_samples_yedges[i_paw] = np.copy(samples_yedges)
            hist_paws_samples_use = np.copy(H)
            hist_paws_samples_use = hist_paws_samples_use - np.min(hist_paws_samples_use)
            hist_paws_samples_use = hist_paws_samples_use / np.max(hist_paws_samples_use)
            img_paws_samples_density[i_paw, :, :, :] = np.array(colors[i_paw], dtype=np.float64)[None, None, :]
            img_paws_samples_density[i_paw,:, :, 3] = np.copy(hist_paws_samples_use * 1.0)
            for i_finger in range(3):
                H, samples_xedges, samples_yedges = \
                    np.histogram2d(fingers_fit[:, i_paw, i_finger, 1], fingers_fit[:, i_paw, i_finger, 0],
                                   bins=bins)
                hist_fingers_samples[i_paw, i_finger] = np.copy(H)
                hist_fingers_samples_xedges[i_paw, i_finger] = np.copy(samples_xedges)
                hist_fingers_samples_yedges[i_paw, i_finger] = np.copy(samples_yedges)
                hist_fingers_samples_use = np.copy(H)
                hist_fingers_samples_use = hist_fingers_samples_use - np.min(hist_fingers_samples_use)
                hist_fingers_samples_use = hist_fingers_samples_use / np.max(hist_fingers_samples_use)
                img_fingers_samples_density[i_paw, i_finger,  :, :, :] = np.array(colors[i_paw], dtype=np.float64)[None, None, :]
                img_fingers_samples_density[i_paw, i_finger,:, :, 3] = np.copy(hist_fingers_samples_use * 1.0)
    
        # manually labeled
        ax.plot(paws_cam[i_paw_inset, 0], paws_cam[i_paw_inset, 1],
                 color=colors[i_paw_inset], alpha=1.0, linestyle='', marker='X', zorder=3,
                 markersize=markersize_inset, markeredgewidth=markeredgewidth, linewidth=linewidth, markeredgecolor='black')
        ax.plot(fingers_cam[i_paw_inset, :, 0], fingers_cam[i_paw_inset, :, 1],
                 color=colors[i_paw_inset], alpha=1.0, linestyle='', marker='X', zorder=3,
                 markersize=markersize_inset, markeredgewidth=markeredgewidth, linewidth=linewidth, markeredgecolor='black')
        # fitted
        ax.plot(paws_fit[0, i_paw_inset, 0], paws_fit[0, i_paw_inset, 1],
                 color=colors[i_paw_inset], alpha=1.0, linestyle='', marker='.', zorder=2,
                 markersize=markersize_inset, markeredgewidth=markeredgewidth, linewidth=linewidth, markeredgecolor='black')
        ax.imshow(img_paws_samples_density[i_paw_inset],
                  interpolation=None, extent=extent_inset, alpha=0.99, vmin=0.0, vmax=1.0, aspect=1)
        ax.plot(fingers_fit[0, i_paw_inset, :, 0], fingers_fit[0, i_paw_inset, :, 1],
                 color=colors[i_paw_inset], alpha=1.0, linestyle='', marker='.', zorder=2,
                 markersize=markersize_inset, markeredgewidth=markeredgewidth, linewidth=linewidth, markeredgecolor='black')
        for i_finger in range(3):
            ax.imshow(img_fingers_samples_density[i_paw_inset, i_finger],
                      interpolation=None, extent=extent_inset, alpha=0.99, vmin=0.0, vmax=1.0, aspect=1)
            
        # distance
        vec_dist_x = np.array([paws_cam[i_paw_inset, 0], paws_fit[0, i_paw_inset, 0]], dtype=np.float64)
        vec_dist_y = np.array([paws_cam[i_paw_inset, 1], paws_fit[0, i_paw_inset, 1]], dtype=np.float64)
        vec_dist = np.array([vec_dist_x[1] - vec_dist_x[0],
                             vec_dist_y[1] - vec_dist_y[0]], dtype=np.float64)
        vec_dist_norm = vec_dist / np.sqrt(np.sum(vec_dist**2))
        alpha = -np.pi/2.0
        R90 = np.array([[np.cos(alpha), -np.sin(alpha)],
                        [np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
        vec_text = np.dot(R90, vec_dist_norm)
        vec_text = -vec_text # to mirror
        #
        dist_linewidth = 1.0
        #
        fac = 2.5
        vec_annotate_x = np.array([fac*vec_text[0]+paws_cam[i_paw_inset, 0],
                                   (fac+1.0)*vec_text[0]+paws_cam[i_paw_inset, 0],
                                   (fac+1.0)*vec_text[0]+(paws_cam[i_paw_inset, 0]+paws_fit[0, i_paw_inset, 0])*0.5,
                                   (fac+2.0)*vec_text[0]+(paws_cam[i_paw_inset, 0]+paws_fit[0, i_paw_inset, 0])*0.5,
                                   (fac+1.0)*vec_text[0]+(paws_cam[i_paw_inset, 0]+paws_fit[0, i_paw_inset, 0])*0.5,
                                   (fac+1.0)*vec_text[0]+paws_fit[0, i_paw_inset, 0],
                                   fac*vec_text[0]+paws_fit[0, i_paw_inset, 0]], dtype=np.float64)
        vec_annotate_y = np.array([fac*vec_text[1]+paws_cam[i_paw_inset, 1],
                                   (fac+1.0)*vec_text[1]+paws_cam[i_paw_inset, 1],
                                   (fac+1.0)*vec_text[1]+(paws_cam[i_paw_inset, 1]+paws_fit[0, i_paw_inset, 1])*0.5,
                                   (fac+2.0)*vec_text[1]+(paws_cam[i_paw_inset, 1]+paws_fit[0, i_paw_inset, 1])*0.5,
                                   (fac+1.0)*vec_text[1]+(paws_cam[i_paw_inset, 1]+paws_fit[0, i_paw_inset, 1])*0.5,
                                   (fac+1.0)*vec_text[1]+paws_fit[0, i_paw_inset, 1],
                                   fac*vec_text[1]+paws_fit[0, i_paw_inset, 1]], dtype=np.float64)
        dist_annotate_h2 = ax.plot(vec_annotate_x, vec_annotate_y,
                                   linewidth=dist_linewidth, color='white', alpha=1.0, linestyle='-', marker='', zorder=4)

        alpha_text = np.arctan2(vec_dist_norm[1], vec_dist_norm[0]) * 180.0/np.pi 
        alpha_text = alpha_text * -1.0 + 180.0
        offset_text = 2.0
        dist_annotate_h3 = ax.text(vec_annotate_x[3] + offset_text * vec_text[0],
                                   vec_annotate_y[3] + offset_text * vec_text[1],
                                   r'$d$',
                                   color='white', va='center', ha='center',
                                   fontsize=fontsize, fontname=fontname)
        dist_annotate_h4 = ax.text(np.shape(img_inset)[0] * 0.5,
                                    1.0,
                                    'position error: '+r'$d$',
                                    color='white', va='top', ha='center',
                                    fontsize=fontsize, fontname=fontname)
        fig.canvas.draw()
        fig.tight_layout()
        fig.canvas.draw()
        plt.pause(2**-10)
        fig.canvas.draw()
        plt.pause(2**-10)
        if save_inset:
            fig.savefig(folder_save+'/underneath_inset_position.svg',
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True,
                        format='svg',
                        pad_inches=0)
            fig.savefig(folder_save+'/underneath_inset_position.tiff',
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True,
                        format='tiff',
                        pad_inches=0)
        
        # angle
        dist_annotate_h2[0].set_visible(False)
        dist_annotate_h3.set_visible(False)
        dist_annotate_h4.set_visible(False)
        #
        i_finger = 0
        vec_dist_cam = fingers_cam[i_paw_inset, i_finger] - paws_cam[i_paw_inset]
        vec_dist_cam_length = np.sqrt(np.sum(vec_dist_cam**2))
        vec_dist_cam_norm = vec_dist_cam / vec_dist_cam_length
        alpha_cam = np.arctan2(vec_dist_cam_norm[1], vec_dist_cam_norm[0])
        vec_dist_cam0 = np.array([vec_dist_cam_length, 0.0], dtype=np.float64) 
        vec_dist_fit = fingers_fit[0, i_paw_inset, i_finger] - paws_fit[0, i_paw_inset] 
        vec_dist_fit_length = np.sqrt(np.sum(vec_dist_fit**2))
        vec_dist_fit_norm = vec_dist_fit / vec_dist_fit_length
        alpha_fit = np.arctan2(vec_dist_fit_norm[1], vec_dist_fit_norm[0])
        vec_dist_fit0 = np.array([vec_dist_fit_length, 0.0], dtype=np.float64)
        #
        angle_length = 3.0/3.0
        #
        angle_cam = paws_cam[i_paw_inset] + vec_dist_cam0 * angle_length
        angle_cam = angle_cam.reshape(1, 2)
        alpha_range = np.sign(alpha_cam) * np.linspace(0.0, abs(alpha_cam), 100, dtype=np.float64)[1:]
        for alpha in alpha_range:
            R = np.array([[np.cos(alpha), -np.sin(alpha)],
                          [np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
            vec_new = np.dot(R, vec_dist_cam0)
            pos_new = paws_cam[i_paw_inset] + vec_new * angle_length
            pos_new = pos_new.reshape(1, 2)
            angle_cam = np.concatenate([angle_cam, pos_new], 0)
        angle_fit = paws_fit[0, i_paw_inset] + vec_dist_fit0 * angle_length
        angle_fit = angle_fit.reshape(1, 2)
        alpha_range = np.sign(alpha_fit) * np.linspace(0.0, abs(alpha_fit), 100, dtype=np.float64)[1:]
        for alpha in alpha_range:
            R = np.array([[np.cos(alpha), -np.sin(alpha)],
                          [np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
            vec_new = np.dot(R, vec_dist_fit0)
            pos_new = paws_fit[0, i_paw_inset] + vec_new * angle_length
            pos_new = pos_new.reshape(1, 2)
            angle_fit = np.concatenate([angle_fit, pos_new], 0)
        angle_center_cam = (paws_cam[i_paw_inset] + angle_cam[0] + angle_cam[-1]) / 3.0
        angle_center_fit = (paws_fit[0, i_paw_inset] + angle_fit[0] + angle_fit[-1]) / 3.0
        #
        angle_color = 'white'
        angle_linewidth = 1.0
        angle_alpha = 1.0
        # fit
        ax.plot(angle_fit[:, 0], angle_fit[:, 1],
                linewidth=angle_linewidth, color=angle_color, alpha=angle_alpha, linestyle='--', marker='', zorder=1)
        ax.plot(np.array([paws_fit[0, i_paw_inset, 0], paws_fit[0, i_paw_inset, 0]+vec_dist_fit_length], dtype=np.float64),
                np.array([paws_fit[0, i_paw_inset, 1], paws_fit[0, i_paw_inset, 1]], dtype=np.float64),
                linewidth=angle_linewidth, color=angle_color, alpha=angle_alpha, linestyle='-', marker='', zorder=1)
        ax.plot(np.array([paws_fit[0, i_paw_inset, 0], fingers_fit[0, i_paw_inset, i_finger, 0]], dtype=np.float64),
                np.array([paws_fit[0, i_paw_inset, 1], fingers_fit[0, i_paw_inset, i_finger, 1]], dtype=np.float64),
                linewidth=angle_linewidth, color=angle_color, alpha=angle_alpha, linestyle='-', marker='', zorder=1)
        # cam
        ax.plot(angle_cam[:, 0], angle_cam[:, 1],
                linewidth=angle_linewidth, color=angle_color, alpha=angle_alpha, linestyle='--', marker='', zorder=1),
        ax.plot(np.array([paws_cam[i_paw_inset, 0], paws_cam[i_paw_inset, 0]+vec_dist_cam_length], dtype=np.float64),
                np.array([paws_cam[i_paw_inset, 1], paws_cam[i_paw_inset, 1]], dtype=np.float64),
                linewidth=angle_linewidth, color=angle_color, alpha=angle_alpha, linestyle='-', marker='', zorder=1),
        ax.plot(np.array([paws_cam[i_paw_inset, 0], fingers_cam[i_paw_inset, i_finger, 0]], dtype=np.float64),
                np.array([paws_cam[i_paw_inset, 1], fingers_cam[i_paw_inset, i_finger, 1]], dtype=np.float64),
                linewidth=angle_linewidth, color=angle_color, alpha=angle_alpha, linestyle='-', marker='', zorder=1)
        #
        ax.text(angle_center_cam[0],
                angle_center_cam[1],
                r'$\alpha_{0}$',
                color='white', va='center', ha='center',
                fontsize=fontsize, fontname=fontname)
        ax.text(angle_center_fit[0],
                angle_center_fit[1],
                r'$\alpha_{1}$',
                color='white', va='center', ha='center',
                fontsize=fontsize, fontname=fontname)
        ax.text(np.shape(img_inset)[0] * 0.5,
                1.0,
                'angle error: '+r'$|\alpha_{1} - \alpha_{0}|$',
                color='white', va='top', ha='center',
                fontsize=fontsize, fontname=fontname)
        #
        ax.set_aspect(1)
        fig.canvas.draw()
        fig.tight_layout()
        fig.canvas.draw()
        plt.pause(2**-10)
        fig.canvas.draw()
        plt.pause(2**-10)
        if save_inset:
            fig.savefig(folder_save+'/underneath_inset_angle.svg',
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True,
                        format='svg',
                        pad_inches=0)
            fig.savefig(folder_save+'/underneath_inset_angle.tiff',
                        dpi=300,
                        bbox_inches='tight',
                        transparent=True,
                        format='tiff',
                        pad_inches=0)
        if verbose:
            plt.show()
