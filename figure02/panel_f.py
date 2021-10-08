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
import model
sys_path0 = np.copy(sys.path)

sys.path.append(os.path.abspath('../ccv'))
import ccv

save = False
verbose = True

folder_save = os.path.abspath('panels')

folder_reconstruction = data.path+'/reconstruction'
add2folder = '__pcutoff9e-01'
folder_list = list(['/20200205/table_20200205_020300_021800', # 0
                    '/20200205/table_20200205_023300_025800', # 1
                    '/20200205/table_20200205_048800_050200', # 2
                    '/20200207/table_20200207_012500_013700', # 3
                    '/20200207/table_20200207_028550_029050', # 4
                    '/20200207/table_20200207_044900_046000', # 5
                    '/20200207/table_20200207_056900_057350', # 6
                    '/20200207/table_20200207_082800_084000', # 7
                   ])
i_cam_down_list = list([0, 0, 0, 1, 0, 0, 0, 1])

file_ccv = data.path_ccv_fig2_f
i_paw_plot = 3
i_finger_angle = 0
threshold = 0.0
dFrame = 1
i_folder = 4
i_cam_down = i_cam_down_list[i_folder]
folder = folder_reconstruction + folder_list[i_folder] + add2folder

sys.path = list(np.copy(sys_path0))
sys.path.append(folder)
importlib.reload(cfg)
cfg.animal_is_large = 0
importlib.reload(anatomy)

nCameras = 6
clim_min = list([0 for i in range(nCameras)])
clim_max = list([127.0, 127.0, 127.0, 127.0, 16.0, 16.0])

nCameras_up = 4
nSamples_use = int(0)

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

fontsize = 6
linewidth = 1.0
fontname = "Arial"
markersize = 2

cmap = plt.cm.tab10
color_mode1 = cmap(5/9)
color_mode2 = cmap(1/9)
color_mode3 = cmap(0/9)
color_mode4 = cmap(2/9)
colors = list([color_mode1, color_mode2, color_mode3, color_mode4])

# look at: Rational Radial Distortion Models with Analytical Undistortion Formulae, Lili Ma et al.
# source: https://arxiv.org/pdf/cs/0307047.pdf
# only works for k = [k1, k2, 0, 0, 0]
def calc_udst(m_dst, k):
    assert np.all(k[2:] == 0.0), 'ERROR: Undistortion only valid for up to two radial distortion coefficients.'
    
    x_2 = m_dst[:, 0]
    y_2 = m_dst[:, 1]
    
    # use r directly instead of c
    nPoints = np.size(m_dst, 0)
    p = np.zeros(6, dtype=np.float64)
    p[4] = 1.0
    sol = np.zeros(3, dtype=np.float64)
    x_1 = np.zeros(nPoints, dtype=np.float64)
    y_1 = np.zeros(nPoints, dtype=np.float64)
    for i_point in range(nPoints):
        cond = (np.abs(x_2[i_point]) > np.abs(y_2[i_point]))
        if (cond):
            c = y_2[i_point] / x_2[i_point]
            p[5] = -x_2[i_point]
        else:
            c = x_2[i_point] / y_2[i_point]
            p[5] = -y_2[i_point]
#        p[4] = 1
        p[2] = k[0] * (1.0 + c**2)
        p[0] = k[1] * (1.0 + c**2)**2
        sol = np.real(np.roots(p))
        # use min(abs(x)) to make your model as accurate as possible
        sol_abs = np.abs(sol)
        if (cond):
            x_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            y_1[i_point] = c * x_1[i_point]
        else:
            y_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            x_1[i_point] = c * y_1[i_point]
    m_udst = np.concatenate([[x_1], [y_1], [m_dst[:, 2]]], 0).T
    return m_udst

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
    # trajectory xy
    fig_w = np.round(mm_in_inch * (88.0), decimals=2)
    fig_h = fig_w
    fig = plt.figure(1, frameon=False, figsize=(fig_w, fig_h))
    fig.canvas.manager.window.move(0, 0)
    fig.clear()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.clear()
    ax.set_axis_off()
    fig.add_axes(ax)
    # X vs time
    fig2_w = np.round(mm_in_inch * 88.0*1.0, decimals=2)
    fig2_h = np.round(mm_in_inch * 88.0*1/2, decimals=2)
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
    # hist
    fig3_w = np.round(mm_in_inch * 88.0*1.0, decimals=2)
    fig3_h = np.round(mm_in_inch * 88.0*1/2, decimals=2)
    fig3 = plt.figure(3, figsize=(fig3_w, fig3_h))
    fig3.canvas.manager.window.move(0, 0)
    fig3.clear()
    ax3_x = left_margin/fig3_w
    ax3_y = bottom_margin/fig3_h
    ax3_w = 1.0 - (left_margin/fig3_w + right_margin/fig3_w)
    ax3_h = 1.0 - (bottom_margin/fig3_h + top_margin/fig3_h)
    ax3 = fig3.add_axes([ax3_x, ax3_y, ax3_w, ax3_h])
    ax3.clear()
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
        
    img_dummy = np.zeros((1024, 1280), dtype=np.float64)
    h_img = ax.imshow(img_dummy,
                      'gray',
                      vmin=0, vmax=127)
    i_nT_list = np.arange(0, cfg.nT, dFrame, dtype=np.int64)
    nT_use = len(i_nT_list)
    for i_nT in range(nT_use):
        print('{:06d} / {:06d} '.format(i_nT+1, nT_use))
        if not(file_ccv == ''):
            img = ccv.get_frame(file_ccv, i_nT_list[i_nT]+cfg.index_frame_start+1).astype(np.float64)
        else:
            img = np.full((1024, 1280), 8.5, dtype=np.float64)
        
        mask_max = (img_dummy < img)
        mask_threshold = (img >= threshold)
        mask = np.logical_and(mask_max, mask_threshold)
        img_dummy[mask] = np.copy(img[mask])
    h_img.set_array(img_dummy)
    h_img.set_clim(0.0, 24.0)

    for i_mode in range(4):
        if (i_mode == 0):
            folder = folder_reconstruction + folder_list[i_folder] + add2folder
        elif (i_mode == 1):
            folder = folder_reconstruction + folder_list[i_folder] + '__mode3' + add2folder
        elif (i_mode == 2):
            folder = folder_reconstruction + folder_list[i_folder] + '__mode2' + add2folder
        elif (i_mode == 3):
            folder = folder_reconstruction + folder_list[i_folder] + '__mode1' + add2folder

        sys.path = list(np.copy(sys_path0))
        sys.path.append(folder)
        importlib.reload(cfg)
        cfg.animal_is_large = 0
        importlib.reload(anatomy)

        folder_reqFiles = data.path + '/required_files' 
        file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
        file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
        file_model = folder_reqFiles + '/model.npy'
        file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_dlc_{:06d}_{:06d}.npy'.format(cfg.index_frame_start, cfg.index_frame_end)
        #
        file_labels_down = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_down_use.npz'
        file_calibration_all = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/calibration_all/multicalibration.npy'
            
        labels_down = np.load(file_labels_down, allow_pickle=True)['arr_0'].item()
        origin_coord = np.load(file_origin_coord, allow_pickle=True).item()
            
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
            var_dummy = np.identity(nPara, dtype=np.float64) * 2**-23
            var_uks = np.tile(var_dummy.ravel(), nT).reshape(nT, nPara, nPara)
            nSamples = int(0)
        frame_list_fit = np.arange(cfg.index_frame_ini,
                                   cfg.index_frame_ini + cfg.nT * cfg.dt,
                                   cfg.dt, dtype=np.int64)
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
        
        # fit
        mu_t = torch.from_numpy(mu_uks)
        markers2d, markers3d, joints3d = model.fcn_emission_free(mu_t, args_torch)
        markers2d = markers2d.cpu().numpy()
        markers3d = markers3d.cpu().numpy()
        joints3d = joints3d.cpu().numpy()
        
        fingers_fit_3d = markers3d[:, index].reshape(cfg.nT, 4, 3, 3)
        paws_fit_3d = markers3d[:, index2]
        #
        fingers_fit = np.full((cfg.nT, 4, 3, 2), np.nan, dtype=np.float64)
        paws_fit = np.full((cfg.nT, 4, 2), np.nan, dtype=np.float64)
        for i_paw in range(4):
            fingers_fit[:, i_paw] = map_m(fingers_fit_3d[:, i_paw, :].reshape(cfg.nT*3, 3),
                                          RX1, tX1, A, k).reshape(cfg.nT, 3, 2) # nSamples, nFingers, 2
            paws_fit[:, i_paw] = map_m(paws_fit_3d[:, i_paw], RX1, tX1, A, k)
        
        # manual labels
        if (i_mode == 0):
            fingers_table_3d = np.full((cfg.nT, nLabels_check, 3), np.nan, dtype=np.float64)
            paws_table_3d = np.full((cfg.nT, 4, 3), np.nan, dtype=np.float64)
            fingers_table = np.full((cfg.nT, nLabels_check, 2), np.nan, dtype=np.float64)
            paws_table = np.full((cfg.nT, 4, 2), np.nan, dtype=np.float64)
            for i_frame in range(nT_check):
                # table
                frame = frame_list_manual[i_frame]
                i_t = frame_list_manual[i_frame] - frame_list_manual[0]
                #
                points_dst = np.full((nLabels_check, 3), np.nan, dtype=np.float64)
                points_udst = np.full((nLabels_check, 3), np.nan, dtype=np.float64)
                for i_label in range(nLabels_check):
                    label_name = label_names[i_label]
                    if label_name in labels_down[frame]:
                        fingers_table[i_t, i_label] = labels_down[frame][label_name][i_cam_down]
                        points_dst[i_label] = np.array([(labels_down[frame][label_name][i_cam_down][0] - A[1]) / A[0],
                                                        (labels_down[frame][label_name][i_cam_down][1] - A[3]) / A[2],
                                                        1.0], dtype=np.float64)
                mask = ~np.any(np.isnan(points_dst), 1)
                points_udst[mask] = calc_udst(points_dst[mask], k)
                line0 = np.einsum('ij,xj->xi', RX1.T, -tX1[None, :])
                line1 = np.einsum('ij,xj->xi', RX1.T, points_udst - tX1[None, :])
                n = line0
                m = line1 - line0
                lambda_val = -n[:, 2] / m[:, 2]
                fingers_table_single = m * lambda_val[:, None] + n
                fingers_table_3d[i_t] = np.copy(fingers_table_single)
                
                points_dst = np.full((4, 3), np.nan, dtype=np.float64)
                points_udst = np.full((4, 3), np.nan, dtype=np.float64)
                for i_label in range(nLabels_check2):
                    label_name = label_names2[i_label]
                    if label_name in labels_down[frame]:
                        paws_table[i_t, i_label] = labels_down[frame][label_name][i_cam_down]
                        points_dst[i_label] = np.array([(labels_down[frame][label_name][i_cam_down][0] - A[1]) / A[0],
                                                        (labels_down[frame][label_name][i_cam_down][1] - A[3]) / A[2],
                                                        1.0], dtype=np.float64)
                mask = ~np.any(np.isnan(points_dst), 1)
                points_udst[mask] = calc_udst(points_dst[mask], k)
                line0 = np.einsum('ij,xj->xi', RX1.T, -tX1[None, :])
                line1 = np.einsum('ij,xj->xi', RX1.T, points_udst - tX1[None, :])
                n = line0
                m = line1 - line0
                lambda_val = -n[:, 2] / m[:, 2]
                paws_table_single = m * lambda_val[:, None] + n
                paws_table_3d[i_t] = np.copy(paws_table_single)
        
        time = np.arange(cfg.nT, dtype=np.float64) / cfg.frame_rate
        angle = np.full(cfg.nT, np.nan, dtype=np.float64)
        
        vec = fingers_fit_3d[:, i_paw_plot, i_finger_angle, :2] - paws_fit_3d[:, i_paw_plot, :2]
        vec = vec / np.sqrt(np.sum(vec**2, 1))[:, None]
        last = 0
        angle = list([])
        for i_t in range(cfg.nT):
            x = vec[i_t, 0]
            y = vec[i_t, 1]
            angle_single = np.arctan2(y, x)
            while angle_single < last-np.pi: angle_single += 2*np.pi
            while angle_single > last+np.pi: angle_single -= 2*np.pi
            last = np.copy(angle_single)
            angle.append(angle_single)
        angle = np.array(angle, dtype=np.float64) * 180.0/np.pi

        angle_bound_up = 180.0 * 2.0
        angle_range = 360.0 * 1.0 * 2.0
        angle_bound_down = angle_bound_up - angle_range
        index_up = list()
        index_down = list()
        if (np.any(angle > angle_bound_up) or np.any(angle < angle_bound_down)):
            for i_t in range(cfg.nT):
                if (angle[i_t] > angle_bound_up):
                    angle[i_t:] = angle[i_t:] - angle_range
                    index_up.append(i_t)
                if (angle[i_t] < angle_bound_down):
                    angle[i_t:] = angle[i_t:] + angle_range
                    index_down.append(i_t)
        index_up = np.array(index_up, dtype=np.int64)
        index_down = np.array(index_down, dtype=np.int64)

        if (np.any(index_up) or np.any(index_down)):
            index_up_down = np.sort(np.concatenate([index_up, index_down]), 0)
            #
            index_fist = index_up_down[0]
            ax2.plot(time[:index_fist], angle[:index_fist],
                     linewidth=1.0, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)
            #
            nSwaps = len(index_up_down)
            if (nSwaps > 1):
                for i in np.arange(1, nSwaps, 1, dtype=np.int64):
                    t_from = index_up_down[i-1]
                    t_to = index_up_down[i]
                    ax2.plot(time[t_from:t_to], angle[t_from:t_to],
                             linewidth=1.0, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)
            #
            index_last = index_up_down[-1]
            ax2.plot(time[index_last:], angle[index_last:],
                     linewidth=1.0, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)
        else:
            ax2.plot(time, angle,
                     linewidth=1.0, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)
            
        for i_t in index_up:
            ax2.plot(np.array([time[i_t-1], time[i_t-1]], dtype=np.float64), 
                     np.array([angle[i_t-1], angle_bound_up], dtype=np.float64),
                     linewidth=1.0, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)  
            ax2.plot(np.array([time[i_t], time[i_t]], dtype=np.float64), 
                     np.array([angle_bound_down, angle[i_t]], dtype=np.float64),
                     linewidth=1.0, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)
        for i_t in index_down:
            ax2.plot(np.array([time[i_t-1], time[i_t-1]], dtype=np.float64), 
                     np.array([angle[i_t-1], angle_bound_down], dtype=np.float64),
                     linewidth=1.0, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)  
            ax2.plot(np.array([time[i_t], time[i_t]], dtype=np.float64), 
                     np.array([angle_bound_up, angle[i_t]], dtype=np.float64),
                     linewidth=1.0, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)
        
        velo = \
                    (+1.0/280.0 * paws_fit_3d[:-8, i_paw_plot, :2] + \
                     -4.0/105.0 * paws_fit_3d[1:-7, i_paw_plot, :2] + \
                     +1.0/5.0 * paws_fit_3d[2:-6, i_paw_plot, :2] + \
                     -4.0/5.0 * paws_fit_3d[3:-5, i_paw_plot, :2] + \
#                      0.0 * paws_fit_3d[4:-4, i_paw_plot, :2] + \
                     +4.0/5.0 * paws_fit_3d[5:-3, i_paw_plot, :2] + \
                     -1.0/5.0 * paws_fit_3d[6:-2, i_paw_plot, :2] + \
                     +4.0/105.0 * paws_fit_3d[7:-1, i_paw_plot, :2] + \
                     -1.0/280.0 * paws_fit_3d[8:, i_paw_plot, :2]) / (1.0/cfg.frame_rate)
        acc = \
                    (-1.0/560.0 * paws_fit_3d[:-8, i_paw_plot, :2] + \
                     +8.0/315.0 * paws_fit_3d[1:-7, i_paw_plot, :2] + \
                     -1.0/5.0 * paws_fit_3d[2:-6, i_paw_plot, :2] + \
                     +8.0/5.0 * paws_fit_3d[3:-5, i_paw_plot, :2] + \
                     -205.0/72.0 * paws_fit_3d[4:-4, i_paw_plot, :2] + \
                     +8.0/5.0 * paws_fit_3d[5:-3, i_paw_plot, :2] + \
                     -1.0/5.0 * paws_fit_3d[6:-2, i_paw_plot, :2] + \
                     +8.0/315.0  * paws_fit_3d[7:-1, i_paw_plot, :2] + \
                     -1.0/560.0 * paws_fit_3d[8:, i_paw_plot, :2]) / (1.0/cfg.frame_rate)**2

        dBin_velo = 1e1 # cm/s**2
        maxBin_velo = 1e3 # cm/s**2
        bin_range_velo = np.arange(-maxBin_velo, maxBin_velo+dBin_velo, dBin_velo, dtype=np.float64)
        nBin_velo = len(bin_range_velo) - 1
        velo_hist = velo[~np.isnan(velo)]
        n_velo_hist = len(velo_hist)
        hist_velo = np.histogram(velo_hist, bins=nBin_velo, range=[bin_range_velo[0], bin_range_velo[-1]], normed=None, weights=None, density=False)
        y_velo = np.zeros(1+2*len(hist_velo[0]), dtype=np.float64)
        y_velo[1::2] = np.copy(hist_velo[0] / n_velo_hist)
        y_velo[2::2] = np.copy(hist_velo[0] / n_velo_hist)
        x_velo = np.zeros(1+2*(len(hist_velo[1])-1), dtype=np.float64)
        x_velo[0] = np.copy(hist_velo[1][0])
        x_velo[1::2] = np.copy(hist_velo[1][:-1])
        x_velo[2::2] = np.copy(hist_velo[1][1:])        
        dBin_acc = 5e4 # cm/s**2
        maxBin_acc = 1e6 # cm/s**2
        bin_range_acc = np.arange(-maxBin_acc, maxBin_acc+dBin_acc, dBin_acc, dtype=np.float64)
        nBin_acc = len(bin_range_acc) - 1
        acc_hist = acc[~np.isnan(acc)]
        n_acc_hist = len(acc_hist)
        hist_acc = np.histogram(acc_hist, bins=nBin_acc, range=[bin_range_acc[0], bin_range_acc[-1]], normed=None, weights=None, density=False)
        y_acc = np.zeros(1+2*len(hist_acc[0]), dtype=np.float64)
        y_acc[1::2] = np.copy(hist_acc[0] / n_acc_hist)
        y_acc[2::2] = np.copy(hist_acc[0] / n_acc_hist)
        x_acc = np.zeros(1+2*(len(hist_acc[1])-1), dtype=np.float64)
        x_acc[0] = np.copy(hist_acc[1][0])
        x_acc[1::2] = np.copy(hist_acc[1][:-1])
        x_acc[2::2] = np.copy(hist_acc[1][1:])

        ax.plot(paws_fit[:, i_paw_plot, 0], paws_fit[:, i_paw_plot, 1],
                linewidth=linewidth, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)
        x3 = x_velo
        y3 = y_velo
        ax3.plot(x3, y3,
                 linewidth=1.0, alpha=1.0, color=colors[::-1][i_mode], zorder=4-i_mode)
            
    ax.set_xlim([200, 1200])
    ax.set_ylim([500, 900])
    ax.set_aspect(1)
    ax.invert_xaxis()
    h_legend = ax.legend(list(['anatomical & temporal', 'temporal', 'anatomical', 'naive']),
                         loc='upper right', frameon=False, fontsize=fontsize)
    for text in h_legend.get_texts():
        text.set_color('white')
        text.set_fontname(fontname)
    
    x2 = np.copy(time)
    ax2.plot(np.array([x2[0], x2[-1]], dtype=np.float64),
             np.array([angle_bound_up, angle_bound_up], dtype=np.float64),
             linestyle='--', linewidth=1.0, alpha=1.0, color='black', zorder=0)
    ax2.plot(np.array([x2[0], x2[-1]], dtype=np.float64),
             np.array([angle_bound_down, angle_bound_down], dtype=np.float64),
             linestyle='--', linewidth=1.0, alpha=1.0, color='black', zorder=0)
    
    h_legend2 = ax2.legend(list(['anatomical & temporal', 'temporal', 'anatomical', 'naive']),
                           loc='upper right', frameon=False, fontsize=fontsize)
    for text in h_legend2.get_texts():
        text.set_fontname(fontname)
        
    h_legend3 = ax3.legend(list(['anatomical & temporal', 'temporal', 'anatomical', 'naive']),
                         loc='upper right', frameon=False, fontsize=fontsize)
    for text in h_legend2.get_texts():
        text.set_fontname(fontname)
      
    plt.pause(2**-10)
    fig.tight_layout()
    plt.pause(2**-10)
    if save:
        fig.savefig(folder_save+'/trajectory_xy.svg',
                     bbox_inches="tight",
                     dpi=300,
                     transparent=True,
                     format='svg',
                     pad_inches=0)
        fig.savefig(folder_save+'/trajectory_xy.tiff',
                     bbox_inches="tight",
                     dpi=300,
                     transparent=True,
                     format='tiff',
                     pad_inches=0)  
    if verbose:
        plt.show()
