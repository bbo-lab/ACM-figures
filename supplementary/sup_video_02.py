#!/usr/bin/env python3

import importlib
from matplotlib import animation
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

folder_save = os.path.abspath('videos')

folder_reconstruction = data.path+'/reconstruction'
folder_data_list = list([
                        'table_20200205_020300_021800__pcutoff9e-01', # 0
                        'table_20200205_023300_025800__pcutoff9e-01', # 1
                        'table_20200205_048800_050200__pcutoff9e-01', # 2
                        'table_20200207_012500_013700__pcutoff9e-01', # 3
                        'table_20200207_028550_029050__pcutoff9e-01', # 4
                        'table_20200207_044900_046000__pcutoff9e-01', # 5
                        'table_20200207_056900_057350__pcutoff9e-01', # 6
                        'table_20200207_082800_084000__pcutoff9e-01', # 7
                        ])
i_cam_down_list = list([0, 0, 0, 1, 0, 0, 0, 1])

i_data = 0
date = folder_data_list[i_data].split('_')[1]
folder_data = folder_reconstruction+'/'+date+'/'+folder_data_list[i_data]
i_cam_down = i_cam_down_list[i_data]
dFrame = 10
#
nCameras_up = 4
file_ccv = '/media/server/bulk/pose_B.EG.1.09/{:s}/cam{:02d}_{:s}_table.ccv'.format(date, 1 + nCameras_up + i_cam_down, date)

cmap = plt.cm.viridis
nSamples = int(1e3)

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
    sys.path = list(np.copy(sys_path0))
    sys.path.append(folder_data)
    importlib.reload(cfg)
    cfg.animal_is_large = 0
    importlib.reload(anatomy)
    
    folder_reqFiles = data.path + '/required_files' 
    file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
    file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
    file_model = folder_reqFiles + '/model.npy'
    file_labelsDLC = folder_reqFiles + '/labels_dlc_dummy.npy' # DLC labels are not used here
    
    file_labels_down = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_down_all.npz'
    file_calibration_all = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/calibration_all/multicalibration.npy'
    
    labels_down = np.load(file_labels_down, allow_pickle=True)['arr_0'].item()
    origin_coord = np.load(file_origin_coord, allow_pickle=True).item()
    mu_ini = np.load(folder_data + '/x_ini.npy', allow_pickle=True)
    save_dict = np.load(folder_data + '/save_dict.npy', allow_pickle=True).item()
    
    label_names = sorted(list(['spot_paw_front_left',
                               'spot_paw_front_right',
                               'spot_paw_hind_left',
                               'spot_paw_hind_right',]))
    nLabels_check = len(label_names)

    mu_uks = save_dict['mu_uks'][1:]
    var_uks = save_dict['var_uks'][1:]
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
    if (dFrame == 1):
        frame_list_manual = np.arange(frame_list_manual[0], frame_list_manual[-1]+1, dFrame, dtype=np.int64)
    
    nT_check = len(frame_list_manual)
    kalman_index = frame_list_fit[np.array(list([i in frame_list_manual for i in frame_list_fit]))] - cfg.index_frame_ini
    
    args_torch = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                      cfg.scale_factor, cfg.pcutoff)
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
    #
    index = list()
    for i_label in range(nLabels_check):
        label_name = label_names[i_label]
        marker_name = 'marker_' + '_'.join(label_name.split('_')[1:]) + '_start'
        index.append(joint_marker_order.index(marker_name))
    index = np.array(index)
    #
    points_fit = np.zeros((nT_check, nSamples+1, nLabels_check, 3), dtype=np.float64)
    points_table = np.full((nT_check, nLabels_check, 3), np.nan, dtype=np.float64)
    for i_frame in range(nT_check):
        # fit
        mu_t = torch.from_numpy(mu_uks[kalman_index[i_frame]])
        var_t = torch.from_numpy(var_uks[kalman_index[i_frame]])
        distribution = torch.distributions.MultivariateNormal(loc=mu_t,
                                                              scale_tril=kalman.cholesky_save(var_t))
        z_samples = distribution.sample((nSamples,))
        
        z_all = torch.cat([mu_t[None, :], z_samples], 0)
        _, marker_pos_torch, _ = model.fcn_emission_free(z_all, args_torch)
        points_fit[i_frame] = marker_pos_torch.numpy()[:, index]
        # table
        frame = frame_list_manual[i_frame]
        points_dst = np.full((nLabels_check, 3), np.nan, dtype=np.float64)
        points_udst = np.full((nLabels_check, 3), np.nan, dtype=np.float64)
        if frame in labels_down:
            for i_label in range(nLabels_check):
                label_name = label_names[i_label]
                if label_name in labels_down[frame]:
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
        points_table[i_frame] = m * lambda_val[:, None] + n
    
    i_frame = 10
    xy_range = 192
    #
    frame = frame_list_manual[i_frame]
    labels_manual_use = np.full((nLabels_check, 2), np.nan, dtype=np.float64)
    for i_label in range(nLabels_check):
        label_name = label_names[i_label]
        if label_name in labels_down[frame]:
            labels_manual_use[i_label] = labels_down[frame][label_name][i_cam_down]
    mask = np.any(np.isnan(labels_manual_use), 1)

    labels_fit_use = map_m(points_fit[i_frame, 0],
                           RX1, tX1, A, k)
    x_mean = np.mean(labels_fit_use[:, 0])
    y_mean = np.mean(labels_fit_use[:, 1])
    center_xy = np.array([x_mean, y_mean], dtype=np.int64)
    labels_fit_use[mask] = np.nan

    samples_fit_proj = map_m(points_fit[i_frame, 1:].reshape(nSamples*nLabels_check, 3),
                             RX1, tX1, A, k).reshape(nSamples, nLabels_check , 2)
    samples_fit_proj[:, mask] = np.nan
    
    def crop_img(img, center_xy, 
                 img_crop, pixel_crop):
        xRes = 1280
        yRes = 1024
        xlim_min = center_xy[0] - xy_range
        xlim_max = center_xy[0] + xy_range
        ylim_min = center_xy[1] - xy_range
        ylim_max = center_xy[1] + xy_range
        xlim_min_use = np.max([xlim_min, 0])
        xlim_max_use = np.min([xlim_max, xRes])
        ylim_min_use = np.max([ylim_min, 0])
        ylim_max_use = np.min([ylim_max, yRes])
        dx = np.int64(xlim_max_use - xlim_min_use)
        dy = np.int64(ylim_max_use - ylim_min_use)

        center_xy_use = center_xy - np.array([xlim_min_use, ylim_min_use])
        center_xy_use = xy_range - center_xy_use
        dx_add = center_xy_use[0]
        dy_add = center_xy_use[1]
        dx_add_int = np.int64(dx_add)
        dy_add_int = np.int64(dy_add)

        img_crop[dy_add_int:dy+dy_add_int, dx_add_int:dx+dx_add_int] = \
            img[ylim_min_use:ylim_max_use, xlim_min_use:xlim_max_use]
        pixel_crop[0] = -xlim_min_use + dx_add_int
        pixel_crop[1] = -ylim_min_use + dy_add_int
        return
    
    img_crop = np.zeros((2*xy_range, 2*xy_range), dtype=np.uint8)
    pixel_crop = np.zeros(2, dtype=np.int64)
    
    fig2 = plt.figure(10, figsize=(8, 8))
    fig2.clear()
    ax2 = fig2.add_subplot(111)
    ax2.clear()
    ax2.set_facecolor('black')
    img = ccv.get_frame(file_ccv, frame+1)
    crop_img(img, center_xy,
             img_crop, pixel_crop)
    h_img = ax2.imshow(img_crop, 'gray', vmin=0, vmax=32)
    h_labels_fit = ax2.plot(labels_fit_use[:, 0] + pixel_crop[0],
                            labels_fit_use[:, 1] + pixel_crop[1],
                            linestyle='', marker='.', 
                            markersize=8, color=cmap(1/3), alpha=1.0, zorder=2,
                            markeredgewidth=2)
    h_samples_fit = ax2.plot(samples_fit_proj[:, :, 0].reshape(nSamples*nLabels_check) + pixel_crop[0],
                             samples_fit_proj[:, :, 1].reshape(nSamples*nLabels_check) + pixel_crop[1],
                             linestyle='', marker='.', 
                             markersize=6, color='red', alpha=0.1, zorder=1)
    ax2.set_xlim([0, 2*xy_range])
    ax2.set_ylim([0, 2*xy_range])
    ax2.invert_yaxis()
    ax2.set_axis_off()
    fig2.tight_layout()
    fig2.canvas.draw()
    
    def update_fig(i_frame):
        frame = frame_list_manual[i_frame]
    
        labels_manual_use = np.full((nLabels_check, 2), np.nan, dtype=np.float64)
        if frame in labels_down:
            for i_label in range(nLabels_check):
                label_name = label_names[i_label]
                if label_name in labels_down[frame]:
                    labels_manual_use[i_label] = labels_down[frame][label_name][i_cam_down]
 
        labels_fit_use = map_m(points_fit[i_frame, 0],
                               RX1, tX1, A, k)
        x_mean = np.mean(labels_fit_use[:, 0])
        y_mean = np.mean(labels_fit_use[:, 1])
        center_xy = np.array([x_mean, y_mean], dtype=np.int64)
        
        samples_fit_proj = map_m(points_fit[i_frame, 1:].reshape(nSamples*nLabels_check, 3),
                                 RX1, tX1, A, k).reshape(nSamples, nLabels_check, 2)
        
        img = ccv.get_frame(file_ccv, frame+1)
        img_crop.fill(0)
        crop_img(img, center_xy,
                 img_crop, pixel_crop)

        h_labels_fit[0].set_data(labels_fit_use[:, 0] + pixel_crop[0],
                                 labels_fit_use[:, 1] + pixel_crop[1])
        h_samples_fit[0].set_data(samples_fit_proj[:, :, 0].reshape(nSamples*nLabels_check) + pixel_crop[0],
                                  samples_fit_proj[:, :, 1].reshape(nSamples*nLabels_check) + pixel_crop[1])
        h_img.set_array(img_crop)
        return 

    slow_factor = 0.2
    frame_rate = cfg.frame_rate
    fps = (float(slow_factor) * float(frame_rate) / float(cfg.dt)) / float(dFrame)
    nFrames = len(frame_list_manual)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=-1)
    #
    ani = animation.FuncAnimation(fig2, update_fig,
                                  frames=nFrames,
                                  interval=1000, blit=False)
    
    file_video = folder_save+'/'+folder_data_list[i_data]+'.mp4'
    if save:
        ani.save(file_video, writer=writer)
    
    if verbose:
        plt.show()
