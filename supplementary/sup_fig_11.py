#!/usr/bin/env python3

import importlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import os
from scipy.optimize import minimize
import sys
import torch

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
import routines_math as rout_m
sys_path0 = np.copy(sys.path)

save = False
verbose = True

saveFolder = os.path.abspath('figures')

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

bottom_margin_x = 0.05 # inch
left_margin_x = 0.05 # inch
left_margin  = 0.2 # inch
right_margin = 0.05 # inch
bottom_margin = 0.2 # inch
top_margin = 0.2 # inch

fontsize = 6
linewidth = 1.0
fontname = "Arial"

folder_data = os.path.abspath(data.path+'/reconstruction')
folder_list = list([
                    '/20200207/gap_20200207_008850_009050__pcutoff9e-01', # wait
                    '/20200207/gap_20200207_044610_044810__pcutoff9e-01', # step
                    '/20200207/gap_20200207_021920_022120__pcutoff9e-01', # reach   
                    ])
nFolders = len(folder_list)
beavior_list = list(['wait', 'step', 'reach'])
nSamples = int(1e4)

def find_closest_3d_point(m, n):
    def obj_func(x):
        estimated_points = x[:-3, None] * m + n
        res = 0.5 * np.sum((estimated_points - x[None, -3:])**2)

        jac = np.zeros(len(x), dtype=np.float64)
        jac[:-3] = np.sum(m**2, 1) * x[:-3] + \
                   np.sum(m * n, 1) - \
                   np.sum(m * x[None, -3:], 1)
        jac[-3:] = (np.float64(len(x)) - 3.0) * x[-3:] - np.sum(estimated_points, 0)
        return res, jac

    nPoints = np.size(m, 0)
    x0 = np.zeros(nPoints + 3, dtype=np.float64)
    tolerance = np.finfo(np.float32).eps # using np.float64 can lead to the optimization not converging (results are the same though)
    min_result = minimize(obj_func,
                          x0,
                          method='l-bfgs-b',
                          jac=True,
                          tol=tolerance,
                          options={'disp':False,
                                   'maxcor':10,
                                   'maxfun':np.inf,
                                   'maxiter':np.inf,
                                   'maxls':20})
    if not(min_result.success):
        print('WARNING: 3D point interpolation did not converge')
        print('\tnPoints\t', nPoints)
        print('\tsuccess:\t', min_result.success)
        print('\tstatus:\t', min_result.status)
        print('\tmessage:\t',min_result.message)
        print('\tnit:\t', min_result.nit)    
    return min_result.x

def calc_dst(m_udst, k):
    x_1 = m_udst[:, 0] / m_udst[:, 2]
    y_1 = m_udst[:, 1] / m_udst[:, 2]
    
    r2 = x_1**2 + y_1**2
    
    x_2 = x_1 * (1 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3) + \
          2 * k[2] * x_1 * y_1 + \
          k[3] * (r2 + 2 * x_1**2)
          
    y_2 = y_1 * (1 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3) + \
          k[2] * (r2 + 2 * y_1**2) + \
          2 * k[3] * x_1 * y_1
    
    nPoints = np.size(m_udst, 0)
    ones = np.ones(nPoints, dtype=np.float64)
    m_dst = np.concatenate([[x_2], [y_2], [ones]], 0).T
    return m_dst

# look at: Rational Radial Distortion Models with Analytical Undistortion Formulae, Lili Ma et al.
# source: https://arxiv.org/pdf/cs/0307047.pdf
# only works for k = [k1, k2, 0, 0, 0]!
def calc_udst(m_dst, k):
    assert np.all(k[2:] == 0.0), 'ERROR: Undistortion only valid for up to two radial distortion coefficients.'
    
    x_2 = m_dst[:, 0]
    y_2 = m_dst[:, 1]
    
    # use r directly instead of c
    nPoints = np.size(m_dst, 0)
    p = np.zeros(6, dtype=np.float64)
    p[4] = 1
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
        p[2] = k[0] * (1 + c**2)
        p[0] = k[1] * (1 + c**2)**2
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

def calc_3d_point(points_2d, A, k, rX1, tX1):
    mask_nan = ~np.any(np.isnan(points_2d), 1)
    mask_zero = ~np.any((points_2d == 0.0), 1)
    mask = np.logical_and(mask_nan, mask_zero)
    nValidPoints = np.sum(mask)
    if (nValidPoints < 2):
        print("WARNING: Less than 2 valid 2D locations for 3D point interpolation.")
    n = np.zeros((nValidPoints, 3))
    m = np.zeros((nValidPoints, 3))
    nCameras = np.size(points_2d, 0)
    
    index = 0
    for i_cam in range(nCameras):
        if (mask[i_cam]):
            point = np.array([[(points_2d[i_cam, 0] - A[i_cam][1]) / A[i_cam][0],
                               (points_2d[i_cam, 1] - A[i_cam][3]) / A[i_cam][2],
                               1.0]], dtype=np.float64)
            point = calc_udst(point, k[i_cam]).T
            line = point * np.array([0.0, 1.0], dtype=np.float64)
            RX1 = rout_m.rodrigues2rotMat_single(rX1[i_cam])
            line = np.dot(RX1.T, line - tX1[i_cam].reshape(3, 1))
            n[index] = line[:, 0]
            m[index] = line[:, 1] - line[:, 0]
            index = index + 1
    x = find_closest_3d_point(m, n)
    
    estimated_points3d = np.full((np.size(points_2d, 0), 3), np.nan, dtype=np.float64)
    estimated_points3d[mask] = x[:-3, None] * m + n
    point3d = x[-3:]
    
    return point3d, estimated_points3d

def get_paw_front(scale_x, scale_y):
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
    if (scale_x != None):
        paw[:, 0] = paw[:, 0] * (scale_x / (np.max(paw[:, 0]) - np.min(paw[:, 0])))
    if (scale_y != None):
        paw[:, 1] = paw[:, 1] * (scale_y / (np.max(paw[:, 1]) - np.min(paw[:, 1])))
    return paw

def get_paw_hind(scale_x, scale_y):
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
    if (scale_x != None):
        paw[:, 0] = paw[:, 0] * (scale_x / (np.max(paw[:, 0]) - np.min(paw[:, 0])))
    if (scale_y != None):
        paw[:, 1] = paw[:, 1] * (scale_y / (np.max(paw[:, 1]) - np.min(paw[:, 1])))
    return paw

def max_finder(trace):
    indices = list()
    for i_index in np.arange(1, len(trace)-1, 1, dtype=np.int64):
        if ((trace[i_index] > trace[i_index-1]) and 
            (trace[i_index] > trace[i_index+1])):
            indices.append(i_index)
    indices = np.array(indices, dtype=np.int64)
    return indices

def min_finder(trace):
    indices = list()
    for i_index in np.arange(1, len(trace)-1, 1, dtype=np.int64):
        if ((trace[i_index] < trace[i_index-1]) and 
            (trace[i_index] < trace[i_index+1])):
            indices.append(i_index)
    indices = np.array(indices, dtype=np.int64)
    return indices

if __name__ == '__main__': 
    fig_w = np.round(mm_in_inch * 88.0 * 0.5, decimals=2)
    fig_h = np.round(mm_in_inch * 88.0 * 0.25, decimals=2)
    fig = plt.figure(1, figsize=(fig_w, fig_h))
    fig.clear()
    fig_x = left_margin/fig_w
    fig_y = bottom_margin/fig_h
    fig_w = 1.0 - (left_margin/fig_w + right_margin/fig_w)
    fig_h = 1.0 - (bottom_margin/fig_h + top_margin/fig_h)
    ax = fig.add_axes([fig_x, fig_y, fig_w, fig_h])
    ax.clear()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #
    fig3d_w = np.round(mm_in_inch * 88.0 * 0.5, decimals=2)
    fig3d_h = np.round(mm_in_inch * 88.0 * 0.5, decimals=2)
    fig3d = plt.figure(2, figsize=(fig3d_w, fig3d_h))
    fig3d.clear()
    fig3d.canvas.manager.window.move(0, 0)
    ax3d = fig3d.add_axes([0, 0, 1, 1], projection='3d')
    ax3d.clear()

    paw_hind_scale_x = 0.5 * (6.405049925951728 + 8.46078506255709) # hard coded (mean of calibration results of 20200205 and 20200207)
    paw_hind_scale_x = paw_hind_scale_x * 0.2
    paw_hind_scale_y = paw_hind_scale_x * 0.5
    paw_hind = get_paw_hind(paw_hind_scale_x, paw_hind_scale_y)
    paw_front = get_paw_front(paw_hind_scale_x, paw_hind_scale_y)
    
    angle_comb_list = list([
                            ['joint_head_001', 'joint_spine_005', 'joint_spine_004'], # 0
                            ['joint_spine_005', 'joint_spine_004', 'joint_spine_003'], # 1
                            ['joint_spine_004', 'joint_spine_003', 'joint_spine_002'], # 2
                            ['joint_spine_003', 'joint_spine_002', 'joint_spine_001'], # 3
                            ['joint_spine_002', 'joint_spine_001', 'joint_tail_005'], # 4
                            ['joint_shoulder_left', 'joint_elbow_left', 'joint_wrist_left'], # 5 # left
                            ['joint_elbow_left', 'joint_wrist_left', 'joint_finger_left_002'], # 6
                            ['joint_hip_left', 'joint_knee_left', 'joint_ankle_left'], # 7
                            ['joint_knee_left', 'joint_ankle_left', 'joint_paw_hind_left'], # 8
                            ['joint_shoulder_right', 'joint_elbow_right', 'joint_wrist_right'], # 9  # right
                            ['joint_elbow_right', 'joint_wrist_right', 'joint_finger_right_002'], # 10
                            ['joint_hip_right', 'joint_knee_right', 'joint_ankle_right'], # 11
                            ['joint_knee_right', 'joint_ankle_right', 'joint_paw_hind_right'], # 12
                           ])
    nAngles = np.size(angle_comb_list, 0)
    index_angle_spring = np.array([7, 8, 11, 12], dtype=np.int64)
        
    nBones = 28
    nJoints = nBones + 1
        
    jump_indices = np.zeros((nFolders, 3), dtype=np.int64)
    jump_distances = np.zeros(nFolders, dtype=np.float64)
    side_jumps = np.zeros(nFolders, dtype=bool)
    for i_folder in np.arange(0, nFolders, 1, dtype=np.int64):
        folder = folder_data+'/'+folder_list[i_folder]
        
        sys.path = list(np.copy(sys_path0))
        sys.path.append(folder)
        importlib.reload(cfg)
        cfg.animal_is_large = 0
        importlib.reload(anatomy)

        folder_reqFiles = data.path + '/required_files' 
        file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
        file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
        file_model = folder_reqFiles + '/model.npy'
        file_labelsDLC = folder_reqFiles + '/labels_dlc_dummy.npy' # DLC labels are not used here
    
        file_save_dict = folder+'/'+'save_dict.npy'
        save_dict = np.load(file_save_dict, allow_pickle=True).item()
        mu_ini = np.load(folder+'/'+'x_ini.npy', allow_pickle=True)

        if ('mu_uks' in save_dict):
            mu_uks = save_dict['mu_uks'][1:]
            var_uks = save_dict['var_uks'][1:]
            nSamples_use= np.copy(nSamples)
            print('{:02d} {:s}'.format(i_folder, folder))
            print(save_dict['message'])
        else:
            mu_uks = save_dict['mu'][1:]
            nT = np.size(mu_uks, 0)
            nPara = np.size(mu_uks, 1)
            var_dummy = np.identity(nPara, dtype=np.float64) * 2**-52
            var_uks = np.tile(var_dummy.ravel(), cfg.nT).reshape(cfg.nT, nPara, nPara)
            nSamples_use = int(0)

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

        t_start = 0
        t_end = 200
        nT_single = t_end - t_start

        position_single = np.full((nT_single, nSamples+1, nJoints, 3), np.nan, dtype=np.float64)
        velocity_single = np.full((nT_single, nSamples+1,  nJoints, 3), np.nan, dtype=np.float64)
        position_single_peak = np.full((nT_single, nSamples+1, nJoints, 3), np.nan, dtype=np.float64)
        velocity_single_peak = np.full((nT_single, nSamples+1,  nJoints, 3), np.nan, dtype=np.float64)
        angle_single_peak = np.full((nT_single, nSamples+1, nAngles), np.nan, dtype=np.float64)
        angle_velocity_single_peak = np.full((nT_single, nSamples+1, nAngles), np.nan, dtype=np.float64)
        for t in range(nT_single):
            if (nSamples > 0):
                mu_t = torch.from_numpy(mu_uks[t])
                var_t = torch.from_numpy(var_uks[t])
                distribution = torch.distributions.MultivariateNormal(loc=mu_t,
                                                                      scale_tril=kalman.cholesky_save(var_t))
                z_samples = distribution.sample((nSamples,))
                z_all = torch.cat([mu_t[None, :], z_samples], 0)
            else:
                z_all = torch.from_numpy(mu_uks[t][None, :])
            markers2d, markers3d, joints3d = model.fcn_emission_free(z_all, args)
            markers2d_fit = markers2d.cpu().numpy().reshape(nSamples+1, nCameras, nMarkers, 2)
            markers2d_fit[:, :, :, 0] = (markers2d_fit[:, :, :, 0] + 1.0) * (cfg.normalize_camera_sensor_x * 0.5)
            markers2d_fit[:, :, :, 1] = (markers2d_fit[:, :, :, 1] + 1.0) * (cfg.normalize_camera_sensor_y * 0.5)
            markers3d_fit = markers3d.cpu().numpy()
            joints3d_fit = joints3d.cpu().numpy()
            #
            derivative_accuracy = 8
            #
            position_single_peak[t] = np.copy(joints3d_fit)
            # do coordinate transformation
            joint_index1 = joint_order.index('joint_spine_002') # pelvis
            joint_index2 = joint_order.index('joint_spine_004') # tibia
            origin = joints3d_fit[:, joint_index1]
            x_direc = joints3d_fit[:, joint_index2] - origin
            x_direc = x_direc[:, :2]
            x_direc = x_direc / np.sqrt(np.sum(x_direc**2, 1))[:, None]
            alpha = np.arctan2(x_direc[:, 1], x_direc[:, 0])
            cos_m_alpha = np.cos(-alpha)
            sin_m_alpha = np.sin(-alpha)
            R = np.stack([np.stack([cos_m_alpha, -sin_m_alpha], 1),
                          np.stack([sin_m_alpha, cos_m_alpha], 1)], 1)
            joints3d_fit = joints3d_fit - origin[:, None, :]
            skeleton_pos_xy = np.einsum('nij,nmj->nm i', R, joints3d_fit[:, :, :2])
            joints3d_fit[:, :, :2] = np.copy(skeleton_pos_xy)
            #
            position_single[t] = np.copy(joints3d_fit)

        velocity_single[4:-4] = \
            (+1/280 * position_single[:-8] + \
             -4/105 * position_single[1:-7] + \
             +1/5 * position_single[2:-6] + \
             -4/5 * position_single[3:-5] + \
#                      0 * position_single[4:-4] + \
             +4/5 * position_single[5:-3] + \
             -1/5 * position_single[6:-2] + \
             +4/105 * position_single[7:-1] + \
             -1/280 * position_single[8:]) / (1.0/cfg.frame_rate)
        velocity_single_peak[4:-4] = \
            (+1/280 * position_single_peak[:-8] + \
             -4/105 * position_single_peak[1:-7] + \
             +1/5 * position_single_peak[2:-6] + \
             -4/5 * position_single_peak[3:-5] + \
#                      0 * position_single_peak[4:-4] + \
             +4/5 * position_single_peak[5:-3] + \
             -1/5 * position_single_peak[6:-2] + \
             +4/105 * position_single_peak[7:-1] + \
             -1/280 * position_single_peak[8:]) / (1.0/cfg.frame_rate)
        for i_ang in range(nAngles):
            index_joint1 = joint_order.index(angle_comb_list[i_ang][0])
            index_joint2 = joint_order.index(angle_comb_list[i_ang][1])
            index_joint3 = joint_order.index(angle_comb_list[i_ang][2])  
            vec1 = position_single_peak[:, :, index_joint2] - position_single_peak[:, :, index_joint1]
            vec2 = position_single_peak[:, :, index_joint3] - position_single_peak[:, :, index_joint2]
            vec1 = vec1 / np.sqrt(np.sum(vec1**2, 2))[:, :, None]
            vec2 = vec2 / np.sqrt(np.sum(vec2**2, 2))[:, :, None]
            ang = np.arccos(np.einsum('nsi,nsi->ns', vec1, vec2)) * 180.0/np.pi
            angle_single_peak[:, :, i_ang] = np.copy(ang)
            angle_velocity_single_peak[4:-4, :, i_ang] = \
                (+1/280 * ang[:-8] + \
                 -4/105 * ang[1:-7] + \
                 +1/5 * ang[2:-6] + \
                 -4/5 * ang[3:-5] + \
#                          0 * ang[4:-4] + \
                 +4/5 * ang[5:-3] + \
                 -1/5 * ang[6:-2] + \
                 +4/105 * ang[7:-1] + \
                 -1/280 * ang[8:]) / (1.0/cfg.frame_rate)

        # find jump
        angle_index_spine = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        angle_index_hind_limbs = np.array([7, 8, 11, 12], dtype=np.int64)
        metric_use_indices = np.concatenate([angle_index_spine,
                                             angle_index_hind_limbs], 0)
        metric_use = np.mean(angle_single_peak[:, 0, metric_use_indices], 1)
        index_ini = np.argmin(metric_use)
        index_jump = np.copy(index_ini)
        #
        index_start = 0
        index_end = np.copy(index_jump)
        indices_maxima1 = max_finder(metric_use[index_start:index_end]) + index_start
        index_start = 0
        index_end = np.copy(indices_maxima1[-1])
        indices_minima1 = min_finder(metric_use[index_start:index_end]) + index_start
        if not(len(indices_minima1) == 0):
            index1 = indices_minima1[-1]
        else:
            index1 = 0
        #
        index_start = np.copy(index_jump)
        index_end = nT_single 
        indices_maxima2 = max_finder(metric_use[index_start:index_end]) + index_start
        index_start = np.copy(indices_maxima2[0])
        index_end = nT_single 
        indices_minimia2 = min_finder(metric_use[index_start:index_end]) + index_start
        if not(len(indices_minimia2) == 0):
            index2 = indices_minimia2[0]
        else:
            index2 = nT_single - 1
        #
        jump_indices = np.array([index1, index_jump, index2], dtype=np.int64)
        print('jump indices:\t{:03d} {:03d} {:03d}'.format(index1, index_jump, index2))
        print('jump time:\t{:03d}'.format(index2 - index1))

        # calculate jump distance
        indices_paws = np.array([joint_order.index('joint_ankle_left'),
                                 joint_order.index('joint_paw_hind_left'),
                                 joint_order.index('joint_toe_left_002'),
                                 joint_order.index('joint_ankle_right'),
                                 joint_order.index('joint_paw_hind_right'),
                                 joint_order.index('joint_toe_right_002'),], dtype=np.int64)
        diff = np.mean(position_single_peak[index2, 0, indices_paws, :2], 0) - np.mean(position_single_peak[index1, 0, indices_paws, :2], 0)
        dist = np.sqrt(np.sum(diff**2))
        jump_distances = np.copy(dist)
        print('jump distance:\t{:08f}'.format(dist))
        #
        side_jumps[i_folder] = (diff[0] > 0.0)        
        gap_edges = np.load(os.path.abspath('../figure04/gap_{:s}__edges.npz'.format(cfg.date)), allow_pickle=True)['arr_0'].item()
        #
        A = args['calibration']['A_fit'].cpu().numpy()
        k = args['calibration']['k_fit'].cpu().numpy()
        rX1 = args['calibration']['rX1_fit'].cpu().numpy()
        tX1 = args['calibration']['tX1_fit'].cpu().numpy()
        gap_edges_3d = dict()
        gap_edges_use = gap_edges[cfg.index_frame_start]
        for key in sorted(list(gap_edges_use.keys())):
            point3d, _ = calc_3d_point(gap_edges_use[key], A, k, rX1, tX1)
            gap_edges_3d[key] = point3d
        edge_left = np.array([gap_edges_3d['lower_left'], gap_edges_3d['upper_left']], dtype=np.float64)
        edge_right = np.array([gap_edges_3d['lower_right'], gap_edges_3d['upper_right']], dtype=np.float64)
        edge_left[:, 0] = np.mean(edge_left[:, 0], 0)
        edge_right[:, 0] = np.mean(edge_right[:, 0], 0)
        edge_left[:, 2] = np.mean(edge_left[:, 2], 0)
        edge_right[:, 2] = np.mean(edge_right[:, 2], 0)
        center_left = np.mean(edge_left, 0)
        center_right = np.mean(edge_right, 0)
        gap_length = np.sqrt(np.sum((center_left[0] - center_right[0])**2))
        print('gap length:\t{:0.8f}'.format(gap_length))

        # PLOT
        ax.clear()
        time = np.arange(nT_single, dtype=np.float64) * 1.0/cfg.frame_rate
        index_use = list([[7], [8],
                          [11], [12],
                          [0, 1, 2, 3, 4, 7, 8, 11, 12]])
        for i_metric in list([4]):
            if (i_metric == 0):
                color='cyan'
            elif (i_metric == 1):
                color = 'blue'
            elif (i_metric == 2):
                color = 'orange'
            elif (i_metric == 3):
                color = 'red'
            elif (i_metric == 4):
                color = 'darkblue'
            metric_use_indices = index_use[i_metric]
            metric_use = np.mean(angle_single_peak[:, :, metric_use_indices], 2)
            avg = metric_use[:, 0] - np.nanmean(metric_use[:, 0], 0)
            std = np.nanstd(metric_use, 1)
            ax.plot(time, avg,
                    linestyle='-', marker='',
                     color=color, linewidth=linewidth,
                     alpha=1.0, zorder=2)
            ax.fill_between(x=time,
                            y1=avg+std,
                            y2=avg-std,
                            color=color, linewidth=linewidth,
                            alpha=0.2, zorder=1,
                           where=None, interpolate=False, step=None, data=None)
        #
        len_x = np.nanmax(time) - np.nanmin(time)
        x_min = np.nanmin(time) - len_x * 0.01
        x_max = np.nanmax(time) + len_x * 0.01
        y_min = -35
        y_max = 20.0
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        #
        ylim_min, ylim_max = ax.get_ylim()
        rectangle_jump = np.array([[time[index1], ylim_min], 
                                   [time[index1], ylim_max],
                                   [time[index2], ylim_max],
                                   [time[index2], ylim_min]], dtype=np.float64)
        poly_jump = plt.Polygon(rectangle_jump, color='black', zorder=0, alpha=0.1)
        ax.add_patch(poly_jump)
        #
        cfg_plt.plot_coord_ax(ax,
                              '100 ms', '15 deg',
                              0.1, 15.0)
        fig.canvas.draw()

        # 3D
        ax3d.clear()
        linewidth_fac = 0.5 # see figure size
        # EDGES
        color = 'darkgray'
        #
        A = args['calibration']['A_fit'].cpu().numpy()
        k = args['calibration']['k_fit'].cpu().numpy()
        rX1 = args['calibration']['rX1_fit'].cpu().numpy()
        tX1 = args['calibration']['tX1_fit'].cpu().numpy()
        gap_edges_3d = dict()
        gap_edges_use = gap_edges[cfg.index_frame_start]
        for key in sorted(list(gap_edges_use.keys())):
            point3d, _ = calc_3d_point(gap_edges_use[key], A, k, rX1, tX1)
            gap_edges_3d[key] = point3d
        edge_left = np.array([gap_edges_3d['lower_left'], gap_edges_3d['upper_left']], dtype=np.float64)
        edge_right = np.array([gap_edges_3d['lower_right'], gap_edges_3d['upper_right']], dtype=np.float64)
        edge_left[:, 0] = np.mean(edge_left[:, 0], 0)
        edge_right[:, 0] = np.mean(edge_right[:, 0], 0)
        edge_left[:, 2] = np.mean(edge_left[:, 2], 0)
        edge_right[:, 2] = np.mean(edge_right[:, 2], 0)
        edge_len = 40.0
        for i_edge in range(2):
            if (i_edge == 0):
                edge_use = edge_left
                x_inf = -edge_len
            else:
                edge_use = edge_right
                x_inf = edge_len
            x = np.array([edge_use[0, 0]+x_inf, edge_use[0, 0], edge_use[1, 0], edge_use[0, 0]+x_inf], dtype=np.float64)
            y = np.array([edge_use[0, 1], edge_use[0, 1], edge_use[1, 1], edge_use[1, 1]], dtype=np.float64)
            z = np.array([edge_use[0, 2], edge_use[0, 2], edge_use[1, 2], edge_use[1, 2]], dtype=np.float64)
            ax3d.plot(x, y, color=color, linewidth=1.0*linewidth_fac, zorder=0)
        # PAWS
        cmap2 = plt.cm.tab10
        color_left_front = cmap2(4/9)
        color_right_front = cmap2(3/9)
        color_left_hind = cmap2(9/9)
        color_right_hind = cmap2(8/9)
        colors_paws = list([color_left_front, color_right_front, color_left_hind, color_right_hind])
        paws_joint_names = list(['joint_wrist_left', 'joint_wrist_right', 'joint_ankle_left', 'joint_ankle_right'])
        nPaws = 4
        index_use = list([[7], [11]])
        window_size = 11
        for i_paw in list([2, 3]):
            metric_use_indices = index_use[i_paw-2]
            metric_use = np.mean(angle_single_peak[:, 0, metric_use_indices] , 1)
            metric_use[5:-5] = 1.0/window_size * (metric_use[:-10] + \
                                                  metric_use[1:-9] + \
                                                  metric_use[2:-8] + \
                                                  metric_use[3:-7] + \
                                                  metric_use[4:-6] + \
                                                  metric_use[5:-5] + \
                                                  metric_use[6:-4] + \
                                                  metric_use[7:-3] + \
                                                  metric_use[8:-2] + \
                                                  metric_use[9:-1] + \
                                                  metric_use[10:])
            for i_win in range(int((window_size-1)/2)):
                metric_use[i_win] = np.nan
                metric_use[-i_win-1] = np.nan
            indices_minima = min_finder(metric_use[:index_jump])
            indices_maxima = max_finder(metric_use[:index_jump])
            extremum = indices_maxima
            if (len(extremum) > 0):
                nExtremum = len(extremum)
                if ((beavior_list[i_folder] == 'wait') or (beavior_list[i_folder] == 'reach')):
                    extremum_list = list([nExtremum-1])
                else:
                    extremum_list = range(nExtremum)
                for i_extremum in extremum_list:
                    time_point_index = extremum[i_extremum]
                    #
                    joint_index = joint_order.index(paws_joint_names[i_paw])
                    paw_pos_xy = position_single_peak[time_point_index, 0, joint_index, :2]
                    #
                    index = np.where(skeleton_edges[:, 0] == joint_index)[0][0]
                    paw_direc = position_single_peak[time_point_index, 0, skeleton_edges[index, 1], :2] - paw_pos_xy
                    alpha = np.arctan2(paw_direc[1], paw_direc[0])
                    #
                    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                                  [np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
                    #
                    if (i_paw < 2):
                        paw_plot = np.einsum('ij,nj->ni', R, paw_front) + paw_pos_xy[None, :]
                    else:
                        paw_plot = np.einsum('ij,nj->ni', R, paw_hind) + paw_pos_xy[None, :]
                    paw_poly = plt.Polygon(paw_plot, color=colors_paws[i_paw], zorder=0, alpha=0.5, linewidth=0.1)   
                    ax3d.add_patch(paw_poly)
                    art3d.pathpatch_2d_to_3d(paw_poly, z=0.0, zdir='z')
        # SKELETON
        color_joint = 'black'
        color_bone = 'black'
        #
        if (side_jumps[i_folder]):
            s1 = 'left'
            s2 = 'right'
        else:
            s1 = 'right'
            s2 = 'left'
        #
        skeleton_pos = np.copy(position_single_peak[index1, 0])
        #
        i_bone = 0
        joint_index_start = skeleton_edges[i_bone, 0]
        joint_index_end = skeleton_edges[i_bone, 1]
        joint_start = skeleton_pos[joint_index_start]
        joint_name = joint_order[joint_index_end]
        joint_name_split = joint_name.split('_')
        zorder_joint = 4 # first joint is always center-joint
        ax3d.plot([joint_start[0]], [joint_start[1]], [joint_start[2]],
                  linestyle='', marker='.', markersize=2.5*linewidth_fac, markeredgewidth=0.5*linewidth_fac, color=color_joint, zorder=zorder_joint)
        for i_bone in range(nJoints-1):
            joint_index_start = skeleton_edges[i_bone, 0]
            joint_index_end = skeleton_edges[i_bone, 1]
            joint_start = skeleton_pos[joint_index_start]
            joint_end = skeleton_pos[joint_index_end]
            vec = np.stack([joint_start, joint_end], 0)
            joint_name = joint_order[joint_index_end]
            joint_name_split = joint_name.split('_')
            if (s1 in joint_name_split):
                zorder_bone = 1
                zorder_joint = 2
            elif (s2 in joint_name_split):
                zorder_bone = 5
                zorder_joint = 6
            else:
                zorder_bone = 3
                zorder_joint = 4
            ax3d.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                    linestyle='-', marker=None, linewidth=1.0*linewidth_fac, color=color_bone, zorder=zorder_bone)
            ax3d.plot([joint_end[0]], [joint_end[1]], [joint_end[2]],
                    linestyle='', marker='.', markersize=2.5*linewidth_fac, markeredgewidth=0.5*linewidth_fac, color=color_joint, zorder=zorder_joint)
                #
        dxyz = 12.5
        pos = np.copy(np.mean(position_single_peak[:, 0], 1))
        dist_left = np.sqrt(np.sum((pos - center_left[None, :])**2, 1))
        dist_right = np.sqrt(np.sum((pos - center_right[None, :])**2, 1))
        if (dist_left[index1] < dist_right[index1]):
            xyz_center = center_left
        else:
            xyz_center = center_right
        ax3d.set_xlim([xyz_center[0]-dxyz, xyz_center[0]+dxyz])
        ax3d.set_ylim([xyz_center[1]-dxyz, xyz_center[1]+dxyz])
        ax3d.set_zlim([xyz_center[2]-dxyz, xyz_center[2]+dxyz])
        ax3d.set_axis_off()
        azim = -40
        elev = 25
        if (side_jumps[i_folder]):
            ax3d.view_init(azim=azim, elev=elev)
        else:
            ax3d.view_init(azim=azim+180, elev=elev)                
        #
        fig3d.canvas.draw()
        if verbose:
            plt.show(block=False)
            print('Press any key to continue')
            input()

        if save:
            fig.savefig(saveFolder+'/{:s}__metric__{:s}.svg'.format(folder_list[i_folder], beavior_list[i_folder]),
    #                          bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)
            fig3d.savefig(saveFolder+'/{:s}__3d__{:s}.svg'.format(folder_list[i_folder], beavior_list[i_folder]),
    #                          bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)