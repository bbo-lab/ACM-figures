#!/usr/bin/env python3

import importlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
import model
import routines_math as rout_m
sys_path0 = np.copy(sys.path)

save = False

folder_save = os.path.abspath('panels')

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

bottom_margin_x = 0.05 # inch
left_margin_x = 0.05 # inch
left_margin  = 0.50 # inch
right_margin = 0.05 # inch
bottom_margin = 0.50 # inch
top_margin = 0.05 # inch

fontsize = 6
linewidth = 1.0
fontname = "Arial"

add2folder = '__pcutoff9e-01'
folder_recon = data.path + '/reconstruction' 
folder_list_wait = list([
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(10050, 10250), # wait
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(30030, 30230), # wait
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(38570, 38770), # wait
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(46420, 46620), # wait
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(48990, 49190), # wait
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(52760, 52960), # wait
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(8850, 9050), # wait
                        ])
folder_list_run = list([
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(33530, 33730), # jump/run
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(159080, 159280), # jump/run
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(176820, 177020), # jump/run
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(188520, 188720), # jump/run
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(27160, 27360), # jump/run
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(74250, 74450), # jump/run
                        ])
folder_list_step = list([
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(41930, 42130), # small step
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(137470, 137670), # small step
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(145240, 145440), # small step
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(152620, 152820), # small step
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(155350, 155550), # small step
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(173910, 174110), # small step
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(179060, 179260), # small step
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(181330, 181530), # small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(31730, 31930), # small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(36280, 36480), # small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(38610, 38810), # small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(44610, 44810), # small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(47360, 47560), # small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(51220, 51420), # small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(55680, 55880), # small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(60410, 60610), # small step
                        ])
folder_list_hick = list([
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(53640, 53840), # hickup
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(62210, 62410), # hickup
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(64080, 64280), # hickup
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(66550, 66750), # hickup
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(71300, 71500), # hickup
                        ])
folder_list_rest = list([
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(5550, 5750), # jump/run or small step
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(14830, 15030), # hickup or small step
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(161790, 161990), # jump/run or hickup
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(2850, 3050), # jump/run or small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(42010, 42210), # wait?
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(58300, 58500), # jump/run or small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(68880, 69080), # jump/run or small step
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(84520, 84720), # small step?
                        ])
folder_list_reach = list([
                        '/20200205/gap_20200205_{:06d}_{:06d}'.format(128720, 128920), # reach
                        '/20200207/gap_20200207_{:06d}_{:06d}'.format(21920, 22120), # reach
                         ])
folder_list = list()
folder_list = folder_list + folder_list_wait
folder_list = folder_list + folder_list_run
folder_list = folder_list + folder_list_step
folder_list = folder_list + folder_list_hick
folder_list = folder_list + folder_list_rest
folder_list = folder_list + folder_list_reach
nFolders = len(folder_list)

list_is_large_animal = list([0 for i in range(nFolders)])

nSamples = int(0e3)

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

if __name__ == '__main__': 
    nT = 200

    fig_xz_w = np.round(mm_in_inch * 88.0 * 0.5, decimals=2)
    fig_xz_h = np.round(mm_in_inch * 88.0 * 0.25, decimals=2)
    fig_xz = plt.figure(1, figsize=(fig_xz_w, fig_xz_h))
    fig_xz.clear()
    fig_xz_x = left_margin/fig_xz_w
    fig_xz_y = bottom_margin/fig_xz_h
    fig_xz_w = 1.0 - (left_margin/fig_xz_w + right_margin/fig_xz_w)
    fig_xz_h = 1.0 - (bottom_margin/fig_xz_h + top_margin/fig_xz_h)
    ax_xz = fig_xz.add_axes([0.1, 0.1, 0.9, 0.9])
    ax_xz.clear()
    ax_xz.spines["top"].set_visible(False)
    ax_xz.spines["right"].set_visible(False)
    #
    fig_xy_w = np.round(mm_in_inch * 88.0 * 0.5, decimals=2)
    fig_xy_h = np.round(mm_in_inch * 88.0 * 0.25, decimals=2)
    fig_xy = plt.figure(2, figsize=(fig_xy_w, fig_xy_h))
    fig_xy.clear()
    fig_xy_x = left_margin/fig_xy_w
    fig_xy_y = bottom_margin/fig_xy_h
    fig_xy_w = 1.0 - (left_margin/fig_xy_w + right_margin/fig_xy_w)
    fig_xy_h = 1.0 - (bottom_margin/fig_xy_h + top_margin/fig_xy_h)
    ax_xy = fig_xy.add_axes([0.1, 0.1, 0.9, 0.9])
    ax_xy.clear()
    ax_xy.spines["top"].set_visible(False)
    ax_xy.spines["right"].set_visible(False)
    #
    fig3d_w = np.round(mm_in_inch * 88.0, decimals=2)
    fig3d_h = np.round(mm_in_inch * 88.0, decimals=2)
    fig3d = plt.figure(4, figsize=(fig3d_w, fig3d_h))
    fig3d.clear()
    fig3d.canvas.manager.window.move(0, 0)
    ax3d = fig3d.add_axes([0, 0, 1, 1], projection='3d')
    ax3d.clear()

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
        
    position = list()
    velocity = list()
    acceleration = list()
    position_mean = list()
    velocity_mean = list()
    acceleration_mean = list()
    angle = list()
    angle_velocity = list()
    angle_acceleration = list()
    position_peak = list()
    velocity_peak = list()
    acceleration_peak = list()
    angle_peak = list()
    angle_velocity_peak = list()
    angle_acceleration_peak = list()
    #
    jump_indices = np.zeros((nFolders, 3), dtype=np.int64)
    jump_distances = np.zeros(nFolders, dtype=np.float64)
    gap_y = 0.0
    gap_z = 0.0
    #
    for i_folder in np.arange(0, nFolders, 1, dtype=np.int64):
        folder = folder_recon+folder_list[i_folder]+add2folder
        sys.path = list(np.copy(sys_path0))
        sys.path.append(folder)
        importlib.reload(cfg)
        cfg.animal_is_large = list_is_large_animal[i_folder]
        importlib.reload(anatomy)
        #
        folder_reqFiles = data.path + '/required_files' 
        file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
        file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
        file_model = folder_reqFiles + '/model.npy'
        file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/' + '/labels_dlc_{:06d}_{:06d}.npy'.format(cfg.index_frame_start, cfg.index_frame_end)
        
        file_save_dict = folder+'/'+'save_dict.npy'
        if not(os.path.isfile(file_save_dict)):
            print('ERROR: {:s} does not contain any results'.format(file_save_dict))
            raise
        else:
            save_dict = np.load(file_save_dict, allow_pickle=True).item()
            file_mu_ini = folder+'/'+'x_ini.npy'
            mu_ini = np.load(file_mu_ini, allow_pickle=True)

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
                var_uks = np.tile(var_dummy.ravel(), nT).reshape(nT, nPara, nPara)
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
            
# # #             # normalize skeletons of different animals
# #             x_joints = list([#'joint_head_001',
# #                              'joint_spine_001',
# #                              'joint_spine_002',
# #                              'joint_spine_003',
# #                              'joint_spine_004',
# #                              'joint_spine_005',
# #                              'joint_tail_001',
# #                              'joint_tail_002',
# #                              'joint_tail_003',
# #                              'joint_tail_004',
# #                              'joint_tail_005',])
# #             y_joints1 = list(['joint_shoulder_left',
# #                               'joint_elbow_left',
# #                               'joint_wrist_left',
# #                               'joint_finger_left_002',])
# #             y_joints2 = list(['joint_hip_left',
# #                               'joint_knee_left',
# #                               'joint_ankle_left',
# #                               'joint_paw_hind_left',
# #                               'joint_toe_left_002',])
# #             bone_lengths_index = args['model']['bone_lengths_index'].cpu().numpy()
# #             bone_lengths_sym = mu_ini[:args['nPara_bones']][bone_lengths_index]
# #             bone_lengths_x = 0.0
# #             bone_lengths_limb_front = 0.0
# #             bone_lengths_limb_hind = 0.0
# #             for i_bone in range(nJoints-1):
# #                 index_bone_end = skeleton_edges[i_bone, 1]
# #                 joint_name = joint_order[index_bone_end]
# #                 if joint_name in x_joints:
# #                     bone_lengths_x = bone_lengths_x + bone_lengths_sym[i_bone]
# #                 elif joint_name in y_joints1:
# #                     bone_lengths_limb_front = bone_lengths_limb_front + bone_lengths_sym[i_bone]
# #                 elif joint_name in y_joints2:
# #                     bone_lengths_limb_hind = bone_lengths_limb_hind + bone_lengths_sym[i_bone]
# #             #
# #             mu_ini[:args['nPara_bones']] = mu_ini[:args['nPara_bones']] / bone_lengths_x
# # #             mask = np.ones_like(bone_lengths_index, dtype=bool)
# # #             for i_bone in range(nJoints-1):
# # #                 index0 = bone_lengths_index[i_bone]
# # #                 if mask[index0]:
# # #                     mu_ini[index0] = mu_ini[index0] / bone_lengths_x
# # #                     mask[index0] = False
#             mu_ini[:args['nPara_bones']] = 1.0

            
            #
#             z_all = np.copy(mu_uks[0])
#             z_all[:3] = 0.0 
#             z_all[3:6] = 0.0
#             z_all[6:] = (args['bounds_free_pose'][6:, 0] + 0.5 * (args['bounds_free_pose'][6:, 1] - args['bounds_free_pose'][6:, 0])).cpu().numpy()
#             z_all = torch.from_numpy(z_all[None, :])
#             _, _, joints3d = model.fcn_emission_free(z_all, args)
#             joints3d_fit = joints3d[0].cpu().numpy()
#             scale_lengths_x = np.max(joints3d_fit[:, 2]) - np.min(joints3d_fit[:, 2]) # use these coordinates because of skeleton orientation [r0 = 0]
#             scale_lengths_y = np.max(joints3d_fit[:, 0]) - np.min(joints3d_fit[:, 0]) # use these coordinates because of skeleton orientation [r0 = 0]
#             scale_lengths_z = np.max(joints3d_fit[:, 1]) - np.min(joints3d_fit[:, 1]) # use these coordinates because of skeleton orientation [r0 = 0]
            
            t_start = 0
            t_end = 200
            nT_single = t_end - t_start

            position_single = np.full((nT_single, nJoints, 3), np.nan, dtype=np.float64)
            velocity_single = np.full((nT_single, nJoints, 3), np.nan, dtype=np.float64)
            acceleration_single = np.full((nT_single, nJoints, 3), np.nan, dtype=np.float64)
            position_mean_single = np.full((nT_single, 3), np.nan, dtype=np.float64)
            velocity_mean_single = np.full((nT_single, 3), np.nan, dtype=np.float64)
            acceleration_mean_single = np.full((nT_single, 3), np.nan, dtype=np.float64)
            angle_single = np.full((nT_single, nAngles), np.nan, dtype=np.float64)
            angle_velocity_single = np.full((nT_single, nAngles), np.nan, dtype=np.float64)
            angle_acceleration_single = np.full((nT_single, nAngles), np.nan, dtype=np.float64)   
            if (nSamples > 0):
                mu_t = torch.from_numpy(mu_uks[t_start:t_end])
                var_t = torch.from_numpy(var_uks[t_start:t_end])
                distribution = torch.distributions.MultivariateNormal(loc=mu_t,
                                                                      scale_tril=kalman.cholesky_save(var_t))
                z_samples = distribution.sample((nSamples,))
                z_all = torch.cat([mu_t[None, :], z_samples], 0)
            else:
                z_all = torch.from_numpy(mu_uks[t_start:t_end])
            markers2d, markers3d, joints3d = model.fcn_emission_free(z_all, args)
            markers2d_fit = markers2d.cpu().numpy().reshape(nT_single, nCameras, nMarkers, 2)
            markers2d_fit[:, :, :, 0] = (markers2d_fit[:, :, :, 0] + 1.0) * (cfg.normalize_camera_sensor_x * 0.5)
            markers2d_fit[:, :, :, 1] = (markers2d_fit[:, :, :, 1] + 1.0) * (cfg.normalize_camera_sensor_y * 0.5)
            markers3d_fit = markers3d.cpu().numpy()
            joints3d_fit = joints3d.cpu().numpy()
            
            position_single_peak = np.full((nT_single, nJoints, 3), np.nan, dtype=np.float64)
            velocity_single_peak = np.full((nT_single, nJoints, 3), np.nan, dtype=np.float64)
            acceleration_single_peak = np.full((nT_single, nJoints, 3), np.nan, dtype=np.float64)
            position_mean_single_peak = np.full((nT_single, 3), np.nan, dtype=np.float64)
            velocity_mean_single_peak = np.full((nT_single, 3), np.nan, dtype=np.float64)
            angle_single_peak = np.full((nT_single, nAngles), np.nan, dtype=np.float64)
            angle_velocity_single_peak = np.full((nT_single, nAngles), np.nan, dtype=np.float64)
            angle_acceleration_single_peak = np.full((nT_single, nAngles), np.nan, dtype=np.float64)
            #
            derivative_accuracy = 8
            #
            position_single_peak = np.copy(joints3d_fit)
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
            acceleration_single_peak[4:-4] = \
                (-1/560 * position_single_peak[:-8] + \
                 +8/315 * position_single_peak[1:-7] + \
                 -1/5 * position_single_peak[2:-6] + \
                 +8/5 * position_single_peak[3:-5] + \
                 -205/72 * position_single_peak[4:-4] + \
                 +8/5 * position_single_peak[5:-3] + \
                 -1/5 * position_single_peak[6:-2] + \
                 +8/315 * position_single_peak[7:-1] + \
                 -1/560 * position_single_peak[8:]) / (1.0/cfg.frame_rate)**2
            position_mean_single_peak = np.mean(joints3d_fit, 1)
            velocity_mean_single_peak[4:-4] = \
                (+1/280 * position_mean_single_peak[:-8] + \
                 -4/105 * position_mean_single_peak[1:-7] + \
                 +1/5 * position_mean_single_peak[2:-6] + \
                 -4/5 * position_mean_single_peak[3:-5] + \
#                      0 * position_mean_single_peak[4:-4] + \
                 +4/5 * position_mean_single_peak[5:-3] + \
                 -1/5 * position_mean_single_peak[6:-2] + \
                 +4/105 * position_mean_single_peak[7:-1] + \
                 -1/280 * position_mean_single_peak[8:]) / (1.0/cfg.frame_rate)
            for i_ang in range(nAngles):
                index_joint1 = joint_order.index(angle_comb_list[i_ang][0])
                index_joint2 = joint_order.index(angle_comb_list[i_ang][1])
                index_joint3 = joint_order.index(angle_comb_list[i_ang][2])  
                vec1 = joints3d_fit[:, index_joint2] - joints3d_fit[:, index_joint1]
                vec2 = joints3d_fit[:, index_joint3] - joints3d_fit[:, index_joint2]
                vec1 = vec1 / np.sqrt(np.sum(vec1**2, 1))[:, None]
                vec2 = vec2 / np.sqrt(np.sum(vec2**2, 1))[:, None]
                ang = np.arccos(np.einsum('ni,ni->n', vec1, vec2)) * 180.0/np.pi
                angle_single_peak[:, i_ang] = np.copy(ang)
                angle_velocity_single_peak[4:-4, i_ang] = \
                    (+1/280 * ang[:-8] + \
                     -4/105 * ang[1:-7] + \
                     +1/5 * ang[2:-6] + \
                     -4/5 * ang[3:-5] + \
#                          0 * ang[4:-4] + \
                     +4/5 * ang[5:-3] + \
                     -1/5 * ang[6:-2] + \
                     +4/105 * ang[7:-1] + \
                     -1/280 * ang[8:]) / (1.0/cfg.frame_rate)
                angle_acceleration_single_peak[4:-4, i_ang] = \
                    (-1/560 * ang[:-8] + \
                     +8/315 * ang[1:-7] + \
                     -1/5 * ang[2:-6] + \
                     +8/5 * ang[3:-5] + \
                     -205/72 * ang[4:-4] + \
                     +8/5 * ang[5:-3] + \
                     -1/5 * ang[6:-2] + \
                     +8/315 * ang[7:-1] + \
                     -1/560 * ang[8:]) / (1.0/cfg.frame_rate)**2

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
            
#             # scale animal skeletons
#             joints3d_fit[:, :, 0] = joints3d_fit[:, :, 0] / scale_lengths_x
#             joints3d_fit[:, :, 1] = joints3d_fit[:, :, 1] / scale_lengths_y
#             joints3d_fit[:, :, 2] = joints3d_fit[:, :, 2] / scale_lengths_z

            #
            position_single = np.copy(joints3d_fit)
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
            acceleration_single[4:-4] = \
                (-1/560 * position_single[:-8] + \
                 +8/315 * position_single[1:-7] + \
                 -1/5 * position_single[2:-6] + \
                 +8/5 * position_single[3:-5] + \
                 -205/72 * position_single[4:-4] + \
                 +8/5 * position_single[5:-3] + \
                 -1/5 * position_single[6:-2] + \
                 +8/315 * position_single[7:-1] + \
                 -1/560 * position_single[8:]) / (1.0/cfg.frame_rate)**2
            position_mean_single = np.mean(joints3d_fit, 1)
            velocity_mean_single[4:-4] = \
                (+1/280 * position_mean_single[:-8] + \
                 -4/105 * position_mean_single[1:-7] + \
                 +1/5 * position_mean_single[2:-6] + \
                 -4/5 * position_mean_single[3:-5] + \
#                      0 * position_mean_single[4:-4] + \
                 +4/5 * position_mean_single[5:-3] + \
                 -1/5 * position_mean_single[6:-2] + \
                 +4/105 * position_mean_single[7:-1] + \
                 -1/280 * position_mean_single[8:]) / (1.0/cfg.frame_rate)
            acceleration_mean_single[4:-4] = \
                (-1/560 * position_mean_single[:-8] + \
                 +8/315 * position_mean_single[1:-7] + \
                 -1/5 * position_mean_single[2:-6] + \
                 +8/5 * position_mean_single[3:-5] + \
                 -205/72 * position_mean_single[4:-4] + \
                 +8/5 * position_mean_single[5:-3] + \
                 -1/5 * position_mean_single[6:-2] + \
                 +8/315 * position_mean_single[7:-1] + \
                 -1/560 * position_mean_single[8:]) / (1.0/cfg.frame_rate)**2
            for i_ang in range(nAngles):
                index_joint1 = joint_order.index(angle_comb_list[i_ang][0])
                index_joint2 = joint_order.index(angle_comb_list[i_ang][1])
                index_joint3 = joint_order.index(angle_comb_list[i_ang][2])  
                vec1 = joints3d_fit[:, index_joint2] - joints3d_fit[:, index_joint1]
                vec2 = joints3d_fit[:, index_joint3] - joints3d_fit[:, index_joint2]
                vec1 = vec1 / np.sqrt(np.sum(vec1**2, 1))[:, None]
                vec2 = vec2 / np.sqrt(np.sum(vec2**2, 1))[:, None]
                ang = np.arccos(np.einsum('ni,ni->n', vec1, vec2)) * 180.0/np.pi
                
#                 # angle between bone and walking direction
#                 vec0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
#                 vec_use = np.copy(vec1)
#                 ang = np.arccos(np.einsum('i,ni->n', vec0, vec_use)) * 180.0/np.pi
#                 # source: https://math.stackexchange.com/questions/2140504/how-to-calculate-signed-angle-between-two-vectors-in-3d
#                 n_cross = np.cross(vec0, vec_use)
#                 n_cross = n_cross / np.sqrt(np.sum(n_cross**2, 1))[:, None]
#                 sin = np.einsum('ni,ni->n', np.cross(n_cross, vec0), vec_use)
#                 mask = (sin < 0.0)
#                 ang[mask] = 360.0 - ang[mask]
#                 if np.any(mask):
#                     print('WARNING: {:07d}/{:07d} signed angles'.format(np.sum(mask), len(mask.ravel())))
                
                angle_single[:, i_ang] = np.copy(ang)
                angle_velocity_single[4:-4, i_ang] = \
                    (+1/280 * ang[:-8] + \
                     -4/105 * ang[1:-7] + \
                     +1/5 * ang[2:-6] + \
                     -4/5 * ang[3:-5] + \
#                          0 * ang[4:-4] + \
                     +4/5 * ang[5:-3] + \
                     -1/5 * ang[6:-2] + \
                     +4/105 * ang[7:-1] + \
                     -1/280 * ang[8:]) / (1.0/cfg.frame_rate)
                angle_acceleration_single[4:-4, i_ang] = \
                    (-1/560 * ang[:-8] + \
                     +8/315 * ang[1:-7] + \
                     -1/5 * ang[2:-6] + \
                     +8/5 * ang[3:-5] + \
                     -205/72 * ang[4:-4] + \
                     +8/5 * ang[5:-3] + \
                     -1/5 * ang[6:-2] + \
                     +8/315 * ang[7:-1] + \
                     -1/560 * ang[8:]) / (1.0/cfg.frame_rate)**2
            #
            position_peak.append(position_single_peak)
            velocity_peak.append(velocity_single_peak)
            acceleration_peak.append(acceleration_single_peak)
            angle_peak.append(angle_single_peak)
            angle_velocity_peak.append(angle_velocity_single_peak)
            angle_acceleration_peak.append(angle_acceleration_single_peak)
            position_mean.append(position_mean_single)
            velocity_mean.append(velocity_mean_single)
            acceleration_mean.append(acceleration_mean_single)
            position.append(position_single)
            velocity.append(velocity_single)
            acceleration.append(acceleration_single)
            angle.append(angle_single)
            angle_velocity.append(angle_velocity_single)
            angle_acceleration.append(angle_acceleration_single)

            # find jump
            angle_index_spine = np.array([0, 1, 2, 3, 4], dtype=np.int64)
            angle_index_hind_limbs = np.array([7, 8, 11, 12], dtype=np.int64)
            metric_use_indices = np.concatenate([angle_index_spine,
                                                 angle_index_hind_limbs], 0)
            metric_use = np.mean(angle[i_folder][:, metric_use_indices], 1)
            index_ini = np.argmin(metric_use)
            index_jump = np.copy(index_ini)
            #
            index_start = 0
            index_end = np.copy(index_jump)
            indices_maxima1 = max_finder(metric_use[index_start:index_end]) + index_start
            index_start = 0
            index_end = np.copy(indices_maxima1[-1])
            index125 = np.copy(index_end)
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
            index175 = np.copy(index_start)
            indices_minimia2 = min_finder(metric_use[index_start:index_end]) + index_start
            if not(len(indices_minimia2) == 0):
                index2 = indices_minimia2[0]
            else:
                index2 = nT_single - 1
            #
            jump_indices[i_folder] = np.array([index1, index_jump, index2], dtype=np.int64)
            print('jump indices:\t{:03d} {:03d} {:03d} {:03d} {:03d}'.format(index1, index125, index_jump, index175, index2))
            print('jump frames:\t{:06d} {:06d} {:06d} {:06d} {:06d}'.format(index1+cfg.index_frame_start,
                                                                            index125+cfg.index_frame_start,
                                                                            index_jump+cfg.index_frame_start,
                                                                            index175+cfg.index_frame_start,
                                                                            index2+cfg.index_frame_start))
            print('jump time:\t{:03d}'.format(index2 - index1))

            # calculate jump distance
            indices_paws = np.array([joint_order.index('joint_ankle_left'),
                                     joint_order.index('joint_paw_hind_left'),
                                     joint_order.index('joint_toe_left_002'),
                                     joint_order.index('joint_ankle_right'),
                                     joint_order.index('joint_paw_hind_right'),
                                     joint_order.index('joint_toe_right_002'),], dtype=np.int64)
            diff = np.mean(position_single_peak[index2, indices_paws, :2], 0) - np.mean(position_single_peak[index1, indices_paws, :2], 0)
            dist = np.sqrt(np.sum(diff**2))
            jump_distances[i_folder] = dist
            print('jump distance:\t{:08f}'.format(dist))
    
            date = folder_list[i_folder].split('_')[1]
            gap_edges = np.load('gap_{:s}__edges.npz'.format(date), allow_pickle=True)['arr_0'].item()
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
            #
            edge_left_len = abs(edge_left[0, 1] - edge_left[1, 1])
            gap_y = gap_y + edge_left_len
            edge_right_len = abs(edge_right[0, 1] - edge_right[1, 1])
            gap_y = gap_y + edge_right_len
            gap_z = gap_z + abs(edge_left[0, 2])
            gap_z = gap_z + abs(edge_left[1, 2])
            gap_z = gap_z + abs(edge_right[0, 2])
            gap_z = gap_z + abs(edge_right[1, 2])
            #
            # PLOT
            cmap = plt.cm.cividis
            pos = np.copy(np.mean(position_peak[-1], 1))
            dist_left = np.sqrt(np.sum((pos - center_left[None, :])**2, 1))
            dist_right = np.sqrt(np.sum((pos - center_right[None, :])**2, 1))
            if (dist_left[index1] < dist_right[index1]):
                pos[:, 0] = pos[:, 0] - center_left[None, 0]
                pos[:, 1] = pos[:, 1] - center_left[None, 1]
                pos[:, 1] = pos[:, 1] - center_left[None, 1]
            else:
                pos[:, 0] = pos[:, 0] - center_right[None, 0]
                pos[:, 1] = pos[:, 1] - center_right[None, 1]
            if (pos[-1, 0] >= 0.0):
                mirror = False
            else:
                mirror = True
            if mirror:   
                pos[:, 0] = -pos[:, 0]
                pos[:, 1] = -pos[:, 1]
            ax_xz.plot(pos[:, 0], pos[:, 2],
                       color=cmap(i_folder/(nFolders-1)),
                       linestyle='-', marker='',
                       linewidth=linewidth, alpha=0.5, zorder=1)
            ax_xy.plot(pos[:, 0], pos[:, 1],
                       color=cmap(i_folder/(nFolders-1)),
                       linestyle='-', marker='',
                       linewidth=linewidth, alpha=0.5, zorder=1)
    #
    gap_y = gap_y / (2.0 * nFolders)
    gap_z = gap_z / (4.0 * nFolders)
    edge_x_start = -30.0
    edge_z_end = gap_z - 0.5
    ax_xz.plot(np.array([edge_x_start, 0.0, 0.0, edge_x_start], dtype=np.float64),
               np.array([gap_z, gap_z, edge_z_end, edge_z_end], dtype=np.float64),
               linestyle='-', marker='',
               linewidth=linewidth, 
               alpha=1.0, zorder=0, color='darkgray')
    ax_xy.plot(np.array([edge_x_start, 0.0, 0.0, edge_x_start], dtype=np.float64),
               np.array([gap_y/2.0, gap_y/2.0, -gap_y/2.0, -gap_y/2.0], dtype=np.float64),
               linestyle='-', marker='',
               linewidth=linewidth, 
               alpha=1.0, zorder=0, color='darkgray')
    #
    ax_xz.set_xlim([-35.0, 52.5])
    ax_xz.set_ylim([-1.0, 6.0])
    cfg_plt.plot_coord_ax(ax_xz, '10 cm', '1 cm', 10.0, 1.0)
    ax_xy.set_xlim([-35.0, 52.5])
    ax_xy.set_ylim([-15, 15.0])
    cfg_plt.plot_coord_ax(ax_xy, '10 cm', '5 cm', 10.0, 5.0)
    #
    # 3d
    z_all = torch.from_numpy(np.zeros_like(mu_uks[0])[None, :])
    _, _, joints3d = model.fcn_emission_free(z_all, args)
    joints3d_fit = joints3d[0].cpu().numpy()
    #
    cmap = plt.cm.viridis
    position_norm = np.array(position, dtype=np.float64)
    position = np.array(position_peak, dtype=np.float64)
    jump_pos1 = abs(position[range(nFolders), jump_indices[:, 1]] - \
                    position[range(nFolders), jump_indices[:, 0]])
    jump_pos1 = np.mean(jump_pos1, (0, 1))
    jump_pos2 = abs(position[range(nFolders), jump_indices[:, 2]] - \
                    position[range(nFolders), jump_indices[:, 0]])
    jump_pos2 = np.mean(jump_pos2, (0, 1))
    jump_dist_x1 = jump_pos1[0] * 2.0
    jump_dist_z1 = jump_pos1[2] * 2.0
    jump_dist_x2 = jump_pos2[0] * 2.0
    jump_dist_z2 = jump_pos2[2] * 2.0
    for i_index in list([0, 1, 2]):
        if (i_index == 0):
            skeleton_pos = np.mean(position_norm[range(nFolders), jump_indices[:, 0]], 0)            
            skeleton_pos_mean = np.mean(skeleton_pos, 0)
            skeleton_pos[:, 2] = skeleton_pos[:, 2] + 3.0
            skeleton_pos_start = np.copy(skeleton_pos)
        elif (i_index == 1):
            skeleton_pos = np.mean(position_norm[range(nFolders), jump_indices[:, 1]], 0)
            skeleton_pos = skeleton_pos - np.mean(skeleton_pos, 0)[None, :]
            skeleton_pos[:, 0] = skeleton_pos[:, 0] + jump_dist_x1
            skeleton_pos[:, 2] = skeleton_pos[:, 2] + jump_dist_z1
            skeleton_pos_mid = np.copy(skeleton_pos)
        elif (i_index == 2):
            skeleton_pos = np.mean(position_norm[range(nFolders), jump_indices[:, 2]], 0)
            skeleton_pos = skeleton_pos - np.mean(skeleton_pos, 0)[None, :]
            skeleton_pos[:, 0] = skeleton_pos[:, 0] + jump_dist_x2
            skeleton_pos[:, 2] = skeleton_pos[:, 2] + jump_dist_z2
            skeleton_pos_end = np.copy(skeleton_pos)
        #
        if (i_index == 0):
            joint_colors_cmap1 = plt.cm.Dark2 # dist
            joint_colors_cmap2 = plt.cm.Set2 # auto
            joint_colors_list = list(['joint_head_001', 'joint_spine_005', 'joint_spine_004', 'joint_spine_003',
                                      'joint_elbow_left', 'joint_wrist_left', 'joint_knee_left', 'joint_ankle_left',
                                      'joint_elbow_right', 'joint_wrist_right', 'joint_knee_right', 'joint_ankle_right'])
            joint_colors = list([joint_colors_cmap1(0.0), joint_colors_cmap1(float(1/7)), joint_colors_cmap1(float(2/7)), joint_colors_cmap1(float(3/7))]) +\
                           list([joint_colors_cmap2(i/7) for i in range(8)])
            color_bone = 'black'
            #
            zorder_offset = 6 * i_index
            i_bone = 0
            joint_index_start = skeleton_edges[i_bone, 0]
            joint_index_end = skeleton_edges[i_bone, 1]
            joint_start = skeleton_pos[joint_index_start]
            joint_name = joint_order[joint_index_start]
            joint_name_split = joint_name.split('_')
            zorder_joint = 4 + zorder_offset # first joint is always center-joint
            markersize_add_constant = 5
            if (joint_name in joint_colors_list):
                color_joint = joint_colors[joint_colors_list.index(joint_name)]
                markersize_add = markersize_add_constant
                zorder_add = 10
            else:
                color_joint = 'black'
                markersize_add = 0
                zorder_add = 0
            ax3d.plot([joint_start[0]], [joint_start[1]], [joint_start[2]],
                      linestyle='', marker='.', markersize=2.5+markersize_add, markeredgewidth=0.5, color=color_joint, zorder=zorder_joint+zorder_add)
            for i_bone in range(nJoints-1):
                joint_index_start = skeleton_edges[i_bone, 0]
                joint_index_end = skeleton_edges[i_bone, 1]
                joint_start = skeleton_pos[joint_index_start]
                joint_end = skeleton_pos[joint_index_end]
                vec = np.stack([joint_start, joint_end], 0)
                joint_name = joint_order[joint_index_end]
                joint_name_split = joint_name.split('_')
                if ('left' in joint_name_split):
                    zorder_bone = 1 + zorder_offset
                    zorder_joint = 2 + zorder_offset
                elif ('right' in joint_name_split):
                    zorder_bone = 5 + zorder_offset
                    zorder_joint = 6 + zorder_offset
                else:
                    zorder_bone = 3 + zorder_offset
                    zorder_joint = 4 + zorder_offset
                ax3d.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                        linestyle='-', marker=None, linewidth=1.0, color=color_bone, zorder=zorder_bone)
                if (joint_name in joint_colors_list):
                    joint_name_use = joint_name
                    
                    # if image gets inverted:
                    joint_name_split = joint_name_use.split('_')
                    if ('left' in joint_name_split):
                        index = joint_name_split.index('left')
                        joint_name_split[index] = 'right'
                        joint_name_use = '_'.join(joint_name_split)
                    elif ('right' in joint_name_split):
                        index = joint_name_split.index('right')
                        joint_name_split[index] = 'left'
                        joint_name_use = '_'.join(joint_name_split)
                    
                    color_joint = joint_colors[joint_colors_list.index(joint_name_use)]
                    markersize_add = markersize_add_constant
                    zorder_add = 10
                else:
                    color_joint = 'black'
                    markersize_add = 0
                    zorder_add = 0
                ax3d.plot([joint_end[0]], [joint_end[1]], [joint_end[2]],
                        linestyle='', marker='.', markersize=2.5+markersize_add, markeredgewidth=0.5, color=color_joint, zorder=zorder_joint+zorder_add)
    dxyz = 10.5
    xyz_center = np.mean(skeleton_pos_start, 0)
    ax3d.set_xlim([xyz_center[0]-dxyz, xyz_center[0]+dxyz])
    ax3d.set_ylim([xyz_center[1]-dxyz, xyz_center[1]+dxyz])
    ax3d.set_zlim([xyz_center[2]-dxyz, xyz_center[2]+dxyz])
    ax3d.set_axis_off()
    ax3d.view_init(azim=60, elev=10) # when inverted
        
    # EDGES
    color = 'darkgray'
    gap_dx1 = 3.0
    gap_dx2 = -2.0
    gap_dy = 15.0
    #
    gap_edges_3d = dict()
    lower_left = np.mean(skeleton_pos_start, 0)
    lower_left[0] = lower_left[0] + gap_dx1
    lower_left[1] = lower_left[1] - gap_dy
    lower_left[2] = 0.0
    upper_left = np.mean(skeleton_pos_start, 0)
    upper_left[0] = upper_left[0] + gap_dx1
    upper_left[1] = upper_left[1] + gap_dy
    upper_left[2] = 0.0
    lower_right = np.mean(skeleton_pos_end, 0)
    lower_right[0] = lower_right[0] + gap_dx2
    lower_right[1] = lower_right[1] - gap_dy
    lower_right[2] = jump_dist_z2
    upper_right = np.mean(skeleton_pos_end, 0)
    upper_right[0] = upper_right[0] + gap_dx2
    upper_right[1] = upper_right[1] + gap_dy
    upper_right[2] = jump_dist_z2
    gap_edges_3d['lower_left'] = lower_left
    gap_edges_3d['upper_left'] = upper_left
    gap_edges_3d['lower_right'] = lower_right
    gap_edges_3d['upper_right'] = upper_right
    edge_left = np.array([gap_edges_3d['lower_left'], gap_edges_3d['upper_left']], dtype=np.float64)
    edge_right = np.array([gap_edges_3d['lower_right'], gap_edges_3d['upper_right']], dtype=np.float64)
    edge_left[:, 0] = np.mean(edge_left[:, 0], 0)
    edge_right[:, 0] = np.mean(edge_right[:, 0], 0)
    edge_left[:, 2] = np.mean(edge_left[:, 2], 0)
    edge_right[:, 2] = np.mean(edge_right[:, 2], 0)
    x_inf0 = -22.5
    x_inf1 = 17.5
    z_inf = -2.5
    for i_edge in range(1):
        if (i_edge == 0):
            edge_use = edge_left
            x_inf = x_inf0
        else:
            edge_use = edge_right
            x_inf = x_inf1
        x = np.array([edge_use[0, 0]+x_inf, edge_use[0, 0], edge_use[1, 0], edge_use[0, 0]+x_inf], dtype=np.float64)
        y = np.array([edge_use[0, 1], edge_use[0, 1], edge_use[1, 1], edge_use[1, 1]], dtype=np.float64)
        z = np.array([edge_use[0, 2], edge_use[0, 2], edge_use[1, 2], edge_use[1, 2]], dtype=np.float64)
        ax3d.plot(x, y, z, color=color, zorder=0)
        ax3d.plot(x, y, z+z_inf, color=color, zorder=0)
        #
        for i_edge2 in range(2):
            x = np.array([edge_use[i_edge2, 0], edge_use[i_edge2, 0]], dtype=np.float64)
            y = np.array([edge_use[i_edge2, 1], edge_use[i_edge2, 1]], dtype=np.float64)
            z = np.array([edge_use[i_edge2, 2], edge_use[i_edge2, 2]+z_inf], dtype=np.float64)
            ax3d.plot(x, y, z, color=color, zorder=0)
    fig3d.canvas.draw()

    fig_xz.canvas.draw()
    fig_xy.canvas.draw()
    plt.pause(2**-10)
    
    if save:
        fig3d.savefig(folder_save+'/skeleton_3d_joints_colored2.svg',
#                          bbox_inches="tight",
                         dpi=300,
                         transparent=True,
                         format='svg',
                         pad_inches=0)
    plt.show()