#!/usr/bin/env python3

import importlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
left_margin  = 0.35 # inch
right_margin = 0.05 # inch
bottom_margin = 0.35 # inch
top_margin = 0.05 # inch
between_margin_h = 0.1 # inch

fontsize = 6
linewidth = 0.5
fontname = "Arial"
markersize = 2

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

cmap = plt.cm.inferno
if (nFolders > 1):
    colors = list([cmap(i/(nFolders-1)) for i in range(nFolders)])
else:
    colors = list([cmap(0.0)])

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
    nT = 200

    # figure for the paw sketch
    figsize_x = np.round(mm_in_inch * 88 * 1/1, decimals=2)
    figsize_y = np.round(mm_in_inch * 88 * 1/2, decimals=2)
    fig3 = plt.figure(3, figsize=(figsize_x, figsize_y))
    fig3.canvas.manager.window.move(0, 0)
    fig3.clear()
    fig3_ax = fig3.add_axes([0, 0, 1, 1])
    fig3_ax.clear()
    fig3_ax.set_aspect(1)
    
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
    
    nBones = 28 # hard coded
    nJoints = nBones + 1

    position_norm = np.full((nFolders, nT, nJoints, 3), np.nan, dtype=np.float64)
    velocity_norm = np.full((nFolders, nT, nJoints, 3), np.nan, dtype=np.float64)
    position = np.full((nFolders, nT, nJoints, 3), np.nan, dtype=np.float64)
    velocity = np.full((nFolders, nT, nJoints, 3), np.nan, dtype=np.float64)
    acceleration = np.full((nFolders, nT, nJoints, 3), np.nan, dtype=np.float64)
    position_mean = np.full((nFolders, nT, 3), np.nan, dtype=np.float64)
    velocity_mean = np.full((nFolders, nT, 3), np.nan, dtype=np.float64)
    acceleration_mean = np.full((nFolders, nT, 3), np.nan, dtype=np.float64)
    angle = np.full((nFolders, nT, nAngles), np.nan, dtype=np.float64)
    angle_velocity = np.full((nFolders, nT, nAngles), np.nan, dtype=np.float64)
    angle_acceleration = np.full((nFolders, nT, nAngles), np.nan, dtype=np.float64)
    angle_norm = np.full((nFolders, nT, nAngles), np.nan, dtype=np.float64)
    index_pose1 = np.zeros(nFolders, dtype=np.int64)
    index_pose2 = np.zeros(nFolders, dtype=np.int64)
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
        file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_dlc_{:06d}_{:06d}.npy'.format(cfg.index_frame_start, cfg.index_frame_end)
           
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
                print(folder)
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

#             # normalize skeletons of different animals
#             x_joints = list([#'joint_head_001',
#                              'joint_spine_001',
#                              'joint_spine_002',
#                              'joint_spine_003',
#                              'joint_spine_004',
#                              'joint_spine_005',
#                              'joint_tail_001',
#                              'joint_tail_002',
#                              'joint_tail_003',
#                              'joint_tail_004',
#                              'joint_tail_005',])
#             y_joints1 = list(['joint_shoulder_left',
#                               'joint_elbow_left',
#                               'joint_wrist_left',
#                               'joint_finger_left_002',])
#             y_joints2 = list(['joint_hip_left',
#                               'joint_knee_left',
#                               'joint_ankle_left',
#                               'joint_paw_hind_left',
#                               'joint_toe_left_002',])
#             bone_lengths_index = args['model']['bone_lengths_index'].cpu().numpy()
#             bone_lengths_sym = mu_ini[:args['nPara_bones']][bone_lengths_index]
#             bone_lengths_x = 0.0
#             bone_lengths_limb_front = 0.0
#             bone_lengths_limb_hind = 0.0
#             for i_bone in range(nJoints-1):
#                 index_bone_end = skeleton_edges[i_bone, 1]
#                 joint_name = joint_order[index_bone_end]
#                 if joint_name in x_joints:
#                     bone_lengths_x = bone_lengths_x + bone_lengths_sym[i_bone]
#                 elif joint_name in y_joints1:
#                     bone_lengths_limb_front = bone_lengths_limb_front + bone_lengths_sym[i_bone]
#                 elif joint_name in y_joints2:
#                     bone_lengths_limb_hind = bone_lengths_limb_hind + bone_lengths_sym[i_bone]
#             #
#             mask = np.ones_like(bone_lengths_index, dtype=bool)
#             for i_bone in range(nJoints-1):
#                 index0 = bone_lengths_index[i_bone]
#                 if mask[index0]:
# #                     index_bone_end = skeleton_edges[i_bone, 1]
# #                     joint_name = joint_order[index_bone_end]
# #                     value0 = mu_ini[index0]
# #                     value1 = bone_lengths_sym[i_bone]
#                     mu_ini[index0] = mu_ini[index0] / bone_lengths_x
#                     mask[index0] = False
            
            if (nSamples > 0):
                mu_t = torch.from_numpy(mu_uks)
                var_t = torch.from_numpy(var_uks)
                distribution = torch.distributions.MultivariateNormal(loc=mu_t,
                                                                      scale_tril=kalman.cholesky_save(var_t))
                z_samples = distribution.sample((nSamples,))
                z_all = torch.cat([mu_t[None, :], z_samples], 0)
            else:
                z_all = torch.from_numpy(mu_uks)
            markers2d, markers3d, joints3d = model.fcn_emission_free(z_all, args)
            markers2d_fit = markers2d.cpu().numpy().reshape(nT, nCameras, nMarkers, 2)
            markers2d_fit[:, :, :, 0] = (markers2d_fit[:, :, :, 0] + 1.0) * (cfg.normalize_camera_sensor_x * 0.5)
            markers2d_fit[:, :, :, 1] = (markers2d_fit[:, :, :, 1] + 1.0) * (cfg.normalize_camera_sensor_y * 0.5)
            markers3d_fit = markers3d.cpu().numpy()
            joints3d_fit = joints3d.cpu().numpy()
            #
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
            joints3d_fit_norm = np.copy(joints3d_fit)
            joints3d_fit_norm = joints3d_fit_norm - origin[:, None, :]
            joints3d_fit_norm_xy = np.einsum('nij,nmj->nmi', R, joints3d_fit_norm[:, :, :2])
            joints3d_fit_norm[:, :, :2] = np.copy(joints3d_fit_norm_xy)
            #
            position_norm[i_folder] = np.copy(joints3d_fit_norm)
            velocity_norm[i_folder, 4:-4] = \
                (+1.0/280.0 * position_norm[i_folder][:-8] + \
                 -4.0/105.0 * position_norm[i_folder][1:-7] + \
                 +1.0/5.0 * position_norm[i_folder][2:-6] + \
                 -4.0/5.0 * position_norm[i_folder][3:-5] + \
#                  0.0 * position_norm[i_folder][4:-4] + \
                 +4.0/5.0 * position_norm[i_folder][5:-3] + \
                 -1.0/5.0 * position_norm[i_folder][6:-2] + \
                 +4.0/105.0 * position_norm[i_folder][7:-1] + \
                 -1.0/280.0 * position_norm[i_folder][8:]) / (1.0/cfg.frame_rate)
            position[i_folder] = np.copy(joints3d_fit)
            velocity[i_folder, 4:-4] = \
                (+1.0/280.0 * position[i_folder][:-8] + \
                 -4.0/105.0 * position[i_folder][1:-7] + \
                 +1.0/5.0 * position[i_folder][2:-6] + \
                 -4.0/5.0 * position[i_folder][3:-5] + \
#                  0.0 * position[i_folder][4:-4] + \
                 +4.0/5.0 * position[i_folder][5:-3] + \
                 -1.0/5.0 * position[i_folder][6:-2] + \
                 +4.0/105.0 * position[i_folder][7:-1] + \
                 -1.0/280.0 * position[i_folder][8:]) / (1.0/cfg.frame_rate)
            acceleration[i_folder, 4:-4] = \
                (-1.0/560.0 * position[i_folder][:-8] + \
                 +8.0/315.0 * position[i_folder][1:-7] + \
                 -1.0/5.0 * position[i_folder][2:-6] + \
                 +8.0/5.0 * position[i_folder][3:-5] + \
                 -205.0/72.0 * position[i_folder][4:-4] + \
                 +8.0/5.0 * position[i_folder][5:-3] + \
                 -1.0/5.0 * position[i_folder][6:-2] + \
                 +8.0/315.0 * position[i_folder][7:-1] + \
                 -1.0/560.0 * position[i_folder][8:]) / (1.0/cfg.frame_rate)**2
            position_mean[i_folder] = np.mean(joints3d_fit, 1)
            velocity_mean[i_folder, 4:-4] = \
                (+1.0/280.0 * position_mean[i_folder][:-8] + \
                 -4.0/105.0 * position_mean[i_folder][1:-7] + \
                 +1.0/5.0 * position_mean[i_folder][2:-6] + \
                 -4.0/5.0 * position_mean[i_folder][3:-5] + \
#                  0.0 * position_mean[i_folder][4:-4] + \
                 +4.0/5.0 * position_mean[i_folder][5:-3] + \
                 -1.0/5.0 * position_mean[i_folder][6:-2] + \
                 +4.0/105.0 * position_mean[i_folder][7:-1] + \
                 -1.0/280.0 * position_mean[i_folder][8:]) / (1.0/cfg.frame_rate)
            acceleration_mean[i_folder, 4:-4] = \
                (-1.0/560.0 * position_mean[i_folder][:-8] + \
                 +8.0/315.0 * position_mean[i_folder][1:-7] + \
                 -1.0/5.0 * position_mean[i_folder][2:-6] + \
                 +8.0/5.0 * position_mean[i_folder][3:-5] + \
                 -205.0/72.0 * position_mean[i_folder][4:-4] + \
                 +8.0/5.0 * position_mean[i_folder][5:-3] + \
                 -1.0/5.0 * position_mean[i_folder][6:-2] + \
                 +8.0/315.0 * position_mean[i_folder][7:-1] + \
                 -1.0/560.0 * position_mean[i_folder][8:]) / (1.0/cfg.frame_rate)**2
            for i_ang in range(nAngles):
                index_joint1 = joint_order.index(angle_comb_list[i_ang][0])
                index_joint2 = joint_order.index(angle_comb_list[i_ang][1])
                index_joint3 = joint_order.index(angle_comb_list[i_ang][2])  
                vec1 = joints3d_fit[:, index_joint2] - joints3d_fit[:, index_joint1]
                vec2 = joints3d_fit[:, index_joint3] - joints3d_fit[:, index_joint2]
                vec1 = vec1 / np.sqrt(np.sum(vec1**2, 1))[:, None]
                vec2 = vec2 / np.sqrt(np.sum(vec2**2, 1))[:, None]
                ang = np.arccos(np.einsum('ni,ni->n', vec1, vec2)) * 180.0/np.pi
                angle[i_folder, :, i_ang] = np.copy(ang)
                
#                 # angle between bone and walking direction
#                 vec1 = joints3d_fit_norm[:, index_joint2] - joints3d_fit_norm[:, index_joint1]
#                 vec2 = joints3d_fit_norm[:, index_joint3] - joints3d_fit_norm[:, index_joint2]
#                 vec1 = vec1 / np.sqrt(np.sum(vec1**2, 1))[:, None]
#                 vec2 = vec2 / np.sqrt(np.sum(vec2**2, 1))[:, None]
#                 vec0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
#                 vec_use = np.copy(vec1)
#                 ang = np.arccos(np.einsum('i,ni->n', vec0, vec_use)) * 180.0/np.pi
#                 # source: https://math.stackexchange.com/questions/2140504/how-to-calculate-signed-angle-between-two-vectors-in-3d
#                 n_cross = np.cross(vec0, vec_use)
#                 n_cross = n_cross / np.sqrt(np.sum(n_cross**2, 1))[:, None]
#                 sin = np.einsum('ni,ni->n', np.cross(n_cross, vec0), vec_use)
#                 mask = (sin < 0.0)
#                 ang[mask] = 360.0 - ang[mask]
#                 angle[i_folder, :, i_ang] = np.copy(ang)
#                 if np.any(mask):
#                     print('WARNING: {:07d}/{:07d} signed angles'.format(np.sum(mask), len(mask.ravel())))
                
#                 index_joint1 = joint_order.index(angle_comb_list[i_ang][0])
#                 index_joint2 = joint_order.index(angle_comb_list[i_ang][1])
#                 index_joint3 = joint_order.index(angle_comb_list[i_ang][2])  
#                 vec1 = joints3d_fit_norm[:, index_joint2] - joints3d_fit_norm[:, index_joint1]
#                 vec2 = joints3d_fit_norm[:, index_joint3] - joints3d_fit_norm[:, index_joint2]
#                 vec1 = vec1 / np.sqrt(np.sum(vec1**2, 1))[:, None]
#                 vec2 = vec2 / np.sqrt(np.sum(vec2**2, 1))[:, None]
#                 ang = np.arccos(np.einsum('ni,ni->n', vec1, vec2)) * 180.0/np.pi
#                 angle_norm[i_folder, :, i_ang] = np.copy(ang)        
        
                angle_velocity[i_folder, 4:-4, i_ang] = \
                    (+1.0/280.0 * ang[:-8] + \
                     -4.0/105.0 * ang[1:-7] + \
                     +1.0/5.0 * ang[2:-6] + \
                     -4.0/5.0 * ang[3:-5] + \
#                      0.0 * ang[4:-4] + \
                     +4.0/5.0 * ang[5:-3] + \
                     -1.0/5.0 * ang[6:-2] + \
                     +4.0/105.0 * ang[7:-1] + \
                     -1.0/280.0 * ang[8:]) / (1.0/cfg.frame_rate)
                angle_acceleration[i_folder, 4:-4, i_ang] = \
                    (-1.0/560.0 * ang[:-8] + \
                     +8.0/315.0 * ang[1:-7] + \
                     -1.0/5.0 * ang[2:-6] + \
                     +8.0/5.0 * ang[3:-5] + \
                     -205.0/72.0 * ang[4:-4] + \
                     +8.0/5.0 * ang[5:-3] + \
                     -1.0/5.0 * ang[6:-2] + \
                     +8.0/315.0 * ang[7:-1] + \
                     -1.0/560.0 * ang[8:]) / (1.0/cfg.frame_rate)**2
            #     
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
            indices_minima1 = min_finder(metric_use[index_start:index_end]) + index_start
            if not(len(indices_minima1) == 0):
                index1 = indices_minima1[-1]
            else:
                index1 = 0
            #
            index_start = np.copy(index_jump)
            index_end = nT
            indices_maxima2 = max_finder(metric_use[index_start:index_end]) + index_start
            index_start = np.copy(indices_maxima2[0])
            index_end = nT
            indices_minimia2 = min_finder(metric_use[index_start:index_end]) + index_start
            if not(len(indices_minimia2) == 0):
                index2 = indices_minimia2[0]
            else:
                index2 = nT - 1
            index_pose1[i_folder] = np.copy(index1)
            index_pose2[i_folder] = np.copy(index2)
            print(index1, index_jump, index2)
            print(cfg.index_frame_start + index1, cfg.index_frame_start + index_jump, cfg.index_frame_start + index2)

    # sort arrays
    joint_order_middle = list()
    joint_order_left = list()
    joint_order_right = list()
    joint_order_middle_index = list()
    joint_order_left_index = list()
    joint_order_right_index = list()
    limb_order = list(['shoulder', 'elbow', 'wrist', 'finger',
                       'hip', 'knee', 'ankle', 'paw', 'toe'])
    for i_joint in range(nJoints):
        joint_name = joint_order[i_joint]
        joint_name_split = joint_name.split('_')
        if ('left' in joint_name_split):
            index = limb_order.index(joint_name_split[1])
            joint_order_left.insert(index, joint_name)
            joint_order_left_index.insert(index, i_joint)
        elif ('right' in joint_name_split):
            index = limb_order.index(joint_name_split[1])
            joint_order_right.insert(index, joint_name)
            joint_order_right_index.insert(index, i_joint)
        else:
            joint_order_middle.append(joint_name)
            joint_order_middle_index.append(i_joint)
    joint_order = joint_order_middle + joint_order_left + joint_order_right
    joint_order_sort_index = joint_order_middle_index + joint_order_left_index + joint_order_right_index
    joint_order_sort_index = np.array(joint_order_sort_index, dtype=np.int64)
    #
    position = position[:, :, joint_order_sort_index]
    velocity = velocity[:, :, joint_order_sort_index]
    acceleration = acceleration[:, :, joint_order_sort_index]

    dummy_x1 = np.full(nFolders, np.nan, dtype=np.float64)
    dummy_x2 = np.full(nFolders, np.nan, dtype=np.float64)
    dummy_y1 = np.full(nFolders, np.nan, dtype=np.float64)
    dummy_y2 = np.full(nFolders, np.nan, dtype=np.float64)
    for i_folder in np.arange(0, nFolders, 1, dtype=np.int64):
        folder = folder_recon+folder_list[i_folder]+add2folder
        file_save_dict = folder+'/'+'save_dict.npy'
        if os.path.isfile(file_save_dict):
            index1 = index_pose1[i_folder]
            index2 = index_pose2[i_folder]
            #
            index_y = abs(velocity_mean[i_folder, :, 0])
            #
            joint_names_list = list([['joint_ankle_left', 'joint_paw_hind_left', 'joint_toe_left_002'],
                                     ['joint_ankle_right', 'joint_paw_hind_right', 'joint_toe_left_002']])
            x_y1 = np.zeros(nT, dtype=np.float64)
            x_y2 = np.zeros(nT, dtype=np.float64)
            for i_joint in range(3):
                index_joint1 = joint_order.index(joint_names_list[0][i_joint])
                index_joint2 = joint_order.index(joint_names_list[1][i_joint])
                x_y1 = x_y1 + np.sqrt(np.sum((position[i_folder, :, index_joint1, :2] - position[i_folder, index1, index_joint1, :2])**2, 1)) / 3.0 # xy-distance
                x_y2 = x_y2 + np.sqrt(np.sum((position[i_folder, :, index_joint2, :2] - position[i_folder, index1, index_joint2, :2])**2, 1)) / 3.0 # xy-distance
            y_y1 = np.copy(np.abs(angle[i_folder, :, 2]))
            y_y2 = np.copy(y_y1)

            # scatter plot
            dIndex = 0
            dummy_x1[i_folder] = np.mean(x_y1[index2-dIndex:index2+dIndex+1])
            dummy_x2[i_folder] = np.mean(x_y2[index2-dIndex:index2+dIndex+1])
            dummy_y1[i_folder] = np.mean(y_y1[index1-dIndex:index1+dIndex+1])
            dummy_y2[i_folder] = np.mean(y_y2[index1-dIndex:index1+dIndex+1])            

            # x-axis plot
            x = np.arange(0, nT, 1, dtype=np.float64)
            x = (x - index1) / (index2  - index1)

            # to have white margin before y-axis
            x_y1_plot = np.copy(x_y1)
            x_y2_plot = np.copy(x_y2)
            dx = 0.025
            mask = np.logical_or((x < -0.5+dx), (x_y1_plot > 1.5-dx))
            x_y1_plot[mask] = np.nan
            mask = np.logical_or((x < -0.5+dx), (x > 1.5-dx))
            x_y2_plot[mask] = np.nan
            
            # plot 2d
            file_x_ini = folder+'/'+'x_ini.npy'
            x_ini = np.load(file_x_ini, allow_pickle=True)
            bone_lengths_index = args['model']['bone_lengths_index'].cpu().numpy()
            bone_lengths = x_ini[:nBones]
            bone_lengths = bone_lengths[bone_lengths_index]
            bone_ends_name = list(np.array(joint_order)[skeleton_edges[:, 1]])
            paw_hind_scale_x = 0.5 * (6.405049925951728 + 8.46078506255709) # hard coded (mean of calibration results for 20200205 and 20200207)
            paw_hind_scale_x = paw_hind_scale_x * 0.2
            paw_hind_scale_y = paw_hind_scale_x * 0.5
            paw_hind = get_paw_hind(paw_hind_scale_x, paw_hind_scale_y)
            #
            paws_names_list1 = list(['joint_ankle_left', 'joint_ankle_right'])
            paws_names_list2 = list(['joint_paw_hind_left', 'joint_paw_hind_right'])
            paws_names_list3 = list(['joint_toe_left_002', 'joint_toe_right_002'])
            for i_paw in range(2):
                joint_index1 = joint_order.index(paws_names_list1[i_paw])
                joint_index2 = joint_order.index(paws_names_list2[i_paw])
                joint_index3 = joint_order.index(paws_names_list3[i_paw])
                position_use = np.copy(position[i_folder])
                velocity_use = np.copy(velocity[i_folder])
                position_paw = 1.0/3.0 * (position[i_folder, index1, joint_index1, 0] + \
                                          position[i_folder, index1, joint_index2, 0] + \
                                          position[i_folder, index1, joint_index3, 0])
                position_use[:, :, 0] = position_use[:, :, 0] - position_paw
                if ((position_use[index2, joint_index1, 0] < 0.0) and \
                    (position_use[index2, joint_index2, 0] < 0.0) and \
                    (position_use[index2, joint_index3, 0] < 0.0)):
                    position_use[:, :, 0] = -position_use[:, :, 0]
                    position_use[:, :, 1] = -position_use[:, :, 1]
                    velocity_use[:, :, 0] = -velocity_use[:, :, 0]
                    velocity_use[:, :, 1] = -velocity_use[:, :, 1]
                velo_all = np.full(2*nFolders, np.nan, dtype=np.float64)
               
                color_coding_min = 18
                color_coding_max = 70
                if (i_paw == 0):
                    color_coding = np.copy(dummy_y1[i_folder])
                elif (i_paw == 1):
                    color_coding = np.copy(dummy_y2[i_folder])
                color_coding = color_coding - color_coding_min
                color_coding = color_coding / (color_coding_max - color_coding_min)
                colors_paws = cmap(color_coding)

                joint_index1 = joint_order.index(paws_names_list1[i_paw])
                joint_index2 = joint_order.index(paws_names_list2[i_paw])
                # before jump
                paw_direc1 = position_use[index1, joint_index2, :2] - position_use[index1, joint_index1, :2]
                paw_direc1 = paw_direc1 / np.sqrt(np.sum(paw_direc1**2))
                # after jump
                paw_direc2 = position_use[index2, joint_index2, :2] - position_use[index2, joint_index1, :2]
                paw_direc2 = paw_direc2 / np.sqrt(np.sum(paw_direc2**2))
        
                # before jump
                paw_pos_xy = position_use[index1, joint_index1, :2]
                alpha = np.arctan2(paw_direc1[1], paw_direc1[0])
                R = np.array([[np.cos(alpha), -np.sin(alpha)],
                              [np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
                paw_plot = np.einsum('ij,nj->ni', R, paw_hind) + paw_pos_xy[None, :]
                paw_poly = plt.Polygon(paw_plot, color=colors_paws, zorder=0, alpha=0.5, fill=True, linewidth=0.1)   
                fig3_ax.add_patch(paw_poly)
                # after jump
                paw_pos_xy = position_use[index2, joint_index1, :2]
                alpha = np.arctan2(paw_direc2[1], paw_direc2[0])
                R = np.array([[np.cos(alpha), -np.sin(alpha)],
                              [np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
                paw_plot = np.einsum('ij,nj->ni', R, paw_hind) + paw_pos_xy[None, :]
                paw_poly = plt.Polygon(paw_plot, color=colors_paws, zorder=0, alpha=0.5, fill=True, linewidth=0.1)   
                fig3_ax.add_patch(paw_poly)
            if (i_folder == 0):
                cfig3_ax = fig3.add_axes([2/3, 0.15, 1/4, 0.025])
                mappable2d = plt.cm.ScalarMappable(cmap=cmap)
                mappable2d.set_clim([color_coding_min, color_coding_max])
                colorbar2d = fig3.colorbar(mappable=mappable2d, ax=fig3_ax, cax=cfig3_ax, orientation='horizontal', ticks=[color_coding_min, color_coding_max])
                colorbar2d.set_label('spine angle (deg)', labelpad=-5, rotation=0,
                                     fontsize=fontsize, fontname=fontname, va='center', ha='center')
                colorbar2d.ax.tick_params(labelsize=fontsize, color='white')
                colorbar2d.outline.set_edgecolor('white')
                for tick in colorbar2d.ax.get_yticklabels():
                    tick.set_fontname(fontname)
                    tick.set_fontsize(fontsize)
                for tick in colorbar2d.ax.get_xticklabels():
                    tick.set_fontname(fontname)
                    tick.set_fontsize(fontsize)    
    
    offset_x_text = 0.75 # cm
    offset_y_text = -0.75 # cm
    offset_x = 2.5 # cm
    offset_y = -9 # cm
    len_x = 5.0 # cm
    len_y = -5.0 # cm
    fig3_ax.plot(np.array([offset_x, offset_x + len_x], dtype=np.float64),
              np.array([offset_y, offset_y], dtype=np.float64),
              linestyle='-', marker='', color='black', zorder=2, linewidth=5*linewidth)
    fig3_ax.text(offset_x + len_x/2.0, offset_y+offset_x_text, '{:0.0f} cm'.format(abs(len_x)),
              fontsize=fontsize, fontname=fontname, ha='center', va='center', rotation='horizontal')
    #
    fig3_ax.set_axis_off()
    fig3_ax.set_aspect(1)
    plt.pause(2**-10)
    fig3.canvas.draw()
    plt.pause(2**-10)
    
    if save:
        fig3.savefig(folder_save+'/gap__population_paw_sketch.svg',
                         bbox_inches="tight",
                         dpi=300,
                         transparent=True,
                         format='svg',
                         pad_inches=0)
    plt.show()
