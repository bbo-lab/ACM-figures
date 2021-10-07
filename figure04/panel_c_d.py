#!/usr/bin/env python3

import importlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
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
sys_path0 = np.copy(sys.path)

save = False

folder_save = os.path.abspath('panels')

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

fontsize = 6
linewidth = 0.25
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

# # perform analysis for a single animal
# date_index = 1
# dates_list = list(['20200205', '20200207'])
# date = dates_list[date_index]
# folder_list_use = list()
# for i_folder in range(nFolders):
#     folder = folder_list[i_folder]
#     if (date in folder.split('_')):
#         folder_list_use.append(folder)
# folder_list = folder_list_use
# nFolders = len(folder_list)

nAnimals = 2
mask_animals = np.zeros((nAnimals, nFolders), dtype=bool)
for i_folder in range(nFolders):
    folder = folder_list[i_folder] 
    if ('20200205' in folder.split('_')):
        i_animal = 0
    elif ('20200207' in folder.split('_')):
        i_animal = 1
    mask_animals[i_animal, i_folder] = True

nSamples = int(0e3)

cmap = plt.cm.viridis
if (nFolders > 1):
    colors = list([cmap(i/(nFolders-1)) for i in range(nFolders)])
else:
    colors = list([cmap(0.0)])

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
    
        file_save_dict = folder+'/save_dict.npy'
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
            if not(len(indices_maxima1) == 0):
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
            if not(len(indices_maxima2) == 0):
                index_start = np.copy(indices_maxima2[0])
            index_end = nT_single 
            indices_minimia2 = min_finder(metric_use[index_start:index_end]) + index_start
            if not(len(indices_minimia2) == 0):
                index2 = indices_minimia2[0]
            else:
                index2 = nT_single - 1
            #
            jump_indices[i_folder] = np.array([index1, index_jump, index2], dtype=np.int64)
            print('jump indices:\t{:03d} {:03d} {:03d}'.format(index1, index_jump, index2))
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
    
#             # PLOT
#             fig_test = plt.figure(123)
#             fig_test.clear()
#             ax_test1 = fig_test.add_subplot(311)
#             ax_test1.clear()
#             ax_test2 = fig_test.add_subplot(312)
#             ax_test2.clear()
#             ax_test3 = fig_test.add_subplot(313)
#             ax_test3.clear()
#             ax_test = list([ax_test1, ax_test2, ax_test3])
#             #
# #             metrtic_use = np.mean(angle[i_folder][:, index_angle_spring], 1)
# #             metrtic_use = np.mean(velocity[i_folder][:, :, :], 1)
# #             metrtic_use = np.mean(np.sqrt(np.sum(velocity[i_folder][:, :, :]**2, 2)), 1)
#             angle_index_spine = np.array([0, 1, 2, 3, 4], dtype=np.int64)
#             angle_index_front_limbs = np.array([5, 6, 9, 10], dtype=np.int64)
#             angle_index_hind_limbs = np.array([7, 8, 11, 12], dtype=np.int64)
#             metric_use_indices = np.concatenate([angle_index_spine,
#                                                  angle_index_hind_limbs], 0)
#             metrtic_use = np.mean(angle[i_folder][:, metric_use_indices], 1)
#             ax_test1.plot(range(nT_single), metrtic_use, color='black', zorder=1)
#             #
# #             color_index = 0
# #             for i_ang in range(nAngles):
# #                 angle_use = np.copy(angle[i_folder][:, i_ang])
# #                 ax_test2.plot(range(nT_single), angle_use, color=cmap(color_index/(nAngles-1)), zorder=1)
# #                 color_index = color_index + 1
# #             ax_test2.legend([angle_comb_list[i][1] for i in range(nAngles)], loc='upper right')
#             #
#             joints_origin = list(['joint_spine_002',])
#             joints_tail = list(['joint_tail_001',
#                                 'joint_tail_002',
#                                 'joint_tail_003',
#                                 'joint_tail_004',
#                                 'joint_tail_005',])
#             joints_spine = list(['joint_spine_001',
#                                  'joint_spine_002',
#                                  'joint_spine_003',
#                                  'joint_spine_004',
#                                  'joint_spine_005',])
#             joints_head = list(['joint_head_001'])
#             joints_front_limbs = list(['joint_shoulder_left',
#                                        'joint_elbow_left',
#                                        'joint_wrist_left',
#                                        'joint_finger_left_002',
#                                        'joint_shoulder_right',
#                                        'joint_elbow_right',
#                                        'joint_wrist_right',
#                                        'joint_finger_right_002',])
#             joints_hind_limbs = list(['joint_hip_left',
#                                       'joint_knee_left',
#                                       'joint_ankle_left',
#                                       'joint_paw_hind_left',
#                                       'joint_toe_left_002',
#                                       'joint_hip_right',
#                                       'joint_knee_right',
#                                       'joint_ankle_right',
#                                       'joint_paw_hind_right',
#                                       'joint_toe_right_002',])
#             joints_extra = list(['joint_hip_left',
#                                  'joint_hip_right',
#                                  'joint_shoulder_left',
#                                  'joint_shoulder_right',
#                                  'joint_toe_left_002',
#                                  'joint_toe_right_002',
#                                  'joint_finger_left_002',
#                                  'joint_finger_right_002',])
#             joints_to_remove = list()
#             joints_to_remove = joints_to_remove + joints_origin
#             joints_to_remove = joints_to_remove + joints_tail
# #             joints_to_remove = joints_to_remove + joints_spine                  
# #             joints_to_remove = joints_to_remove + joints_front_limbs
# #             joints_to_remove = joints_to_remove + joints_hind_limbs
#             joints_to_remove = joints_to_remove + joints_extra
#             joints_to_remove = np.unique(joints_to_remove)
#             mask_joints = np.ones(nJoints, dtype=bool)
#             for i_joint in range(nJoints):
#                 if joint_order[i_joint] in joints_to_remove:
#                     mask_joints[i_joint] = False
#             nJoints_use = np.sum(mask_joints)
#             joint_order_use = np.array(joint_order)[mask_joints]
#             sort_index = np.argsort(joint_order_use)
#             joint_order_use = joint_order_use[sort_index]
#             metric_use2 = position[i_folder][:, mask_joints, 0]
#             metric_use2 = metric_use2[:, sort_index]
#             metric_use3 = position[i_folder][:, mask_joints, 1]
#             metric_use3 = metric_use3[:, sort_index]
#             color_index = 0
#             for i_joint in range(nJoints_use):
#                 ax_test2.plot(range(nT_single), metric_use2[:, i_joint], color=cmap(color_index/(nJoints_use-1)), zorder=1)
#                 ax_test3.plot(range(nT_single), metric_use3[:, i_joint], color=cmap(color_index/(nJoints_use-1)), zorder=1)
#                 color_index = color_index + 1
#             ax_test3.legend([joint_order_use[i] for i in range(nJoints_use)], loc='upper right')
#             #
#             for i_ax in range(len(ax_test)):
#                     ax_use = ax_test[i_ax]
#                     ylim_min, ylim_max = ax_use.get_ylim()
#                     rectangle_jump = np.array([[index1, ylim_min], 
#                                                [index1, ylim_max],
#                                                [index2, ylim_max],
#                                                [index2, ylim_min]], dtype=np.float64)
#                     poly_jump = plt.Polygon(rectangle_jump, color='black', zorder=0, alpha=0.1)
                    
#                     ax_use.plot(np.array([index_ini, index_ini], dtype=np.float64),
#                                 np.array([ylim_min, ylim_max], dtype=np.float64),
#                                 linestyle='--', color='red')
                    
#                     ax_use.add_patch(poly_jump)
#                     ax_use.set_xlim([0, nT_single])
#                     ax_use.set_ylim([ylim_min, ylim_max])
#             ax_test[0].set_ylabel('metric #0')
#             ax_test[1].set_ylabel('metric #1')
#             ax_test[2].set_ylabel('metric #2')
#             ax_test[2].set_xlabel('time (frames)')
#             fig_test.canvas.draw()
#             plt.show(block=False)
#             input()
  
    angle_index_spine = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    angle_index_hind_limbs = np.array([7, 8, 11, 12], dtype=np.int64)
    metric_use_indices = np.concatenate([angle_index_spine,
                                         angle_index_hind_limbs], 0)
    
    time = np.arange(-nT_single, nT_single+1, 1, dtype=np.float64) * 1.0/cfg.frame_rate # s
    metric = np.full((nFolders, 2*nT_single+1), np.nan, dtype=np.float64)
    for i_folder in range(nFolders):
        metric_single = np.mean(angle[i_folder][:, metric_use_indices], 1)
        index = jump_indices[i_folder, 1]
        metric[i_folder, nT_single-index:nT_single-index+nT_single] = np.copy(metric_single)
    metric_avg = np.nanmean(metric, 0)
    metric_std = np.nanstd(metric, 0)
    #
    n = np.sum(~np.isnan(metric), 0)
    mask = (n == nFolders)
    time = time[mask]
    metric_avg = metric_avg[mask]   
    metric_std = metric_std[mask]
    #
    index_ini = np.argmin(metric_avg)
    index_jump = np.copy(index_ini)
    #
    index_start = 0
    index_end = np.copy(index_jump)
    indices_maxima1 = max_finder(metric_avg[index_start:index_end]) + index_start
    index_start = 0
    index_end = np.copy(indices_maxima1[-1])
    indices_minima1 = min_finder(metric_avg[index_start:index_end]) + index_start
    if not(len(indices_minima1) == 0):
        index1 = indices_minima1[-1]
    else:
        index1 = 0
    #
    index_start = np.copy(index_jump)
    index_end = nT_single 
    indices_maxima2 = max_finder(metric_avg[index_start:index_end]) + index_start
    index_start = np.copy(indices_maxima2[0])
    index_end = nT_single 
    indices_minimia2 = min_finder(metric_avg[index_start:index_end]) + index_start
    if not(len(indices_minimia2) == 0):
        index2 = indices_minimia2[0]
    else:
        index2 = nT_single - 1


    # PLOT
    x = np.copy(time)
    y_avg = np.copy(metric_avg)
    y_std = np.copy(metric_std)
    #
    color = 'black'
    #
    figsize_x = np.round(mm_in_inch * 88.0, decimals=2)
    figsize_y = np.round(mm_in_inch * 88.0 * 0.25, decimals=2)
    fig = plt.figure(5, figsize=(figsize_x, figsize_y))
    fig.canvas.manager.window.move(0, 0)
    fig.clear()
    ax = fig.add_subplot(111)
    ax.clear()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(x, y_avg, linestyle='-', marker='', linewidth=linewidth, color=color, alpha=1.0, zorder=1)
    ax.fill_between(x=x,
                    y1=y_avg+y_std,
                    y2=y_avg-y_std,
                    color=color, linewidth=linewidth,
                    alpha=0.2, zorder=2,
                    where=None, interpolate=False, step=None, data=None)
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylim([29.0, 72.0])
    #
    ylim_min, ylim_max = ax.get_ylim()
    ax.plot([time[index1], time[index1]], [ylim_min, ylim_max], linestyle='-', marker='', linewidth=2*linewidth, color='green', alpha=1.0, zorder=1)
    ax.plot([time[index_jump], time[index_jump]], [ylim_min, ylim_max], linestyle='-', marker='', linewidth=2*linewidth, color='orange', alpha=1.0, zorder=1)
    ax.plot([time[index2], time[index2]], [ylim_min, ylim_max], linestyle='-', marker='', linewidth=2*linewidth, color='red', alpha=1.0, zorder=1)
    #
#     ylim_min, ylim_max = ax_use.get_ylim()
#     rectangle_jump = np.array([[index1, ylim_min], 
#                                [index1, ylim_max],
#                                [index2, ylim_max],
#                                [index2, ylim_min]], dtype=np.float64)
#     poly_jump = plt.Polygon(rectangle_jump, color='black', zorder=0, alpha=0.1)

#     ax_use.plot(np.array([index_ini, index_ini], dtype=np.float64),
#                 np.array([ylim_min, ylim_max], dtype=np.float64),
#                 linestyle='--', color='red')
#     ax_use.add_patch(poly_jump)
    #
    cfg_plt.plot_coord_ax(ax, '100 ms', '15 deg', 0.1, 15.0)
    fig.canvas.draw()
    
    
    
    
    figsize_x = np.round(mm_in_inch * 88.0*0.5, decimals=2)
    figsize_y = np.round(mm_in_inch * 88.0*2/3, decimals=2)
    fig2 = plt.figure(6, figsize=(figsize_x, figsize_y))
    fig2.canvas.manager.window.move(0, 0)
    fig2.clear()
    ax2 = fig2.add_subplot(111)
    ax2.clear()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    #
    for i_folder in np.arange(1, nFolders, 2, dtype=np.int64):
        metric_use = np.copy(metric[i_folder])
        metric_use = metric_use - metric_use[np.nanargmin(metric_use)]
        metric_use = metric_use + float(i_folder * 10)
        time_use = np.arange(len(metric_use), dtype=np.float64) * 1.0/cfg.frame_rate # s
        ax2.plot(time_use, metric_use,
                 linestyle='-', marker='', linewidth=1, color='black')
    cfg_plt.plot_coord_ax(ax2, '100 ms', '30 deg', 0.1, 30.0)
    fig2.canvas.draw()
    
    if save:
        fig.savefig(folder_save+'/jump_metric__population_average.svg',
                     dpi=300,
                     transparent=True,
                     format='svg',
                     pad_inches=0)
        fig2.savefig(folder_save+'/jump_metric__population_traces.svg',
                     dpi=300,
                     transparent=True,
                     format='svg',
                     pad_inches=0)
        
    plt.show()
