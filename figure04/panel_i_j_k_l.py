#!/usr/bin/env python3

import importlib
import numpy as np
from matplotlib.lines import Line2D
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
verbose = True

folder_save = os.path.abspath('panels')

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

bottom_margin_x = 0.10 # inch
left_margin_x = 0.10 # inch
left_margin  = 0.30 # inch
right_margin = 0.05 # inch
bottom_margin = 0.30 # inch
top_margin = 0.05 # inch

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
    side_jumps = np.zeros(nFolders, dtype=bool)
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
            metric_use = np.mean(angle_peak[i_folder][:, metric_use_indices], 1)
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
            #
            side_jumps[i_folder] = (diff[0] > 0.0)
    
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
#             print('Press any key to continue')
#             input()
     
    # save arrays for 3D plot
    positions_norm0 = np.copy(position)
    positions0 = np.copy(position_peak)
    nJoints0 = np.copy(nJoints)
    joint_order0 = np.copy(joint_order)
    
    # get rid of joints we are not interested in for the correlation
    joints_origin = list(['joint_spine_002'])
    joints_tail = list(['joint_tail_001',
                        'joint_tail_002',
                        'joint_tail_003',
                        'joint_tail_004',
                        'joint_tail_005',])
    joints_tail_extra = joints_tail + list(['joint_spine_001'])
    joints_spine = list(['joint_spine_001',
                         'joint_spine_002',
                         'joint_spine_003',
                         'joint_spine_004',
                         'joint_spine_005',])
    joints_head = list(['joint_head_001',])
    joints_front_limbs = list(['joint_shoulder_left',
                               'joint_elbow_left',
                               'joint_wrist_left',
                               'joint_finger_left_002',
                               'joint_shoulder_right',
                               'joint_elbow_right',
                               'joint_wrist_right',
                               'joint_finger_right_002',])
    joints_hind_limbs = list(['joint_hip_left',
                              'joint_knee_left',
                              'joint_ankle_left',
                              'joint_paw_hind_left',
                              'joint_toe_left_002',
                              'joint_hip_right',
                              'joint_knee_right',
                              'joint_ankle_right',
                              'joint_paw_hind_right',
                              'joint_toe_right_002',])
    # pos, velo, acc
    joints_to_remove = list()
    joints_to_remove = joints_to_remove + joints_origin
    joints_to_remove = joints_to_remove + joints_tail
    joints_to_remove = joints_to_remove + joints_tail_extra
# #     joints_to_remove = joints_to_remove + joints_spine
    joints_to_remove = joints_to_remove + joints_front_limbs 
    joints_to_remove = joints_to_remove + joints_hind_limbs 
    joints_to_remove = np.unique(joints_to_remove)
    nJoints_remove = len(joints_to_remove)
    mask = np.ones(nJoints, dtype=bool)
    for joint_name in joints_to_remove:
        index = joint_order.index(joint_name)
        mask[index] = False
    for i_folder in range(nFolders):
        position[i_folder] = np.copy(position[i_folder][:, mask])
        velocity[i_folder] = np.copy(velocity[i_folder][:, mask])
        acceleration[i_folder] = np.copy(acceleration[i_folder][:, mask])
        position_peak[i_folder] = np.copy(position_peak[i_folder][:, mask])
        velocity_peak[i_folder] = np.copy(velocity_peak[i_folder][:, mask])
        acceleration_peak[i_folder] = np.copy(acceleration_peak[i_folder][:, mask])
    for joint_name in joints_to_remove:
        index = joint_order.index(joint_name)
        joint_order.pop(index)
    nJoints = nJoints - nJoints_remove
    # ang, ang velo, ang acc
    joints_to_remove = list()
    joints_to_remove = joints_to_remove + joints_front_limbs 
    joints_to_remove = joints_to_remove + joints_hind_limbs 
    joints_to_remove = joints_to_remove + list(['joint_spine_001', 'joint_spine_002']) # remove joints per hand as a quick fix
    mask = np.ones(nAngles, dtype=bool)
    for i_ang in range(nAngles):
        if (angle_comb_list[i_ang][1] in joints_to_remove):
            mask[i_ang] = False
    for i_folder in range(nFolders):
        angle[i_folder] = np.copy(angle[i_folder][:, mask])
        angle_velocity[i_folder] = np.copy(angle_velocity[i_folder][:, mask])
        angle_acceleration[i_folder] = np.copy(angle_acceleration[i_folder][:, mask])
        angle_peak[i_folder] = np.copy(angle_peak[i_folder][:, mask])
        angle_velocity_peak[i_folder] = np.copy(angle_velocity_peak[i_folder][:, mask])
        angle_acceleration_peak[i_folder] = np.copy(angle_acceleration_peak[i_folder][:, mask])
    indices_remove = np.arange(nAngles, dtype=np.int64)[~mask]
    nAngles_remove = np.sum(~mask, dtype=np.int64)
    for i_ang in range(nAngles_remove):
        angle_comb_list.pop(indices_remove[i_ang])
        indices_remove = indices_remove - 1
    nAngles = nAngles - nAngles_remove

    # sort arrays
    joint_order_middle = list()
    joint_order_left = list(['dummy' for i in range(9)]) # 9 is the number of joints on one side
    joint_order_right = list(['dummy' for i in range(9)]) # 9 is the number of joints on one side
    joint_order_middle_index = list()
    joint_order_left_index = list(['dummy' for i in range(9)]) # 9 is the number of joints on one side
    joint_order_right_index = list(['dummy' for i in range(9)]) # 9 is the number of joints on one side
    limb_order = list(['head',
                       'spine',
                       'shoulder', 'elbow', 'wrist', 'finger',
                       'hip', 'knee', 'ankle', 'paw', 'toe',
                       'tail'])
    for i_joint in range(nJoints):
        joint_name = joint_order[i_joint]
        joint_name_split = joint_name.split('_')
        if ('left' in joint_name_split):
            index = limb_order.index(joint_name_split[1])
            joint_order_left[index] = joint_name
            joint_order_left_index[index] = i_joint
        elif ('right' in joint_name_split):
            index = limb_order.index(joint_name_split[1])
            joint_order_right[index] = joint_name
            joint_order_right_index[index] = i_joint
        else:
            joint_order_middle.append(joint_name)
            joint_order_middle_index.append(i_joint)
    while ('dummy' in joint_order_left_index):
        joint_order_left_index.remove('dummy')
    while ('dummy' in joint_order_right_index):
        joint_order_right_index.remove('dummy')
    while ('dummy' in joint_order_left):
        joint_order_left.remove('dummy')
    while ('dummy' in joint_order_right):
        joint_order_right.remove('dummy')
    joint_order_sort_index = joint_order_middle_index + joint_order_left_index + joint_order_right_index
    joint_order_sort_index = np.array(joint_order_sort_index, dtype=np.int64)
    joint_order = joint_order_middle + joint_order_left + joint_order_right
    #
    for i_folder in range(nFolders):
        position[i_folder] = np.copy(position[i_folder][:, joint_order_sort_index])
        velocity[i_folder] = np.copy(velocity[i_folder][:, joint_order_sort_index])
        acceleration[i_folder] = np.copy(acceleration[i_folder][:, joint_order_sort_index])
        position_peak[i_folder] = np.copy(position_peak[i_folder][:, joint_order_sort_index])
        velocity_peak[i_folder] = np.copy(velocity_peak[i_folder][:, joint_order_sort_index])
        acceleration_peak[i_folder] = np.copy(acceleration_peak[i_folder][:, joint_order_sort_index]) 

    
    # plot ids of metrics
    index_corr = 0
    metric_names_list = list()
    #
    for s1 in list(['velocity (cm/s)',]):
#     for s1 in list(['position (cm)', 'velocity (cm/s)',]):
        for i_joint in range(nJoints):
#             for s2 in list(['']):
            for s2 in list(['z-']):
#             for s2 in list(['x-', 'z-']):
#             for s2 in list(['x-', 'y-', 'z-']):
                s = s2+s1
                metric_name = '{:03d}\t{:s}:\t{:s}'.format(index_corr, s, joint_order[i_joint])
                metric_names_list.append(metric_name.replace('\t', ' '))
                index_corr = index_corr + 1
                print(metric_name)
    for s in list(['angular velocity (deg/s)',]):
#     for s in list(['angle (deg)', 'angular velocity (deg/s)',]):
        for i_angle in range(nAngles):
            metric_name = '{:03d}\t{:s}:\t{:s}'.format(index_corr, s, angle_comb_list[i_angle][1])
            metric_names_list.append(metric_name.replace('\t', ' '))
            index_corr = index_corr + 1
            print(metric_name)
        
    # correlate with distance
    dIndex = 0
    time_shift_start = -80
    time_shift_end = 0
    index_dim = np.array([2], dtype=np.int64)
    nBackward = (time_shift_end - time_shift_start) + 1
    dx_dist = (len(index_dim) * nJoints) * 1 + nAngles * 1
    dy_dist = np.copy(nBackward)
    corr_dist = np.full((dy_dist, dx_dist), np.nan, dtype=np.float64)
    #
    time_until_jump = np.arange(time_shift_start, time_shift_end+1, 1, dtype=np.int64)
    index_jump_start = np.copy(jump_indices[:, 0])
    index_jump_middle = np.copy(jump_indices[:, 1])
    index_jump_end = np.copy(jump_indices[:, 2])
    index_zero = np.copy(index_jump_end)
    #
    x_all = np.full((dy_dist, nFolders, dx_dist), np.nan, dtype=np.float64)
    for i_dt in range(nBackward):
        dt = time_until_jump[i_dt]
        #
        x_avg = np.zeros((nAnimals, dx_dist), dtype=np.float64)
        x2_avg = np.zeros((nAnimals, dx_dist), dtype=np.float64)
        x_std = np.ones((nAnimals, dx_dist), dtype=np.float64)
#         for i_folder in range(nFolders):
#             folder = folder_list[i_folder] 
#             if ('20200205' in folder.split('_')):
#                 i_animal = 0
#             elif ('20200207' in folder.split('_')):
#                 i_animal = 1
#             if mask_animals[i_animal, i_folder]:
#                 index1 = index_zero[i_folder] + dt
#                 index_start = index1 - dIndex
#                 index_end = index1 + dIndex + 1
#                 pos_use = np.nanmean(np.array(position)[i_folder, index_start:index_end, :, index_dim], 0)
#                 velo_use = np.nanmean(np.array(velocity)[i_folder, index_start:index_end, :, index_dim], 0)
# #                 pos_use = np.sqrt(np.sum(pos_use[:, np.array([0, 1, 2], dtype=np.int64)]**2, 1))
# #                 velo_use = np.sqrt(np.sum(velo_use[:, np.array([0, 1, 2], dtype=np.int64)]**2, 1))
#                 pos_use = pos_use.ravel()
#                 velo_use = velo_use.ravel()
#                 ang_use = np.nanmean(np.array(angle)[i_folder, index_start:index_end], 0)
#                 ang_velo_use = np.nanmean(np.array(angle_velocity)[i_folder, index_start:index_end], 0)
#                 x_use = np.concatenate([pos_use, velo_use, ang_use, ang_velo_use], 0)
#                 x_avg[i_animal] = x_avg[i_animal] + x_use
#                 x2_avg[i_animal] = x2_avg[i_animal] + x_use**2
#         nFolders_use = np.sum(mask_animals, 1, dtype=np.float64)
#         x_avg = x_avg / nFolders_use[:, None]
#         x2_avg = x2_avg / nFolders_use[:, None]
#         x_std = np.sqrt(x2_avg - x_avg**2)
        #
        x0 = np.full((nFolders, dx_dist), np.nan, dtype=np.float64)
        x = np.full((nFolders, dx_dist), np.nan, dtype=np.float64)
        mask_folders = np.zeros(nFolders, dtype=bool)
        for i_folder in range(nFolders):
            folder = folder_list[i_folder] 
            if ('20200205' in folder.split('_')):
                i_animal = 0
            elif ('20200207' in folder.split('_')):
                i_animal = 1
            index_use = index_zero[i_folder] + dt
            if ((index_use >= int(derivative_accuracy/2)) and (index_use < 200 - int(derivative_accuracy/2))):
#                 index_start0 = index_jump_start - dIndex
#                 index_end0 = index_jump_start + dIndex + 1
#                 x0_pos = np.mean(position_peak[i_folder][index_start0:index_end0], 0).ravel()
#                 x0_velo = np.mean(velocity_peak[i_folder][index_start0:index_end0], 0).ravel()
#                 x0_ang = np.mean(angle_peak[i_folder][index_start0:index_end0], 0).ravel()
#                 x0_ang_velo = np.mean(angle_velocity_peak[i_folder][index_start0:index_end0], 0).ravel()
#                 x0[i_folder] = np.concatenate([x0_pos, x0_velo, x0_ang, x0_ang_velo], 0)
#                 index_start = index_use - dIndex
#                 index_end = index_use + dIndex + 1
#                 x_pos = np.mean(position_peak[i_folder][index_start:index_end], 0).ravel()
#                 x_velo = np.mean(velocity_peak[i_folder][index_start:index_end], 0).ravel()
#                 x_ang = np.mean(angle_peak[i_folder][index_start:index_end], 0).ravel()
#                 x_ang_velo = np.mean(angle_velocity_peak[i_folder][index_start:index_end], 0).ravel()
#                 x[i_folder] = np.concatenate([x_pos, x_velo, x_ang, x_ang_velo], 0)
#                 x[i_folder] = (x[i_folder] - x0[i_folder])
                #
                index_start = index_use - dIndex
                index_end = index_use + dIndex + 1
                #
#                 x_pos = np.copy(position[i_folder][:, :, index_dim])
                x_pos = np.copy(position_peak[i_folder][:, :, index_dim])
                x_pos = np.mean(x_pos[index_start:index_end], 0).ravel()
#                 x_velo = np.copy(velocity[i_folder][:, :, index_dim])
                x_velo = np.copy(velocity_peak[i_folder][:, :, index_dim])
                x_velo = np.mean(x_velo[index_start:index_end], 0).ravel()
                x_acc = np.copy(acceleration[i_folder][:, :, index_dim])
                x_acc = np.mean(x_acc[index_start:index_end], 0).ravel()
                x_ang = np.copy(angle[i_folder])
#                 x_ang = np.copy(angle_peak[i_folder])
                x_ang = np.mean(x_ang[index_start:index_end], 0).ravel()
                x_ang = np.mean(x_ang[index_start:index_end], 0).ravel()
                x_ang_velo = np.copy(angle_velocity[i_folder])
#                 x_ang_velo = np.copy(angle_velocity_peak[i_folder])
                x_ang_velo = np.mean(x_ang_velo[index_start:index_end], 0).ravel()
                x_ang_acc = np.copy(angle_acceleration[i_folder])
                x_ang_acc = np.mean(x_ang_acc[index_start:index_end], 0).ravel()
                x[i_folder] = np.concatenate([
#                                               x_pos,
                                              x_velo,
#                                               x_ang,
                                              x_ang_velo,
                                             ], 0)
                x[i_folder] = (x[i_folder] - x_avg[i_animal]) / x_std[i_animal]
                #
#                 x_velo = np.sqrt(np.sum(velocity_peak[i_folder]**2, 2))
#                 x_velo = np.mean(x_velo[index_start:index_end], 0).ravel()
#                 x_ang_velo = np.copy(angle_velocity[i_folder])
#                 x_ang_velo = np.mean(x_ang_velo[index_start:index_end], 0).ravel()
#                 x[i_folder] = np.concatenate([
# #                                               x_pos,
#                                               x_velo,
# #                                               x_ang,
#                                               x_ang_velo,
#                                              ], 0)
#                 x[i_folder] = (x[i_folder] - x_avg[i_animal]) / x_std[i_animal]
                #
                mask_folders[i_folder] = True
        nFolders_use = np.sum(mask_folders, dtype=np.float64)
        #
        x = x[mask_folders]
        avg_x = np.sum(x, 0) / nFolders_use
        avg_x2 = np.sum(x**2, 0) / nFolders_use
        var_x = avg_x2 - avg_x**2
        sigma_x = np.sqrt(var_x)
        x_norm = x - avg_x[None, :]
        #
        y = np.copy(jump_distances) # jump distance
#         y = (jump_indices[:, 2] - jump_indices[:, 0]) # jump duration #1
#         y = (jump_indices[:, 2] - jump_indices[:, 1]) # jump duration #2
#         y = (jump_indices[:, 1] - jump_indices[:, 0]) # jump duration #3
        #
#         y = (y - np.mean(y)) / np.std(y)
        #
        y = y[mask_folders]
        avg_y = np.sum(y, 0) / nFolders_use
        avg_y2 = np.sum(y**2, 0) / nFolders_use
        var_y = avg_y2 - avg_y**2
        sigma_y = np.sqrt(var_y)
        y_norm = y - avg_y
        #
        avg_outer_all = np.sum(np.einsum('ni,n->ni', x_norm, y_norm), 0) / nFolders_use
        corr_dist[i_dt] = avg_outer_all / (sigma_x * sigma_y)
        #
        x_all[i_dt, mask_folders] = np.copy(x)
    
    nIndex = 2
    scatter_index_x_list = list([6, 2])
    scatter_index_y_list = list([39, 45])
    
    # STATS
    import scipy.stats
    #
    for i_index in range(nIndex):
        scatter_index_x = scatter_index_x_list[i_index]
        scatter_index_y = scatter_index_y_list[i_index]
        scatter_x = x_all[scatter_index_y, :, scatter_index_x]
        scatter_y = np.copy(y)
        stat, p = scipy.stats.spearmanr(scatter_x, scatter_y) # does not assume data is normally distributed
        print('scatter_index_x:\t\t{:03d} ({:s})'.format(scatter_index_x, metric_names_list[scatter_index_x]))
        print('scatter_index_y:\t\t{:03d}'.format(scatter_index_y))
        print('time before end of jump:\t{:02f} (ms)'.format((-((dy_dist-1) - scatter_index_y))*(1.0/200.0)*1e3))
        print('stat={:0.8f}, p={:0.2e}'.format(stat, p))
        if p > 0.05:
            print('Probably independent')
        else:
            print('Probably dependent')
        print()
    
    # PLOT
    y_axis_lim = np.array([np.min(y), np.max(y)], dtype=np.float64)
    x_axis_lim = np.full((nIndex, 2), np.nan, dtype=np.float64)
    for i_index in range(nIndex):
        x_axis_lim[i_index] = np.array([np.min(x_all[scatter_index_y_list[i_index], :, scatter_index_x_list[i_index]]),
                                        np.max(x_all[scatter_index_y_list[i_index], :, scatter_index_x_list[i_index]])],
                                       dtype=np.float64)
#     print(y_axis_lim)
#     print(x_axis_lim)
    #
    y_axis_lim = np.array([12.5, 27.5], dtype=np.float64)
    y_axis_ticks = np.array([15, 20, 25], dtype=np.int64)
    y_axis_label = 'jump distance (cm)'

    x_axis_lim = np.array([[-650, 90],
                           [-5, 45],
                           [-30, 40]], dtype=np.float64)
    x_axis_ticks = np.array([[-600, -280, 40],
                             [0, 20, 40],
                             [-25, 5, 35]], dtype=np.int64)
    x_axis_label = list(['angular velocity of\nspine joint #3 (deg/s)',
                         'z-velocity of\nspine joint #4 (cm/s)',
                         'z-velocity of\nspine joint #3 (cm/s)'])
    #
    # correlation matrix
    fig_corr_w = np.round(mm_in_inch * 88.0, decimals=2)
    fig_corr_h = np.round(mm_in_inch * 88.0 * 0.5, decimals=2)
    fig_corr = plt.figure(5, figsize=(fig_corr_w, fig_corr_h))
    fig_corr.canvas.manager.window.move(0, 0)
    fig_corr.clear()
    ax_corr_x = left_margin/fig_corr_w
    ax_corr_y = bottom_margin/fig_corr_h
    ax_corr_w = 1.0 - (left_margin/fig_corr_w + right_margin/fig_corr_w)
    ax_corr_h = 1.0 - (bottom_margin/fig_corr_h + top_margin/fig_corr_h)
    ax_corr = fig_corr.add_axes([ax_corr_x, ax_corr_y, ax_corr_w, ax_corr_h])
    ax_corr.clear()
    ax_corr.spines["top"].set_visible(False)
    ax_corr.spines["right"].set_visible(False)
    ax_corr.spines['left'].set_visible(False)
    ax_corr.spines['bottom'].set_visible(False)
    corr_dist_plot = np.copy(corr_dist.T)
    corr_h = ax_corr.imshow(corr_dist_plot,
                            cmap='viridis',
                            aspect='auto',
                            vmin=-1.0,
                            vmax=1.0,
                            zorder=0,
                            interpolation=None,
                            extent=(0.0, dy_dist, 0.0, dx_dist),
                            origin='lower')
    for i_index in range(nIndex):
        pixel_x = scatter_index_y_list[i_index]
        pixel_y = scatter_index_x_list[i_index]
        rect = np.array([[pixel_x-0.5, pixel_y-0.5],
                         [pixel_x-0.5, pixel_y+0.5],
                         [pixel_x+0.5, pixel_y+0.5],
                         [pixel_x+0.5, pixel_y-0.5],
                         [pixel_x-0.5, pixel_y-0.5]], dtype=np.float64) + 0.5
        ax_corr.plot(rect[:, 0], rect[:, 1],
                     linestyle='-', marker='',
                     linewidth=0.75,
                     color='white', zorder=1)
    ax_corr.set_xlabel('time until end of jump (ms)', fontsize=fontsize)

    ax_corr.set_yticks([])
    ax_corr.set_yticklabels([])
    x_pos = -2.5
    markersize = 3
    markeredgewidth = 3
    y_pos_list = np.arange(nJoints+nAngles, dtype=np.float64)+0.5
    joint_colors_cmap = plt.cm.Dark2
#     joint_colors_list = list([joint_colors_cmap(float(i)/(nJoints+nAngles-1)) for i in range(nJoints+nAngles)])
    joint_colors_list = list([joint_colors_cmap(0.0),
                              joint_colors_cmap(float(1/7)), joint_colors_cmap(float(2/7)), joint_colors_cmap(float(3/7)),
                              joint_colors_cmap(float(1/7)), joint_colors_cmap(float(2/7)), joint_colors_cmap(float(3/7))])
    for i_joint in range(nJoints):
        y_pos = y_pos_list[i_joint]
        ax_corr.plot([x_pos], [y_pos], linestyle='', marker='o', markersize=markersize, markeredgewidth=markeredgewidth, color=joint_colors_list[i_joint])
    for i_joint in range(nAngles):
        y_pos = y_pos_list[nJoints+i_joint]
        ax_corr.plot([x_pos], [y_pos], linestyle='', marker='s', markersize=markersize, markeredgewidth=markeredgewidth, color=joint_colors_list[nJoints+i_joint])

    ax_corr.set_xticks([0.0+0.5, 40.0+0.5, 80.0+0.5])
    ax_corr.set_xticklabels([400, 200, 0])
    for tick in ax_corr.get_xticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    ax_corr.xaxis.set_label_coords(x=ax_corr_x+0.5*ax_corr_w, y=bottom_margin_x/fig_corr_h, transform=fig_corr.transFigure)
    ax_corr.yaxis.set_label_coords(x=left_margin_x/fig_corr_w, y=ax_corr_y+0.5*ax_corr_h, transform=fig_corr.transFigure)
    fig_corr.canvas.draw()
    if save:
        fig_corr.savefig(folder_save+'/correlation_distance.svg',
#                          bbox_inches="tight",
                         dpi=300,
                         transparent=True,
                         format='svg',
                         pad_inches=0)

    # correlation scatter        
    fig_scatter_w = np.round(mm_in_inch * 88.0 * 1.0/3.0, decimals=2)
    fig_scatter_h = np.round(mm_in_inch * 88.0 * 1.0/3.0, decimals=2)
    fig_scatter = plt.figure(6, figsize=(fig_scatter_w, fig_scatter_h))
    fig_scatter.canvas.manager.window.move(0, 0)
    fig_scatter.clear()
    ax_scatter_x = left_margin/fig_scatter_w
    ax_scatter_y = bottom_margin/fig_scatter_h
    ax_scatter_w = 1.0 - (left_margin/fig_scatter_w + right_margin/fig_scatter_w)
    ax_scatter_h = 1.0 - (bottom_margin/fig_scatter_h + top_margin/fig_scatter_h)
    ax_scatter = fig_scatter.add_axes([ax_scatter_x, ax_scatter_y, ax_scatter_w, ax_scatter_h])
    # 3D
    fig3d_w = np.round(mm_in_inch * 88.0 * 0.5, decimals=2)
    fig3d_h = np.round(mm_in_inch * 88.0 * 0.5, decimals=2)
    fig3d = plt.figure(7, figsize=(fig3d_w, fig3d_h))
    fig3d.clear()
    fig3d.canvas.manager.window.move(0, 0)
    ax3d = fig3d.add_axes([0, 0, 1, 1], projection='3d')
    ax3d.clear()
    #
    cmap2 = plt.cm.tab10
    color_animal1 = 'black'
    color_animal2 = 'black'
    #
    for i_index in range(nIndex):
        scatter_index_x = scatter_index_x_list[i_index]
        scatter_index_y = scatter_index_y_list[i_index]
        #
        print('scatter_index_x:\t\t{:03d} ({:s})'.format(scatter_index_x, metric_names_list[scatter_index_x]))
        print('scatter_index_y:\t\t{:03d}'.format(scatter_index_y))
        print('time before end of jump:\t{:01d} (frames)'.format(-((dy_dist-1) - scatter_index_y)))
        print(corr_dist[scatter_index_y, scatter_index_x])
        #
        indices_x = range(dx_dist)
        indices_y = range(dy_dist)
        index_min_x = np.tile(indices_x, dy_dist).ravel()[np.nanargmin(corr_dist)]
        index_min_y = np.repeat(indices_y, dx_dist).ravel()[np.nanargmin(corr_dist)]
        index_max_x = np.tile(indices_x, dy_dist).ravel()[np.nanargmax(corr_dist)]
        index_max_y = np.repeat(indices_y, dx_dist).ravel()[np.nanargmax(corr_dist)]
        print('distance correlation:')
        print('min.:\t{:0.8f} {:03d} {:03d}'.format(np.nanmin(corr_dist), index_min_x, index_min_y))
        print('max.:\t{:0.8f} {:03d} {:03d}'.format(np.nanmax(corr_dist), index_max_x, index_max_y))
        #
        ax_scatter.clear()
        ax_scatter.spines["top"].set_visible(False)
        ax_scatter.spines["right"].set_visible(False)
        #
#         custom_lines = list([Line2D([0], [0], linestyle='', marker='o', color=colors0[i]) for i in range(len(colors0))])    
        #
        for i_folder in range(nFolders):
            folder = folder_list[i_folder]
            if ('20200205' in folder.split('_')):
                i_animal = 0
                color_use = color_animal1
            elif ('20200207' in folder.split('_')):
                i_animal = 1
                color_use = color_animal2
            #
            ax_scatter.plot([x_all[scatter_index_y, i_folder, scatter_index_x]],
                            [y[i_folder]],
                            linestyle='', marker='o', markersize=1, color=color_use)
        ax_scatter.set_xlim(x_axis_lim[i_index])
        ax_scatter.set_ylim(y_axis_lim)
        ax_scatter.set_xlabel(x_axis_label[i_index], fontsize=fontsize)
        ax_scatter.set_ylabel(y_axis_label, fontsize=fontsize)
        ax_scatter.set_xticks(x_axis_ticks[i_index])
        ax_scatter.set_xticklabels(x_axis_ticks[i_index])
        ax_scatter.set_yticks(y_axis_ticks)
        ax_scatter.set_yticklabels(y_axis_ticks)
        for tick in ax_scatter.get_xticklabels():
            tick.set_fontsize(fontsize)
            tick.set_fontname(fontname)
        for tick in ax_scatter.get_yticklabels():
            tick.set_fontname(fontname)
            tick.set_fontsize(fontsize)
        ax_scatter.xaxis.set_label_coords(x=ax_scatter_x+0.5*ax_scatter_w, y=bottom_margin_x/fig_scatter_h, transform=fig_scatter.transFigure)
        ax_scatter.yaxis.set_label_coords(x=left_margin_x/fig_scatter_w, y=ax_scatter_y+0.5*ax_scatter_h, transform=fig_scatter.transFigure)
        fig_scatter.canvas.draw()
        plt.show(block=False)
        if save:
            fig_scatter.savefig(folder_save+'/distance_vs_metric_x{:03d}_y{:03d}.svg'.format(scatter_index_x, scatter_index_y),
    #                          bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)

        # 3D
        linewidth_fac = 0.25
        cmap_time = plt.cm.plasma
        #
        for i_min_max in range(2):
            ax3d.clear()
            #
            if (i_min_max == 0):
                s_min_max = 'min'
                i_folder = np.argmin(x_all[scatter_index_y, :, scatter_index_x])
            elif (i_min_max == 1):
                s_min_max = 'max'
                i_folder = np.argmax(x_all[scatter_index_y, :, scatter_index_x])
                
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
#             index1 = jump_indices[i_folder, 0]# - 10
#             index2 = jump_indices[i_folder, 2]# + 10
#             t_list = np.arange(index1, index2+1, 10, dtype=np.int64)
            
#             dIndex = 0
#             time_shift_start = -80
#             time_shift_end = 0
#             nBackward = (time_shift_end - time_shift_start) + 1
#             index_jump_start = np.copy(jump_indices[:, 0])
#             index_jump_middle = np.copy(jump_indices[:, 1])
#             index_jump_end = np.copy(jump_indices[:, 2])
#             index_zero = np.copy(index_jump_end)
#             time_until_jump = np.arange(time_shift_start, time_shift_end+1, 1, dtype=np.int64)
#             for i_dt in range(nBackward):

            index_use = index_zero[i_folder] + time_until_jump[scatter_index_y]
            dT = 8
            t_list = np.arange(index_use-dT, index_use+dT+1, 1, dtype=np.int64)
            nT_plot = len(t_list)
            #
            colors = cmap_time(np.linspace(0.0, 1.0, nT_plot))
            skeleton_pos_all = list()
            #
            for i_t in range(nT_plot):
                t = t_list[i_t]
                skeleton_pos = np.copy(positions0[i_folder, t])
                metric = metric_names_list[scatter_index_x]
                joint_name = metric.split(' ')[-1]
                joint_index = list(joint_order0).index(joint_name)
                origin = np.copy(skeleton_pos[joint_index])
                if not('angular' in metric):
                    origin[2] = 0.0
                skeleton_pos = skeleton_pos - origin[None, :]
                skeleton_pos_all.append(skeleton_pos)
                #
                color_joint = colors[i_t]
                color_bone = colors[i_t]
                zorder_offset = 6 * i_t
                #
                i_bone = 0
                joint_index_start = skeleton_edges[i_bone, 0]
                joint_index_end = skeleton_edges[i_bone, 1]
                joint_start = skeleton_pos[joint_index_start]
                joint_name = joint_order0[joint_index_end]
                joint_name_split = joint_name.split('_')
                zorder_joint = 4 # first joint is always center-joint
                ax3d.plot([joint_start[0]], [joint_start[1]], [joint_start[2]],
                          linestyle='', marker='.', markersize=2.5*linewidth_fac, markeredgewidth=0.5*linewidth_fac, color=color_joint, zorder=zorder_offset+zorder_joint)
                for i_bone in range(nJoints0-1):
                    joint_index_start = skeleton_edges[i_bone, 0]
                    joint_index_end = skeleton_edges[i_bone, 1]
                    joint_start = skeleton_pos[joint_index_start]
                    joint_end = skeleton_pos[joint_index_end]
                    vec = np.stack([joint_start, joint_end], 0)
                    joint_name = joint_order0[joint_index_end]
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
                            linestyle='-', marker=None, linewidth=1.0*linewidth_fac, color=color_bone, zorder=zorder_offset+zorder_bone)
                    ax3d.plot([joint_end[0]], [joint_end[1]], [joint_end[2]],
                            linestyle='', marker='.', markersize=2.5*linewidth_fac, markeredgewidth=0.5*linewidth_fac, color=color_joint, zorder=zorder_offset+zorder_joint)
                    #
            dxyz = 12.5
            xyz_center = np.mean(np.array(skeleton_pos_all), (0, 1))
            ax3d.set_xlim([xyz_center[0]-dxyz, xyz_center[0]+dxyz])
            ax3d.set_ylim([xyz_center[1]-dxyz, xyz_center[1]+dxyz])
            ax3d.set_zlim([xyz_center[2]-dxyz, xyz_center[2]+dxyz])
            ax3d.set_axis_off()
            azim = -90
            elev = 0
            if (side_jumps[i_folder]):
                ax3d.view_init(azim=azim, elev=elev)
            else:
                ax3d.view_init(azim=azim+180, elev=elev)    
            #
            fig3d.canvas.draw()
            plt.show(block=False)
            if save:
                fig3d.savefig(fodler_save+'/distance_vs_metric_x{:03d}_y{:03d}_3d_{:s}.svg'.format(scatter_index_x, scatter_index_y, s_min_max),
        #                          bbox_inches="tight",
                                 dpi=300,
                                 transparent=True,
                                 format='svg',
                                 pad_inches=0)
                if verbose:
                    plt.show(block=False)
        if verbose:
            print('Press any key to continue')
            input()

    plt.pause(2**-10)
    if verbose:
        plt.show()
