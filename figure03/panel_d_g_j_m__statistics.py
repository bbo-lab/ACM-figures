#!/usr/bin/env python3

import importlib
import matplotlib.pyplot as plt
import numpy as np
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
verbose = True


species = 'rat'

if species=='rat':
    folder_recon = data.path + '/datasets_figures/reconstruction'
    folder_list = list(['/20200205/arena_20200205_033300_033500', # 0 # 20200205
                        '/20200205/arena_20200205_034400_034650', # 1
                        '/20200205/arena_20200205_038400_039000', # 2
                        '/20200205/arena_20200205_039200_039500', # 3
                        '/20200205/arena_20200205_039800_040000', # 4
                        '/20200205/arena_20200205_040500_040700', # 5
                        '/20200205/arena_20200205_041500_042000', # 6
                        '/20200205/arena_20200205_045250_045500', # 7
                        '/20200205/arena_20200205_046850_047050', # 8
                        '/20200205/arena_20200205_048500_048700', # 9
                        '/20200205/arena_20200205_049200_050300', # 10
                        '/20200205/arena_20200205_050900_051300', # 11
                        '/20200205/arena_20200205_052050_052200', # 12
                        '/20200205/arena_20200205_055200_056000', # 13
                        '/20200207/arena_20200207_032200_032550', # 14 # 20200207
                        '/20200207/arena_20200207_044500_045400', # 15
                        '/20200207/arena_20200207_046300_046900', # 16
                        '/20200207/arena_20200207_048000_049800', # 17
                        '/20200207/arena_20200207_050000_050400', # 18
                        '/20200207/arena_20200207_050800_051300', # 19
                        '/20200207/arena_20200207_052250_052650', # 20
                        '/20200207/arena_20200207_053050_053500', # 21
                        '/20200207/arena_20200207_055100_056200', # 22
                        '/20200207/arena_20200207_058400_058900', # 23
                        '/20200207/arena_20200207_059450_059950', # 24
                        '/20200207/arena_20200207_060400_060800', # 25
                        '/20200207/arena_20200207_061000_062100', # 26
                        '/20200207/arena_20200207_064100_064400', # 27
                        ])
    folder_list_use = list([i.split('/')[2] for i in folder_list])
    print(folder_list_use)
    folder_list_indices = list([[[int(i.split('_')[-2]), int(i.split('_')[-1])]] for i in folder_list_use])
    print(folder_list_indices)
    frame_rate = 100.0

    add2cfg = ''

elif species=='mouse':
    folder_recon = data.path + '/dataset_analysis/'
    folder_list = list(['/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_003412_004000',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_006254_006598',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_008018_008508',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_016543_017033',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_019581_020169',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_021051_021639',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_032712_033398',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_042610_043198',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_053291_054271',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_059955_060641',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_074948_075634',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_078377_078965',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_086609_087687',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_109050_109441',
                        '/M220217_DW01/20220217/ACM/M220217_DW01/results/M220217_DW01_111401_112185',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_000300_000900',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_001500_001900',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_002900_003400',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_004600_005100',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_006100_006500',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_009800_010700',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_011600_012100',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_013200_014000',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_015600_016100',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_016700_017400',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_021600_022300',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_044600_045300',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_050700_051200',
                        '/M220217_DW03/20220217/ACM/M220217_DW03/results/M220217_DW03_060700_061800',
                        ])
    folder_list_use = list([i.split('/')[6] for i in folder_list])
    print(folder_list_use)
    folder_list_indices = list([[[int(i.split('_')[-2]), int(i.split('_')[-1])]] for i in folder_list_use])
    print(folder_list_indices)
    frame_rate = 196.0

    add2cfg = '/configuration/'

#
list_is_large_animal = list([0 for i in folder_list])

modes_list = list(['mode1', 'mode4'])

#
threshold_velocity = 25.0 # cm/s
dTime = 0.2 # s
axis = 0
if (axis == 0):
    axis_s = 'x'
elif (axis == 1):
    axis_s = 'y'
elif (axis == 2):
    axis_s = 'z'
    
paws_joint_names = list(['joint_wrist_left', 'joint_wrist_right', 'joint_ankle_left', 'joint_ankle_right',])
paws_joint_names_legend = list(['left wrist joint', 'right wrist joint', 'left ankle joint', 'right ankle joint',])
angle_joint_list = list([['joint_wrist_left', 'joint_wrist_right', 'joint_ankle_left', 'joint_ankle_right',]])
angle_joint_legend_list = list([['left wrist joint', 'right wrist joint', 'left ankle joint', 'right ankle joint',]])
nt_legend_list = list([['left elbow joint', 'right elbow joint', 'left knee joint', 'right knee joint',]])
nAngles_pairs = np.size(angle_joint_list, 0)

if __name__ == '__main__':    
    nFolders = len(folder_list)
    #
    dIndex = int(np.ceil(dTime * frame_rate))
    nAll = int(1+2*dIndex)
    #
#     for i_paw_timing in list([0, 1, 2, 3]):
    for i_paw_timing in list([2]): # this is the one shown in the figures (color: cyan)
        if (i_paw_timing == 0):
            joints_list_limb = list(['joint_shoulder_left', 'joint_elbow_left', 'joint_wrist_left', 'joint_finger_left_002'])
            joints_list_legend_limb = list(['left shoulder joint', 'left elbow joint', 'left wrist joint', 'left finger joint'])
        elif (i_paw_timing == 1):
            joints_list_limb = list(['joint_shoulder_right', 'joint_elbow_right', 'joint_wrist_right', 'joint_finger_right_002'])
            joints_list_legend_limb = list(['right shoulder joint', 'right elbow joint', 'right wrist joint', 'right finger joint'])
        elif (i_paw_timing == 2):
            joints_list_limb = list(['joint_hip_left', 'joint_knee_left', 'joint_ankle_left', 'joint_paw_hind_left', 'joint_toe_left_002'])
            joints_list_legend_limb = list(['left hip joint', 'left knee joint', 'left ankle joint', 'left hind paw', 'left toe joint'])
        elif (i_paw_timing == 3):
            joints_list_limb = list(['joint_hip_right', 'joint_knee_right', 'joint_ankle_right', 'joint_paw_hind_right', 'joint_toe_right_002'])
            joints_list_legend_limb = list(['right hip joint', 'right knee joint', 'right ankle joint', 'right hind paw', 'right toe joint'])
        nJointsLimb = len(joints_list_limb)
        #
        pos_std_all = list()
        velo_std_all = list()
        ang_std_all= list()
        ang_velo_std_all = list()
        for i_mode in range(len(modes_list)):
            mode = modes_list[i_mode]
            if (mode == 'mode4'):
                if species == 'rat':
                    add2folder = '__pcutoff9e-01'
                elif species == 'mouse':
                    add2folder = '__mode4__pcutoff9e-01'
            elif (mode == 'mode1'):
                add2folder = '__mode1__pcutoff9e-01'
            else:
                raise
            #
            pos_power_1 = np.zeros((4, nAll), dtype=np.float64)
            pos_power_2 = np.zeros((4, nAll), dtype=np.float64)
            velo_power_1 = np.zeros((4, nAll), dtype=np.float64)
            velo_power_2 = np.zeros((4, nAll), dtype=np.float64)
            acc_power_1 = np.zeros((4, nAll), dtype=np.float64)
            acc_power_2 = np.zeros((4, nAll), dtype=np.float64)
            ang_power_1 = np.zeros((nAngles_pairs, 4, nAll), dtype=np.float64)
            ang_power_2 = np.zeros((nAngles_pairs, 4, nAll), dtype=np.float64)
            ang_velo_power_1 = np.zeros((nAngles_pairs, 4, nAll), dtype=np.float64)
            ang_velo_power_2 = np.zeros((nAngles_pairs, 4, nAll), dtype=np.float64)
            detec_power_1 = np.zeros((4, nAll), dtype=np.float64)
            detec_power_2 = np.zeros((4, nAll), dtype=np.float64)
            #
            pos_power_1_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
            pos_power_2_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
            velo_power_1_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
            velo_power_2_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
            acc_power_1_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
            acc_power_2_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
            ang_power_1_limb = np.zeros((nJointsLimb-1, nAll), dtype=np.float64)
            ang_power_2_limb = np.zeros((nJointsLimb-1, nAll), dtype=np.float64)
            ang_velo_power_1_limb = np.zeros((nJointsLimb-1, nAll), dtype=np.float64)
            ang_velo_power_2_limb = np.zeros((nJointsLimb-1, nAll), dtype=np.float64)
            detec_power_1_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
            detec_power_2_limb = np.zeros((nJointsLimb, nAll), dtype=np.float64)
            #
            nPeaks_all = np.zeros(nAll, dtype=np.float64)
            nPeaks_all2 = np.zeros(nAll, dtype=np.float64)
            #
            for i_folder in range(nFolders):
                folder = folder_recon+folder_list[i_folder]+add2folder
                sys.path = list(np.copy(sys_path0))
                sys.path.append(folder+add2cfg)
                print(folder+add2cfg)
                importlib.reload(cfg)
                cfg.animal_is_large = list_is_large_animal[i_folder]
                importlib.reload(anatomy)
            
                if os.path.isfile(folder+'/save_dict.npy'):
                    print(folder)

                    # get arguments
                    folder_reqFiles = data.path + '/datasets_figures/required_files'

                    file_origin_coord = folder+'/configuration/file_origin_coord.npy'
                    if not os.path.isfile(file_origin_coord):
                        file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
                    file_calibration = folder+'/configuration/file_calibration.npy'
                    if not os.path.isfile(file_calibration):
                        file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
                    file_model = folder+'/configuration/file_model.npy'
                    if not os.path.isfile(file_model):
                        file_model = folder_reqFiles + '/model.npy'
                    file_labelsDLC = folder+'/configuration/file_labelsDLC.npy'
                    if not os.path.isfile(file_labelsDLC):
                        file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/' + cfg.file_labelsDLC.split('/')[-1]

                    args_model = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                                      cfg.scale_factor, cfg.pcutoff)
                    if ((cfg.mode == 1) or (cfg.mode == 2)):
                        args_model['use_custom_clip'] = False
                    elif ((cfg.mode == 3) or (cfg.mode == 4)):
                        args_model['use_custom_clip'] = True
                    args_model['plot'] = True
                    nBones = args_model['numbers']['nBones']
                    nMarkers = args_model['numbers']['nMarkers']
                    skeleton_edges = args_model['model']['skeleton_edges'].cpu().numpy()
                    joint_order = args_model['model']['joint_order']
                    bone_lengths_index = args_model['model']['bone_lengths_index'].cpu().numpy()
                    joint_marker_index = args_model['model']['joint_marker_index'].cpu().numpy()
                    labels_mask = args_model['labels_mask'].cpu().numpy().astype(np.float64)
                    # get save_dict
                    save_dict = np.load(folder+'/save_dict.npy', allow_pickle=True).item()
                    if ('mu_uks' in save_dict):
                        mu_uks_norm_all = np.copy(save_dict['mu_uks'][1:])
                    elif ('mu_fit' in save_dict):
                        mu_uks_norm_all = np.copy(save_dict['mu_fit'][1:])
                    else:
                        mu_uks_norm_all = np.copy(save_dict['mu'][1:])
                    #
                    # get joint connections for angle calculation
                    nJoints = args_model['numbers']['nBones']+1
                    angle_joint_connections_names = list([])
                    angle_joint_connections_names_middle = list([])
                    for i_joint in range(nJoints):
                        mask_from = (skeleton_edges[:, 0] == i_joint)
                        mask_to = (skeleton_edges[:, 1] == i_joint)
                        for i_joint_from in skeleton_edges[:, 0][mask_to]:
                            for i_joint_to in skeleton_edges[:, 1][mask_from]:
                                angle_joint_connections_names.append(list([joint_order[i_joint_from],
                                                                           joint_order[i_joint],
                                                                           joint_order[i_joint_to]]))
                                angle_joint_connections_names_middle.append(joint_order[i_joint])
                    nAngles = np.size(angle_joint_connections_names, 0)

                    nSeq = np.size(folder_list_indices[i_folder], 0)
                    for i_seq in range(nSeq):
                        frame_start = folder_list_indices[i_folder][i_seq][0]
                        frame_end = folder_list_indices[i_folder][i_seq][1]

                        mu_uks_norm = mu_uks_norm_all[frame_start-cfg.index_frame_ini:frame_end-cfg.index_frame_ini]
                        mu_uks = model.undo_normalization(torch.from_numpy(mu_uks_norm), args_model).numpy() # reverse normalization
                        labels_mask_sum = np.sum(labels_mask[frame_start-cfg.index_frame_ini:frame_end-cfg.index_frame_ini], 1)
                        # get x_ini
                        x_ini = np.load(folder+'/x_ini.npy', allow_pickle=True)

                        # free parameters
                        free_para_pose = args_model['free_para_pose'].cpu().numpy()
                        free_para_bones = np.zeros(nBones, dtype=bool)
                        free_para_markers = np.zeros(nMarkers*3, dtype=bool)    
                        free_para = np.concatenate([free_para_bones,
                                                    free_para_markers,
                                                    free_para_pose], 0)
                        # update args
                        args_model['x_torch'] = torch.from_numpy(x_ini).type(model.float_type)
                        args_model['x_free_torch'] = torch.from_numpy(x_ini[free_para]).type(model.float_type)
                        # full x
                        nT_use = np.size(mu_uks_norm, 0)
                        x = np.tile(x_ini, nT_use).reshape(nT_use, len(x_ini))
                        x[:, free_para] = mu_uks

                        # get poses
                        _, _, skeleton_pos = model.fcn_emission(torch.from_numpy(x), args_model)
                        skeleton_pos = skeleton_pos.cpu().numpy()                

                        # do coordinate transformation
                        joint_index1 = joint_order.index('joint_spine_002') # pelvis
                        joint_index2 = joint_order.index('joint_spine_004') # tibia
                        origin = skeleton_pos[:, joint_index1]
                        x_direc = skeleton_pos[:, joint_index2] - origin
                        x_direc = x_direc[:, :2]
                        x_direc = x_direc / np.sqrt(np.sum(x_direc**2, 1))[:, None]
                        alpha = np.arctan2(x_direc[:, 1], x_direc[:, 0])
                        cos_m_alpha = np.cos(-alpha)
                        sin_m_alpha = np.sin(-alpha)
                        R = np.stack([np.stack([cos_m_alpha, -sin_m_alpha], 1),
                                      np.stack([sin_m_alpha, cos_m_alpha], 1)], 1)
                        skeleton_pos = skeleton_pos - origin[:, None, :]
                        skeleton_pos_xy = np.einsum('nij,nmj->nmi', R, skeleton_pos[:, :, :2])
                        skeleton_pos[:, :, :2] = np.copy(skeleton_pos_xy)



                        position_single = np.full((nT_use, nJoints, 3), np.nan, dtype=np.float64)
                        velocity_single = np.full((nT_use, nJoints, 3), np.nan, dtype=np.float64)
                        acceleration_single = np.full((nT_use, nJoints, 3), np.nan, dtype=np.float64)
                        angle_single = np.full((nT_use, nAngles), np.nan, dtype=np.float64)
                        angle_velocity_single = np.full((nT_use, nAngles), np.nan, dtype=np.float64)
                        joints_ang_3d = np.zeros((nAngles_pairs, 4, nT_use), dtype=np.float64)
                        joints_ang_3d_limb = np.zeros((nJointsLimb-1, nT_use), dtype=np.float64)
                        joints_ang_velo_3d = np.zeros((nAngles_pairs, 4, nT_use), dtype=np.float64)
                        joints_ang_velo_3d_limb = np.zeros((nJointsLimb-1, nT_use), dtype=np.float64)
                        #
                        derivative_accuracy = 8
                        derivative_index0 = int(derivative_accuracy/2)
                        derivative_index1 = nT_use - int(derivative_accuracy/2)
                        #
                        position_single = np.copy(skeleton_pos)
                        velocity_single[4:-4] = \
                            (+1.0/280.0 * position_single[:-8] + \
                             -4.0/105.0 * position_single[1:-7] + \
                             +1.0/5.0 * position_single[2:-6] + \
                             -4.0/5.0 * position_single[3:-5] + \
    #                          0.0 * position_single[4:-4] + \
                             +4.0/5.0 * position_single[5:-3] + \
                             -1.0/5.0 * position_single[6:-2] + \
                             +4.0/105.0 * position_single[7:-1] + \
                             -1.0/280.0 * position_single[8:]) / (1.0/cfg.frame_rate)
                        acceleration_single[4:-4] = \
                            (-1.0/560.0 * position_single[:-8] + \
                             +8.0/315.0 * position_single[1:-7] + \
                             -1.0/5.0 * position_single[2:-6] + \
                             +8.0/5.0 * position_single[3:-5] + \
                             -205.0/72.0 * position_single[4:-4] + \
                             +8.0/5.0 * position_single[5:-3] + \
                             -1.0/5.0 * position_single[6:-2] + \
                             +8.0/315.0 * position_single[7:-1] + \
                             -1.0/560.0 * position_single[8:]) / (1.0/cfg.frame_rate)**2
                        for i_ang in range(nAngles):
    #                         index_joint1 = joint_order.index(angle_joint_connections_names[i_ang][0])
    #                         index_joint2 = joint_order.index(angle_joint_connections_names[i_ang][1])
    #                         index_joint3 = joint_order.index(angle_joint_connections_names[i_ang][2])  
    #                         vec1 = position_single[:, index_joint2] - position_single[:, index_joint1]
    #                         vec2 = position_single[:, index_joint3] - position_single[:, index_joint2]
    #                         vec1 = vec1 / np.sqrt(np.sum(vec1**2, 1))[:, None]
    #                         vec2 = vec2 / np.sqrt(np.sum(vec2**2, 1))[:, None]
    #                         ang = np.arccos(np.einsum('ni,ni->n', vec1, vec2)) * 180.0/np.pi

                            index_joint1 = joint_order.index(angle_joint_connections_names[i_ang][0])
                            index_joint2 = joint_order.index(angle_joint_connections_names[i_ang][1])
                            vec1 = position_single[:, index_joint2] - position_single[:, index_joint1]
                            vec1 = vec1 / np.sqrt(np.sum(vec1**2, 1))[:, None]
                            # angle between bone and walking direction
                            vec0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                            vec_use = np.copy(vec1)
                            ang = np.arccos(np.einsum('i,ni->n', vec0, vec_use)) * 180.0/np.pi
                            # source: https://math.stackexchange.com/questions/2140504/how-to-calculate-signed-angle-between-two-vectors-in-3d
                            n_cross = np.cross(vec0, vec_use)
                            n_cross = n_cross / np.sqrt(np.sum(n_cross**2, 1))[:, None]
                            sin = np.einsum('ni,ni->n', np.cross(n_cross, vec0), vec_use)
                            mask = (sin < 0.0)
                            ang[mask] = 360.0 - ang[mask]
                            if np.any(mask):
                                print('WARNING: {:07d}/{:07d} signed angles'.format(np.sum(mask), len(mask.ravel())))

                            angle_single[:, i_ang] = np.copy(ang)
                            angle_velocity_single[4:-4, i_ang] =  \
                                (+1.0/280.0 * ang[:-8] + \
                                 -4.0/105.0 * ang[1:-7] + \
                                 +1.0/5.0 * ang[2:-6] + \
                                 -4.0/5.0 * ang[3:-5] + \
        #                          0.0 * ang[4:-4] + \
                                 +4.0/5.0 * ang[5:-3] + \
                                 -1.0/5.0 * ang[6:-2] + \
                                 +4.0/105.0 * ang[7:-1] + \
                                 -1.0/280.0 * ang[8:]) / (1.0/cfg.frame_rate)
                        #
                        joints_pos_1d = np.zeros((4, nT_use), dtype=np.float64)
                        dy_dx = np.full(np.shape(joints_pos_1d), np.nan, dtype=np.float64)
                        dy_dx2 = np.full(np.shape(joints_pos_1d), np.nan, dtype=np.float64)
                        for i_paw in range(4):
                            joint_index = joint_order.index(paws_joint_names[i_paw])
                            joints_pos_1d[i_paw] = position_single[:, joint_index, axis]
                            dy_dx[i_paw] = velocity_single[:, joint_index, axis]
                            dy_dx2[i_paw] = acceleration_single[:, joint_index, axis]
                        joints_pos_1d_limb = np.zeros((nJointsLimb, nT_use), dtype=np.float64)
                        dy_dx_limb = np.full(np.shape(joints_pos_1d_limb), np.nan, dtype=np.float64)
                        dy_dx2_limb = np.full(np.shape(joints_pos_1d_limb), np.nan, dtype=np.float64)
                        for i_joint in range(nJointsLimb):
                            joint_index = joint_order.index(joints_list_limb[i_joint])
                            joints_pos_1d_limb[i_joint] = position_single[:, joint_index, axis]
                            dy_dx_limb[i_joint] = velocity_single[:, joint_index, axis]
                            dy_dx2_limb[i_joint] = acceleration_single[:, joint_index, axis] 
    #                     joints_ang_3d = np.zeros((nAngles_pairs, 4, nT_use), dtype=np.float64)
                        for i_ang_pair in range(nAngles_pairs):
                            for i_paw in range(4):
                                joint_index = angle_joint_connections_names_middle.index(angle_joint_list[i_ang_pair][i_paw])
                                joints_ang_3d[i_ang_pair, i_paw] = angle_single[:, joint_index]
                                joints_ang_velo_3d[i_ang_pair, i_paw] = angle_velocity_single[:, joint_index]
    #                     joints_ang_3d_limb = np.zeros((nJointsLimb-1, nT_use), dtype=np.float64)
                        for i_joint in range(nJointsLimb-1):
                            joint_index = angle_joint_connections_names_middle.index(joints_list_limb[:-1][i_joint])
                            joints_ang_3d_limb[i_joint] = angle_single[:, joint_index]
                            joints_ang_velo_3d_limb[i_joint] = angle_velocity_single[:, joint_index]
    #                     # 1D position (i.e. velocity of x/y/z combined)
    #                     joints_pos_1d = np.zeros((4, nT_use), dtype=np.float64)
    #                     for i_paw in range(4):
    #                         joint_index = joint_order.index(paws_joint_names[i_paw])
    #                         joints_pos_1d[i_paw] = skeleton_pos[:, joint_index, axis] # cm
    #                     joints_pos_1d_limb = np.zeros((nJointsLimb, nT_use), dtype=np.float64)
    #                     for i_joint in range(nJointsLimb):
    #                         joint_index = joint_order.index(joints_list_limb[i_joint])
    #                         joints_pos_1d_limb[i_joint] = skeleton_pos[:, joint_index, axis] # cm
    #                     # 1st 1D derivative (i.e. x-, y-, or z-velocity & -acceleration)
    #                     derivative_accuracy = 2
    #                     derivative_index0 = int(derivative_accuracy/2)
    #                     derivative_index1 = nT_use - int(derivative_accuracy/2)
    #                     dy_dx = np.full(np.shape(joints_pos_1d), np.nan, dtype=np.float64)
    #                     dy_dx2 = np.full(np.shape(joints_pos_1d), np.nan, dtype=np.float64)
    #                     dy_dx_limb = np.full(np.shape(joints_pos_1d_limb), np.nan, dtype=np.float64)
    #                     dy_dx2_limb = np.full(np.shape(joints_pos_1d_limb), np.nan, dtype=np.float64)
    #                     dy_dx[:, derivative_index0:derivative_index1] = (0.5 * joints_pos_1d[:, 2:] - 0.5 * joints_pos_1d[:, :-2]) / (1.0/frame_rate) # cm/s
    #                     dy_dx2[:, derivative_index0:derivative_index1] = (joints_pos_1d[:, 2:] - 2.0 * joints_pos_1d[:, 1:-1] + joints_pos_1d[:, :-2]) / (1.0/frame_rate)**2 # cm/s^2
    #                     dy_dx_limb[:, derivative_index0:derivative_index1] = (0.5 * joints_pos_1d_limb[:, 2:] - 0.5 * joints_pos_1d_limb[:, :-2]) / (1.0/frame_rate) # cm/s
    #                     dy_dx2_limb[:, derivative_index0:derivative_index1] = (joints_pos_1d_limb[:, 2:] - 2.0 * joints_pos_1d_limb[:, 1:-1] + joints_pos_1d_limb[:, :-2]) / (1.0/frame_rate)**2 # cm/s^2
    #                     # angles
    #                     joints_ang_3d = np.zeros((nAngles_pairs, 4, nT_use), dtype=np.float64)
    #                     for i_ang_pair in range(nAngles_pairs):
    #                         for i_paw in range(4):
    #                             joint_index = angle_joint_connections_names_middle.index(angle_joint_list[i_ang_pair][i_paw])
    #                             joints_ang_3d[i_ang_pair, i_paw] = angles_joints_3d_use[:, joint_index] # deg
    #                     joints_ang_3d_limb = np.zeros((nJointsLimb-1, nT_use), dtype=np.float64)
    #                     for i_joint in range(nJointsLimb-1):
    #                         joint_index = angle_joint_connections_names_middle.index(joints_list_limb[:-1][i_joint])
    #                         joints_ang_3d_limb[i_joint] = angles_joints_3d_use[:, joint_index] # deg
                        # detections
                        detections = np.zeros((4, nT_use), dtype=np.float64)
                        for i_paw in range(4):
                            joint_index = joint_order.index(paws_joint_names[i_paw])
                            marker_index = np.where(joint_index == abs(joint_marker_index))[0][0]
                            detections[i_paw] = labels_mask_sum[:, marker_index]
                        detections_limb = np.zeros((nJointsLimb, nT_use), dtype=np.float64)
                        for i_joint in range(nJointsLimb):
                            joint_index = joint_order.index(joints_list_limb[i_joint])
                            marker_index = np.where(joint_index == abs(joint_marker_index))[0][0]
                            detections_limb[i_joint] = labels_mask_sum[:, marker_index]

                        # find peak indices
                        derivative_accuracy_index = int(derivative_accuracy/2)
                        dy_dx_use = dy_dx[:, derivative_index0:derivative_index1]
                        mask = (dy_dx_use > threshold_velocity)
                        index = np.arange(nT_use - 2 * derivative_accuracy_index, dtype=np.int64)
                        indices_peaks = list()
                        for i_paw in range(4):
                            indices_peaks.append(list())
                            if np.any(mask[i_paw]):
                                diff_index = np.diff(index[mask[i_paw]])
                                nPeaks_mask = (diff_index > 1)
                                nPeaks = np.sum(nPeaks_mask) + 1
                                if (nPeaks > 1):
                                    for i_peak in range(nPeaks):
                                        if (i_peak == 0):
                                            index_start = 0
                                            index_end = index[mask[i_paw]][1:][nPeaks_mask][i_peak]
                                        elif (i_peak == nPeaks-1):
                                            index_start = np.copy(index_end)
                                            index_end = nT_use - 2 * derivative_accuracy_index
                                        else:
                                            index_start = np.copy(index_end)
                                            index_end = index[mask[i_paw]][1:][nPeaks_mask][i_peak]
                                        dy_dx_index = index[index_start:index_end]
                                        if (np.size(dy_dx_index) > 0):
                                            index_peak = np.argmax(dy_dx_use[i_paw][dy_dx_index]) + dy_dx_index[0] + derivative_accuracy_index
                                            indices_peaks[i_paw].append(index_peak)
                                else:
                                    index_peak = np.argmax(dy_dx_use[i_paw]) + derivative_accuracy_index
                                    indices_peaks[i_paw].append(index_peak)

                        nPeaks = len(indices_peaks[i_paw_timing])
                        for i_peak in range(nPeaks):
                            index_peak = indices_peaks[i_paw_timing][i_peak]
                            # index for position and angle
                            index0 = index_peak - dIndex
                            index1 = index_peak + dIndex + 1
                            index0_use = 0
                            index1_use = nAll
                            if (index0 < 0):
                                index0_use = abs(index0)
                                index0 = 0
                            if (index1 > nT_use):
                                index1_use = nAll - (index1 - nT_use)
                                index1 = nT_use
                            # index for velocity and acceleration
                            index0_2 = np.copy(index0)
                            index1_2 = np.copy(index1)
                            index0_use2 = np.copy(index0_use)
                            index1_use2 = np.copy(index1_use)
                            if (index0_2 < derivative_index0):
                                diff = derivative_index0 - index0_2
                                index0_2 = index0_2 + diff
                                index0_use2 = index0_use2 + diff
                            if (index1_2 > derivative_index1):
                                diff = index1_2 - derivative_index1
                                index1_2 = index1_2 - diff
                                index1_use2 = index1_use2 - diff
                            #
                            nPeaks_all[index0_use:index1_use] = nPeaks_all[index0_use:index1_use] + 1
                            nPeaks_all2[index0_use2:index1_use2] = nPeaks_all2[index0_use2:index1_use2] + 1
                            # compare all four limbs
                            for i_paw in range(4):
                                pos = joints_pos_1d[i_paw, index0:index1]
                                pos_power_1[i_paw, index0_use:index1_use] = pos_power_1[i_paw, index0_use:index1_use] + pos
                                pos_power_2[i_paw, index0_use:index1_use] = pos_power_2[i_paw, index0_use:index1_use] + pos**2
                                velo = dy_dx[i_paw, index0_2:index1_2]
                                velo_power_1[i_paw, index0_use2:index1_use2] = velo_power_1[i_paw, index0_use2:index1_use2] + velo
                                velo_power_2[i_paw, index0_use2:index1_use2] = velo_power_2[i_paw, index0_use2:index1_use2] + velo**2
                                acc = dy_dx2[i_paw, index0_2:index1_2]
                                acc_power_1[i_paw, index0_use2:index1_use2] = acc_power_1[i_paw, index0_use2:index1_use2] + acc
                                acc_power_2[i_paw, index0_use2:index1_use2] = acc_power_2[i_paw, index0_use2:index1_use2] + acc**2
                                detec = detections[i_paw, index0:index1]
                                detec_power_1[i_paw, index0_use:index1_use] = detec_power_1[i_paw, index0_use:index1_use] + detec
                                detec_power_2[i_paw, index0_use:index1_use] = detec_power_2[i_paw, index0_use:index1_use] + detec**2
                                #
                                for i_ang_pairs in range(nAngles_pairs):
                                    ang = joints_ang_3d[i_ang_pairs, i_paw, index0:index1]
                                    ang_power_1[i_ang_pairs, i_paw, index0_use:index1_use] = ang_power_1[i_ang_pairs, i_paw, index0_use:index1_use] + ang
                                    ang_power_2[i_ang_pairs, i_paw, index0_use:index1_use] = ang_power_2[i_ang_pairs, i_paw, index0_use:index1_use] + ang**2
                                    ang_velo = joints_ang_velo_3d[i_ang_pairs, i_paw, index0_2:index1_2]
                                    ang_velo_power_1[i_ang_pairs, i_paw, index0_use2:index1_use2] = ang_velo_power_1[i_ang_pairs, i_paw, index0_use2:index1_use2] + ang_velo
                                    ang_velo_power_2[i_ang_pairs, i_paw, index0_use2:index1_use2] = ang_velo_power_2[i_ang_pairs, i_paw, index0_use2:index1_use2] + ang_velo**2
                            # compare joints of single limb
                            for i_joint in range(nJointsLimb):
                                pos = joints_pos_1d_limb[i_joint, index0:index1]
                                pos_power_1_limb[i_joint, index0_use:index1_use] = pos_power_1_limb[i_joint, index0_use:index1_use] + pos
                                pos_power_2_limb[i_joint, index0_use:index1_use] = pos_power_2_limb[i_joint, index0_use:index1_use] + pos**2
                                velo = dy_dx_limb[i_joint, index0_2:index1_2]
                                velo_power_1_limb[i_joint, index0_use2:index1_use2] = velo_power_1_limb[i_joint, index0_use2:index1_use2] + velo
                                velo_power_2_limb[i_joint, index0_use2:index1_use2] = velo_power_2_limb[i_joint, index0_use2:index1_use2] + velo**2
                                acc = dy_dx2_limb[i_joint, index0_2:index1_2]
                                acc_power_1_limb[i_joint, index0_use2:index1_use2] = acc_power_1_limb[i_joint, index0_use2:index1_use2] + acc
                                acc_power_2_limb[i_joint, index0_use2:index1_use2] = acc_power_2_limb[i_joint, index0_use2:index1_use2] + acc**2
                                detec = detections_limb[i_joint, index0:index1]
                                detec_power_1_limb[i_joint, index0_use:index1_use] = detec_power_1_limb[i_joint, index0_use:index1_use] + detec
                                detec_power_2_limb[i_joint, index0_use:index1_use] = detec_power_2_limb[i_joint, index0_use:index1_use] + detec**2
                            for i_joint in range(nJointsLimb-1):
                                ang = joints_ang_3d_limb[i_joint, index0:index1]
                                ang_power_1_limb[i_joint, index0_use:index1_use] = ang_power_1_limb[i_joint, index0_use:index1_use] + ang
                                ang_power_2_limb[i_joint, index0_use:index1_use] = ang_power_2_limb[i_joint, index0_use:index1_use] + ang**2    
                                ang_velo = joints_ang_velo_3d_limb[i_joint, index0_2:index1_2]
                                ang_velo_power_1_limb[i_joint, index0_use2:index1_use2] = ang_velo_power_1_limb[i_joint, index0_use2:index1_use2] + ang_velo
                                ang_velo_power_2_limb[i_joint, index0_use2:index1_use2] = ang_velo_power_2_limb[i_joint, index0_use2:index1_use2] + ang_velo**2 

            time_fac = 1e3
            time = (np.arange(nAll, dtype=np.float64) - float(dIndex)) * (1.0/frame_rate) * time_fac
            velo_power_1_limb = velo_power_1_limb / time_fac
            velo_power_2_limb = velo_power_2_limb / time_fac**2
            acc_power_1_limb = acc_power_1_limb / (time_fac**2)
            acc_power_2_limb = acc_power_2_limb / (time_fac**2)**2
            ang_velo_power_1_limb = ang_velo_power_1_limb / time_fac
            ang_velo_power_2_limb = ang_velo_power_2_limb / time_fac**2
            velo_power_1 = velo_power_1 / time_fac
            velo_power_2 = velo_power_2 / time_fac**2
            acc_power_1 = acc_power_1 / (time_fac**2)
            acc_power_2 = acc_power_2 / (time_fac**2)**2
            ang_velo_power_1 = ang_velo_power_1 / time_fac
            ang_velo_power_2 = ang_velo_power_2 / time_fac**2
            #
            pos_avg = pos_power_1 / nPeaks_all[None, :]
            pos_std = np.sqrt((pos_power_2 - (pos_power_1**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])
            velo_avg = velo_power_1 / nPeaks_all2[None :]
            velo_std = np.sqrt((velo_power_2 - (velo_power_1**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
            acc_avg = acc_power_1 / nPeaks_all2[None, :]
            acc_std = np.sqrt((acc_power_2 - (acc_power_1**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
            ang_avg = ang_power_1 / nPeaks_all[None, :]
            ang_std = np.sqrt((ang_power_2 - (ang_power_1**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])
            ang_velo_avg = ang_velo_power_1 / nPeaks_all2[None, :]
            ang_velo_std = np.sqrt((ang_velo_power_2 - (ang_velo_power_1**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
            detec_avg = detec_power_1 / nPeaks_all[None, :]
            detec_std = np.sqrt((detec_power_2 - (detec_power_1**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])
            #
            pos_avg_limb = pos_power_1_limb / nPeaks_all[None, :]
            pos_std_limb = np.sqrt((pos_power_2_limb - (pos_power_1_limb**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])
            velo_avg_limb = velo_power_1_limb / nPeaks_all2[None, :]
            velo_std_limb = np.sqrt((velo_power_2_limb - (velo_power_1_limb**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
            acc_avg_limb = acc_power_1_limb / nPeaks_all2[None, :]
            acc_std_limb = np.sqrt((acc_power_2_limb - (acc_power_1_limb**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
            ang_avg_limb = ang_power_1_limb / nPeaks_all[None, :]
            ang_std_limb = np.sqrt((ang_power_2_limb - (ang_power_1_limb**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])
            ang_velo_avg_limb = ang_velo_power_1_limb / nPeaks_all2[None, :]
            ang_velo_std_limb = np.sqrt((ang_velo_power_2_limb - (ang_velo_power_1_limb**2 / nPeaks_all2[None, :])) / nPeaks_all2[None, :])
            detec_avg_limb = detec_power_1_limb / nPeaks_all[None, :]
            detec_std_limb = np.sqrt((detec_power_2_limb - (detec_power_1_limb**2 / nPeaks_all[None, :])) / nPeaks_all[None, :])

            # save different modes
            pos_std_all.append(pos_std)
            velo_std_all.append(velo_std)
            ang_std_all.append(ang_std[0])
            ang_velo_std_all.append(ang_velo_std[0])
            
        # stats
        import scipy.stats
        #
        print()
        print('timed paw:')
        print('{:s}'.format(paws_joint_names_legend[i_paw_timing]))
        print()
        #
        i_paw = np.ones(4, dtype=bool)
        #
        print('Summary:')
        print('\tposition:')
        x = pos_std_all[0][i_paw].ravel()
        y = pos_std_all[1][i_paw].ravel()
        print('\t\tmode1:\t{:0.8e} +/- {:0.2e}\t{:0.2e}'.format(np.nanmean(x), np.nanstd(x), np.nanmedian(x)))
        print('\t\tmode4:\t{:0.8e} +/- {:0.2e}\t{:0.2e}'.format(np.nanmean(y), np.nanstd(y), np.nanmedian(y)))
        print('\tvelocity:')
        x = velo_std_all[0][i_paw].ravel()
        y = velo_std_all[1][i_paw].ravel()
        print('\t\tmode1:\t{:0.8e} +/- {:0.2e}\t{:0.2e}'.format(np.nanmean(x), np.nanstd(x), np.nanmedian(x)))
        print('\t\tmode4:\t{:0.8e} +/- {:0.2e}\t{:0.2e}'.format(np.nanmean(y), np.nanstd(y), np.nanmedian(y)))
        print('\tangle:')
        x = ang_std_all[0][i_paw].ravel()
        y = ang_std_all[1][i_paw].ravel()
        print('\t\tmode1:\t{:0.8e} +/- {:0.2e}\t{:0.2e}'.format(np.nanmean(x), np.nanstd(x), np.nanmedian(x)))
        print('\t\tmode4:\t{:0.8e} +/- {:0.2e}\t{:0.2e}'.format(np.nanmean(y), np.nanstd(y), np.nanmedian(y)))
        print('\tangular velocity:')
        x = ang_velo_std_all[0][i_paw].ravel()
        y = ang_velo_std_all[1][i_paw].ravel()
        print('\t\tmode1:\t{:0.8e} +/- {:0.2e}\t{:0.2e}'.format(np.nanmean(x), np.nanstd(x), np.nanmedian(x)))
        print('\t\tmode4:\t{:0.8e} +/- {:0.2e}\t{:0.2e}'.format(np.nanmean(y), np.nanstd(y), np.nanmedian(y)))
        print()
        #
        print('Mann-Whitney rank test:')
        alternative = 'greater'
        #
        print('\tposition:')
        x = pos_std_all[0][i_paw].ravel()
        y = pos_std_all[1][i_paw].ravel()
        mask = ~np.logical_or(np.isnan(x), np.isnan(y))
        statistic, pvalue = scipy.stats.mannwhitneyu(x[mask], y[mask],
                                                 alternative=alternative)
        print('\t\tstatistic:\t{:0.8e}'.format(statistic))
        print('\t\tpvalue:\t\t{:0.8e}'.format(pvalue))
        print('\tvelocity:')
        x = velo_std_all[0][i_paw].ravel()
        y = velo_std_all[1][i_paw].ravel()
        mask = ~np.logical_or(np.isnan(x), np.isnan(y))
        statistic, pvalue = scipy.stats.mannwhitneyu(x[mask], y[mask],
                                                 alternative=alternative)
        print('\t\tstatistic:\t{:0.8e}'.format(statistic))
        print('\t\tpvalue:\t\t{:0.8e}'.format(pvalue))
        print('\tangle:')
        x = ang_std_all[0][i_paw].ravel()
        y = ang_std_all[1][i_paw].ravel()
        mask = ~np.logical_or(np.isnan(x), np.isnan(y))
        statistic, pvalue = scipy.stats.mannwhitneyu(x[mask], y[mask],
                                                 alternative=alternative)
        print('\t\tstatistic:\t{:0.8e}'.format(statistic))
        print('\t\tpvalue:\t\t{:0.8e}'.format(pvalue))
        print('\tangular velocity:')
        x = ang_velo_std_all[0][i_paw].ravel()
        y = ang_velo_std_all[1][i_paw].ravel()
        mask = ~np.logical_or(np.isnan(x), np.isnan(y))
        statistic, pvalue = scipy.stats.mannwhitneyu(x[mask], y[mask],
                                                 alternative=alternative)
        print('\t\tstatistic:\t{:0.8e}'.format(statistic))
        print('\t\tpvalue:\t\t{:0.8e}'.format(pvalue))
        print()
