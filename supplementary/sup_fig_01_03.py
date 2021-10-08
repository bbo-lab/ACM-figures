#!/usr/bin/env python3

import importlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
import routines_math as rout_m
sys_path0 = np.copy(sys.path)

save = False
verbose = True

saveFolder = os.path.abspath('figures')

folder_reconstruction = data.path+'/reconstruction'
folders = list(['arena_20200205_calibration_on',
                'arena_20200207_calibration_on',
                'table_1_20210511_calibration',
                'table_2_20210511_calibration__more',
                'table_3_20210511_calibration',
                'table_4_20210511_calibration__more',])
dates_list = list(['20200205', '20200207', '20210511_1', '20210511_2', '20210511_3', '20210511_4'])

if __name__ == '__main__': 
    fig = plt.figure(1, figsize=(8, 8))
    fig.canvas.manager.window.move(0, 0)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    
    for i_folder in range(len(folders)):
        folder = folder_reconstruction + '/' + dates_list[i_folder] + '/' + folders[i_folder]

        sys.path = list(np.copy(sys_path0))
        sys.path.append(folder)
        importlib.reload(cfg)
        cfg.animal_is_large = 0
        importlib.reload(anatomy)

        folder_reqFiles = data.path + '/required_files' 
        file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
        file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
        file_model = folder_reqFiles + '/model.npy'
        file_labelsDLC = folder_reqFiles + '/labels_dlc_dummy.npy' # DLC labels are actually never needed here
        
        file_calib = folder+'/'+'x_calib.npy'
        x_calib = np.load(file_calib, allow_pickle=True)
    
        # get arguments
        args_model = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                          cfg.scale_factor, cfg.pcutoff)
        if ((cfg.mode == 1) or (cfg.mode == 2)):
            args_model['use_custom_clip'] = False
        elif ((cfg.mode == 3) or (cfg.mode == 4)):
            args_model['use_custom_clip'] = True
        args_model['plot'] = True
        nBones = args_model['numbers']['nBones']
        nMarkers = args_model['numbers']['nMarkers']
        nPara_bones = args_model['nPara_bones']
        nPara_markers = args_model['nPara_markers']
        nPara_pose = args_model['nPara_pose']
        skeleton_edges = args_model['model']['skeleton_edges'].cpu().numpy()
        joint_order = args_model['model']['joint_order']
        joint_marker_index = args_model['model']['joint_marker_index'].cpu().numpy()
        #
        bounds_pose = args_model['bounds_pose'][6:].cpu().numpy() # get rid of t0 and r0

        # get x_calib
        x_align = np.load(folder+'/x_align.npy', allow_pickle=True)
        # free parameters
        free_para_pose = args_model['free_para_pose'].cpu().numpy()
        free_para_bones = np.zeros(nBones, dtype=bool)
        free_para_markers = np.zeros(nMarkers*3, dtype=bool)    
        free_para = np.concatenate([free_para_bones,
                                    free_para_markers,
                                    free_para_pose], 0)
        
        # change global rotation
        R = np.identity(3, dtype=np.float64)
        R = np.array([[0.0, 0.0, -1.0],
                       [-1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]], dtype=np.float64)
        r = rout_m.rotMat2rodrigues_single(R)
        
        # get pose
        x = np.copy(x_align)
        x[nPara_bones+nPara_markers:nPara_bones+nPara_markers+3] = 0.0
        x[nPara_bones+nPara_markers+3:nPara_bones+nPara_markers+6] = r
        x[nPara_bones+nPara_markers+6:] = bounds_pose[:, 0] + 0.5 * (bounds_pose[:, 1] - bounds_pose[:, 0])
        #
        marker_pos2d, marker_pos3d, skeleton_pos3d = model.fcn_emission(torch.from_numpy(x[None, :]), args_model)
        marker_pos2d = marker_pos2d[0].cpu().numpy()
        marker_pos3d = marker_pos3d[0].cpu().numpy()
        skeleton_pos3d = skeleton_pos3d[0].cpu().numpy()

        ax.clear()
        ax.set_axis_off()
        ax.set_proj_type('ortho')
        
        for i_bone in range(nBones):
            joint_index_start = skeleton_edges[i_bone, 0]
            joint_index_end = skeleton_edges[i_bone, 1]
            joint_start = skeleton_pos3d[joint_index_start]
            joint_end = skeleton_pos3d[joint_index_end]
            vec = np.stack([joint_start, joint_end], 0)
            color_bone = 'black'
            ax.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                    linestyle='-', marker='', linewidth=2, color=color_bone, zorder=1)
            ax.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                    linestyle='', marker='.', markersize=4, markeredgewidth=1, color='darkgray', zorder=2)
        # add scales 
        linewidth = 0.5
        fontsize = 9
        fontname = 'Arial'
        #
        scale_length = 5.0 # cm
        scale_start = np.array([-9.0, -7.5, -7.5], dtype=np.float64)
        for i_ax in list([0, 1, 2]):
            scale_end = np.copy(scale_start)
            scale_end[i_ax] = scale_end[i_ax] + scale_length
            vec = np.stack([scale_start, scale_end], 0)
            ax.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                        linestyle='-', marker='', linewidth=5.0*linewidth, color='black', zorder=3, alpha=1.0)
        color_marker_line = 'blue'
        color_marker = 'green'
        for i_marker in range(nMarkers):
            marker = marker_pos3d[i_marker]
            joint_index = joint_marker_index[i_marker]
            joint = skeleton_pos3d[joint_index]
            vec = np.stack([joint, marker], 0)
            ax.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                    linestyle='-', marker='', linewidth=2, color=color_marker_line, zorder=3, alpha=0.5)
            ax.plot([vec[1, 0]], [vec[1, 1]], [vec[1, 2]],
                    linestyle='', marker='.', markersize=6, markeredgewidth=1, color=color_marker, zorder=4, alpha=1.0)
        
        lim_value = 35.0
        ax.set_xlim([-lim_value, 0.0])
        ax.set_ylim([-lim_value/2, lim_value/2])
        ax.set_zlim([-lim_value/2, lim_value/2])
        #
        elev_list = list([0, 90])
        proj_list = list(['xy', 'xz'])
        date = dates_list[i_folder]
        for i_proj in range(2):
            proj = proj_list[i_proj]
            elev = elev_list[i_proj]
            ax.view_init(elev, -90)  
            fig.canvas.draw()
            if save:
                fig.savefig(saveFolder+'/skeleton_{:s}_{:s}.svg'.format(proj, date),
        #                     bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0)
            if verbose:
                plt.show(block=False)
                print('Press any key to continue')
                input()
            
    # get ini pose
    x = np.copy(x_align)
    x[:nPara_bones] = 3.0
    x[nPara_bones+nPara_markers:] = 0.0
    x[nPara_bones+nPara_markers:nPara_bones+nPara_markers+3] = 0.0
    x[nPara_bones+nPara_markers+3:nPara_bones+nPara_markers+6] = r
    x[nPara_bones+nPara_markers+6:] = bounds_pose[:, 0] + 0.5 * (bounds_pose[:, 1] - bounds_pose[:, 0])
    #
    marker_pos2d, marker_pos3d, skeleton_pos3d = model.fcn_emission(torch.from_numpy(x[None, :]), args_model)
    marker_pos2d = marker_pos2d[0].cpu().numpy()
    marker_pos3d = marker_pos3d[0].cpu().numpy()
    skeleton_pos3d = skeleton_pos3d[0].cpu().numpy()

    ax.clear()
    ax.set_axis_off()
    ax.set_proj_type('ortho')
    #
    color_bone = 'black'
    for i_bone in range(nBones):
        joint_index_start = skeleton_edges[i_bone, 0]
        joint_index_end = skeleton_edges[i_bone, 1]
        joint_start = skeleton_pos3d[joint_index_start]
        joint_end = skeleton_pos3d[joint_index_end]
        vec = np.stack([joint_start, joint_end], 0)
        ax.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                linestyle='-', marker='', linewidth=2, color=color_bone, zorder=1)
        ax.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                linestyle='', marker='.', markersize=4, markeredgewidth=1, color='darkgray', zorder=2)
        
    # add scales 
    linewidth = 0.5
    fontsize = 9
    fontname = 'Arial'
    #
    scale_length = 5.0 # cm
    scale_start = np.array([-25.0, -7.5, -7.5], dtype=np.float64)
    for i_ax in list([0, 1, 2]):
        scale_end = np.copy(scale_start)
        scale_end[i_ax] = scale_end[i_ax] + scale_length
        vec = np.stack([scale_start, scale_end], 0)
        ax.plot(vec[:, 0], vec[:, 1], vec[:, 2],
                    linestyle='-', marker='', linewidth=5.0*linewidth, color='black', zorder=3, alpha=1.0)
    #
    lim_value = 30.0
    ax.set_xlim([-lim_value, 0.0])
    ax.set_ylim([-lim_value/2, lim_value/2])
    ax.set_zlim([-lim_value/2, lim_value/2])

    elev_list = list([0, 90])
    proj_list = list(['xy', 'xz'])
    date = 'ini'
    for i_proj in range(2):
        proj = proj_list[i_proj]
        elev = elev_list[i_proj]
        ax.view_init(elev, -90)  
        fig.canvas.draw()
        #
        if save:
            fig.savefig(saveFolder+'/skeleton_{:s}_{:s}.svg'.format(proj, date),
    #                     bbox_inches="tight",
                         dpi=300,
                         transparent=True,
                         format='svg',
                         pad_inches=0)       

    # sort and replace items in joint_order if required
    list_replace = list()
    nReplace = np.shape(list_replace)[0]
    joint_order0 = list(np.copy(joint_order))
    for i_replace in range(nReplace):
        joint_order[joint_order0.index(list_replace[i_replace][0])] = list_replace[i_replace][1]

    joint_list_legend = list()
    for i_joint in range(nBones+1):
        joint_name = sorted(joint_order)[i_joint]
        joint_name_split = joint_name.split('_')[1:]
        if ('left' in joint_name_split):
            joint_name_split[joint_name_split.index('left')] = '(left)'
        elif ('right' in joint_name_split):
            joint_name_split[joint_name_split.index('right')] = '(right)'
        joint_name_split = ' '.join(joint_name_split)
        joint_name_use = '{:02d}. {:s}'.format(i_joint, joint_name_split)
        joint_list_legend.append(joint_name_use)
    joint_list_legend = '\n'.join(joint_list_legend)
    
    fig2 = plt.figure(2, figsize=(8, 8))
    fig2.canvas.manager.window.move(0, 0)
    ax2 = fig2.add_axes([0, 0, 1, 1])
    
    # get pose
    x = np.copy(x_align)
    x[:nPara_bones] = 3.0
    x[nPara_bones+nPara_markers:] = 0.0
    x[nPara_bones+nPara_markers:nPara_bones+nPara_markers+3] = 0.0
    x[nPara_bones+nPara_markers+3:nPara_bones+nPara_markers+6] = r
    x[nPara_bones+nPara_markers+6:] = bounds_pose[:, 0] + 0.5 * (bounds_pose[:, 1] - bounds_pose[:, 0])
    #
    marker_pos2d, marker_pos3d, skeleton_pos3d = model.fcn_emission(torch.from_numpy(x[None, :]), args_model)
    marker_pos2d = marker_pos2d[0].cpu().numpy()
    marker_pos3d = marker_pos3d[0].cpu().numpy()
    skeleton_pos3d = skeleton_pos3d[0].cpu().numpy()

    ax2.clear()
    ax2.set_axis_off()
    ax2.set_aspect(1)
    #
    color_bone = 'black'
    for i_bone in range(nBones):
        joint_index_start = skeleton_edges[i_bone, 0]
        joint_index_end = skeleton_edges[i_bone, 1]
        joint_start = skeleton_pos3d[joint_index_start]
        joint_end = skeleton_pos3d[joint_index_end]
        vec = np.stack([joint_start, joint_end], 0)
        ax2.plot(vec[:, 0], vec[:, 1],
                linestyle='-', marker='', linewidth=2, color=color_bone, zorder=1)
        ax2.plot(vec[:, 0], vec[:, 1],
                linestyle='', marker='.', markersize=4, markeredgewidth=1, color='darkgray', zorder=2)
    #
    loc_max_x = 0.0
    loc_max_y = 0.0
    skeleton_pos3d_use = np.copy(skeleton_pos3d)
    skeleton_pos3d_use = skeleton_pos3d_use[np.argsort(joint_order)]
    for i_joint in range(nBones+1):
        ax2.text(skeleton_pos3d_use[i_joint, 0], skeleton_pos3d_use[i_joint, 1], i_joint,
                 va='center', ha='center', color='red', alpha=1.0,
                 fontsize=9)
        loc_max_x = max(loc_max_x, skeleton_pos3d_use[i_joint, 0])
        loc_max_y = max(loc_max_y, skeleton_pos3d_use[i_joint, 1])
    #
    offset_x = 1
    offset_y = 0
    ax2.text(loc_max_x + offset_x, 0.0 + offset_y, joint_list_legend,
             va='center', ha='left', color='black', alpha=1.0,
             fontsize=9)
    #
    ax2.set_xlim([-32.50, 10.0])
    ax2.set_ylim([-10.0, 10.0])
    #
    if save:
        fig2.savefig(saveFolder+'/joints.svg',
    #                     bbox_inches="tight",
                      dpi=300,
                      transparent=True,
                      format='svg',
                      pad_inches=0)     
    fig2.canvas.draw()

    if verbose:
        plt.show()