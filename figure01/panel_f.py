#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../ACM'))
import data

save = True
verbose = True # WARNING: Script reuses figure. Needs to be saved to be able to see relevant figures.

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

fontsize = 8
linewidth_scalebar = 1.0
linewidth_bone = 0.5
markersize_joint = 1.0
fontname = "Arial"

def draw_mri2d(file_mri, file_skeleton, mri_resolution, folder_save):
    print(file_mri)
    markers_dict = np.load(file_skeleton, allow_pickle=True).item()
    links = markers_dict['links']
    joints = markers_dict['joints']    
    nBones = len(links.keys())
    nJoints = nBones + 1
    
    mri_data = np.load(file_mri, allow_pickle=True)
    mri_data = mri_data - np.nanmin(mri_data)
    mri_data = mri_data / np.nanmax(mri_data)


    mri_data_use = np.zeros((403, 938, 208), dtype=np.float64)
    mri_data_use_shape = np.shape(mri_data_use)
    #
    offset = 16
    joint_pos_max = np.full(2, -np.inf, dtype=np.float64)
    joint_pos_min = np.full(2, np.inf, dtype=np.float64)
    joint_names_list = sorted(list(joints.keys()))
    for i_joint in range(nJoints):
        joint_pos = joints[joint_names_list[i_joint]]
        for d in range(2):
            joint_pos_max[d] = max(joint_pos_max[d], joint_pos[d])
            joint_pos_min[d] = min(joint_pos_min[d], joint_pos[d])
    joint_pos_max = joint_pos_max.astype(np.int64) + offset
    joint_pos_min = joint_pos_min.astype(np.int64) - offset
    for i_joint in range(nJoints):
        joints[joint_names_list[i_joint]][0] = joints[joint_names_list[i_joint]][0] - joint_pos_min[0]
        joints[joint_names_list[i_joint]][1] = joints[joint_names_list[i_joint]][1] - joint_pos_min[1]
    mri_data_cut = mri_data[joint_pos_min[0]:joint_pos_max[0],
                            joint_pos_min[1]:joint_pos_max[1],
                            :]
    mri_data_cut_shape = np.shape(mri_data_cut)
    mri_data_use[0:mri_data_cut_shape[0],
                 0:mri_data_cut_shape[1],
                 0:mri_data_cut_shape[2]] = np.copy(mri_data_cut)
    mri_data_use = mri_data_use - np.nanmin(mri_data_use)
    mri_data_use = mri_data_use / np.nanmax(mri_data_use)
    mri_data_xy = np.nanmax(mri_data_use, axis=2)
    mri_data_xz = np.nanmax(mri_data_use, axis=0)

    mri_data_xy = mri_data_xy - np.nanmin(mri_data_xy)
    mri_data_xy = mri_data_xy / np.nanmax(mri_data_xy)
    mri_data_xz = mri_data_xz - np.nanmin(mri_data_xz)
    mri_data_xz = mri_data_xz / np.nanmax(mri_data_xz)

    # get transposed
    mri_data_xz_T = mri_data_xz.T

    mri_data_xz_shape = np.shape(mri_data_xz)
    mri_data_xy_shape = np.shape(mri_data_xy)
    ratio1 = mri_data_xz_shape[1] / mri_data_xz_shape[0]
    ratio2 = mri_data_xy_shape[0] / mri_data_xy_shape[1]

    ratio_inkscape = 1.275
        
    fig_w = np.round(mm_in_inch * 86.0*2.0/3.0 * ratio_inkscape, decimals=2)
    fig_w2 = np.round(mm_in_inch * (86.0*1.0/3.0)*1.5 * 0.5 * ratio_inkscape, decimals=2) # inset
    fig_h2 = np.copy(fig_w2) # inset
    #
    fig1 = plt.figure(1, frameon=False, figsize=(fig_w, np.round(fig_w*ratio1, decimals=2)))
    fig1.canvas.manager.window.move(0, 0)
    fig1.clear()
    fig1.set_facecolor('white')
    ax1 = fig1.add_subplot(111, frameon=False)
    ax1.clear()
    ax1.set_facecolor('white')
    ax1.set_position([0, 0, 1, 1])
    ax1.set_axis_off()
    ax1.set_aspect(1)
    #
    fig2 = plt.figure(2, frameon=False, figsize=(fig_w, np.round(fig_w*ratio2, decimals=2)))
    fig2.canvas.manager.window.move(0, 0)
    fig2.clear()
    fig2.set_facecolor('white')
    ax2 = fig2.add_subplot(111, frameon=False)
    ax2.clear()
    ax2.set_facecolor('white')
    ax2.set_position([0, 0, 1, 1])
    ax2.set_axis_off()
    ax2.set_aspect(1)
    #
    fig = list([fig1, fig2])
    ax = list([ax1, ax2])
    #
    plt.show(block=False)
    im0 = ax[0].imshow(mri_data_xz.T,
                 cmap='gray_r',
                 vmin=0.0,
                 vmax=1.0,
                 aspect=1)
    im1 = ax[1].imshow(mri_data_xy,
                 cmap='gray_r',
                 vmin=0.0,
                 vmax=1.0,
                 aspect=1)

    scalebar_dxyz = 50
    scalebar_text_dxyz = 30
    scalebar_pixel_length = 125
    resolution = mri_resolution * 1e-1 # cm
    scalebar_length = scalebar_pixel_length * resolution
    #
    scalebar_xz = ax[0].plot(np.array([scalebar_dxyz, scalebar_dxyz+scalebar_pixel_length], dtype=np.float64),
                             np.array([scalebar_dxyz, scalebar_dxyz], dtype=np.float64),
                             linestyle='-',
                             marker='',
                             color='black',
                             linewidth=linewidth_scalebar,
                             zorder=3)
    scalebar_xz_text = ax[0].annotate('{:0.1f} cm'.format(scalebar_length),
                                      [scalebar_dxyz+scalebar_pixel_length/2.0, scalebar_dxyz-scalebar_text_dxyz],
                                      color='black', ha='center', va='center',
                                      xycoords='data',
                                      fontname=fontname, fontsize=fontsize)
    scalebar_xy = ax[1].plot(np.array([scalebar_dxyz, scalebar_dxyz+scalebar_pixel_length], dtype=np.float64),
                             np.array([scalebar_dxyz, scalebar_dxyz], dtype=np.float64),
                             linestyle='-',
                             marker='',
                             color='black',
                             linewidth=linewidth_scalebar,
                             zorder=3)
    scalebar_xy_text = ax[1].annotate('{:0.1f} cm'.format(scalebar_length),
                                      [scalebar_dxyz+scalebar_pixel_length/2.0, scalebar_dxyz-scalebar_text_dxyz],
                                      color='black', ha='center', va='center',
                                      xycoords='data',
                                      fontname=fontname, fontsize=fontsize)

    cmap = plt.get_cmap('viridis')
    skeleton_color_val = 'white'
    joints_visited = list()
    h_bones11 = list()
    h_bones12 = list()
    h_bones21 = list()
    h_bones22 = list()
    h_joints11 = list()
    h_joints12 = list()
    h_joints21 = list()
    h_joints22 = list()
    #

    black_white_line_fac = 1.5
    black_white_marker_fac = 1.5
    #
    for joint_name in joints:
        joints_visited.append(joint_name)
        if (joint_name.split('_')[0] == 'joint'):            
            joint_pos = joints[joint_name]
            
            joint_pos_xz = np.array([joint_pos[1], joint_pos[2]], dtype=np.float64)
            # bones
            for joint_name_end in links[joint_name]:
                if not(joint_name_end in joints_visited):                    
                    joint_pos_end = joints[joint_name_end]
                    joint_pos_end_xz = np.array([joint_pos_end[1], joint_pos_end[2]], dtype=np.float64)
                    h_bones11_single = ax[0].plot(np.array([joint_pos_xz[0], joint_pos_end_xz[0]], dtype=np.float64),
                               np.array([joint_pos_xz[1], joint_pos_end_xz[1]], dtype=np.float64),
                               linestyle='-',
                               marker='',
                               color='black',
                               linewidth=linewidth_bone*black_white_line_fac,
                               zorder=1)
                    h_bones12_single = ax[0].plot(np.array([joint_pos_xz[0], joint_pos_end_xz[0]], dtype=np.float64),
                               np.array([joint_pos_xz[1], joint_pos_end_xz[1]], dtype=np.float64),
                               linestyle='-',
                               marker='',
                               color=skeleton_color_val,
                               linewidth=linewidth_bone,
                               zorder=2)
                    h_bones11.append(h_bones11_single)
                    h_bones12.append(h_bones12_single)
            # joints
            h_joints11_single = ax[0].plot(joint_pos_xz[0],
                       joint_pos_xz[1],
                       linestyle='',
                       marker='.',
                       color='black',
                       markersize=markersize_joint*black_white_marker_fac,
                       zorder=1)
            h_joints12_single = ax[0].plot(joint_pos_xz[0],
                       joint_pos_xz[1],
                       linestyle='',
                       marker='.',
                       color=skeleton_color_val,
                       markersize=markersize_joint,
                       zorder=2)
            h_joints11.append(h_joints11_single)
            h_joints12.append(h_joints12_single)

            joint_pos_xy = np.array([joint_pos[1], joint_pos[0]], dtype=np.float64)
            joint_pos_xy = joint_pos_xy.T
            # bones
            for joint_name_end in links[joint_name]:
                if not(joint_name_end in joints_visited):                    
                    joint_pos_end = joints[joint_name_end]
                    joint_pos_end_xy = np.array([joint_pos_end[1], joint_pos_end[0]], dtype=np.float64)
                    joint_pos_end_xy = joint_pos_end_xy.T
                    h_bones21_single = ax[1].plot(np.array([joint_pos_xy[0], joint_pos_end_xy[0]], dtype=np.float64),
                               np.array([joint_pos_xy[1], joint_pos_end_xy[1]], dtype=np.float64),
                               linestyle='-',
                               marker='',
                               color='black',
                               linewidth=linewidth_bone*black_white_line_fac,
                               zorder=1)
                    h_bones22_single = ax[1].plot(np.array([joint_pos_xy[0], joint_pos_end_xy[0]], dtype=np.float64),
                               np.array([joint_pos_xy[1], joint_pos_end_xy[1]], dtype=np.float64),
                               linestyle='-',
                               marker='',
                               color=skeleton_color_val,
                               linewidth=linewidth_bone,
                               zorder=2)
                    h_bones21.append(h_bones21_single)
                    h_bones22.append(h_bones22_single)
            # joints
            h_joints21_single = ax[1].plot(joint_pos_xy[0],
                                   joint_pos_xy[1],
                                   linestyle='',
                                   marker='.',
                                   color='black',
                                   markersize=markersize_joint*black_white_marker_fac,
                                   zorder=1)
            h_joints22_single = ax[1].plot(joint_pos_xy[0],
                                   joint_pos_xy[1],
                                   linestyle='',
                                   marker='.',
                                   color=skeleton_color_val,
                                   markersize=markersize_joint,
                                   zorder=2)
            h_joints21.append(h_joints21_single)
            h_joints22.append(h_joints22_single)

    ax[0].invert_yaxis()
    ax[0].invert_xaxis()
    ax[1].invert_xaxis()
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    dxz = 32
    dxy = 32
    joint_name_inset1 = 'joint_spine_010'
    joint_name_inset2 = 'joint_elbow_left'
    joint_inset1 = np.array([joints[joint_name_inset1][1],
                             joints[joint_name_inset1][2]], dtype=np.float64)
    joint_inset2 = np.array([joints[joint_name_inset2][1],
                             joints[joint_name_inset2][0]], dtype=np.float64)
    inset1 = np.array([[joint_inset1[0]+dxz, joint_inset1[1]+dxz],
                       [joint_inset1[0]+dxz, joint_inset1[1]-dxz],
                       [joint_inset1[0]-dxz, joint_inset1[1]-dxz],
                       [joint_inset1[0]-dxz, joint_inset1[1]+dxz],
                       [joint_inset1[0]+dxz, joint_inset1[1]+dxz]], dtype=np.float64)
    inset2 = np.array([[joint_inset2[0]+dxy, joint_inset2[1]+dxy],
                       [joint_inset2[0]+dxy, joint_inset2[1]-dxy],
                       [joint_inset2[0]-dxy, joint_inset2[1]-dxy],
                       [joint_inset2[0]-dxy, joint_inset2[1]+dxy],
                       [joint_inset2[0]+dxy, joint_inset2[1]+dxy]], dtype=np.float64)

    h_isnet1 = ax[0].plot(inset1[:, 0], inset1[:, 1],
                          linestyle='--',
                          marker='',
                          color='black',
                          linewidth=linewidth_bone*1.0,
                          zorder=3)
    h_isnet2 = ax[1].plot(inset2[:, 0], inset2[:, 1],
                          linestyle='--',
                          marker='',
                          color='black',
                          linewidth=linewidth_bone*1.0,
                          zorder=3)
    plt.pause(2**-10)
    fig[0].canvas.draw()
    fig[1].canvas.draw()
    plt.pause(2**-10)
    if save:
        fig[0].savefig(folder_save + '/mri2d_xz_0.svg',
#                        bbox_inches=0,
                       dpi=1200,
                       transparent=True,
                       format='svg',
                       pad_inches=0) 
        fig[1].savefig(folder_save + '/mri2d_xy_0.svg',
#                        bbox_inches=0,
                       dpi=1200,
                       transparent=True,
                       format='svg',
                       pad_inches=0)
    plt.pause(0.1)

    size_fac = 2.0
    nBones = np.size(h_bones11, 0)
    for i_bone in range(nBones):
        h_bones21[i_bone][0].set(linewidth=linewidth_bone*black_white_line_fac*size_fac)
        h_bones22[i_bone][0].set(linewidth=linewidth_bone*size_fac)
        h_bones11[i_bone][0].set(linewidth=linewidth_bone*black_white_line_fac*size_fac)
        h_bones12[i_bone][0].set(linewidth=linewidth_bone*size_fac)
    nJoints = np.size(h_joints11, 0)
    for i_joint in range(nJoints):
        h_joints21[i_joint][0].set(markersize=markersize_joint*black_white_marker_fac*size_fac)
        h_joints22[i_joint][0].set(markersize=markersize_joint*size_fac)
        h_joints11[i_joint][0].set(markersize=markersize_joint*black_white_marker_fac*size_fac)
        h_joints12[i_joint][0].set(markersize=markersize_joint*size_fac)
    fig[0].set_size_inches(fig_w2, fig_h2)
    fig[1].set_size_inches(fig_w2, fig_h2)
    h_isnet1[0].set(visible=False)
    h_isnet2[0].set(visible=False)
    plt.pause(2**-10)
    fig[0].canvas.draw()
    fig[1].canvas.draw()
    plt.pause(2**-10)
    #
    plt.show(block=False)

    scalebar_dxyz = 16
    scalebar_text_dxyz = 8
    scalebar_pixel_length = 25
    scalebar_length = resolution * scalebar_pixel_length
    scalebar_xz_text.set_text('{:0.1f} cm'.format(scalebar_length))
    scalebar_xy_text.set_text('{:0.1f} cm'.format(scalebar_length))
    mri_data_xz_use = np.zeros_like(mri_data_xz, dtype=np.float64)
    mri_data_xy_use = np.zeros_like(mri_data_xy, dtype=np.float64)
    for joint_name in joints:
        if ((joint_name == joint_name_inset1)) or (joint_name == joint_name_inset2):
            #
            joint_pos = joints[joint_name]
            #
            joint_pos_xz = np.array([joint_pos[1], joint_pos[2]], dtype=np.float64)
            joint_pos_xy = np.array([joint_pos[1], joint_pos[0]], dtype=np.float64)

            joint_pos_x = int(joint_pos[0])
            joint_pos_y = int(joint_pos[1])
            joint_pos_z = int(joint_pos[2])
            dxyz = 5
            mri_data_xy = np.nanmean(mri_data_use[:, :, joint_pos_z-dxyz:joint_pos_z+dxyz+1], 2)
            mri_data_xz = np.nanmean(mri_data_use[joint_pos_x-dxyz:joint_pos_x+dxyz+1, :, :], 0)
            mri_data_xy = mri_data_xy - np.nanmin(mri_data_xy[joint_pos_x-dxz:joint_pos_x+dxz, joint_pos_y-dxz:joint_pos_y+dxz])
            mri_data_xy = mri_data_xy / np.nanmax(mri_data_xy[joint_pos_x-dxz:joint_pos_x+dxz, joint_pos_y-dxz:joint_pos_y+dxz])
            mri_data_xz = mri_data_xz - np.nanmin(mri_data_xz[joint_pos_y-dxz:joint_pos_y+dxz, joint_pos_z-dxz:joint_pos_z+dxz])
            mri_data_xz = mri_data_xz / np.nanmax(mri_data_xz[joint_pos_y-dxz:joint_pos_y+dxz, joint_pos_z-dxz:joint_pos_z+dxz])

            im0.set_array(mri_data_xz.T)
            im1.set_array(mri_data_xy)
    
            ax[0].set_xlim([joint_pos_xz[0]-dxz, joint_pos_xz[0]+dxz])
            ax[0].set_ylim([joint_pos_xz[1]-dxz, joint_pos_xz[1]+dxz])
            ax[1].set_xlim([joint_pos_xy[0]-dxy, joint_pos_xy[0]+dxy])
            ax[1].set_ylim([joint_pos_xy[1]-dxy, joint_pos_xy[1]+dxy])
            #
            ax[0].invert_xaxis()
            ax[1].invert_xaxis()
            ax[1].invert_yaxis()
            #
            scalebar_xz[0].set_data(np.array([-scalebar_dxyz+joint_pos_xz[0],
                                              -scalebar_dxyz+joint_pos_xz[0]+scalebar_pixel_length], dtype=np.float64),
                                    np.array([-scalebar_dxyz+joint_pos_xz[1],
                                              -scalebar_dxyz+joint_pos_xz[1]], dtype=np.float64))
            scalebar_xz_text.set(alpha=0.0)
            scalebar_xz_text = ax[0].annotate('{:0.1f} cm'.format(scalebar_length),
                                              [-scalebar_dxyz+joint_pos_xz[0]+scalebar_pixel_length/2.0,
                                               -scalebar_dxyz+joint_pos_xz[1]-scalebar_text_dxyz],
                                              color='black', ha='center', va='center',
                                              xycoords='data',
                                              zorder=3,
                                              fontname=fontname, fontsize=fontsize)
            scalebar_xy[0].set_data(np.array([-scalebar_dxyz+joint_pos_xy[0],
                                              -scalebar_dxyz+joint_pos_xy[0]+scalebar_pixel_length], dtype=np.float64),
                                    np.array([-scalebar_dxyz+joint_pos_xy[1],
                                              -scalebar_dxyz+joint_pos_xy[1]], dtype=np.float64))
            scalebar_xy_text.set(alpha=0.0)
            scalebar_xy_text = ax[1].annotate('{:0.1f} cm'.format(scalebar_length),
                                              [-scalebar_dxyz+joint_pos_xy[0]+scalebar_pixel_length/2.0,
                                               -scalebar_dxyz+joint_pos_xy[1]-scalebar_text_dxyz],
                                              color='black', ha='center', va='center',
                                              xycoords='data',
                                              zorder=3,
                                              fontname=fontname, fontsize=fontsize)
            plt.pause(2**-10)
            fig[0].canvas.draw()
            fig[1].canvas.draw()
            plt.pause(2**-10)
            if save:
                fig[0].savefig(folder_save + '/mri2d_xz__{:s}_0.svg'.format(joint_name),
                               bbox_inches=0,
                               dpi=1200,
                               transparent=True,
                               format='svg',
                               pad_inches=0)
                fig[1].savefig(folder_save + '/mri2d_xy__{:s}_0.svg'.format(joint_name),
                               bbox_inches=0,
                               dpi=1200,
                               transparent=True,
                               format='svg',
                               pad_inches=0)
    return

if __name__ == '__main__':
    folder_reqFiles = data.path + '/datasets_figures/required_files' 

    animal = 'M220217_DW03_20220217'
    folder_save = data.path+"/dataset_analysis/M220217_DW03/MRI/panels"
    os.makedirs(folder_save,exist_ok=True)
    file_mri = data.path+"/dataset_analysis/M220217_DW03/MRI/mean_mri_data__01_flip.npy"
    file_skeleton = data.path+"/dataset_analysis/M220217_DW03/MRI/labels_m3_base.npy"
    mri_resolution = 0.3
    draw_mri2d(file_mri, file_skeleton, mri_resolution, folder_save)
    if verbose:
        plt.show(block=True)

    animal = 'M220217_DW01_20220217'
    folder_save = data.path+"/dataset_analysis/M220217_DW01/MRI/panels"
    os.makedirs(folder_save,exist_ok=True)
    file_mri = "/home/voit/tmp"+"/dataset_analysis/M220217_DW01/MRI/mean_mri_data__01_flip.npy"
    file_skeleton = "/home/voit/tmp"+"/dataset_analysis/M220217_DW01/MRI/labels_m1_flip.npy"
    mri_resolution = 0.3
    draw_mri2d(file_mri, file_skeleton, mri_resolution, folder_save)
    if verbose:
        plt.show(block=True)


    exit()
    #
    date = '20200207' # '20200207' or '20200205'
    folder_save = os.path.abspath('panels') + '/mri/' + date
    file_mri = folder_reqFiles+'/'+date+'/mri_data.npy'
    file_skeleton = folder_reqFiles+'/'+date+'/mri_skeleton0_full.npy'
    mri_resolution = 0.4
    draw_mri2d(file_mri, file_skeleton, mri_resolution, folder_save)
    if verbose:
        plt.show(block=True)
    #
    date = '20210511_1' # '20210511_1' or '20210511_2' or '20210511_3' or '20210511_4'
    folder_save = os.path.abspath('panels') + '/mri/' + date
    animal_id = date.split('_')[-1]
    file_mri = folder_reqFiles+'/'+date.split('_')[0]+'/'+'mri_'+animal_id+'/mri_data.npy'
    file_skeleton = folder_reqFiles+'/'+date.split('_')[0]+'/'+'mri_'+animal_id+'/mri_skeleton0_full.npy'
    mri_resolution = 0.4
    draw_mri2d(file_mri, file_skeleton, mri_resolution, folder_save)
    if verbose:
        plt.show(block=True)
    #
    date = '20210511_3' # '20210511_1' or '20210511_2' or '20210511_3' or '20210511_4'
    folder_save = os.path.abspath('panels') + '/mri/' + date
    animal_id = date.split('_')[-1]
    file_mri = folder_reqFiles+'/'+date.split('_')[0]+'/'+'mri_'+animal_id+'/mri_data.npy'
    file_skeleton = folder_reqFiles+'/'+date.split('_')[0]+'/'+'mri_'+animal_id+'/mri_skeleton0_full.npy'
    mri_resolution = 0.4
    draw_mri2d(file_mri, file_skeleton, mri_resolution, folder_save)
    if verbose:
        plt.show(block=True)
    ##

    if verbose:
        plt.show()
