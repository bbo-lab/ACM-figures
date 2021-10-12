#!/usr/bin/env python3

import importlib
import numpy as np
import matplotlib.pyplot as plt
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

saveFolder = os.path.abspath('figures')

folder_reconstruction = data.path+'/reconstruction'
folder = folder_reconstruction+'/20200205/'+'arena_20200205_calibration_on'

index_frame_plot = 17900

def get_origin_coord(file_origin_coord, scale_factor):
    # load
    origin_coord = np.load(file_origin_coord, allow_pickle=True).item()
    # arena coordinate system
    origin = origin_coord['origin']
    coord = origin_coord['coord']
    # scaling (calibration board square size -> cm)
    origin = origin * scale_factor
    return origin, coord

if __name__ == "__main__":
    # get arguments
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
    file_labelsManual = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels.npz'

    args = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                cfg.scale_factor, cfg.pcutoff)
    args['use_custom_clip'] = False
    
    nBones = args['numbers']['nBones']
    skeleton_edges = args['model']['skeleton_edges']
    joint_order = args['model']['joint_order']
    bounds_bones, bounds_free_bones, free_para_bones = anatomy.get_bounds_bones(nBones, skeleton_edges, joint_order)

    # get relevant information from arguments
    nBones = args['numbers']['nBones']
    nMarkers = args['numbers']['nMarkers']
    nCameras = args['numbers']['nCameras']
    joint_order = args['model']['joint_order'] # list
    joint_marker_order = args['model']['joint_marker_order'] # list
    skeleton_edges = args['model']['skeleton_edges'].cpu().numpy()
    bone_lengths_index = args['model']['bone_lengths_index'].cpu().numpy()
    joint_marker_index = args['model']['joint_marker_index'].cpu().numpy()
    #
    free_para_bones = args['free_para_bones'].cpu().numpy()
    free_para_markers = args['free_para_markers'].cpu().numpy()
    free_para_pose = args['free_para_pose'].cpu().numpy()
    nPara_bones = args['nPara_bones']
    nPara_markers = args['nPara_markers']
    nPara_pose = args['nPara_pose']
    nFree_bones = args['nFree_bones']
    nFree_markers = args['nFree_markers']
    nFree_pose = args['nFree_pose']

    # load frame list according to manual labels
    labels_manual = np.load(file_labelsManual, allow_pickle=True)['arr_0'].item()
    
    # get calibration frame list
    frame_list_calib = np.array([], dtype=np.int64)
    for i in range(np.size(cfg.index_frames_calib, 0)):
        framesList_single = np.arange(cfg.index_frames_calib[i][0],
                                      cfg.index_frames_calib[i][1] + cfg.dFrames_calib,
                                      cfg.dFrames_calib,
                                      dtype=np.int64)
        frame_list_calib = np.concatenate([frame_list_calib, framesList_single], 0)
    nFrames = int(np.size(frame_list_calib))

    # create correct free_para
    free_para = np.concatenate([free_para_bones,
                                free_para_markers], 0)
    for i_frame in frame_list_calib:
        free_para = np.concatenate([free_para,
                                    free_para_pose], 0)

    # load arena coordinate system
    origin, coord = get_origin_coord(file_origin_coord, cfg.scale_factor)
    #
    labels_frame = np.zeros((nCameras, nMarkers, 3), dtype=np.float64)
    labels_use = labels_manual[frame_list_calib[0]]
    for i_marker in range(nMarkers):
        marker_name = joint_marker_order[i_marker]
        marker_name_split = marker_name.split('_')
        label_name = 'spot_' + '_'.join(marker_name_split[1:-1])
        if label_name in labels_use:
            labels_frame[:, i_marker, :2] = labels_use[label_name]
            labels_frame[:, i_marker, 2] = 1.0
    x_pose = np.zeros(len(free_para_pose), dtype=np.float64)[None, :]
    for i_frame in frame_list_calib[1:]:
        labels_frame = np.zeros((nCameras, nMarkers, 3), dtype=np.float64)
        labels_use = labels_manual[frame_list_calib[0]]
        for i_marker in range(nMarkers):
            marker_name = joint_marker_order[i_marker]
            marker_name_split = marker_name.split('_')
            label_name = 'spot_' + '_'.join(marker_name_split[1:-1])
            if label_name in labels_use:
                labels_frame[:, i_marker, :2] = labels_use[label_name]
                labels_frame[:, i_marker, 2] = 1.0
        x_pose_single = np.zeros(len(free_para_pose), dtype=np.float64)[None, :]
        x_pose = np.concatenate([x_pose, x_pose_single], 0)
    x_free_pose = x_pose[:, free_para_pose].ravel()
    x_pose = x_pose.ravel()
    
    # BOUNDS
    # bone_lengths
    bounds_free_bones = args['bounds_free_bones']
    bounds_free_low_bones = model.do_normalization_bones(bounds_free_bones[:, 0])
    bounds_free_high_bones = model.do_normalization_bones(bounds_free_bones[:, 1])
    # joint_marker_vec
    bounds_free_markers = args['bounds_free_markers']
    bounds_free_low_markers = model.do_normalization_markers(bounds_free_markers[:, 0])
    bounds_free_high_markers = model.do_normalization_markers(bounds_free_markers[:, 1])
    # pose
    bounds_free_pose = args['bounds_free_pose']
    bounds_free_low_pose_single = model.do_normalization(bounds_free_pose[:, 0][None, :], args).numpy().ravel()
    bounds_free_high_pose_single = model.do_normalization(bounds_free_pose[:, 1][None, :], args).numpy().ravel()
    bounds_free_low_pose = np.tile(bounds_free_low_pose_single, nFrames)
    bounds_free_high_pose = np.tile(bounds_free_high_pose_single, nFrames)
    # all
    bounds_free_low = np.concatenate([bounds_free_low_bones,
                                      bounds_free_low_markers,
                                      bounds_free_low_pose], 0)
    bounds_free_high = np.concatenate([bounds_free_high_bones,
                                       bounds_free_high_markers,
                                       bounds_free_high_pose], 0)
    bounds_free = np.stack([bounds_free_low, bounds_free_high], 1)
    args['bounds_free'] = bounds_free
    
    # INITIALIZE X
    inital_bone_length = 0.0
    inital_marker_length = 0.0
    # initialize bone_lengths and joint_marker_vec
    x_bones = bounds_free_low_bones.numpy() + (bounds_free_high_bones.numpy() - bounds_free_low_bones.numpy()) * 0.5
    x_bones[np.isinf(x_bones)] = inital_bone_length  

    x_free_bones = x_bones[free_para_bones]
    x_markers = np.full(nPara_markers, 0.0, dtype=np.float64)
    x_free_markers = np.zeros(nFree_markers ,dtype=np.float64)
    x_free_markers[(bounds_free_low_markers != 0.0) & (bounds_free_high_markers == 0.0)] = -inital_marker_length
    x_free_markers[(bounds_free_low_markers == 0.0) & (bounds_free_high_markers != 0.0)] = inital_marker_length
    x_free_markers[(bounds_free_low_markers != 0.0) & (bounds_free_high_markers != 0.0)] = 0.0
    x_free_markers[(bounds_free_low_markers == 0.0) & (bounds_free_high_markers == 0.0)] = 0.0
    x_markers[free_para_markers] = np.copy(x_free_markers)
    #
    x = np.concatenate([x_bones,
                        x_markers,
                        x_pose], 0)
    
    # update args regarding fixed tensors
    args['plot'] = False
    args['nFrames'] = nFrames
    
    # update args regarding x0 and labels
    # ARGS X
    args['x_torch'] = torch.from_numpy(np.concatenate([x_bones,
                                                       x_markers,
                                                       x_pose[:nPara_pose]], 0))
    args['x_free_torch'] = torch.from_numpy(np.concatenate([x_free_bones,
                                                            x_free_markers,
                                                            x_free_pose], 0))
    args['x_free_torch'].requires_grad = True
    # ARGS LABELS MANUAL
    args['labels_single_torch'] = torch.zeros((nFrames, nCameras, nMarkers, 3), dtype=model.float_type)
    args['labels_mask_single_torch'] = torch.zeros((nFrames, nCameras, nMarkers), dtype=torch.bool)
    for i in range(nFrames):
        index_frame = frame_list_calib[i]
        if index_frame in labels_manual:
            labels_manual_frame = labels_manual[index_frame]
            for marker_index in range(nMarkers):
                marker_name = joint_marker_order[marker_index]
                string = 'spot_' + '_'.join(marker_name.split('_')[1:-1])
                if string in labels_manual_frame:
                    mask = ~np.any(np.isnan(labels_manual[index_frame][string]), 1)
                    args['labels_mask_single_torch'][i, :, marker_index] = torch.from_numpy(mask)
                    args['labels_single_torch'][i, :, marker_index, :2][mask] = torch.from_numpy(labels_manual[index_frame][string][mask])
                    args['labels_single_torch'][i, :, marker_index, 2][mask] = 1.0
    
    # plot calibration
    print('Plotting frame with index:\t{:06d}'.format(index_frame_plot))
    # load calibration
    x_calib = np.load(folder+'/x_calib.npy', allow_pickle=True)

    #0 get indices
    index_x = np.where(frame_list_calib == index_frame_plot)[0][0]
    # update args
    args['labels_single_torch'] = args['labels_single_torch'][index_x].clone()
    args['labels_mask_single_torch'] = args['labels_mask_single_torch'][index_x].clone() 
    args['model']['skeleton_vertices_new'] = args['model']['skeleton_vertices_new'][0][None, :, :]
    args['model']['joint_marker_vectors_new'] = args['model']['joint_marker_vectors_new'][0][None, :, :]

    bone_lengths_new = x_calib[:nPara_bones]
    bone_lengths = args['model']['bone_lengths'].numpy()
    bone_lengths_index = args['model']['bone_lengths_index'].numpy()
    skeleton_edges = args['model']['skeleton_edges'].numpy()
    skeleton_vertices = args['model']['skeleton_vertices'].numpy()
    skeleton_coords = args['model']['skeleton_coords'].numpy()
    surface_vertices = args['model']['surface_vertices'].numpy()
    surface_vertices_weights = args['model']['surface_vertices_weights'].numpy()
    surface_vertices_new = np.copy(surface_vertices)
    surface_connect_index = np.argmax(surface_vertices_weights, 1)
    for i_bone in range(nBones):
        scale_factor = float(bone_lengths_new[bone_lengths_index[i_bone]] / bone_lengths[i_bone])
        R = skeleton_coords[i_bone]
        #
        index_joint_end = skeleton_edges[i_bone, 1]
        mask = (surface_connect_index == index_joint_end)
        vertices_new = surface_vertices[mask]
        vertices_new = vertices_new - skeleton_vertices[index_joint_end]
        vertices_new = np.einsum('ij,nj->ni', R.T, vertices_new)
        vertices_new = vertices_new * scale_factor
        vertices_new = np.einsum('ij,nj->ni', R, vertices_new)
        vertices_new = vertices_new + skeleton_vertices[index_joint_end]
        surface_vertices_new[mask] = vertices_new
    args['model']['surface_vertices'] = torch.from_numpy(surface_vertices_new)

    # get correct x
    x_skel_plot = x_calib[:nPara_bones+nPara_markers]
    x_pose_plot = x_calib[nPara_bones+nPara_markers:].reshape(nFrames, int(np.size(free_para_pose)))
    x_plot = np.concatenate([x_skel_plot, x_pose_plot[index_x]], 0)
    x_torch = torch.from_numpy(x_plot)
    x = x_torch
    index_frame = index_frame_plot
    # PLOT_MODEL
    model_dict = args['model']
    numbers = args['numbers']
    nBones = numbers['nBones']
    nMarkers = numbers['nMarkers']
    nPara_bones = args['nPara_bones']
    nPara_markers = args['nPara_markers']
    nPara_skel = nPara_bones + nPara_markers
    bone_lenghts = torch.reshape(x_torch[:nPara_bones], (1, nPara_bones))
    joint_marker_vec = torch.reshape(x_torch[nPara_bones:nPara_skel], (1, nMarkers, 3))
    model_t0_new = torch.reshape(x_torch[nPara_skel:nPara_skel+3], (1, 3))
    model_r0_new = torch.reshape(x_torch[nPara_skel+3:nPara_skel+6], (1, 3))
    model_r_new = torch.reshape(x_torch[nPara_skel+6:], (1, nBones-1, 3))
    skel_coords_new, skel_verts_new, surf_verts_new, joint_marker_pos = \
        model.adjust_joint_marker_pos2(model_dict,
                                       bone_lenghts, joint_marker_vec,
                                       model_t0_new, model_r0_new, model_r_new,
                                       nBones,
                                       True)
    skel_coords_new = skel_coords_new.squeeze()
    skel_verts_new = skel_verts_new.squeeze()
    surf_verts_new = surf_verts_new.squeeze()
    joint_marker_pos = joint_marker_pos.squeeze()
    # PLOT_MODEL_3D_PROJ
    numbers = args['numbers']
    nCams = numbers['nCameras']
    nBones = numbers['nBones']
    nMarkers = numbers['nMarkers']
    model_dict = args['model']
    skel_edges = model_dict['skeleton_edges']
    skel_verts = model_dict['skeleton_vertices']
    calibration = args['calibration']
    A = calibration['A_fit']
    k = calibration['k_fit']
    RX1 = calibration['RX1_fit']
    tX1 = calibration['tX1_fit']
    label_pos = args['labels_single_torch']
    mask_all = args['labels_mask_single_torch']
    nPara_bones = args['nPara_bones']
    nPara_markers = args['nPara_markers']
    nPara_skel = nPara_bones + nPara_markers
    # plot
    fig = plt.figure(1, figsize=(10, 10))
    fig.clear()
    fig.canvas.manager.window.move(0, 0)
    fig.subplots_adjust(bottom=0,
                        top=1,
                        left=0,
                        right=1)
    fig.set_facecolor('black')
    ax = list()
    n = np.int64(np.ceil(np.sqrt(nCams)))
    for i_cam in range(nCams):
        a = fig.add_subplot(n, n, i_cam + 1)
        a.set_title('camera: {:01d}'.format(i_cam),
                    color='gray')
        a.set_axis_off()
        a.set_facecolor('black')
        ax.append(a)
    fig.tight_layout()
    #
    bone_lenghts = torch.reshape(x[:nPara_bones], (1, nPara_bones))
    joint_marker_vec = torch.reshape(x[nPara_bones:nPara_skel], (1, nMarkers, 3))
    model_t0_new = torch.reshape(x[nPara_skel:nPara_skel+3], (1, 3))
    model_r0_new = torch.reshape(x[nPara_skel+3:nPara_skel+6], (1, 3))
    model_r_new = torch.reshape(x[nPara_skel+6:], (1, nBones-1, 3))
    skel_coords_new, skel_verts_new, surf_verts_new, marker_pos_new = \
        model.adjust_joint_marker_pos2(model_dict,
                                       bone_lenghts, joint_marker_vec,
                                       model_t0_new, model_r0_new, model_r_new,
                                       nBones,
                                       True)
    marker_pos_new_proj_all = model.map_m(RX1, tX1, A, k,
                                          marker_pos_new)
    skel_verts_new_proj_all = model.map_m(RX1, tX1, A, k,
                                          skel_verts_new)
    skel_verts_new_proj_all = skel_verts_new_proj_all.squeeze()
    marker_pos_new_proj_all = marker_pos_new_proj_all.squeeze()
    # manual labels
    labels_order = model_dict['joint_marker_order']
    labels_manual_dict = np.load(file_labelsManual, allow_pickle=True)['arr_0'].item()
    labels_manual = np.full((nCams, nMarkers, 2), float('nan'), dtype=np.float64)
    manual_labels_exist = False
    if (index_frame in labels_manual_dict):
        manual_labels_exist = True
        labels_use = labels_manual_dict[index_frame]
        for i_label in range(nMarkers):
            label_name = labels_order[i_label]
            label_name_split = label_name.split('_')[1:-1]
            label_name = 'spot_' + '_'.join(label_name_split)
            if (label_name in labels_use):
                labels_manual[:, i_label] = labels_use[label_name]
    for i_cam in range(nCams):
        labels_mask = mask_all[i_cam]
        # ATTENTION: apply the masks before converting to numpy
        marker_pos_new_proj = marker_pos_new_proj_all[i_cam][labels_mask].cpu().numpy()
        skel_verts_new_proj = skel_verts_new_proj_all[i_cam].cpu().numpy()
        label_pos_use = label_pos[i_cam][labels_mask].cpu().numpy()
        if (manual_labels_exist):
            label_manual_use = labels_manual[i_cam][labels_mask.cpu().numpy()]   
        # plot
        file_ccv = data.path_ccv_sup_fig4_list[i_cam]
        img = ccv.get_frame(file_ccv, index_frame + 1)
        # image
        ax[i_cam].imshow(img,
                 cmap='gray',
                 vmin=0,
                 vmax=127)
        # dlc labels
        ax[i_cam].plot(label_pos_use[:, 0],
                label_pos_use[:, 1],
                color='green',
                marker='x',
                linestyle='',
                markersize=5,
                alpha=0.75)
        # projected markers
        ax[i_cam].plot(marker_pos_new_proj[:, 0],
                marker_pos_new_proj[:, 1],
                color='blue',
                marker='x',
                linestyle='',
                markersize=5,
                alpha=0.75)     
        # projected distances between features
        nFeatures = np.size(marker_pos_new_proj, 0)
        for i_feat in range(nFeatures):
            ax[i_cam].plot(np.array([label_pos_use[i_feat, 0], marker_pos_new_proj[i_feat, 0]]),
                           np.array([label_pos_use[i_feat, 1], marker_pos_new_proj[i_feat, 1]]),
                           color='cyan',
                           marker='',
                           linestyle='-',
                           alpha=0.25)
        #
        dxy = 200
        center = np.mean(label_pos_use, 0)
        ax[i_cam].set_aspect(1)
        ax[i_cam].set_xlim([center[0] - dxy, center[0] + dxy])
        ax[i_cam].set_ylim([center[1] - dxy, center[1] + dxy])
        ax[i_cam].invert_yaxis()
    fig.canvas.draw()
    plt.pause(1e-16)
    if save:
        fig.savefig(saveFolder+'/skeleton_calibration.svg',
    #                     bbox_inches="tight",
                      dpi=300,
                      transparent=True,
                      format='svg',
                      pad_inches=0,
                      facecolor=fig.get_facecolor(), edgecolor='none')
        fig.savefig(saveFolder+'/skeleton_calibration.tiff',
    #                     bbox_inches="tight",
                      dpi=300,
                      transparent=True,
                      format='tiff',
                      pad_inches=0,
                      facecolor=fig.get_facecolor(), edgecolor='none')
    if verbose:
        plt.show()