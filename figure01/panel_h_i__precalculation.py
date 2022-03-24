#!/usr/bin/env python3

import copy
import importlib
from mayavi import mlab
import numpy as np
import os
import scipy
import scipy.optimize
import sys
import time
import torch

ACMconfig_path = os.path.dirname(os.path.abspath(__file__))+'/../ACM/config'
#'/media/smb/soma-fs.ad01.caesar.de/bbo/projects/monsees-pose/dataset_analysis/M220217_DW01/ACM/M220217_DW01_20220309_173500_DWchecked/'
ACMcode_path = os.path.dirname(os.path.abspath(__file__))+'/../ACM'

sys.path.append(ACMconfig_path)
import configuration as cfg
sys.path.pop(sys.path.index(ACMconfig_path))
#
sys.path.append(os.path.abspath(ACMcode_path))
import data
import helper
import anatomy
import model
sys_path0 = np.copy(sys.path)

verbose = True
save = False

folder_base = data.path+'/../dataset_analysis/'
folder_list = list([
    folder_base+'/M220217_DW01/ACM/',
    folder_base+'/20200205/arena_20200205_calibration_on/',
    ])

file_acm_list = list([
    folder_base+'/M220217_DW01/ACM/M220217_DW01_20220309_173500_DWchecked/',
    folder_base+'/../datasets_figures/reconstruction/20200205/arena_20200205_calibration_on/',
    ])

file_acmres_list = list([
    folder_base+'/M220217_DW01/ACM/M220217_DW01_20220309_173500_DWchecked/results/M220217_DW01_20220309_173500_DWchecked_20220314-221557/',
    folder_base+'/../datasets_figures/reconstruction/20200205/arena_20200205_calibration_on/',
    ])

file_mri_list = list([
    folder_base+'/M220217_DW01/MRI/labels_m1.npy',
    folder_base+'/../datasets_figures/required_files/20200205/mri_skeleton0_full.npy',
    ])

resolution_mri_list = [0.2 0.4]
list_is_large_animal = list([0, 0])

# MATH
def rodrigues2rotMat_single(r):
    sqrt_arg = np.sum(r**2)
    if (sqrt_arg <= 3*model.num_tol**2):
        rotMat = np.identity(3, dtype=np.float64)
        print('WARNING: rodrigues2rotMat_single')
    else:
        theta = np.sqrt(sqrt_arg)
        u = r / theta
        # row 1
        rotMat_00 = np.cos(theta) + u[0]**2 * (1.0 - np.cos(theta))
        rotMat_01 = u[0] * u[1] * (1.0 - np.cos(theta)) - u[2] * np.sin(theta)
        rotMat_02 = u[0] * u[2] * (1.0 - np.cos(theta)) + u[1] * np.sin(theta)
        # row 2
        rotMat_10 = u[0] * u[1] * (1.0 - np.cos(theta)) + u[2] * np.sin(theta)
        rotMat_11 = np.cos(theta) + u[1]**2 * (1.0 - np.cos(theta))
        rotMat_12 = u[1] * u[2] * (1.0 - np.cos(theta)) - u[0] * np.sin(theta)
        # row 3
        rotMat_20 = u[0] * u[2] * (1.0 - np.cos(theta)) - u[1] * np.sin(theta)
        rotMat_21 = u[1] * u[2] * (1.0 - np.cos(theta)) + u[0] * np.sin(theta)
        rotMat_22 = np.cos(theta) + u[2]**2 * (1.0 - np.cos(theta))
        # output
        rotMat = np.array([[rotMat_00, rotMat_01, rotMat_02],
                           [rotMat_10, rotMat_11, rotMat_12],
                           [rotMat_20, rotMat_21, rotMat_22]], dtype=np.float64)
    return rotMat

def rotMat2rodrigues_single(R):
    if (abs(3.0 - np.trace(R)) <= model.num_tol):
        r = np.zeros(3, dtype=np.float64)
    else:
        theta_norm = np.arccos((np.trace(R) - 1.0) / 2.0)
        r = (theta_norm / (2.0 * np.sin(theta_norm))) *  \
            np.array([R[2,1] - R[1,2],
                      R[0,2] - R[2,0],
                      R[1,0] - R[0,1]], dtype=np.float64)
#         print(R)
#         r = np.array([R[2,1] - R[1,2],
#                       R[0,2] - R[2,0],
#                       R[1,0] - R[0,1]], dtype=np.float64)
#         print(r)
#         r = r / np.sqrt(np.sum(r**2))
#         r = r * theta_norm
    return r

# def rotMat2rodrigues_single(R):
#     r = np.zeros(3, dtype=np.float64)
#     K = (R - R.T) / 2.0
#     r[0] = K[2, 1]
#     r[1] = K[0, 2]
#     r[2] = K[1, 0]

#     if not(np.all(R == np.identity(3, dtype=np.float64))):
#         R_logm = np.real(scipy.linalg.logm(R))
#         thetaM_1 = R_logm[2, 1] / (r[0] + -np.abs(np.sign(r[0])) + 1.0)
#         thetaM_2 = R_logm[0, 2] / (r[1] + -np.abs(np.sign(r[1])) + 1.0)
#         thetaM_3 = R_logm[1, 0] / (r[2] + -np.abs(np.sign(r[2])) + 1.0)
#         thetaM = np.array([thetaM_1, thetaM_2, thetaM_3])

#         theta = np.mean(thetaM[thetaM != 0.0])
#         r = r * theta

#     return r

# MRI

def extract_mri_labeling(file_mri_labeling, resolution, joint_name_start):
    print(file_mri_labeling)
    model = np.load(file_mri_labeling, allow_pickle=True).item()
    labels3d = model['joints']
    links = model['links']
    
    for k,v in labels3d.items():
        labels3d[k][labels3d[k]<0] = 0
    
    markers = dict()
    joints = dict()
    for i in labels3d.keys():
        if (i.split('_')[0] == 'marker'):
            markers[i] = labels3d[i]
        elif (i.split('_')[0] == 'joint'):
            joints[i] = labels3d[i]
            
    # this generates skeleton_verts and skeleton_edges which are needed later on in the optimization
    skeleton_verts = list()
    joints_left = list([joint_name_start])
    joints_visited = list()
    while (np.size(joints_left) > 0):
        i_joint = joints_left[0]
        skeleton_verts.append(joints[i_joint] * resolution)
        next_joint = list(np.sort(links[i_joint]))
        for i_joint_next in np.sort(links[i_joint]):
            if not(i_joint_next in joints_visited):
                joints_left.append(i_joint_next)
        joints_visited.append(i_joint)
        joints_left.pop(0)
        joints_left = list(np.sort(joints_left))
    skeleton_verts = np.array(skeleton_verts)
    
    skeleton_edges = list()
    for index in range(np.size(joints_visited)):
        for i_joint in links[joints_visited[index]]:
            if not(i_joint in joints_visited[:index]):
                index_joint2 = np.where([i == i_joint for i in joints_visited])[0][0]
                edge = np.array([index, index_joint2])
                skeleton_edges.append(edge)
    skeleton_edges = np.array(skeleton_edges)
    
    return joints, markers, skeleton_verts, skeleton_edges, joints_visited

# MODEL
def fcn(x_torch, args_torch):
    # numbers
    nBones = args_torch['numbers']['nBones']
    nMarkers = args_torch['numbers']['nMarkers']
    nCameras = args_torch['numbers']['nCameras']
    
    nPara_bones = args_torch['nPara_bones']
    nPara_markers = args_torch['nPara_markers']
#     nPara_pose = args_torch['nPara_pose']
    
    nPara_skel = nPara_bones + nPara_markers
        
    nSigmaPoints = x_torch.size()[0]
    model_bone_lengths = x_torch[:, :nPara_bones].reshape(nSigmaPoints, nPara_bones)
    joint_marker_vec = x_torch[:, nPara_bones:nPara_skel].reshape(nSigmaPoints, nMarkers, 3)
    model_t0_torch = x_torch[:, nPara_skel:nPara_skel+3].reshape(nSigmaPoints, 3)
    model_r0_torch = x_torch[:, nPara_skel+3:nPara_skel+6].reshape(nSigmaPoints, 3)
    model_r_torch = x_torch[:, nPara_skel+6:].reshape(nSigmaPoints, (nBones-1), 3) 

    marker_pos_torch = model.adjust_joint_marker_pos2(args_torch['model'],
                                                      model_bone_lengths, joint_marker_vec,
                                                      model_t0_torch, model_r0_torch, model_r_torch,
                                                      nBones)
    return marker_pos_torch

def fcn_free(x_free_torch, args_torch):
    nPara_bones = args_torch['nPara_bones']
    nPara_markers = args_torch['nPara_markers']
#     nPara_pose = args_torch['nPara_pose']
    nFree_bones = args_torch['nFree_bones']
    nFree_markers = args_torch['nFree_markers']
#     nFree_pose = args_torch['nFree_pose']
    
    nPara_skel = nPara_bones + nPara_markers
    nFree_skel = nFree_bones + nFree_markers
    
    x_bones_torch = args_torch['x_torch'][:, :nPara_bones].clone()
    x_bones_torch[:, args_torch['free_para_bones']] = model.undo_normalization_bones(x_free_torch[:, :nFree_bones])
    x_markers_torch = args_torch['x_torch'][:, nPara_bones:nPara_skel].clone()
    x_markers_torch[:, args_torch['free_para_markers']] = model.undo_normalization_markers(x_free_torch[:, nFree_bones:nFree_skel])
    x_pose_torch = args_torch['x_torch'][:, nPara_skel:].clone()
    x_pose_torch[:, args_torch['free_para_pose']] = model.undo_normalization(x_free_torch[:, nFree_skel:], args_torch)
    x_torch = torch.cat([x_bones_torch, x_markers_torch, x_pose_torch], 1)

    marker_pos_torch = fcn(x_torch, args_torch)
    return marker_pos_torch


def obj_fcn(x_free_torch, args_torch):
    nFree_bones = args_torch['nFree_bones']
    nFree_markers = args_torch['nFree_markers']
#     nFree_pose = args_torch['nFree_pose']
    nMarkers = args_torch['numbers']['nMarkers']
    nBones = args_torch['numbers']['nBones']
    nJoints= nBones + 1
    weights = args_torch['weights']
    
    x_free_bones_torch = x_free_torch[:, :nFree_bones]
    x_free_markers_torch = x_free_torch[:, nFree_bones:nFree_bones+nFree_markers]
    x_free_pose_torch = x_free_torch[:, nFree_bones+nFree_markers:]
    x_free_use_torch = torch.cat([x_free_bones_torch, x_free_markers_torch, x_free_pose_torch], 1)
    marker_pos_torch = fcn_free(x_free_use_torch, args_torch)
    
    diff_x_torch = (marker_pos_torch[:, :, 0] - args_torch['labels_single_torch'][:, :, 0])
    diff_y_torch = (marker_pos_torch[:, :, 1] - args_torch['labels_single_torch'][:, :, 1])
    diff_z_torch = (marker_pos_torch[:, :, 2] - args_torch['labels_single_torch'][:, :, 2])

    dist_torch = ((weights[None, :] * diff_x_torch)**2 + \
                  (weights[None, :] * diff_y_torch)**2 + \
                  (weights[None, :] * diff_z_torch)**2) # do not take the square root here!
    res_torch = torch.sum(dist_torch) / nMarkers
    return res_torch

# OPTIMIZATION

def obj_fcn__wrap(x_free, args): 
    x_free_torch = args['x_free_torch']
    x_free_torch.data.copy_(torch.from_numpy(x_free[None, :]).data)
    loss_torch = obj_fcn(x_free_torch, args)
    
    loss_torch.backward()
    grad_free = np.copy(x_free_torch.grad.detach().cpu().numpy())
    x_free_torch.grad.data.zero_()
    
    loss = loss_torch.item()
    return loss, grad_free

def optimize__scipy(x_free, args,
                    opt_dict):
    bounds_free_pose = args['bounds_free_pose']
    bounds_free_low_pose = model.do_normalization(bounds_free_pose[:, 0][None, :], args).numpy().ravel()
    bounds_free_high_pose = model.do_normalization(bounds_free_pose[:, 1][None, :], args).numpy().ravel()
    bounds_free = np.stack([bounds_free_low_pose, bounds_free_high_pose], 1)
    
    time_start = time.time()
    min_result = scipy.optimize.minimize(obj_fcn__wrap,
                                         x_free,
                                         args=args,
                                         method=opt_dict['opt_method'],
                                         jac=True,
                                         hess=None,
                                         hessp=None,
                                         bounds=bounds_free,
                                         constraints=(),
                                         tol=None,
                                         callback=None,
                                         options=opt_dict['opt_options'])
    time_end = time.time()
    
    print('iterations:\t{:06d}'.format(min_result.nit))
    print('residual:\t{:0.8e}'.format(min_result.fun))
    print('success:\t{}'.format(min_result.success))
    print('message:\t{}'.format(min_result.message))
    print('time needed:\t{:0.3f} seconds'.format(time_end - time_start))
    return min_result

# PLOTTING
skeleton3d_scale_factor = 1/3
color_surf = (75/255, 75/255, 75/255)
bones3d_line_width = 2
bones3d_tube_radius = 0.1
bones3d_tube_sides = 16

def plot_model_3d(model_3d,
                  fig, color=(0, 0, 0)):
#     coords = model_3d_new['skeleton_coords']
    joint_marker_pos = model_3d['joint_marker_pos']
    joint_marker_index = model_3d['joint_marker_index']
    #
    surf_verts = model_3d['surface_vertices']
    surf_tri = model_3d['surface_triangles']
    skel_verts = model_3d['skeleton_vertices']
    skel_edges = model_3d['skeleton_edges']
    
    nBones = np.size(skel_edges, 0)
    nJoints = nBones + 1

#     fig = mlab.figure(figN,
#                       size=(1280, 1024),
#                       bgcolor=(1, 1, 1))
#     mlab.clf()


        
#     # surface
#     surf = mlab.triangular_mesh(surf_verts[:, 0],
#                                 surf_verts[:, 1],
#                                 surf_verts[:, 2],
#                                 surf_tri,
#                                 color=color_surf,
#                                 figure=fig,
#                                 opacity=0.5,
#                                 transparent=True,
#                                 vmin=0,
#                                 vmax=1)
    # joints
    skel = mlab.points3d(skel_verts[:, 0],
                         skel_verts[:, 1],
                         skel_verts[:, 2],
                         color=color,
                         figure=fig,
                         scale_factor=skeleton3d_scale_factor)
    # bones
    for edge in skel_edges:
        i_joint = skel_verts[edge[0]]
        joint_to = skel_verts[edge[1]]
        i_bone = mlab.plot3d(np.array([i_joint[0], joint_to[0]]),
                             np.array([i_joint[1], joint_to[1]]),
                             np.array([i_joint[2], joint_to[2]]),
                             color=color,
                             figure=fig,
                             line_width=bones3d_line_width,
                             tube_radius=bones3d_tube_radius,
                             tube_sides=bones3d_tube_sides)
    # joint-marker vectors
    nMarkers = np.size(joint_marker_index.cpu().numpy())
    for marker_index in range(nMarkers):
        joint_index = joint_marker_index[marker_index]
        vec_start = skel_verts[joint_index]
        vec_end = joint_marker_pos[marker_index]
        mlab.plot3d(np.array([vec_start[0], vec_end[0]]),
                    np.array([vec_start[1], vec_end[1]]),
                    np.array([vec_start[2], vec_end[2]]),
                    color=color,
                    figure=fig,
                    opacity=0.75,
                    transparent=True,
                    line_width=bones3d_line_width,
                    tube_radius=bones3d_tube_radius/2,
                    tube_sides=bones3d_tube_sides)
    return

def plot_model(x_torch, argsv,
               fig, color=(0, 0, 0)):
    model_3d = args['model']
    
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
        model.adjust_joint_marker_pos2(model_3d,
                                       bone_lenghts, joint_marker_vec,
                                       model_t0_new, model_r0_new, model_r_new,
                                       nBones,
                                       True)

    skel_coords_new = skel_coords_new.squeeze()
    skel_verts_new = skel_verts_new.squeeze()
    surf_verts_new = surf_verts_new.squeeze()
    joint_marker_pos = joint_marker_pos.squeeze()
        
    model_3d_new = copy.deepcopy(model_3d)
    model_3d_new['surface_vertices'] = surf_verts_new.detach().cpu().numpy()
    model_3d_new['skeleton_vertices'] = skel_verts_new.detach().cpu().numpy()
    model_3d_new['skeleton_coords'] = skel_coords_new.detach().cpu().numpy()
    model_3d_new['joint_marker_pos'] = joint_marker_pos.detach().cpu().numpy()
    plot_model_3d(model_3d_new,
                  color, fig)

    return

if __name__ == "__main__":
    for i_folder in range(len(folder_list)):
        folder = folder_list[i_folder]
        folder_acmres = file_acmres_list[i_folder]
        folder_acm = file_acm_list[i_folder]

        sys.path = list(np.copy(sys_path0))
        sys.path.append(folder_acm)

        importlib.reload(cfg)
        cfg.animal_is_large = list_is_large_animal[i_folder]
        importlib.reload(anatomy)
        #
        folder_reqFiles = data.path + '/required_files'
        file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy' # Backwards compatibiliy
        if not os.path.isfile(file_origin_coord):
            file_origin_coord = cfg.file_origin_coord
        file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy' # Backwards compatibiliy
        if not os.path.isfile(file_calibration):
            file_calibration = cfg.file_calibration
        file_model = folder_reqFiles + '/model.npy' # Backwards compatibiliy
        if not os.path.isfile(file_model):
            file_model = cfg.file_model
        file_labelsDLC = folder_reqFiles + '/labels_dlc_dummy.npy' # DLC labels are actually never needed here  # Backwards compatibiliy
        if not os.path.isfile(file_labelsDLC):
            file_labelsDLC = cfg.file_labelsDLC
        
        # GET THE DATA FROM THE MRI SCAN
        file_mri = file_mri_list[i_folder]
    
        resolution_mri = resolution_mri_list[i_folder]
        joint_start = 'joint_head_001'
        joints_mri, markers_mri, skeleton_verts_mri, skeleton_edges_mri, joints_visited_mri = \
            extract_mri_labeling(file_mri, resolution_mri, joint_start)


        # get arguments
        args = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                    cfg.scale_factor, cfg.pcutoff)
        args['use_custom_clip'] = False

#         nMarkers = args['numbers']['nMarkers']
#         joint_marker_index = args['model']['joint_marker_index'].cpu().numpy()
#         weights = np.ones(nMarkers, dtype=np.float64)
#         for i_joint in np.unique(joint_marker_index):
#             mask = (joint_marker_index == i_joint)
#             weights[mask] = 1.0 / np.sum(mask)
#         args['weights'] = torch.from_numpy(weights)

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
    #     nFree_bones = args['nFree_bones']
    #     nFree_markers = args['nFree_markers']
        nFree_pose = args['nFree_pose']

        # remove all free parameters that do not modify the pose
        free_para_bones = np.zeros_like(free_para_bones, dtype=bool)
        free_para_markers = np.zeros_like(free_para_markers, dtype=bool)
        nFree_bones = int(0)
        nFree_markers = int(0)
        args['free_para_bones'] = torch.from_numpy(free_para_bones)
        args['free_para_markers'] = torch.from_numpy(free_para_markers)
        args['nFree_bones'] = nFree_bones
        args['nFree_markers'] = nFree_markers

        # get ground truth locations (from MRI)
        markers0 = np.zeros((nMarkers, 3), dtype=np.float64)
        for i_marker in range(nMarkers):
            markers0[i_marker] = np.copy(markers_mri[joint_marker_order[i_marker]])
        markers0 = markers0 * resolution_mri * 1e-1 # mri -> cm
        #
        joints0 = np.zeros((nBones+1, 3), dtype=np.float64)
        for i_joint in range(nBones+1):
            joints0[i_joint] = np.copy(joints_mri[joint_order[i_joint]])
        joints0 = joints0 * resolution_mri * 1e-1 # mri -> cm
        #
        skel_verts0 = skeleton_verts_mri * 1e-1 # mri -> cm
        skel_edges0 = skeleton_edges_mri


        # load calibrated model and initalize the pose
        print(folder_acmres + '/x_calib.npy')
        x_calib = np.load(folder_acmres + '/x_calib.npy', allow_pickle=True)
        x_bones = x_calib[:nPara_bones]
        x_markers = x_calib[nPara_bones:nPara_bones+nPara_markers]
    #     x_pose = x_calib[nPara_bones+nPara_markers:]
        #
        x_pose = np.random.randn(nBones * 3 + 3).astype(np.float64) * 1e-3
        x_pose[np.logical_not(free_para_pose)] = 0.0
        # t0
        x_pose[0:3] = joints0[0]
        # r0
    #     R = np.array([[-1.0, 0.0, 0.0],
    #                   [0.0, 0.0, -1.0],
    #                   [0.0, 1.0, 0.0]], dtype=np.float64)
        r1 = np.array([0.0, 0.0, np.pi], dtype=np.float64)
        R1 = rodrigues2rotMat_single(r1)
        r2 = np.array([-np.pi/2.0, 0.0, 0.0], dtype=np.float64)
    #     R2 = rodrigues2rotMat_single(r2)
    #     R = np.dot(R2, R1)
        R = R1
        x_pose[3:6] = rotMat2rodrigues_single(R)

        free_para = np.concatenate([free_para_bones,
                                    free_para_markers,
                                    free_para_pose], 0)
        x = np.concatenate([x_bones, x_markers, x_pose], 0)

        # normalize x
        x_free = model.do_normalization(torch.from_numpy(x_pose[free_para_pose].reshape(1, nFree_pose)), args).numpy().ravel()
        x_align_free = np.copy(x_free) # for testing

        args['x_torch'] = torch.from_numpy(x[None, :])
        args['x_free_torch'] = torch.from_numpy(x_free[None, :])    
        args['x_free_torch'].requires_grad = True
        args['labels_single_torch'] = torch.from_numpy(markers0[None, :, :])
        args['labels_mask_single_torch'] = torch.ones_like(args['labels_single_torch'], dtype=torch.bool)

        # OPTIMIZE
        # create optimization dictonary
        opt_options = dict()
        opt_options['disp'] = False#verbose
        opt_options['maxiter'] = float('inf')
        opt_options['maxcor'] = 100 # scipy default value: 10
        opt_options['ftol'] = 2**-23 # scipy default value: 2.220446049250313e-09
        opt_options['gtol'] = 0.0 # scipy default value: 1e-05
        opt_options['maxfun'] = float('inf')
        opt_options['iprint'] = -1
        opt_options['maxls'] = 200 # scipy default value: 20
        opt_dict = dict()
        opt_dict['opt_method'] = 'L-BFGS-B'
        opt_dict['opt_options'] = opt_options
        print('Alignment')
        min_result = optimize__scipy(x_free, args,
                                     opt_dict)
        print('Finished alignment')
        print()

        # copy fitting result into correct arrary
        x_align_free = np.copy(min_result.x)

        # reverse normalization of x
        x_align_free = model.undo_normalization(torch.from_numpy(x_align_free).reshape(1, nFree_pose), args).numpy().ravel()

        # add free variables
        x_align = np.copy(x)
        x_align[free_para] = x_align_free
        # save
        if save:
            savefolder = folder + '/figures/figure1/panel_h_i/'
            os.makedirs(savefolder)
            np.save(savefolder + '/x_align.npy', x_align)
            print('Saved aligned 3D model ({:s})'.format(savefolder + '/x_align.npy'))
            print()

        if (verbose):
            # PRINT BOUNDS
            buffer1 = 25
            #
            bounds_free_0 = np.zeros((nBones+1)*3, dtype=np.float64)
            bounds_free_range = np.zeros((nBones+1)*3, dtype=np.float64)
            bounds_free_0[free_para_pose] = args['bounds_free_pose_0']
            bounds_free_range[free_para_pose] = args['bounds_free_pose_range']
            bounds_low = bounds_free_0 - bounds_free_range
            bounds_up = bounds_free_0 + bounds_free_range
            bounds_low = bounds_low.reshape(nBones+1, 3)
            bounds_up = bounds_up.reshape(nBones+1, 3)
            bounds_low = bounds_low[1:] * 180.0/np.pi
            bounds_up= bounds_up[1:] * 180.0/np.pi
            #
            x_use = x_align[nPara_bones+nPara_markers:].reshape(nBones+1, 3)
            x_use = x_use[1:] * 180.0/np.pi
            #
            free_para_pose_use = free_para_pose.reshape(nBones+1, 3)
            free_para_pose_use = free_para_pose_use[1:].astype(bool)
            for i in range(nBones):
                joint0 = skel_verts0[skel_edges0[i, 0]]
                joint1 = skel_verts0[skel_edges0[i, 1]]
                bone_lengths0 = np.sum((joint1 - joint0)**2)**0.5
                bone_lengths = x_align[:nPara_bones+nPara_markers][bone_lengths_index[i]]

#                 print('{:s}{:s}---{:s}{:s}:'.format(joint_order[skeleton_edges[i, 0]],
#                                                     ' ' * (buffer1 - len(joint_order[skeleton_edges[i, 0]])),
#                                                     ' ' * (buffer1 - len(joint_order[skeleton_edges[i, 1]])),
#                                                     joint_order[skeleton_edges[i, 1]]))
#                 print('low:\t\t{:0.3f},{:0.3f}, {:0.3f}'.format(bounds_low[i, 0], bounds_low[i, 1], bounds_low[i, 2]))
#                 print('value:\t\t{:0.3f}, {:0.3f}, {:0.3f}'.format(x_use[i, 0], x_use[i, 1], x_use[i, 2]))
#                 print('up:\t\t{:0.3f}, {:0.3f}, {:0.3f}'.format(bounds_up[i, 0], bounds_up[i, 1], bounds_up[i, 2]))
#                 print('free:\t\t{:0.3f}, {:0.3f}, {:0.3f}'.format(free_para_pose_use[i, 0], free_para_pose_use[i, 1], free_para_pose_use[i, 2]))
#                 print('bone_length0:\t{:0.3f}'.format(bone_lengths0))
#                 print('bone_length:\t{:0.3f}'.format(bone_lengths))
#                 print('ratio:\t\t{:0.3f}'.format(bone_lengths/bone_lengths0))
#                 print()

            # PLOT
            color0 = (0.0, 1.0, 0.0)
            color1 = (1.0, 0.0, 0.0)

            fig = mlab.figure(1,
                              size=(1280, 1024),
                              bgcolor=(1, 1, 1))
            mlab.clf()
            # joints0
        #     skel = mlab.points3d(joints0[:, 0],
        #                          joints0[:, 1],
        #                          joints0[:, 2],
        #                          color=color0,
        #                          figure=fig,
        #                          scale_factor=skeleton3d_scale_factor)
            skel = mlab.points3d(skel_verts0[:, 0],
                                 skel_verts0[:, 1],
                                 skel_verts0[:, 2],
                                 color=color0,
                                 figure=fig,
                                 scale_factor=skeleton3d_scale_factor*0.5)
            # bones0
            for edge in skel_edges0:
                joint0 = skel_verts0[edge[0]]
                joint1 = skel_verts0[edge[1]]
                i_bone = mlab.plot3d(np.array([joint0[0], joint1[0]], dtype=np.float64),
                                     np.array([joint0[1], joint1[1]], dtype=np.float64),
                                     np.array([joint0[2], joint1[2]], dtype=np.float64),
                                     color=color0,
                                     figure=fig,
                                     line_width=bones3d_line_width,
                                     tube_radius=bones3d_tube_radius,
                                     tube_sides=bones3d_tube_sides)
            # markers0
            mlab.points3d(markers0[:, 0],
                          markers0[:, 1],
                          markers0[:, 2],
                          color=(0.0, 0.0, 1.0),
                          figure=fig,
                          scale_factor=skeleton3d_scale_factor)
            # aligned model
            plot_model(torch.from_numpy(x_align), args,
                       color1, fig)
            #
            mlab.show()
