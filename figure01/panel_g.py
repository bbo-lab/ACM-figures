#!/usr/bin/env python3

import importlib
import matplotlib.pyplot as plt
from mayavi import mlab
import numpy as np
import os
from skimage import measure
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
show = True

skeleton3d_scale_factor = 1/3
color_skel = (0.0, 0.0, 0.0)
color_surf = 0.1
bones3d_line_width = 2
bones3d_tube_radius = 0.1
bones3d_tube_sides = 16

mm_in_inch = 5.0/127.0

def save_fig3d(file, fig):
    fig.scene.anti_aliasing_frames = 20
    # fix to avoid error when using mlab.screenshot
    # see: https://github.com/enthought/mayavi/issues/702
    fig.scene._lift()
    screenshot = mlab.screenshot(figure=fig,
                                 mode='rgba',
                                 antialiased=True)
    #
    fig0_w = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    fig0_h = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    fig0 = plt.figure(100, figsize=(fig0_w, fig0_h))
    fig0.canvas.manager.window.move(0, 0)
    fig0.clear()
    fig0.set_facecolor('white')
    fig0.set_dpi(300)
    ax0 = fig0.add_subplot(111, frameon=False)
    ax0.clear()
    ax0.set_facecolor('white')
    ax0.set_position([0, 0, 1, 1])
    #
    im = ax0.imshow(screenshot,
                    aspect='equal',
                    zorder=1)
    ax0.set_axis_off()
    #
    plt.pause(2**-10)
    fig0.canvas.draw()
    plt.pause(2**-10)
    #    
    fig0.savefig(file,
#                  bbox_inches="tight",
                 dpi=300,
                 transparent=True,
                 format='svg',
                 pad_inches=0)   
    
    
    size_fac = 4
    mlab.savefig(file.split('.')[0]+'_mlab.tiff',
                 size=(1280*size_fac, 1280*size_fac),
                 figure=fig,
                 magnification='auto',)
    return

def perform_marching_cubes(file_mri_data,
                           resolution, marker_size,
                           fac_remove_markers,
                           threshold_low,
                           markers):
    mri_data = np.load(file_mri_data, allow_pickle=True)
    mri_shape = np.shape(mri_data)

    # this gets rid of the markers
    mri_data_use = np.copy(mri_data)
    
    dxyz = np.int64(np.ceil(marker_size / resolution))
    sim_size = 2 * dxyz + 1
    x = np.linspace(-dxyz, dxyz, sim_size)
    y = np.linspace(-dxyz, dxyz, sim_size)
    z = np.linspace(-dxyz, dxyz, sim_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    shape_use = np.array([sim_size,
                          sim_size,
                          sim_size,
                          1])
    xyz = np.concatenate([X.reshape(shape_use),
                          Y.reshape(shape_use),
                          Z.reshape(shape_use)], 3)
    xyz_shape = np.shape(xyz)
    xyz_use = xyz.reshape(np.prod(xyz_shape[:-1]), 3)
    
    for i in np.sort(list(markers.keys())):        
        i_split = i.split('_')
        if ((i_split[-1] == 'start') and ('_'.join(i_split[:-1]) + '_end' in np.sort(list(markers.keys())))):
            marker_start = markers[i]
            marker_end = markers['_'.join(i_split[:-1]) + '_end']

            t = marker_start
            nVec = marker_end - marker_start
            nVec_norm = np.sqrt(np.sum(nVec**2))
            nVec = nVec / nVec_norm
            zVec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            zVec_norm = np.sqrt(np.sum(zVec**2))
            zVec = zVec / zVec_norm

            # get rotation matrix that rotates z onto the vertex normal
            # c.f. https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
            if np.all(nVec == np.array([0.0, 0.0, 1.0], dtype=np.float32)):
                R = np.identity(3)
            elif np.all(nVec == np.array([0.0, 0.0, -1.0], dtype=np.float32)):
                R = np.array([[1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0],
                              [0.0, 0.0, -1.0]])
            else:
                v = np.cross(zVec, nVec)
                s = np.sqrt(np.sum(v**2))
                c = np.dot(zVec, nVec)
                vx = np.array([[0.0, -v[2], v[1]],
                               [v[2], 0.0, -v[0]],
                               [-v[1], v[0], 0.0]])
                R = np.identity(3) + vx + np.dot(vx, vx) * (1 - c) / s**2
            r = marker_size / resolution * fac_remove_markers # mm

            vec_t = xyz_use# - t
            vec_rt = np.einsum('ij,kj->ki', R, vec_t)
            eq = np.einsum('ij,ij->i', vec_rt, vec_rt) - r**2
            mask1 = (eq <= 0).reshape(xyz_shape[:-1])

            # project points onto new z-direction and check if the lambda value is positive
            vec_proj_lambda = np.dot(vec_t, R[:, 2]) / np.dot(R[:, 2], R[:, 2])
            mask2 = (vec_proj_lambda >= 0).reshape(xyz_shape[:-1])
            mask = mask1 & mask2

            t_use = t - (marker_size / resolution * (fac_remove_markers - 1) / 2) * nVec
            mri_data_use[int(t_use[0])-dxyz:int(t_use[0])+dxyz+1,
                         int(t_use[1])-dxyz:int(t_use[1])+dxyz+1,
                         int(t_use[2])-dxyz:int(t_use[2])+dxyz+1][mask] = 0

    # this creates a binary image only containing the voxels that belong to the rat
    def largest_label_volume(im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)
    
        counts = counts[vals != bg]
        vals = vals[vals != bg]
    
        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    threshold_up = np.max(mri_data_use)
    
    binary_image = np.array((mri_data_use > threshold_low) & ((mri_data_use < threshold_up)), dtype=np.int16)
    
    labeled_areas = measure.label(binary_image, connectivity=1)
    vals, counts = np.unique(labeled_areas, return_counts=True)
    counts_argsort = np.argsort(counts)
    index_air = vals[counts_argsort][-1]
    
    binary_image_use = np.ones(mri_shape, dtype=bool)
    binary_image_use[labeled_areas == index_air] = False
    
    # For every slice we determine the largest solid structure
    for i, axial_slice in enumerate(binary_image_use):
        axial_slice = axial_slice - 1
        labeling = measure.label(axial_slice, connectivity=1)
        l_max = largest_label_volume(labeling, bg=0)
        if l_max is not None: #This slice contains some lung
            binary_image_use[i][labeling != l_max] = True

    # this only keeps the labeled structure with the most counts (excluding the background)
    labeled_areas = measure.label(binary_image_use, connectivity=2)
    vals, counts = np.unique(labeled_areas, return_counts=True)
    counts_argsort = np.argsort(counts)
    counts = counts[counts_argsort]
    vals = vals[counts_argsort]
    index_test = vals[-2]
    binary_image_use2 = np.ones(mri_shape, dtype=bool)
    binary_image_use2[labeled_areas != index_test] = False

    # threshold must be 0.5 when marching cubes is performed on a binary image
    threshold = 0.5
    
    # Use marching cubes to obtain the surface mesh
    surface_vertices, faces, normals, values = measure.marching_cubes_lewiner(binary_image_use2,
                                                                   level=threshold,
                                                                   spacing=(resolution, resolution, resolution),
                                                                   gradient_direction='descent',
                                                                   step_size=1,
                                                                   allow_degenerate=True,
                                                                   use_classic=False)
    surface_triangles = [tuple(faces[i]) for i in range(len(faces))]
    
    return surface_vertices, surface_triangles

def extract_mri_labeling(file_mri_labeling, resolution, joint_name_start):
    model = np.load(file_mri_labeling, allow_pickle=True).item()
    labels3d = model['joints']
    links = model['links']
    
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

def rodrigues2rotMat(r): # 3
    sqrt_arg = np.sum(r**2, 0)
    theta = np.sqrt(sqrt_arg)
    if (theta > 2**-23):
        omega = r / theta
        omega_hat = np.array([[0.0, -omega[2], omega[1]],
                              [omega[2], 0.0, -omega[0]],
                              [-omega[1], omega[0], 0.0]], dtype=np.float64)
        rotMat = np.eye(3, dtype=np.float64) + \
                 np.sin(theta) * omega_hat + \
                 (1.0 - np.cos(theta)) * np.einsum('ij,jk->ik', omega_hat, omega_hat)
    else:
        rotMat = np.eye(3, dtype=np.float64)
    return rotMat # 3, 3

def get_bone_lengths_index(model_ini):
    skeleton_edges = model_ini['skeleton_edges']
    joint_order = model_ini['joint_order']
    nBones = int(np.size(skeleton_edges, 0))

    bone_lengths_index = np.ones(nBones, dtype=np.int64)
    bone_lengths_index_entry = 0
    for i_bone in range(nBones):
        joint_name = joint_order[skeleton_edges[i_bone, 1]]
        joint_name_split = joint_name.split('_')
        mask = np.array([i == 'right' for i in joint_name_split], dtype=bool)
        joint_is_symmetric = np.any(mask)
        if (joint_is_symmetric):
            index_left_right = np.arange(len(mask), dtype=np.int64)[mask][0]
            index1 = joint_order.index('_'.join(joint_name_split[:index_left_right]) + 
                                       '_left' + '_' * len(joint_name_split[index_left_right+1:]) + 
                                       '_'.join(joint_name_split[index_left_right+1:]))
            index2 = np.where(skeleton_edges[:, 1] == index1)[0][0]
            bone_lengths_index[i_bone] = bone_lengths_index[index2]
        else:
            bone_lengths_index[i_bone] = bone_lengths_index_entry
            bone_lengths_index_entry += 1
    return bone_lengths_index

def plot_model(x_torch, args):    
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

    skel_coords_new = skel_coords_new.numpy().squeeze()
    skel_verts_new = skel_verts_new.numpy().squeeze()
    surf_verts_new = surf_verts_new.numpy().squeeze()
    joint_marker_pos = joint_marker_pos.numpy().squeeze()
    return skel_coords_new, skel_verts_new, surf_verts_new, joint_marker_pos

if __name__ == '__main__':
    date = '20200207' # '20200205' or '20200207'
    task = 'arena'
    folder_save = os.path.abspath('panels')
    folder_recon = data.path+'/reconstruction'
    folder = folder_recon+'/'+date+'/'+task+'_'+date+'_calibration_on'

    sys.path.append(folder)
    importlib.reload(cfg)
    cfg.animal_is_large = 0
    importlib.reload(anatomy)

    folder_reqFiles = data.path + '/required_files' 
    file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
    file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
    file_model = folder_reqFiles + '/model.npy'
#     file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/' + '/labels_dlc_{:06d}_{:06d}.npy'.format(cfg.index_frame_start, cfg.index_frame_end)
    file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/' + cfg.file_labelsDLC.split('/')[-1]

    file_align = folder + '/x_align.npy'
    file_mri =  folder_reqFiles + '/' + date + '/' + '/mri_data.npy'
    file_skeleton = folder_reqFiles + '/' + date + '/' + '/mri_skeleton0_full.npy'
    model_ini = np.load(file_model, allow_pickle=True).item()
    model_torch = helper.get_model3d(file_model)
    model_torch['bone_lengths_index'] = torch.from_numpy(get_bone_lengths_index(model_ini))

    x_align = np.load(file_align, allow_pickle=True)
    x_align_torch = torch.from_numpy(x_align)

    args = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                cfg.scale_factor, cfg.pcutoff)
    skeleton_coords_new, skeleton_vertices_new, surface_vertices_new, marker_positions_new = \
        plot_model(x_align_torch, args)
    joint_order_new = args['model']['joint_order']
    skeleton_edges_new = args['model']['skeleton_edges'].numpy()

    resolution_mri = 0.4 # resolution in mm
    marker_size = 3.0 # radius in mm (needs to be the same unit as the resolution)
    fac_remove_markers = 1.25 # simulates a somewhat bigger half spherical marker to remove the most of the intensity produced by the markers in the mri scan
    threshold_low = 50.0 # every voxel lower than this threshold is considered noise and set to zero when the marching cubes algorithm is used
    joint_start = 'joint_head_001' # defines where the skeleton graph starts

    joints_mri, markers_mri, skeleton_verts_mri, skeleton_edges_mri, joint_order = \
        extract_mri_labeling(file_skeleton, resolution_mri, joint_start)
    surface_vertices, surface_triangles = \
        perform_marching_cubes(file_mri,
                               resolution_mri, marker_size,
                               fac_remove_markers,
                               threshold_low,
                               markers_mri)
    #
    nBones = int(np.shape(skeleton_edges_mri)[0])
    nMarkers = int(np.size(list(markers_mri.keys())) / 2)
    markers0 = np.zeros((nMarkers, 3), dtype=np.float64)
    for i_marker in range(nMarkers):
        marker_name = list(markers_mri.keys())[i_marker]
        markers0[i_marker] = np.copy(markers_mri[marker_name])
    joints0 = np.zeros((nBones+1, 3), dtype=np.float64)
    for i_joint in range(nBones+1):
        joint_name = list(joints_mri.keys())[i_joint]
        joints0[i_joint] = np.copy(joints_mri[joint_name])

    surf_verts0 = np.copy(surface_vertices)
    surf_verts0 = surf_verts0 * 1e-1 # mri -> cm
    origin = joints_mri[joint_start] * resolution_mri * 1e-1 # mri -> cm
    surf_verts0 = surf_verts0 - origin
    
    skel_edges0 = np.copy(skeleton_edges_new)
    skel_verts0 = np.copy(skeleton_vertices_new)
    origin_new = skeleton_vertices_new[0]
    skel_verts0 = skel_verts0 - origin_new

    fig = mlab.figure(1,
                      size=(1280, 1280),
                      bgcolor=(1, 1, 1))
    mlab.clf()

    colors_limts = list([(1.0, 0.0, 0.0),
                         (0.0, 1.0, 0.0),
                         (0.0, 0.0, 1.0)])
    limit_length = 2/3
    nAng = 20
    
    bounds_pose, _, _, _ = anatomy.get_bounds_pose(args['numbers']['nBones'], skeleton_edges_new, joint_order_new)
    bounds_pose = bounds_pose[6:] # get rid of t0 and r0

    bounds_pose = bounds_pose.reshape(args['numbers']['nBones']-1, 3, 2)
    for i_bone in range(1, args['numbers']['nBones']):
        joint_index_start = skeleton_edges_new[i_bone, 0]
        joint_index_end = skeleton_edges_new[i_bone, 1]
        joint_start = skeleton_vertices_new[joint_index_start] - origin_new
        joint_end = skeleton_vertices_new[joint_index_end] - origin_new
        #
        joint_name = joint_order_new[joint_index_start]
        joint_name_split = joint_name.split('_')
        if (True):
            skeleton_coords_index = np.where(skeleton_edges_new[:, 1] == joint_index_start)[0][0]
            skeleton_coords = skeleton_coords_new[skeleton_coords_index]
            r = np.zeros(3, dtype=np.float64)
            if (joint_name == 'joint_shoulder_left'):
                r = np.array([0.0, np.pi * -1/2, 0.0], dtype=np.float64)
            elif (joint_name == 'joint_hip_left'):
                r = np.array([0.0, np.pi * -1/2, 0.0], dtype=np.float64)
            elif (joint_name == 'joint_shoulder_right'):
                r = np.array([0.0, np.pi * 1/2, 0.0], dtype=np.float64)
            elif (joint_name == 'joint_hip_right'):
                r = np.array([0.0, np.pi * 1/2, 0.0], dtype=np.float64)
            if not(np.all(r == 0.0)):
                R = rodrigues2rotMat(r)
                skeleton_coords = np.dot(skeleton_coords, R.T)

            for d in range(3):
                limit_down = bounds_pose[i_bone-1, d, 0]
                limit_up = bounds_pose[i_bone-1, d, 1]
                if (limit_down != limit_up):
                    vec0 = skeleton_coords[:, 2]
                    vec0 = vec0 * limit_length
                    alpha_range = np.linspace(limit_down, limit_up, nAng)
                    angle_plot = np.zeros((nAng, 3), dtype=np.float64)
                    for i_alpha in range(nAng):
                        r = skeleton_coords[:, d]
                        if (d == 2):
                            if ('left' in joint_name_split):
                                vec0_use = skeleton_coords[:, 0] * limit_length * -1.0
                            elif ('right' in joint_name_split):
                                vec0_use = skeleton_coords[:, 0] * limit_length
                        else:
                            vec0_use = np.copy(vec0)
                        r = r * alpha_range[i_alpha]
                        R = rodrigues2rotMat(r)
                        angle_plot[i_alpha] = np.dot(R, vec0_use)
                    angle_plot = np.concatenate([joint_start[None, :],
                                                 angle_plot + joint_start[None, :],
                                                 joint_start[None, :]], 0)
                    i_ang = mlab.plot3d(angle_plot[:, 0], angle_plot[:, 1], angle_plot[:, 2],
                                        color=colors_limts[d],
                                        figure=fig,
                                        line_width=bones3d_line_width,
                                        tube_radius=bones3d_tube_radius*0.5,
                                        tube_sides=bones3d_tube_sides)
    skel = mlab.points3d(skel_verts0[:, 0],
                         skel_verts0[:, 1],
                         skel_verts0[:, 2],
                         color=color_skel,
                         figure=fig,
                         scale_factor=skeleton3d_scale_factor*0.5)
    # bones0
    for edge in skel_edges0:
        joint0 = skel_verts0[edge[0]]
        joint1 = skel_verts0[edge[1]]
        i_bone = mlab.plot3d(np.array([joint0[0], joint1[0]], dtype=np.float64),
                             np.array([joint0[1], joint1[1]], dtype=np.float64),
                             np.array([joint0[2], joint1[2]], dtype=np.float64),
                             color=color_skel,
                             figure=fig,
                             line_width=bones3d_line_width,
                             tube_radius=bones3d_tube_radius,
                             tube_sides=bones3d_tube_sides)
    # surface
    nSurf = np.size(surf_verts0, 0)
    scalars = np.full(nSurf, color_surf, dtype=np.float64)
    surf = mlab.triangular_mesh(surf_verts0[:, 0],
                                surf_verts0[:, 1],
                                surf_verts0[:, 2],
                                surface_triangles,
                                colormap='gray',
                                figure=fig,
                                opacity=0.1,
                                scalars=scalars,
                                transparent=True,
                                vmin=0,
                                vmax=1)
    fig.scene.parallel_projection = True
    mlab.view(figure=fig,
              azimuth=float(90),
              elevation=float(0),
              distance=float(5),
              focalpoint=np.mean(skel_verts0, 0))
    fig.scene.camera.zoom(1.0)
    mlab.draw()
    plt.pause(2**-10)
    if save:
        save_fig3d(folder_save + '/mri3d_limits.svg', fig)
    if show:
        mlab.show()
        plt.show()