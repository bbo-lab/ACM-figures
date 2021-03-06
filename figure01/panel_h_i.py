#!/usr/bin/env python3

import importlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mayavi import mlab
import numpy as np
from skimage import measure
import os
import sys
import torch

sys.path.append(os.path.abspath('../ACM/config'))
import configuration as cfg
sys.path.pop(sys.path.index(os.path.abspath('../ACM/config')))
#
sys.path.append(os.path.abspath('../ACM'))
import data
import helper
import anatomy
import model
sys_path0 = np.copy(sys.path)

# save = False # LEGACY: comment to prevent memory issues (probably only occuring on Windows machines?)
save_all = False
verbose = True

folder_save = os.path.abspath('panels')

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

left_margin_x = 0.01 # inch
left_margin  = 0.4 # inch
right_margin = 0.05 # inch
bottom_margin = 0.25 # inch
top_margin = 0.05 # inch

fontsize = 6
linewidth = 1.0
linewidth_hist = 1.0
fontname = "Arial"
markersize = 1

#
folder_reconstruction = data.path+'/reconstruction'
folder_list = list([[folder_reconstruction+'/20200205/arena_20200205_calibration_on',
                     folder_reconstruction+'/20200207/arena_20200207_calibration_on',
                     folder_reconstruction+'/20210511_1/table_{:01d}_20210511_calibration'.format(1),
                     folder_reconstruction+'/20210511_2/table_{:01d}_20210511_calibration__more'.format(2),
                     folder_reconstruction+'/20210511_3/table_{:01d}_20210511_calibration'.format(3),
                     folder_reconstruction+'/20210511_4/table_{:01d}_20210511_calibration__more'.format(4),]])
color_index_list = list([1, 1, 0, 0, 2, 2])
file_mri_list =  list(['/mri_data.npy',
                       '/mri_data.npy',
                       '/mri_{:01d}'.format(1) + '/mri_data.npy',
                       '/mri_{:01d}'.format(2) + '/mri_data.npy',
                       '/mri_{:01d}'.format(3) + '/mri_data.npy',
                       '/mri_{:01d}'.format(4) + '/mri_data.npy'])
file_skeleton_list = list(['/mri_skeleton0_full.npy',
                           '/mri_skeleton0_full.npy',
                           '/mri_{:01d}'.format(1) + '/mri_skeleton0_full.npy',
                           '/mri_{:01d}'.format(2) + '/mri_skeleton0_full.npy',
                           '/mri_{:01d}'.format(3) + '/mri_skeleton0_full.npy',
                           '/mri_{:01d}'.format(4) + '/mri_skeleton0_full.npy'])
list_is_large_animal = list([0, 0, 0, 0, 1, 1])

cmap = plt.cm.viridis
color_mri = cmap(2/3)
color_fit = cmap(0.0)
cmap2 = plt.cm.tab10
color_animal_size1 = cmap2(0/9)
color_animal_size2 = cmap2(1/9)
color_animal_size3 = cmap2(2/9)
color_error = 'gray'
color_animals = list([color_animal_size1, color_animal_size2, color_animal_size3])

skeleton3d_scale_factor = 1/3
color_skel = (0.5, 0.5, 0.5)
color_surf = 0.0
bones3d_line_width = 2
bones3d_tube_radius = 0.1
bones3d_tube_sides = 16

def save_fig3d(file, fig, bool_legend=False):
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
    fig0 = plt.figure(1, figsize=(fig0_w, fig0_h))
    fig0.clear()
    fig0.set_facecolor('white')
    fig0.set_dpi(300)
    ax0 = fig0.add_subplot(111, frameon=True)
    ax0.clear()
    ax0.set_facecolor('white')
    ax0.set_position([0, 0, 1, 1])
    #
    im = ax0.imshow(screenshot,
                    aspect='equal',
                    zorder=1)
    ax0.set_axis_off()
    #
    if bool_legend:
        linewidth3d = 10
        legend_0 = mlines.Line2D([], [],
                                 color=color_mri,
                                 marker='',
                                 linestyle='-',
                                 linewidth=linewidth3d,
                                 label='MRI')
        legend_fit = mlines.Line2D([], [],
                                 color=color_fit,
                                 marker='',
                                 linestyle='-',
                                 linewidth=linewidth3d,
                                 label='Fit')
    h_legend_list = list([legend_0, legend_fit])
    h_legend = ax0.legend(handles=h_legend_list, loc='upper left', fontsize=fontsize, frameon=False)
    for text in h_legend.get_texts():
        text.set_fontname(fontname)

    #
#     fig0.tight_layout()
    fig0.canvas.draw()
    #    
    fig0.savefig(file,
                 bbox_inches="tight",
                 dpi=300,
                 transparent=True,
                 format='svg')     
    return

def perform_marching_cubes(file_mri_data,
                           resolution, marker_size,
                           fac_remove_markers,
                           threshold_low,
                           markers):
    mri_data = np.load(file_mri_data, allow_pickle=True)
    mri_shape = np.shape(mri_data)
    
    
    #%% this gets rid of the markers
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

    #%% this creates a binary image only containing the voxels that belong to the rat
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

def plot_model_3d(fig,
                  skel_edges, skel_verts,
                  surf_tri=None, surf_verts=None,
                  color=(0.0, 0.0, 0.0)):
    nBones = np.size(skel_edges, 0)
    nJoints = nBones + 1
        
    # surface
    if not(np.any((surf_tri == None ) or (surf_verts == None))):
        nSurf = np.size(surf_verts, 0)
        scalars = np.full(nSurf, color_surf, dtype=np.float64)
        surf = mlab.triangular_mesh(surf_verts[:, 0],
                                    surf_verts[:, 1],
                                    surf_verts[:, 2],
                                    surf_tri,
                                    colormap='gray',
                                    figure=fig,
                                    opacity=0.5,
                                    scalars=scalars,
                                    transparent=True,
                                    vmin=0,
                                    vmax=1)
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
    return

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

if __name__ == '__main__':
    fig_w = np.round(mm_in_inch * 88.0*1/3, decimals=2)
    fig_h = np.round(mm_in_inch * 88.0*1/3, decimals=2)
    #
    fig31_w = fig_w
    fig31_h = fig_h
    fig31 = plt.figure(31, figsize=(fig31_w, fig31_h))
    fig31.canvas.manager.window.move(0, 0)
    fig31.clear()
    ax31_x = left_margin/fig31_w
    ax31_y = bottom_margin/fig31_h
    ax31_w = 1.0 - (left_margin/fig31_w + right_margin/fig31_w)
    ax31_h = 1.0 - (bottom_margin/fig31_h + top_margin/fig31_h)
    ax31 = fig31.add_axes([ax31_x, ax31_y, ax31_w, ax31_h])
    ax31.clear()
    ax31.spines["top"].set_visible(False)
    ax31.spines["right"].set_visible(False)

    fig61_w = fig_w
    fig61_h = fig_h
    fig61 = plt.figure(61, figsize=(fig61_w, fig61_h))
    fig61.canvas.manager.window.move(0, 0)
    fig61.clear()    
    ax61_x = left_margin/fig61_w
    ax61_y = bottom_margin/fig61_h
    ax61_w = 1.0 - (left_margin/fig61_w + right_margin/fig61_w)
    ax61_h = 1.0 - (bottom_margin/fig61_h + top_margin/fig61_h)
    ax61 = fig61.add_axes([ax61_x, ax61_y, ax61_w, ax61_h])
    ax61.clear()
    ax61.spines["top"].set_visible(False)
    ax61.spines["right"].set_visible(False)

    fig81_w = fig_w
    fig81_h = fig_h
    fig81 = plt.figure(81, figsize=(fig81_w, fig81_h))
    fig81.canvas.manager.window.move(0, 0)
    fig81.clear()
    ax81_x = left_margin/fig81_w
    ax81_y = bottom_margin/fig81_h
    ax81_w = 1.0 - (left_margin/fig81_w + right_margin/fig81_w)
    ax81_h = 1.0 - (bottom_margin/fig81_h + top_margin/fig81_h)
    ax81 = fig81.add_axes([ax81_x, ax81_y, ax81_w, ax81_h])
    ax81.clear()
    ax81.spines["top"].set_visible(False)
    ax81.spines["right"].set_visible(False)
    
    
    for i_mode in range(np.size(folder_list, 0)):
        mode = list(['on', 'off'])[i_mode]
        #
        bone_length0_all = list()
        bone_length1_all = list()
        bone_angles0_all = list()
        bone_angles1_all = list()
        #
        joint_pos_error_all = list()
        bone_lengths_error_all = list()
        bone_angles_error_all = list()
        for i_folder in range(len(folder_list[i_mode])):
            folder = folder_list[i_mode][i_folder]

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
            file_labelsDLC = folder_reqFiles + '/labels_dlc_dummy.npy' # DLC labels are actually never needed here

            file_mri = folder_reqFiles + '/' + cfg.date + '/' + file_mri_list[i_folder]
            file_skeleton = folder_reqFiles + '/' + cfg.date + '/' + file_skeleton_list[i_folder]
        
            file_align = folder + '/x_align.npy'
        
            file_ini = file_model
            model_ini = np.load(file_ini, allow_pickle=True).item()
            model_torch = helper.get_model3d(file_ini)
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
        #         joints0[i_joint] = np.copy(joints_mri[joint_order[i_joint]])
                joint_name = list(joints_mri.keys())[i_joint]
                joints0[i_joint] = np.copy(joints_mri[joint_name])
            skel_edges0 = np.copy(skeleton_edges_mri)
            skel_verts0 = np.copy(skeleton_verts_mri)
            surf_verts0 = np.copy(surface_vertices)
            #
            markers0 = markers0 * resolution_mri * 1e-1 # mri -> cm
            joints0 = joints0 * resolution_mri * 1e-1 # mri -> cm
            skel_verts0 = skel_verts0 * 1e-1 # mri -> cm
            surf_verts0 = surf_verts0 * 1e-1 # mri -> cm
            #
        #     origin = joints_mri[joint_start] * resolution_mri * 1e-1 # mri -> cm
        #     markers0 = markers0 - origin
        #     joints0 = joints0 - origin
        #     skel_verts0 = skel_verts0 - origin
        #     surf_verts0 = surf_verts0 - origin
            #
            skeleton_edges0 = skeleton_edges_mri
        #     skeleton_coords0 = skeleton_coords_new
            skeleton_vertices0 = skel_verts0
            surface_triangles0 = surface_triangles
            surface_vertices0 = surf_verts0
            marker_positions0 = markers0

#             # LEGACY: comment to prevent memory issues (probably only occuring on Windows machines?)
#         #     mlab.options.offscreen = True
#             # PLOT
#             fig = mlab.figure(1,
#                               size=(1080, 1080),
#                               bgcolor=(1, 1, 1))
#         #     fig = mlab.figure(1,
#         #                       size=(10240, 10240),
#         #                       bgcolor=(1, 1, 1))
#             mlab.clf()
#             plot_model_3d(fig,
#                           skeleton_edges0, skeleton_vertices0,
#                           surface_triangles0, surface_vertices0,
#                           color=color_mri[:3])
#             plot_model_3d(fig,
#                           skeleton_edges_new, skeleton_vertices_new,
#                           None, None,
#                           color=color_fit[:3])

#             # scalebar
#             scalebar_location_start = np.array([2.5, 0.0, 1.0], dtype=np.float64)
#             scalebar_location_end = np.array([2.5, 4.0, 1.0], dtype=np.float64)
#             scalebar = mlab.plot3d(np.array([scalebar_location_start[0], scalebar_location_end[0]], dtype=np.float64),
#                                    np.array([scalebar_location_start[1], scalebar_location_end[1]], dtype=np.float64),
#                                    np.array([scalebar_location_start[2], scalebar_location_end[2]], dtype=np.float64),
#                                    color=(0.0, 0.0, 0.0),
#                                    figure=fig,
#                                    line_width=bones3d_line_width,
#                                    tube_radius=bones3d_tube_radius,
#                                    tube_sides=bones3d_tube_sides)
#         #     scalebar_text_location = scalebar_location_start + (scalebar_location_end - scalebar_location_start) / 2.0
#             scalebar_text_location = scalebar_location_end + np.array([0.5, -0.75, 0.5], dtype=np.float)
#             scalebar_length = np.sqrt(np.sum((scalebar_location_end - scalebar_location_start)**2))
#             scalebar_text = mlab.text3d(scalebar_text_location[0],
#                                         scalebar_text_location[1],
#                                         scalebar_text_location[2],
#                                         '{:0.1f} cm'.format(scalebar_length),
#                                         figure=fig,
#                                         color=(0.0, 0.0, 0.0),
#                                         line_width=1.0,
#                                         scale=0.5)

#             fig.scene.parallel_projection = True

#             mlab.view(figure=fig,
#                       azimuth=90,
#                       elevation=0,
#                       distance=25,
#                       focalpoint=np.mean(joints0, 0))
#             if save:
#                 save_fig3d(folder+'/model_aligned_xy.svg', fig, True)

#             mlab.view(figure=fig,
#                       azimuth=0,
#                       elevation=-90,
#                       distance=25,
#                       focalpoint=np.mean(joints0, 0))
#             if save:
#                 save_fig3d(folder+'/model_aligned_xz.svg', fig, True)

#         #     mlab.show()


            # COMPARE LENGTHS & ANGLES
            nBones0 = np.size(skeleton_edges0, 0)
            nBones1 = np.size(skeleton_edges_new, 0)
            # get joint connections for angle calculation
            angle_joint_connections = list([])
            for i_joint in range(nBones0+1):
                mask_from = (skeleton_edges0[:, 0] == i_joint)
                mask_to = (skeleton_edges0[:, 1] == i_joint)
                for i_joint_from in skeleton_edges0[:, 0][mask_to]:
                    for i_joint_to in skeleton_edges0[:, 1][mask_from]:
                        angle_joint_connections.append(np.array([i_joint_from, i_joint, i_joint_to], dtype=np.int64))
            angle_joint_connections = np.array(angle_joint_connections, dtype=np.int64)
            nAngles = len(angle_joint_connections)
            #
            angle_joint_connections_new = list([])
            for i_joint in range(nBones1+1):
                mask_from = (skeleton_edges_new[:, 0] == i_joint)
                mask_to = (skeleton_edges_new[:, 1] == i_joint)
                for i_joint_from in skeleton_edges_new[:, 0][mask_to]:
                    for i_joint_to in skeleton_edges_new[:, 1][mask_from]:
                        angle_joint_connections_new.append(np.array([i_joint_from, i_joint, i_joint_to], dtype=np.int64))
            angle_joint_connections_new = np.array(angle_joint_connections_new, dtype=np.int64)
            nAngles_new = len(angle_joint_connections_new)
            #
            bone_name0 = list()
            joint_name0 = list()
            #
            bone_name_joint0 = list()
            joint_pos0 = list()
            # head to cervical
            bone_name_joint0.append('begin cervical vertebrae')
            i_joint = joint_order.index('joint_spine_031')
            pos = skeleton_vertices0[i_joint]
            joint_pos0.append(pos)
            # cervical to thoracic
            bone_name_joint0.append('begin thoracic vertebrae')
            i_joint = joint_order.index('joint_spine_026')
            pos1 = skeleton_vertices0[i_joint]
            i_joint = joint_order.index('joint_spine_025')
            pos2 = skeleton_vertices0[i_joint]
            pos = 0.5 * (pos1 + pos2)
            joint_pos0.append(pos2)
            # thoracic to lumbar
            bone_name_joint0.append('begin lumbar vertebrae')
            i_joint = joint_order.index('joint_spine_013')
            pos1 = skeleton_vertices0[i_joint]
            i_joint = joint_order.index('joint_spine_012')
            pos2 = skeleton_vertices0[i_joint]
            pos = 0.5 * (pos1 + pos2)
            joint_pos0.append(pos2)
            # lumbar to sacrum
            bone_name_joint0.append('begin sacrum')
            i_joint = joint_order.index('joint_spine_006')
            pos1 = skeleton_vertices0[i_joint]
            i_joint = joint_order.index('joint_spine_005')
            pos2 = skeleton_vertices0[i_joint]
            pos = 0.5 * (pos1 + pos2)
            joint_pos0.append(pos2)
            # sacrum to tail
            bone_name_joint0.append('begin tail')
            i_joint = joint_order.index('joint_tail_021') # joint_tail_021 still part of sacrum
            pos1 = skeleton_vertices0[i_joint]
            i_joint = joint_order.index('joint_tail_020')
            pos2 = skeleton_vertices0[i_joint]
            pos = 0.5 * (pos1 + pos2)
            joint_pos0.append(pos2)
            #
            bone_length0 = list()
            bone_angles0 = list()
            for i_bone in range(nBones0):
                joint_name_start = joint_order[skeleton_edges0[i_bone, 0]]
                joint_name_end = joint_order[skeleton_edges0[i_bone, 1]]
                joint_name_start_split = joint_name_start.split('_')
                joint_name_end_split = joint_name_end.split('_')
                cond_start = (('left' in joint_name_start_split) or ('right' in joint_name_start_split)) 
                cond_end = (('left' in joint_name_end_split) or ('right' in joint_name_end_split))
                if (cond_start or cond_end):
                    joint_name_start_use = '_'.join(joint_name_start_split[1:])
                    joint_name_end_use = '_'.join(joint_name_end_split[1:])
                    # angles
                    if skeleton_edges0[i_bone, 1] in angle_joint_connections[:, 1]:
                        index = np.where(angle_joint_connections[:, 1] == skeleton_edges0[i_bone, 1])[0][0]
                        joint_pos_0 = skeleton_vertices0[angle_joint_connections[index, 0]]
                        joint_pos_1 = skeleton_vertices0[angle_joint_connections[index, 1]]
                        joint_pos_2 = skeleton_vertices0[angle_joint_connections[index, 2]]
                        vec1 = joint_pos_1 - joint_pos_0
                        vec2 = joint_pos_2 - joint_pos_1
                        vec1 /= np.sqrt(np.sum(vec1**2))
                        vec2 /= np.sqrt(np.sum(vec2**2))
                        angle = np.arccos(np.dot(vec1, vec2))
                        bone_angles0.append(angle)
                        joint_name0.append(joint_name_end_use)
                    # length
                    if ('spine' in joint_name_start_use):
                        joint_name_start_use = 'spine'
                    bone_name0.append(joint_name_start_use + \
                                      ' --- ' + \
                                      '_'.join(joint_name_end_split[1:]))
                    bone_name_joint0.append(' '.join(joint_name_end_split[1:]))
                    #
                    joint_pos_start = skeleton_vertices0[skeleton_edges0[i_bone, 0]]
                    joint_pos_end = skeleton_vertices0[skeleton_edges0[i_bone, 1]]
                    length = np.sqrt(np.sum((joint_pos_end - joint_pos_start)**2))
                    bone_length0.append(length)
                    joint_pos0.append(joint_pos_end)
            bone_name0 = np.array(bone_name0)
            bone_name_joint0 = np.array(bone_name_joint0)
            bone_length0 = np.array(bone_length0)
            joint_pos0 = np.array(joint_pos0)
            bone_angles0 = np.array(bone_angles0)
            index_sort = np.argsort(bone_name0)
            bone_name0 = bone_name0[index_sort]
            bone_length0 = bone_length0[index_sort]
            #

            bone_name1 = list()
            joint_name1 = list()
            #
            bone_name_joint1 = list()
            joint_pos1 = list()
            # head to cervical
            bone_name_joint1.append('begin cervical vertebrae')
            i_joint = joint_order_new.index('joint_spine_005')
            pos = skeleton_vertices_new[i_joint]
            joint_pos1.append(pos)
            # cervical to thoracic
            bone_name_joint1.append('begin thoracic vertebrae')
            i_joint = joint_order_new.index('joint_spine_004')
            pos = skeleton_vertices_new[i_joint]
            joint_pos1.append(pos)
            # thoracic to lumbar
            bone_name_joint1.append('begin lumbar vertebrae')
            i_joint = joint_order_new.index('joint_spine_003')
            pos = skeleton_vertices_new[i_joint]
            joint_pos1.append(pos)
            # lumbar to sacrum
            bone_name_joint1.append('begin sacrum')
            i_joint = joint_order_new.index('joint_spine_002')
            pos = skeleton_vertices_new[i_joint]
            joint_pos1.append(pos)
            # sacrum to tail
            bone_name_joint1.append('begin tail')
            i_joint = joint_order_new.index('joint_spine_001')
            pos = skeleton_vertices_new[i_joint]
            joint_pos1.append(pos)
            #
            bone_length1 = list()        
            bone_angles1 = list()
            for i_bone in range(nBones1):
                joint_name_start = joint_order_new[skeleton_edges_new[i_bone, 0]]
                joint_name_end = joint_order_new[skeleton_edges_new[i_bone, 1]]
                joint_name_start_split = joint_name_start.split('_')
                joint_name_end_split = joint_name_end.split('_')
                cond_start = (('left' in joint_name_start_split) or ('right' in joint_name_start_split)) 
                cond_end = (('left' in joint_name_end_split) or ('right' in joint_name_end_split)) 
                if (cond_start or cond_end):
                    joint_name_start_use = '_'.join(joint_name_start_split[1:])
                    joint_name_end_use = '_'.join(joint_name_end_split[1:])
                    # angles
                    if skeleton_edges_new[i_bone, 1] in angle_joint_connections_new[:, 1]:
                        index = np.where(angle_joint_connections_new[:, 1] == skeleton_edges_new[i_bone, 1])[0][0]
                        joint_pos_0 = skeleton_vertices_new[angle_joint_connections_new[index, 0]]
                        joint_pos_1 = skeleton_vertices_new[angle_joint_connections_new[index, 1]]
                        joint_pos_2 = skeleton_vertices_new[angle_joint_connections_new[index, 2]]
                        vec1 = joint_pos_1 - joint_pos_0
                        vec2 = joint_pos_2 - joint_pos_1
                        vec1_len = np.sqrt(np.sum(vec1**2))
                        vec2_len = np.sqrt(np.sum(vec2**2))
                        if ((vec1_len > 2**-23) and (vec2_len > 2**-23)):
                            vec1 = vec1 / vec1_len
                            vec2 = vec2 / vec2_len
                            vec1_vec2_dot = np.dot(vec1, vec2)
                            
                            # when vectors are perfectly aligned numerical erros might cause the dot product to be >1
                            if (vec1_vec2_dot > 1.0):
                                vec1_vec2_dot = 1.0
                            
                            angle = np.arccos(vec1_vec2_dot)
                        else:
                            name1 = joint_order_new[angle_joint_connections_new[index, 0]]
                            name2 = joint_order_new[angle_joint_connections_new[index, 1]]
                            name3 = joint_order_new[angle_joint_connections_new[index, 2]]
                            print(name1, ' --- ', name2, ' --- ', name3)
                            print(vec1_len, vec2_len)
                            angle = np.nan
                        bone_angles1.append(angle)
                        joint_name1.append(joint_name_end_use)
                    # length
    #                 if ('spine' in joint_name_start_use):
    #                     joint_name_start_use = 'spine'
    #                 bone_name1.append('from: ' + joint_name_start_use + \
    #                                   '\n' + \
    #                                   'to: ' + '_'.join(joint_name_end_split[1:]))
                    if ('spine' in joint_name_start_use):
                        joint_name_start_use = 'spine'
                    bone_name1.append(joint_name_start_use + \
                                      ' --- ' + \
                                      '_'.join(joint_name_end_split[1:]))
                    bone_name_joint1.append(' '.join(joint_name_end_split[1:]))
                    #
                    joint_pos_start = skeleton_vertices_new[skeleton_edges_new[i_bone, 0]]
                    joint_pos_end = skeleton_vertices_new[skeleton_edges_new[i_bone, 1]]
                    length = np.sqrt(np.sum((joint_pos_end - joint_pos_start)**2))
                    bone_length1.append(length)
                    joint_pos1.append(joint_pos_end)
            bone_name1 = np.array(bone_name1)
            bone_name_joint1 = np.array(bone_name_joint1)
            bone_length1 = np.array(bone_length1)
            joint_pos1 = np.array(joint_pos1)
            bone_angles1 = np.array(bone_angles1)
            index_sort = np.argsort(bone_name1)
            bone_name1 = bone_name1[index_sort]
            bone_length1 = bone_length1[index_sort]

            # to only compare bones that are labeled in ground truth data set
            index_bones = np.array([i in list(bone_name0) for i in bone_name1], dtype=bool)
            index_bones2 = np.array([i in list(bone_name_joint0) for i in bone_name_joint1], dtype=bool)
            index_joints = np.array([i in list(joint_name0) for i in joint_name1], dtype=bool)
            bone_length1 = bone_length1[index_bones]
            joint_pos1 = joint_pos1[index_bones2]            
            bone_angles1 = bone_angles1[index_joints]
            #
            joint_pos_error = np.sqrt(np.sum((joint_pos1 - joint_pos0)**2, 1))
            joint_pos_error_all.append(joint_pos_error.ravel())
            bone_lengths_error = abs(bone_length1 - bone_length0)
            bone_lengths_error_all.append(bone_lengths_error.ravel())
            bone_angles_error = abs(bone_angles1 - bone_angles0) * 180.0/np.pi
            bone_angles_error_all.append(bone_angles_error.ravel())
            #
            bone_length0_all = bone_length0_all + list(bone_length0)
            bone_length1_all = bone_length1_all + list(bone_length1)
            bone_angles0_all = bone_angles0_all + list(bone_angles0)
            bone_angles1_all = bone_angles1_all + list(bone_angles1)

            # SAVE
            dict_aligned_pose = dict()
            dict_aligned_pose['skeleton_edges'] = skeleton_edges_new
            dict_aligned_pose['joint_order'] = joint_order_new
            dict_aligned_pose['skeleton_coords'] = skeleton_coords_new
            dict_aligned_pose['skeleton_vertices'] = skeleton_vertices_new
            dict_aligned_pose['surface_vertices'] = surface_vertices_new
            dict_aligned_pose['marker_positions'] = marker_positions_new
            #
            dict_aligned_pose['bone_angles0'] = bone_angles0
            dict_aligned_pose['bone_angles1'] = bone_angles1
            dict_aligned_pose['bone_length0'] = bone_length0
            dict_aligned_pose['bone_length1'] = bone_length1
            dict_aligned_pose['joint_pos0'] = joint_pos0
            dict_aligned_pose['joint_pos1'] = joint_pos1
            # 
#             # LEGACY: comment to prevent memory issues (probably only occuring on Windows machines?)
#             if save:
#                 np.save(folder + '/pos_align.npy', dict_aligned_pose)
#                 print('Saved aligned 3D positions ({:s})'.format(folder + '/pos_align.npy'))
#                 print()
            
            # PLOT
    #         X = np.arange(np.size(bone_length0))
    #         fig = plt.figure(1, figsize=(16, 9))
    #         fig.clear()
    #         ax = fig.add_subplot(111)
    #         ax.clear()
    #         ax.spines["top"].set_visible(False)
    #         ax.spines["right"].set_visible(False)
    #         ax.bar(X+0.00, bone_length0, color=color_mri, width=0.25)
    #         ax.bar(X+0.25, bone_length1, color=color_fit, width=0.25)
    #         ax.set_xticks(X+0.125)
    #         ax.set_xticklabels(bone_name0, rotation='vertical', fontname=fontname, fontsize=fontsize)
    #         ax.set_yticks(np.arange(0.0, 4.5, 0.5, dtype=np.float64))
    #         ax.set_yticklabels(np.arange(0.0, 4.5, 0.5, dtype=np.float64), fontname=fontname, fontsize=fontsize)
    #         ax.set_ylim([0.0, 4.1])
    #         ax.set_ylabel('bone length (cm)', fontname=fontname, fontsize=fontsize)
    #         linewidth = 10
    #         legend_0 = mlines.Line2D([], [],
    #                                      color=color_mri,
    #                                      marker='',
    #                                      linestyle='-',
    #                                      linewidth=linewidth,
    #                                      label='MRI')
    #         legend_fit = mlines.Line2D([], [],
    #                                      color=color_fit,
    #                                      marker='',
    #                                      linestyle='-',
    #                                      linewidth=linewidth,
    #                                      label='Fit')
    #         h_legend = list([legend_0, legend_fit])
    #         ax.legend(handles=h_legend, loc='upper right', fontname=fontname, fontsize=fontsize, frameon=False)
    #         fig.tight_layout()
    #         fig.canvas.draw()


    #         Y = np.arange(np.size(bone_length0))
    #         fig2_w = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig2_h = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig2 = plt.figure(2, figsize=(fig2_w, fig2_h))
    #         fig2.canvas.manager.window.move(0, 0)
    #         fig2.clear()
    #         ax2 = fig2.add_subplot(111)
    #         ax2.clear()
    #         ax2.spines["top"].set_visible(False)
    #         ax2.spines["right"].set_visible(False)
    #         ax2.barh(Y+0.25, bone_length0[::-1], color=color_mri, height=0.25)
    #         ax2.barh(Y+0.00, bone_length1[::-1], color=color_fit, height=0.25)    
    #         ax2.set_yticks(Y+0.125)
    #         ax2.set_yticklabels(bone_name0[::-1], rotation='horizontal', horizontalalignment ='right', fontname=fontname, fontsize=fontsize)
    #         ax2.set_xticks(np.arange(0.0, 4.5, 0.5, dtype=np.float64))
    #         ax2.set_xticklabels(np.arange(0.0, 4.5, 0.5, dtype=np.float64), fontname=fontname, fontsize=fontsize)
    #         ax2.set_xlim([0.0, 4.1])
    #         ax2.set_xlabel('bone length (cm)', fontname=fontname, fontsize=fontsize)
    #         linewidth2d = 10
    #         legend_0 = mlines.Line2D([], [],
    #                                      color=color_mri,
    #                                      marker='',
    #                                      linestyle='-',
    #                                      linewidth=linewidth2d,
    #                                      label='MRI')
    #         legend_fit = mlines.Line2D([], [],
    #                                      color=color_fit,
    #                                      marker='',
    #                                      linestyle='-',
    #                                      linewidth=linewidth2d,
    #                                      label='Fit')
    #         h_legend = list([legend_0, legend_fit])
    #         ax2.legend(handles=h_legend, loc='upper right', fontsize=fontsize, frameon=False)
    # #         fig2.tight_layout()
    #         fig2.canvas.draw()    

    #         fig3_w = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig3_h = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig3 = plt.figure(3, figsize=(fig3_w, fig3_h))
    #         fig3.canvas.manager.window.move(0, 0)
    #         fig3.clear()
    #         ax3 = fig3.add_subplot(111)
    #         ax3.clear()
    #         ax3.spines["top"].set_visible(False)
    #         ax3.spines["right"].set_visible(False)
    #         ax3.plot(bone_length0, bone_length1,
    #                 linestyle='', marker='o',
    #                 color=cmap(1/3), zorder=2, markersize=markersize)
    #         dxy_offset = 0.1
    #     #     bone_length_all = np.concatenate([bone_length0, bone_length1], 0)
    #         unity_line = np.array([-10.0, 10.0], dtype=np.float64)
    #         ax3.plot(unity_line, unity_line,
    #                 linestyle='-', marker='',
    #                 color='black', alpha=1.0, zorder=1)
    # #         ax3.set_xticks(np.arange(0.0, 4.5, 0.5, dtype=np.float64))
    # #         ax3.set_yticks(np.arange(0.0, 4.5, 0.5, dtype=np.float64))
    # #         ax3.set_xticklabels(np.arange(0.0, 4.5, 0.5, dtype=np.float64), fontname=fontname, fontsize=fontsize)
    # #         ax3.set_yticklabels(np.arange(0.0, 4.5, 0.5, dtype=np.float64), fontname=fontname, fontsize=fontsize)
    #         ax3.set_xlim([0.0-0.1, 5.0+0.1])
    #         ax3.set_ylim([0.0-0.1, 5.0+0.1])
    #         ax3.set_xlabel('true bone length (cm)', fontname=fontname, fontsize=fontsize)
    #         ax3.set_ylabel('learned bone length (cm)', fontname=fontname, fontsize=fontsize)
    # #         fig3.tight_layout()
    #         fig3.canvas.draw()

# #             if (mode == 'on'):
# #                 color_use = color_mode4
# #             elif (mode == 'off'):
# #                 color_use = color_mode1
#             if (i_folder == 0):
#                 color_use = color_animal1
#             elif (i_folder == 1):
#                 color_use = color_animal2
            color_use = color_animals[color_index_list[i_folder]]
            ax31.plot(bone_length0, bone_length1,
                    linestyle='', marker='o',
                    color=color_use, zorder=2-i_mode, markersize=markersize)
            dxy_offset = 0.1
        #     bone_length_all = np.concatenate([bone_length0, bone_length1], 0)
            ax31.set_xticks(list([0.0, 2.5, 5.0]))
            ax31.set_yticks(list([0.0, 2.5, 5.0]))
            ax31.set_xticklabels(list([0, 2.5, 5.0]), fontname=fontname, fontsize=fontsize)
            ax31.set_yticklabels(list([0, 2.5, 5.0]), fontname=fontname, fontsize=fontsize)
            ax31.set_xlim([0.0, 5.0])
            ax31.set_ylim([0.0, 5.0])
            ax31.set_xlabel('true bone length (cm)', va='center', ha='center', fontname=fontname, fontsize=fontsize)
            ax31.set_ylabel('learned bone length (cm)', va='top', ha='center', fontname=fontname, fontsize=fontsize)
            for tick in ax31.get_xticklabels():
                tick.set_fontsize(fontsize)
                tick.set_fontname(fontname)
            for tick in ax31.get_yticklabels():
                tick.set_fontname(fontname)
                tick.set_fontsize(fontsize)
            ax31.yaxis.set_label_coords(x=left_margin_x/fig31_w, y=ax31_y+0.5*ax31_h, transform=fig31.transFigure)
    #         fig31.tight_layout()
            fig31.canvas.draw()

    #         Y = np.arange(np.size(bone_angles0))
    #         fig5_w = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig5_h = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig5 = plt.figure(5, figsize=(fig5_w, fig5_h))
    #         fig5.canvas.manager.window.move(0, 0)
    #         fig5.clear()
    #         ax5 = fig5.add_subplot(111)
    #         ax5.clear()
    #         ax5.spines["top"].set_visible(False)
    #         ax5.spines["right"].set_visible(False)
    #         ax5.barh(Y+0.25, bone_angles0[::-1]*180.0/np.pi, color=color_mri, height=0.25)
    #         ax5.barh(Y+0.00, bone_angles1[::-1]*180.0/np.pi, color=color_fit, height=0.25)    
    #         ax5.set_yticks(Y+0.125)
    #         ax5.set_yticklabels(joint_name0[::-1], rotation='horizontal', horizontalalignment ='right', fontname=fontname, fontsize=fontsize)
    #         ax5.set_xticks(np.arange(0.0, 120, 20, dtype=np.int64))
    #         ax5.set_xticklabels(np.arange(0.0, 120, 20, dtype=np.int64), fontname=fontname, fontsize=fontsize)
    #         ax5.set_xlim([0.0, 115.0])
    #         ax5.set_xlabel('joint angles (deg)', fontname=fontname, fontsize=fontsize)
    #         linewidth3d = 10
    #         legend_0 = mlines.Line2D([], [],
    #                                      color=color_mri,
    #                                      marker='',
    #                                      linestyle='-',
    #                                      linewidth=linewidth3d,
    #                                      label='MRI')
    #         legend_fit = mlines.Line2D([], [],
    #                                      color=color_fit,
    #                                      marker='',
    #                                      linestyle='-',
    #                                      linewidth=linewidth3d,
    #                                      label='Fit')
    #         h_legend = list([legend_0, legend_fit])
    #         ax5.legend(handles=h_legend, loc='upper right', fontsize=fontsize, frameon=False)
    # #         fig5.tight_layout()
    #         fig5.canvas.draw()

    #         fig6_w = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig6_h = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig6 = plt.figure(6, figsize=(fig6_w, fig6_h))
    #         fig6.canvas.manager.window.move(0, 0)
    #         fig6.clear()
    #         ax6 = fig6.add_subplot(111)
    #         ax6.clear()
    #         ax6.spines["top"].set_visible(False)
    #         ax6.spines["right"].set_visible(False)
    #         ax6.plot(bone_angles0*180.0/np.pi, bone_angles1*180.0/np.pi,
    #                 linestyle='', marker='o',
    #                 color=cmap(1/3), zorder=2, markersize=markersize)
    #         dxy_offset = 0.1
    #     #     bone_angle_all = np.concatenate([bone_angles0*180.0/np.pi, bone_angles1*180.0/np.pi], 0)
    #         unity_line = np.array([-360.0, 360.0], dtype=np.float64)
    #         ax6.plot(unity_line, unity_line,
    #                  linestyle='-', marker='',
    #                  color='black', alpha=1.0, zorder=1)
    #         ax6.set_xticks(list([0, 45, 90]))
    #         ax6.set_yticks(list([0, 45, 90]))
    #         ax6.set_xticklabels(list([0, 45, 90]), fontname=fontname, fontsize=fontsize)
    #         ax6.set_yticklabels(list([0, 45, 90]), fontname=fontname, fontsize=fontsize)
    #         ax6.set_xlim([0.0-5.0, 90.0+5.0])
    #         ax6.set_ylim([0.0-5.0, 90.0+5.0])
    #         ax6.set_xlabel('true joint angle (deg)', fontname=fontname, fontsize=fontsize)
    #         ax6.set_ylabel('learned joint angle (deg)', fontname=fontname, fontsize=fontsize)
    # #         fig6.tight_layout()
    #         fig6.canvas.draw()

# #             if (mode == 'on'):
# #                 color_use = color_mode4
# #             elif (mode == 'off'):
# #                 color_use = color_mode1
#             if (i_folder == 0):
#                 color_use = color_animal1
#             elif (i_folder == 1):
#                 color_use = color_animal2
            color_use = color_animals[color_index_list[i_folder]]
            ax61.plot(bone_angles0*180.0/np.pi, bone_angles1*180.0/np.pi,
                    linestyle='', marker='o',
                    color=color_use, zorder=2-i_mode, markersize=markersize)
            dxy_offset = 0.1
        #     bone_angle_all = np.concatenate([bone_angles0*180.0/np.pi, bone_angles1*180.0/np.pi], 0)
            ax61.set_xticks(list([-10, 60, 130]))
            ax61.set_yticks(list([-10, 60, 130]))
            ax61.set_xticklabels(list([-10, 60, 130]), fontname=fontname, fontsize=fontsize)
            ax61.set_yticklabels(list([-10, 60, 130]), fontname=fontname, fontsize=fontsize)
            ax61.set_xlim([-10.0, 130.0])
            ax61.set_ylim([-10.0, 130.0])
            ax61.set_xlabel('true joint angle (deg)', va='center', ha='center', fontname=fontname, fontsize=fontsize)
            ax61.set_ylabel('learned joint angle (deg)', va='top', ha='center', fontname=fontname, fontsize=fontsize)
            for tick in ax61.get_xticklabels():
                tick.set_fontsize(fontsize)
                tick.set_fontname(fontname)
            for tick in ax61.get_yticklabels():
                tick.set_fontname(fontname)
                tick.set_fontsize(fontsize)
            ax61.yaxis.set_label_coords(x=left_margin_x/fig61_w, y=ax61_y+0.5*ax61_h, transform=fig61.transFigure)
    #         fig61.tight_layout()
            fig61.canvas.draw()


    #         Y = np.arange(np.size(joint_pos_error))
    #         fig7_w = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig7_h = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig7 = plt.figure(81, figsize=(fig7_w, fig7_h))
    #         fig7.canvas.manager.window.move(0, 0)
    #         fig7.clear()
    #         ax7 = fig7.add_subplot(111)
    #         ax7.clear()
    #         ax7.spines["top"].set_visible(False)
    #         ax7.spines["right"].set_visible(False)
    #         ax7.barh(Y, joint_pos_error[::-1], color=color_error, height=0.25)
    #         ax7.set_yticks(Y)
    #         ax7.set_yticklabels(bone_name_joint0[::-1], rotation='horizontal', horizontalalignment ='right', fontname=fontname, fontsize=fontsize)
    # #         ax7.set_xticks(np.arange(0.0, 4.5, 0.5, dtype=np.float64))
    # #         ax7.set_xticklabels(np.arange(0.0, 4.5, 0.5, dtype=np.float64), fontname=fontname, fontsize=fontsize)
    # #         ax7.set_xlim([0.0, 4.1])
    #         ax7.set_xlabel('joint postion error (cm)', fontname=fontname, fontsize=fontsize)
    # #         linewidth = 10
    # #         legend = mlines.Line2D([], [],
    # #                                      color=color_fit,
    # #                                      marker='',
    # #                                      linestyle='-',
    # #                                      linewidth=linewidth,
    # #                                      label='MRI')

    # #         h_legend = list([legend_0, legend_fit])
    # #         ax7.legend(handles=h_legend, loc='upper right', fontname=fontname, fontsize=fontsize, frameon=False)
    # #         fig7.tight_layout()
    #         fig7.canvas.draw()        

    #         dBin = 0.5
    #         bin_range = np.arange(0.0, 10.0+dBin, dBin, dtype=np.float64)
    #         fig8_w = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig8_h = np.round(mm_in_inch * 86.0*0.5, decimals=2)
    #         fig8 = plt.figure(8, figsize=(fig8_w, fig8_h))
    #         fig8.canvas.manager.window.move(0, 0)
    #         fig8.clear()
    #         ax8 = fig8.add_subplot(111)
    #         ax8.clear()
    #         ax8.spines["top"].set_visible(False)
    #         ax8.spines["right"].set_visible(False)

    #         hist = np.histogram(joint_pos_error, bins=len(bin_range)-1, range=[bin_range[0], bin_range[-1]], normed=None, weights=None, density=True)
    #         y = np.zeros(1+2*len(hist[0]), dtype=np.float64)
    #         y[1::2] = np.copy(hist[0])
    #         y[2::2] = np.copy(hist[0])
    #         x = np.zeros(1+2*(len(hist[1])-1), dtype=np.float64)
    #         x[0] = np.copy(hist[1][0])
    #         x[1::2] = np.copy(hist[1][:-1])
    #         x[2::2] = np.copy(hist[1][1:])
    #         ax8.plot(x, y, color=color_error, linestyle='-', marker='', linewidth=linewidth, alpha=1.0)
    # #         ax8.hist(joint_pos_error,
    # #                  bins=bin_range,
    # #                  range=None,
    # #                  density=True,
    # #                  weights=None,
    # #                  cumulative=False,
    # #                  bottom=None,
    # #                  histtype='bar',
    # #                  align='mid',
    # #                  orientation='vertical',
    # #                  rwidth=None,
    # #                  log=False,
    # #                  color=color_error,
    # #                  label=None,
    # #                  stacked=False,
    # #                  data=None,
    # #                  zorder=1,
    # #                  alpha=1.0,
    # #                  edgecolor='black',
    # #                  linewidth=1.0)
    # #         ax8.set_xticks(np.arange(0, 105, 10, dtype=np.int64).astype(np.float64)/10.0)
    # #         ax8.set_yticks(np.arange(0, 24, 4, dtype=np.int64).astype(np.float64)/10.0)
    # #         ax8.set_xticklabels(np.arange(0, 105, 10, dtype=np.int64).astype(np.float64)/10.0, fontname=fontname, fontsize=fontsize)
    # #         ax8.set_yticklabels(np.arange(0, 24, 4, dtype=np.int64).astype(np.float64)/10.0, fontname=fontname, fontsize=fontsize)
    #         ax8.set_xlim([0.0, 5.0])
    #         ax8.set_ylim([0.0, 2.0])
    #         ax8.set_xlabel('joint position error (cm)', fontname=fontname, fontsize=fontsize)
    #         ax8.set_ylabel('propability', fontname=fontname, fontsize=fontsize)
    # #         fig8.tight_layout()
    #         fig8.canvas.draw()   

    #         if save:
    # #             fig.savefig(folder+'/align_bar1.svg',
    # #                         bbox_inches="tight",
    # #                         dpi=300,
    # #                         transparent=True,
    # #                         format='svg')
    #             fig2.savefig(folder+'/align_bar_bone_length.svg',
    #                          bbox_inches="tight",
    #                          dpi=300,
    #                          transparent=True,
    #                          format='svg')
    #             fig3.savefig(folder+'/align_unity_bone_length.svg',
    #                          bbox_inches="tight",
    #                          dpi=300,
    #                          transparent=True,
    #                          format='svg') 
    #             fig5.savefig(folder+'/align_bar_angle.svg',
    #                          bbox_inches="tight",
    #                          dpi=300,
    #                          transparent=True,
    #                          format='svg')
    #             fig6.savefig(folder+'/align_unity_angle.svg',
    #                          bbox_inches="tight",
    #                          dpi=300,
    #                          transparent=True,
    #                          format='svg')
    #             fig7.savefig(folder+'/align_bar_joint_position_error.svg',
    #                      bbox_inches="tight",
    #                      dpi=300,
    #                      transparent=True,
    #                      format='svg')
    #             fig8.savefig(folder+'/align_hist_joint_position_error.svg',
    #                      bbox_inches="tight",
    #                      dpi=300,
    #                      transparent=True,
    #                      format='svg')

        unity_line = np.array([-10.0, 10.0], dtype=np.float64)
        ax31.plot(unity_line, unity_line,
                linestyle='-', marker='',
                color='darkgray', alpha=1.0, zorder=0, linewidth=linewidth_hist)
    #     ax31.legend(['animal #{:01d}'.format(i+1) for i in range(len(folder_list))], loc='upper left', fontsize=fontsize, frameon=False)
        unity_line = np.array([-360.0, 360.0], dtype=np.float64)
        ax61.plot(unity_line, unity_line,
                linestyle='-', marker='',
                color='darkgray', alpha=1.0, zorder=0, linewidth=linewidth_hist)
    #     ax61.legend(['animal #{:01d}'.format(i+1) for i in range(len(folder_list))], loc='upper left', fontsize=fontsize, frameon=False)

        dBin = 0.5
        bin_range = np.arange(0.0, 10.0+dBin, dBin, dtype=np.float64)
        joint_pos_error_all = np.array(joint_pos_error_all).ravel()
        hist = np.histogram(joint_pos_error_all, bins=len(bin_range)-1, range=[bin_range[0], bin_range[-1]], normed=None, weights=None, density=False)
        n = np.float64(len(joint_pos_error_all))
        y = np.zeros(1+2*len(hist[0]), dtype=np.float64)
        y[1::2] = np.copy(hist[0] / n)
        y[2::2] = np.copy(hist[0] / n)
        x = np.zeros(1+2*(len(hist[1])-1), dtype=np.float64)
        x[0] = np.copy(hist[1][0])
        x[1::2] = np.copy(hist[1][:-1])
        x[2::2] = np.copy(hist[1][1:])
#         if (mode == 'on'):
#             color_error = color_mode4
#         elif (mode == 'off'):
#             color_error = color_mode1
        ax81.plot(x, y, color=color_error, linestyle='-', marker='',
                  linewidth=linewidth_hist, alpha=1.0, zorder=2-i_mode)
    #     ax81.hist(joint_pos_error_all,
    #              bins=bin_range,
    #              range=None,
    #              density=True,
    #              weights=None,
    #              cumulative=False,
    #              bottom=None,
    #              histtype='bar',
    #              align='mid',
    #              orientation='vertical',
    #              rwidth=None,
    #              log=False,
    #              color=color_error,
    #              label=None,
    #              stacked=False,
    #              data=None,
    #              zorder=1,
    #              alpha=1.0,
    #              edgecolor='black',
    #              linewidth=1.0)
        ax81.set_xticks(list([0.0, 2.5, 5.0]))
        ax81.set_xticklabels(list([0, 2.5, 5.0]), fontname=fontname, fontsize=fontsize)
        ax81.set_yticks(list([0.0, 0.2, 0.4]))
        ax81.set_yticklabels(list([0, 0.2, 0.4]), fontname=fontname, fontsize=fontsize)
        ax81.set_xlim([0.0, 5.0])
        ax81.set_ylim([0.0, 0.4])
    #     ax81.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax81.yaxis.get_offset_text().set_fontsize(fontsize)
        ax81.set_xlabel('joint position error (cm)', va='center', ha='center', fontname=fontname, fontsize=fontsize)
        ax81.set_ylabel('propability', va='top', ha='center', fontname=fontname, fontsize=fontsize)
        for tick in ax81.get_xticklabels():
            tick.set_fontsize(fontsize)
            tick.set_fontname(fontname)
        for tick in ax81.get_yticklabels():
            tick.set_fontname(fontname)
            tick.set_fontsize(fontsize)
        ax81.yaxis.set_label_coords(x=left_margin_x/fig81_w, y=ax81_y+0.5*ax81_h, transform=fig81.transFigure)
    #     fig81.tight_layout()
        fig81.canvas.draw()   
        
        
#     from matplotlib.lines import Line2D
#     custom_lines = [Line2D([0], [0], label='fully constrained',
#                            color=color_mode4, linestyle='-', marker='', linewidth=5.0*linewidth),
#                     Line2D([0], [0], label='unconstrained',
#                            color=color_mode1, linestyle='-', marker='', linewidth=5.0*linewidth)]
#     h_legend = ax31.legend(handles=custom_lines, fontsize=fontsize, frameon=False, loc='upper right')
#     for text in h_legend.get_texts():
#         text.set_fontname(fontname)
#     h_legend = ax61.legend(handles=custom_lines, fontsize=fontsize, frameon=False, loc='upper right')
#     for text in h_legend.get_texts():
#         text.set_fontname(fontname)
#     h_legend = ax81.legend(handles=custom_lines, fontsize=fontsize, frameon=False, loc='upper right')
#     for text in h_legend.get_texts():
#         text.set_fontname(fontname)
    
    
    import scipy.stats

    print('joint position error (cm):')
    error = np.array(joint_pos_error_all).ravel()
    print('n:\t{:02d}'.format(len(error)))
    print('avg.:\t{:0.8f}'.format(np.mean(error)))
    print('sd.:\t{:0.8f}'.format(np.std(error)))
    print('median:\t{:0.8f}'.format(np.median(error)))
    print()
    #
    print('bone length error (cm):')
    error = np.array(bone_lengths_error_all).ravel()
    x = np.array(bone_length0_all, dtype=np.float64)
    y = np.array(bone_length1_all, dtype=np.float64)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    pcc = cov_xy / (std_x * std_y)
    print('n:\t{:02d}'.format(len(error)))
    print('avg.:\t{:0.8f}'.format(np.mean(error)))
    print('sd.:\t{:0.8f}'.format(np.std(error)))
    print('median:\t{:0.8f}'.format(np.median(error)))
    print('pcc:\t{:0.8f}'.format(pcc))
#     stat, p = scipy.stats.pearsonr(x, y)
    stat, p = scipy.stats.spearmanr(x, y) # does not assume data is normally distributed
    print('stat={:0.8f}, p={:0.2e}'.format(stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')
    print('measured bone length interval:')
    print('{:0.2f} cm to {:0.2f} cm'.format(np.min(np.array(bone_length0_all, dtype=np.float64)),
                                            np.max(np.array(bone_length0_all, dtype=np.float64))))
    print()
    #
    print('bone angle error (deg):')
    error = np.array(bone_angles_error_all).ravel()
    x = np.array(bone_angles0_all, dtype=np.float64)
    y = np.array(bone_angles1_all, dtype=np.float64)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    pcc = cov_xy / (std_x * std_y)
    print('n:\t{:02d}'.format(len(error)))
    print('avg.:\t{:0.8f}'.format(np.mean(error)))
    print('sd.:\t{:0.8f}'.format(np.std(error)))
    print('median:\t{:0.8f}'.format(np.median(error)))
    print('pcc:\t{:0.8f}'.format(pcc))
#     stat, p = scipy.stats.pearsonr(x, y)
    stat, p = scipy.stats.spearmanr(x, y) # does not assume data is normally distributed
    print('stat={:0.8f}, p={:0.2e}'.format(stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')
    print('measured bone angle interval:')
    print('{:0.2f} deg to {:0.2f} deg'.format(np.min(np.array(bone_angles0_all, dtype=np.float64)*180.0/np.pi),
                                              np.max(np.array(bone_angles0_all, dtype=np.float64)*180.0/np.pi)))
    print()    
    
    if save_all:
        fig31.savefig(folder_save+'/align_unity__bone_length.svg',
#                      bbox_inches="tight",
                 dpi=300,
                 transparent=True,
                 format='svg',
                  pad_inches=0)
        fig61.savefig(folder_save+'/align_unity__joint_angle.svg',
#                      bbox_inches="tight",
                 dpi=300,
                 transparent=True,
                 format='svg',
                  pad_inches=0) 
        fig81.savefig(folder_save+'/align_hist__joint_position_error.svg',
#                      bbox_inches="tight",
                 dpi=300,
                 transparent=True,
                 format='svg',
                 pad_inches=0)    

        
        
        
#     dBin = 10.0
#     bin_range = np.arange(0.0, 180.0+dBin, dBin, dtype=np.float64)
#     fig91_w = np.round(mm_in_inch * 86.0*0.5, decimals=2)
#     fig91_h = np.round(mm_in_inch * 86.0*0.5, decimals=2)
#     fig91 = plt.figure(91, figsize=(fig91_w, fig91_h))
#     fig91.canvas.manager.window.move(0, 0)
#     fig91.clear()
#     ax91 = fig91.add_subplot(111)
#     ax91.clear()
#     ax91.spines["top"].set_visible(False)
#     ax91.spines["right"].set_visible(False)
#     bone_angles_error_all = np.array(bone_angles_error_all).ravel()
#     hist = np.histogram(bone_angles_error_all, bins=len(bin_range)-1, range=[bin_range[0], bin_range[-1]], normed=None, weights=None, density=True)
#     y = np.zeros(1+2*len(hist[0]), dtype=np.float64)
#     y[1::2] = np.copy(hist[0])
#     y[2::2] = np.copy(hist[0])
#     x = np.zeros(1+2*(len(hist[1])-1), dtype=np.float64)
#     x[0] = np.copy(hist[1][0])
#     x[1::2] = np.copy(hist[1][:-1])
#     x[2::2] = np.copy(hist[1][1:])
#     ax91.plot(x, y, color=color_error, linestyle='-', marker='', linewidth=linewidth, alpha=1.0)
# #     ax81.hist(joint_pos_error_all,
# #              bins=bin_range,
# #              range=None,
# #              density=True,
# #              weights=None,
# #              cumulative=False,
# #              bottom=None,
# #              histtype='bar',
# #              align='mid',
# #              orientation='vertical',
# #              rwidth=None,
# #              log=False,
# #              color=color_error,
# #              label=None,
# #              stacked=False,
# #              data=None,
# #              zorder=1,
# #              alpha=1.0,
# #              edgecolor='black',
# #              linewidth=1.0)
#     ax91.set_xticks(list([0, 45, 90]))
# #     ax91.set_xticklabels(list([0, 45, 90]), fontname=fontname, fontsize=fontsize)
#     ax91.set_yticks(list([0.00, 0.025, 0.05]))
# #     ax91.set_yticklabels(list([0.00, 0.025, 0.05]), fontname=fontname, fontsize=fontsize)
#     ax91.set_xlim([0.0-dBin/4, 90.0+dBin/4])
#     ax91.set_ylim([0.0-0.001, 0.05+0.0025])
#     ax91.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#     ax91.set_xlabel('joint angle error (deg)', fontname=fontname, fontsize=fontsize)
#     ax91.set_ylabel('propability', fontname=fontname, fontsize=fontsize)
# #     fig91.tight_layout()
#     fig91.canvas.draw()   
#     if save_all:
#         for i_folder in range(len(folder_list)):
#             folder = folder_list[i_folder]
#             fig91.savefig(folder_save+'/align_hist__joint_angle_error_all.svg',
#                      bbox_inches="tight",
#                      dpi=300,
#                      transparent=True,
#                      format='svg')   

    if (verbose):
        plt.show()
