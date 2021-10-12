#!/usr/bin/env python3

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import animation
from mayavi import mlab
import numpy as np
import os
import sys
import torch

import sup_video_01_helper as helper

sys.path.append(os.path.abspath('../ACM'))
import data

save = True
verbose = False

folder_save = os.path.abspath('videos')

start_frame_x = 3200
end_frame_x = 13200
dFrame_x = 1
ini_frame = 4150
scale_factor = 3.7
pcutoff = 0.9
folder_sup = data.path + '/supplementary/schematic_mathematical_model'
file_pose = folder_sup + '/ref.npy'
file_origin_coord = folder_sup + '/origin_coord.npy'
file_calibration = folder_sup + '/multicalibration.npy'
file_model = folder_sup + '/model_3d.npy'
file_labelsDLC = folder_sup + '/dlc_labels.npy'

def create_figure(figIndex):
    fig = plt.figure(figIndex, figsize=(8, 8))
    fig.clear()
    fig.set_facecolor('white')
    fig.set_dpi(300)
    mng = plt.get_current_fig_manager()
    mng.window.setGeometry(0, 0, 1, 1)
    if verbose:
        mng.window.showMaximized()
        plt.pause(1e-16)
        plt.show(block=False)
    return fig

def create_axes(fig, nAxis):
    ax = list()
    if (nAxis > 1):
        nRows = 2 
    else:
        nRows = 1
    nCols = np.int64(np.ceil(nAxis / nRows))
    for i_ax in range(nAxis):
        if (i_ax < nAxis / 2):
            i_row = 0
        else:
            i_row = nRows - 1
        gs = gridspec.GridSpec(nRows, nCols)
        ss = gs.new_subplotspec(loc=(i_row, i_ax % nCols),
                                rowspan=1, colspan=1)
        ax.append(fig.add_subplot(ss, frameon=True))
        ax[-1].set_facecolor('white')     
    if verbose:
        plt.pause(1e-16)
        plt.show(block=False)
    return ax

def create_figure3d(figIndex, bgColor=40/255):
    # size depends on the screen (-50 because the figure has a task bar I guess)
    if not(verbose):
        mlab.options.offscreen = True
    fig3d = mlab.figure(figIndex,
                        size=(990-50, 990),
                        fgcolor=(1, 1, 1),
                        bgcolor=(bgColor, bgColor, bgColor))
    mlab.clf()
    return fig3d

def take_screenshot(fig3d):
    fig3d.scene.anti_aliasing_frames = 20
    # fix to avoid error when using mlab.screenshot
    # see: https://github.com/enthought/mayavi/issues/702
    fig3d.scene._lift()
    screenshot = mlab.screenshot(figure=fig3d,
                                 mode='rgba',
                                 antialiased=True)
    return screenshot

def plot_model_3d__ini(fig3d,
                       surface_vertices, skeleton_vertices,
                       grad_surf, grad_joints,
                       args):   
    model = args['model']
    skeleton_edges = model['skeleton_edges']
    surface_triangles = model['surface_triangles']
    # bones
    h_bones_3d = list()
    for edge in skeleton_edges:
        i_joint = skeleton_vertices[edge[0]]
        joint_to = skeleton_vertices[edge[1]]
        h_bone = mlab.plot3d(np.array([i_joint[0], joint_to[0]]),
                             np.array([i_joint[1], joint_to[1]]),
                             np.array([i_joint[2], joint_to[2]]),
                             color=(1, 1, 1),
                             figure=fig3d,
                             line_width=2,
                             opacity=1.0,
                             transparent=False,
                             tube_radius=0.1,
                             tube_sides=16)
        h_bone.visible = bool(grad_joints[edge[1]])
        h_bones_3d.append(h_bone)
    # joints
    h_joints_3d = mlab.points3d(skeleton_vertices[:, 0],
                                skeleton_vertices[:, 1],
                                skeleton_vertices[:, 2],
                                grad_joints,
                                color=(1, 1, 1),
                                figure=fig3d,
                                opacity=1.0,
                                scale_factor=1/3,
                                transparent=True)
        
    # surface
    h_surface_3d = mlab.triangular_mesh(surface_vertices[:, 0],
                                        surface_vertices[:, 1],
                                        surface_vertices[:, 2],
                                        surface_triangles,
                                        colormap='viridis',
                                        figure=fig3d,
                                        opacity=1.0,
                                        scalars=grad_surf,
                                        transparent=True,
                                        vmin=0,
                                        vmax=1)
    # to modify the alpha channel of the colormap
    lut = h_surface_3d.module_manager.scalar_lut_manager.lut.table.to_array()
    lut[:, -1] = np.linspace(0, 127, 256) # transparent
    h_surface_3d.module_manager.scalar_lut_manager.lut.table = lut
    return h_surface_3d, h_joints_3d, h_bones_3d

def plot_model_3d__update(surface_vertices, skeleton_vertices,
                          grad_surf, grad_joints,
                          h_surface_3d, h_joints_3d, h_bones_3d,
                          args):
    model = args['model']
    skeleton_edges = model['skeleton_edges']
    # numbers
    numbers = args['numbers']
    nBones = numbers['nBones']
    # bones
    for index_bone in range(nBones):
        edge = skeleton_edges[index_bone]
        i_joint = skeleton_vertices[edge[0]]
        joint_to = skeleton_vertices[edge[1]]
        # had to manually fix a bug for this in the mayavi code base
        # see:
        # https://github.com/enthought/mayavi/issues/696
        # https://github.com/enthought/mayavi/pull/699/commits/88c7f9ffc7da8b0b5e591ba73c0b039ce62e960c
        h_bone = h_bones_3d[index_bone]
        h_bone.mlab_source.set(x=np.array([i_joint[0], joint_to[0]]),
                               y=np.array([i_joint[1], joint_to[1]]),
                               z=np.array([i_joint[2], joint_to[2]]))
        h_bone.visible = bool(grad_joints[edge[1]])
    # joints
    h_joints_3d.mlab_source.set(x=skeleton_vertices[:, 0],
                                y=skeleton_vertices[:, 1],
                                z=skeleton_vertices[:, 2],
                                scalars=grad_joints)
    # surface
    h_surface_3d.mlab_source.set(x=surface_vertices[:, 0],
                                 y=surface_vertices[:, 1],
                                 z=surface_vertices[:, 2],
                                 scalars=grad_surf)
    return

def adjust_model(x_torch, args):
    # model3d
    model = args['model']
    # numbers
    numbers = args['numbers']
    nBones = numbers['nBones']
    # project model
    model_t_frame = torch.reshape(x_torch.data[:3], (3,))
    model_r_frame = torch.reshape(x_torch.data[3:], (nBones, 3))
    
    skel_coords_new, skel_verts_new, surf_verts_new, marker_pos_new = \
        helper.adjust_all(model, model_t_frame, model_r_frame)
    return skel_coords_new, skel_verts_new, surf_verts_new, marker_pos_new

def draw_camera(fig, origin, coord):
    line_width = 5
    tube_radius = 1
    tube_sides = 16
    fac = 10
    # x
    h_x = mlab.plot3d(np.array([origin[0], origin[0] + fac * coord[0, 0]]),
                np.array([origin[1], origin[1] + fac * coord[1, 0]]),
                np.array([origin[2], origin[2] + fac * coord[2, 0]]),
                color=(1, 0, 0),
                figure=fig,
                opacity=1.0,
                transparent=False,
                line_width=line_width,
                tube_radius=tube_radius,
                tube_sides=tube_sides)
    # y
    h_y = mlab.plot3d(np.array([origin[0], origin[0] + fac * coord[0, 1]]),
                np.array([origin[1], origin[1] + fac * coord[1, 1]]),
                np.array([origin[2], origin[2] + fac * coord[2, 1]]),
                color=(0, 1, 0),
                figure=fig,
                opacity=1.0,
                transparent=False,
                line_width=line_width,
                tube_radius=tube_radius,
                tube_sides=tube_sides)
    # z
    h_z = mlab.plot3d(np.array([origin[0], origin[0] + fac * coord[0, 2]]),
                np.array([origin[1], origin[1] + fac * coord[1, 2]]),
                np.array([origin[2], origin[2] + fac * coord[2, 2]]),
                color=(0, 0, 1),
                figure=fig,
                opacity=1.0,
                transparent=False,
                line_width=line_width,
                tube_radius=tube_radius,
                tube_sides=tube_sides)
    h_coord = list([h_x, h_y, h_z])
    return h_coord

def draw_arena(fig, args):
    line_width = 0.15
    tube_radius = 0.15
    tube_sides = 16
    
    # numbers
    nCameras = args['numbers']['nCameras']
    
    # calibration
    calibration = args['calibration']
    A_entries = calibration['A_fit'].numpy()
    A = np.zeros((nCameras, 3, 3), dtype=np.float64)
    for i_cam in range(nCameras):
        A[i_cam, 0, 0] = A_entries[i_cam, 0]
        A[i_cam, 0, 2] = A_entries[i_cam, 1]
        A[i_cam, 1, 1] = A_entries[i_cam, 2]
        A[i_cam, 1, 2] = A_entries[i_cam, 3]
        A[i_cam, 2, 2] = 1.0
    k = calibration['k_fit'].numpy()
    RX1 = calibration['RX1_fit'].numpy()
    tX1 = calibration['tX1_fit'].numpy()
    indexRefCam = calibration['indexRefCam']
    
    origin, coord = helper.get_origin_coord(file_origin_coord, scale_factor)
    arenaCorners = np.array([[ 14.12450104,  11.54737429,   0.06192083],
                             [ 14.95736319,  -9.59530272,   0.14634636],
                             [-13.0273297 , -10.63827857,   0.06758548],
                             [-13.74729673,  10.54606616,  -0.11839662]]) * scale_factor # hard coded
    arenaCorners_proj = np.zeros((4, 3), dtype=np.float64)
    for i_corner in range(4):
        arenaCorners_proj[i_corner] = origin + np.dot(coord, arenaCorners[i_corner])

    for i_corner in range(nCameras):
        vec = arenaCorners_proj[(i_corner + 1) % nCameras] - arenaCorners_proj[i_corner]
        mlab.plot3d(np.array([arenaCorners_proj[i_corner, 0],
                                 arenaCorners_proj[i_corner, 0] + vec[0]]),
                    np.array([arenaCorners_proj[i_corner, 1],
                                 arenaCorners_proj[i_corner, 1] + vec[1]]),
                    np.array([arenaCorners_proj[i_corner, 2],
                                 arenaCorners_proj[i_corner, 2] + vec[2]]),
                    color=(1, 1, 1),
                    figure=fig,
                    opacity=0.5,
                    transparent=True,
                    line_width=line_width,
                    tube_radius=tube_radius,
                    tube_sides=tube_sides)
    return

def draw_pose_adjustment(i_frame):
    args = helper.get_arguments(file_calibration, file_model, file_labelsDLC,
                                scale_factor, pcutoff,
                                True, True)
    # get numbers from args
    numbers = args['numbers']
    nCameras = numbers['nCameras']
    nBones = numbers['nBones']
    nMarkers = numbers['nMarkers']
    #
    model = args['model']
    joint_marker_index = model['joint_marker_index']
    joint_marker_order = model['joint_marker_order']
    grad_surf = np.ones(np.size(model['surface_vertices'], 0))
    grad_joints = np.ones(np.size(model['skeleton_vertices'], 0))
    
    # calibration
    calibration = args['calibration']
    A_torch = calibration['A_fit']
    k_torch = calibration['k_fit']
    RX1_torch = calibration['RX1_fit']
    tX1_torch = calibration['tX1_fit']
    indexRefCam = calibration['indexRefCam']
    
    # load x
    x_all = np.load(file_pose, allow_pickle=True)
    frame_list = np.arange(start_frame_x,
                           end_frame_x + dFrame_x,
                           dFrame_x)
    
    # plotting
    mlab.options.offscreen = True
    fig3d = create_figure3d(1, bgColor=40/255)
    fig = create_figure(1)
    ax = create_axes(fig, 1)
    
    # initial model adjustment
    index = np.where(i_frame == frame_list)[0][0]
    x = np.copy(x_all[index])
    helper.initialize_args__x(x, args, True)
    x_torch = args['x_torch']
    
    coords_ini, skeleton_vertices_ini, surface_vertices_ini, joint_marker_pos_ini = \
        adjust_model(x_torch, args)
    t_correction = torch.median(skeleton_vertices_ini, dim=0)[0]
    
    h_surface_3d, h_joints_3d, h_bones_3d = \
        plot_model_3d__ini(fig3d,
                                  surface_vertices_ini, skeleton_vertices_ini,
                                  grad_surf, grad_joints,
                                  args)
    # joint-marker vectors
    h_joint_marker_vec = list()
    for marker_index in range(nMarkers):
        joint_index = joint_marker_index[marker_index]
        vec_start = skeleton_vertices_ini[joint_index]
        vec_end = joint_marker_pos_ini[marker_index]
        h = mlab.plot3d(np.array([vec_start[0], vec_end[0]]),
                        np.array([vec_start[1], vec_end[1]]),
                        np.array([vec_start[2], vec_end[2]]),
                        color=(0, 0, 0),
                        figure=fig3d,
                        opacity=0.75,
                        transparent=True,
                        line_width=2,
                        tube_radius=0.05,
                        tube_sides=16) 
        h_joint_marker_vec.append(h)
    # parameter vector
    cmap = plt.get_cmap('viridis')
    vec_start = np.zeros(3, dtype=np.float64)
    vec_end = np.zeros(3, dtype=np.float64)
    h_para = mlab.quiver3d(vec_start[0],
                           vec_start[1],
                           vec_start[2],
                           vec_end[0],
                           vec_end[1],
                           vec_end[2],
                           color=cmap(0.5)[:3],
                           figure=fig3d,
                           opacity=1.0,
                           reset_zoom=True,
                           resolution=64,
                           transparent=False,
                           line_width=3,
                           scale_mode='vector',
                           scale_factor=1)
    h_para.glyph.glyph.clamping = False # for scaling

    # draw cameras
    h_proj = list()
    for i_cam in range(nCameras):
        coord = RX1_torch[i_cam].transpose(1, 0)
        origin = tX1_torch[indexRefCam] - torch.einsum('ij,j->i', (RX1_torch[i_cam].transpose(1, 0), tX1_torch[i_cam]))# + tX1_torch[indexRefCam]
        draw_camera(fig3d, origin, coord)
        
        vec_start = np.zeros(3, dtype=np.float64)
        vec_end = np.zeros(3, dtype=np.float64)
        h_proj_single = mlab.plot3d(np.array([vec_start[0], vec_end[0]]),
                                    np.array([vec_start[1], vec_end[1]]),
                                    np.array([vec_start[2], vec_end[2]]),
                                    color=cmap(2/3)[:3],
                                    figure=fig3d,
                                    opacity=0.75,
                                    transparent=True,
                                    line_width=2.5,
                                    tube_radius=0.25,
                                    tube_sides=16)
        h_proj.append(h_proj_single)
    # draw arena
    draw_arena(fig3d, args)

    # view
    az0 = 0
    elev0 = 180
    dist0 = 120
    roll0 = 180
    mlab.view(figure=fig3d,
              azimuth=az0,
              elevation=elev0,
              distance=dist0,
              focalpoint=t_correction.numpy())
    mlab.roll(figure=fig3d,
              roll=roll0)
    
    # turn distortion on/off
    mlab.gcf().scene.parallel_projection = False
    # take screenshot
    screenshot3d = take_screenshot(fig3d)
    
    # set 3d figure background color    
    bg3d_img = np.zeros_like(screenshot3d, dtype=np.float32)
    bg3d_img[:, :, :3] = 40 / 255
    bg3d_img[:, :, 3] = 1.0
    im2d_bg = ax[0].imshow(bg3d_img,
                           aspect='equal',
                           zorder=0)
    # plot within matplotlib figure
    im2d = ax[0].imshow(screenshot3d,
                        aspect='equal',
                        zorder=1)
    
    fig.subplots_adjust(bottom=0,
                        top=1,
                        left=0,
                        right=1)
    ax[0].set_axis_off()
    fig.canvas.draw()
    if verbose:
        plt.pause(0.1)
        plt.show(block=False)

    # PARAMETERS
    x_ini = np.copy(x)
    x_rand = np.random.randn((1+nBones)*3) * np.finfo(np.float64).eps
    free_para = args['free_para']
    free_para_mask = np.all(np.reshape(free_para, (nBones+1, 3)), 1)[1:]
    
    bones_index = np.arange(nBones, dtype=np.int64)
    bones_index_free = bones_index[free_para_mask]
    nBones_free = np.size(bones_index_free)

    para_vecs = np.reshape(x_ini[free_para][3:], (nBones_free, 3))
    para_vec_norm = np.sqrt(np.sum(para_vecs**2, 1))
    dAng_use = np.min(para_vec_norm) / 2
    frac_list = np.array([], dtype=np.float64)
    para_list = np.array([], dtype=np.int64)
    # rotations
    nFrac = np.full(nBones_free + 1, np.nan, dtype=np.int64)    
    index_frac = 1
    for i_para in range(1, nBones+1, 1):
        if (i_para in bones_index_free + 1):
            para_vec = x_ini[3*i_para:3*(i_para+1)]
            para_vec_norm = np.sqrt(np.sum(para_vec**2))
            nFrac[index_frac] = np.int64(np.round(para_vec_norm / dAng_use))
            frac = np.linspace(1/nFrac[index_frac], 1, nFrac[index_frac])
            frac_list = np.concatenate([frac_list, frac], 0)
            para_list = np.concatenate([para_list, np.repeat(i_para, nFrac[index_frac])], 0)
            index_frac = index_frac +  1
    # translation
    index_frac = 0
    i_para = 0
    nFrac[index_frac] = np.int64(np.max(nFrac[~np.isnan(nFrac)]))
    frac = np.linspace(0.45, 1, nFrac[index_frac]) # to not start from actual camera position
    frac_list = np.concatenate([frac, frac_list], 0)
    para_list = np.concatenate([np.repeat(i_para, nFrac[index_frac]), para_list], 0)
    nFrames_para = np.size(frac_list, 0)
        
    # ZOOM
    marker_name_target = 'marker_spine_003_start'
    index_target_marker = model['joint_marker_order'].index(marker_name_target)
    nFrames_zoom = np.int64(np.max(nFrac[~np.isnan(nFrac)]) * 1.5)
    frac_zoom = np.linspace(1/nFrames_zoom, 1.0, nFrames_zoom)
    nFrames_wait = np.int64(np.max(nFrac[~np.isnan(nFrac)]) * 1.5)
    frac_zoom = np.concatenate([frac_zoom, np.ones(nFrames_wait, dtype=np.float64)], 0)

    # VIDEO MAKING
    fps = 30
    nFrames_video = nFrames_para + nFrames_zoom + nFrames_wait
    duration = nFrames_video / fps
    
    print('duration:\t', duration)
    print('nFrames:\t', nFrames_video)
    print('fps:\t\t', fps)
        
    def update_model3d(t):
        print('{:03d}/{:03d}'.format(int(t), int(nFrames_video)), end='\r')
        if (t < nFrames_para):
            i_para = para_list[t]
            frac = frac_list[t]

            x_use = np.zeros_like(x_ini, dtype=np.float64)
            x_use[:3*(i_para+1)] = np.copy(x_ini[:3*(i_para+1)])
            x_use[3*(i_para+1):] = np.copy(x_rand[3*(i_para+1):])

            para_vec = frac * x_ini[3*i_para:3*(i_para+1)]
            x_use[3*i_para:3*(i_para+1)] = np.copy(para_vec)

            helper.update_args__x(x_use, args, True)
            x_torch = args['x_torch']

            coords, skeleton_vertices, surface_vertices, joint_marker_pos = \
                adjust_model(x_torch, args)
            surface_vertices = surface_vertices.numpy()
            skeleton_vertices = skeleton_vertices.numpy()

            plot_model_3d__update(surface_vertices, skeleton_vertices,
                                         grad_surf, grad_joints,
                                         h_surface_3d, h_joints_3d, h_bones_3d,
                                         args)            
            for marker_index in range(nMarkers):
                joint_index = joint_marker_index[marker_index]
                vec_start = skeleton_vertices[joint_index]
                vec_end = joint_marker_pos[marker_index]
                h_vec = h_joint_marker_vec[marker_index]
                h_vec.mlab_source.set(x=np.array([vec_start[0], vec_end[0]]),
                                      y=np.array([vec_start[1], vec_end[1]]),
                                      z=np.array([vec_start[2], vec_end[2]]))
            if (i_para != 0.0):
                i_bone = i_para - 1
                joint_index = model['skeleton_edges'][i_bone, 0]
                vec_start = skeleton_vertices[joint_index]
                vec = para_vec * 5
            else:
                joint_index = model['skeleton_edges'][0, 0]
                vec0 = skeleton_vertices[joint_index]
                vec_start = tX1_torch[indexRefCam].numpy() + vec0 * frac / 2
                vec = vec0 - vec_start
            h_para.mlab_source.set(x=vec_start[0], y=vec_start[1], z=vec_start[2],
                                   u=vec[0], v=vec[1], w=vec[2])
        else:
            h_para.mlab_source.set(x=0.0, y=0.0, z=0.0,
                                   u=0.0, v=0.0, w=0.0)
            frac = frac_zoom[t - nFrames_para]
            
            dist1 = 450
            mlab.view(figure=fig3d,
                      azimuth=az0,
                      elevation=elev0,
                      distance=dist0 + (dist1 - dist0) * frac,
                      focalpoint=t_correction.numpy())
            mlab.roll(figure=fig3d,
                      roll=roll0)
                
            vec_start = joint_marker_pos_ini[index_target_marker]
            vec_start_use = vec_start.numpy()
            for i_cam in range(nCameras):
                origin = tX1_torch[indexRefCam] - torch.einsum('ij,j->i', (RX1_torch[i_cam].transpose(1, 0), tX1_torch[i_cam]))
                vec_end =  vec_start + (origin - vec_start) * frac
                vec_end_use = vec_end.numpy()
                h_proj[i_cam].mlab_source.set(x=np.array([vec_start_use[0], vec_end_use[0]]),
                                              y=np.array([vec_start_use[1], vec_end_use[1]]),
                                              z=np.array([vec_start_use[2], vec_end_use[2]]))
                
        screenshot3d = take_screenshot(fig3d)
        # plot within matplotlib figure
        im2d.set_data(screenshot3d)
        fig.canvas.draw()
        return ax
    #
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=-1)
    #
    ani = animation.FuncAnimation(fig, update_model3d,
                                  frames=nFrames_video,
                                  interval=1, blit=False)
    if save:
        ani.save(folder_save + '/math_model.mp4', writer=writer)
    return 

if __name__ == '__main__':
    draw_pose_adjustment(ini_frame)
