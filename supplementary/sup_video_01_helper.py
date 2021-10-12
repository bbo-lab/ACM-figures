import numpy as np 
import torch
from torch.autograd import Variable

def get_calibration(file_calibration, scale_factor,
                    use_torch):
    # load
    calibration = np.load(file_calibration, allow_pickle=True).item()
    # scaling (calibration board square size -> cm)
    calibration['tX1_fit'] *= scale_factor
    if use_torch:
        # calibration numpy
        A_entries = calibration['A_fit']
        k = calibration['k_fit']
        RX1 = calibration['RX1_fit']
        tX1 = calibration['tX1_fit']
        # calibration torch        
        calibration['A_fit'] = Variable(torch.from_numpy(A_entries), requires_grad=False)
        calibration['k_fit'] = Variable(torch.from_numpy(k), requires_grad=False)
        calibration['RX1_fit'] = Variable(torch.from_numpy(RX1), requires_grad=False)
        calibration['tX1_fit'] = Variable(torch.from_numpy(tX1), requires_grad=False)
    return calibration

def get_origin_coord(file_origin_coord, scale_factor):
    # load
    origin_coord = np.load(file_origin_coord, allow_pickle=True).item()
    # arena coordinate system
    origin = origin_coord['origin']
    coord = origin_coord['coord']
    # scaling (calibration board square size -> cm)
    origin = origin * scale_factor
    return origin, coord

def get_model3d(file_model,
                use_torch, use_torch_plot):
    # load
    model3d = np.load(file_model, allow_pickle=True).item()
    if use_torch:
        # 3d model numpy
        surf_verts = model3d['surface_vertices']
        surf_tris = model3d['surface_triangles']
        skel_verts = model3d['skeleton_vertices']
        skel_edges = model3d['skeleton_edges']
        skel_coords = model3d['skeleton_coords']
        skel_coords_index = model3d['skeleton_coords_index']
        surf_verts_weights = model3d['surface_vertices_weights']
        skel_verts_links = model3d['skeleton_vertices_links']
        joint_marker_vec = model3d['joint_marker_vectors']
        joint_order = model3d['joint_order']
        joint_marker_index = model3d['joint_marker_index']
        joint_marker_order = model3d['joint_marker_order'] 
        # model torch
        model3d['skeleton_edges'] = Variable(torch.from_numpy(skel_edges), requires_grad=False)
        model3d['skeleton_vertices_links'] = Variable(torch.from_numpy(skel_verts_links.astype(np.int64)).type_as(torch.ByteTensor()),
                                                      requires_grad=False)
        model3d['joint_marker_index'] = Variable(torch.from_numpy(joint_marker_index), requires_grad=False)
        model3d['skeleton_coords_index'] = Variable(torch.from_numpy(skel_coords_index), requires_grad=False)
        model3d['skeleton_vertices'] = Variable(torch.from_numpy(skel_verts), requires_grad=False)
        model3d['skeleton_coords'] = Variable(torch.from_numpy(skel_coords), requires_grad=False)
        model3d['joint_marker_vectors'] = Variable(torch.from_numpy(joint_marker_vec), requires_grad=False)
        model3d['surf_vertices_weights'] = Variable(torch.from_numpy(surf_verts_weights), requires_grad=False)
        # include this within model3d to have as little function calls as possible within the objective function
        model3d['skeleton_coords_new'] = Variable(torch.from_numpy(skel_coords), requires_grad=False)
        model3d['skeleton_vertices_new'] = Variable(torch.from_numpy(skel_verts), requires_grad=False)
        # only needed for plotting (i.e. do not use if you run into memory limitations)
        if use_torch_plot:
            model3d['surface_triangles'] = surf_tris
            model3d['surface_vertices'] = Variable(torch.from_numpy(surf_verts), requires_grad=False)
            model3d['surface_vertices_weights'] = Variable(torch.from_numpy(surf_verts_weights), requires_grad=False)
    return model3d

def get_numbers(calibration=None,
                model3d=None):
    numbers = dict()
    # calibration
    if (calibration != None):
        nCameras = calibration['nCameras']
        numbers['nCameras'] = nCameras
    # model3d
    if (model3d != None):
        nBones = np.size(model3d['skeleton_edges'], 0)
        nJoints = np.size(model3d['skeleton_vertices'], 0)
        nMarkers = np.size(model3d['joint_marker_vectors'], 0)
        numbers['nBones'] = nBones
        numbers['nJoints'] = nJoints
        numbers['nMarkers'] = nMarkers
    return numbers

def get_free_parameters(model3d):
    nBones = np.size(model3d['skeleton_edges'], 0)
    free_para = np.ones((1 + nBones, 3), dtype=bool)
    joint_order = model3d['joint_order']
    for i_joint in range(nBones):
        joint_name_split = joint_order[i_joint].split('_')
        if ((joint_name_split[1] == 'spine') and (joint_name_split[2] == '002')):
            # +1 because first three parameters are translation
            free_para[1 + (i_joint + 0), :] = False # pelvis left
            free_para[1 + (i_joint + 1), :] = False # pelvis left
        if ((joint_name_split[1] == 'spine') and (joint_name_split[2] == '005')):
            # +1 because first three parameters are translation
            free_para[1 + (i_joint + 1), :] = False # collarbone left
            free_para[1 + (i_joint + 2), :] = False # collarbone right
    free_para = free_para.ravel()
    return free_para

def get_arguments(file_calibration, file_model, file_labelsDLC,
                  scale_factor, pcutoff=0.9,
                  use_torch=True, use_torch_plot=False):
    # calibration
    calibration = get_calibration(file_calibration, scale_factor,
                                  use_torch)
    numbers = get_numbers(calibration=calibration)
    # model3d
    model3d = get_model3d(file_model,
                          use_torch, use_torch_plot)
    numbers = get_numbers(calibration=calibration,
                          model3d=model3d)
    # labels
    numbers['nFrames'] = 0 # legacy
    numbers['nLabels'] = 0 # legacy
    # add batch size to numbers
    numbers['nBatch'] = 1
    # numbers
    nCameras = numbers['nCameras']
    nBones = numbers['nBones']
    nFrames = numbers['nFrames']
    nMarkers = numbers['nMarkers']
    # free parameters
    free_para = get_free_parameters(model3d)
    free_para_batch = np.tile(free_para, 1)
    #
    args = dict()
    # numbers
    args['numbers'] = numbers
    # calibration
    args['calibration'] = calibration
    # model3d
    args['model'] = model3d
    # free parameters
    args['free_para'] = free_para
    args['free_para_batch'] = free_para_batch
    return args

def initialize_args__x(x, args, use_torch):
    free_para_batch = args['free_para_batch']
    x_free = x[free_para_batch] 
    if (use_torch):
        args['x_torch'] = Variable(torch.from_numpy(np.copy(x)),
                                   requires_grad=True)
        args['x_free_torch'] = Variable(torch.from_numpy(np.copy(x_free)),
                                        requires_grad=True)
    return

def update_args__x(x, args, use_torch):
    free_para_batch = args['free_para_batch']
    x_free = x[free_para_batch]
    if (use_torch):
        args['x_torch'].data.copy_(torch.from_numpy(np.copy(x)))
        args['x_free_torch'].data.copy_(torch.from_numpy(np.copy(x_free)))
    return

def rodrigues2rotMat(r):    
    theta = torch.norm(r, 2)
    u = r / theta
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    one_MINUS_cos_theta = 1 - cos_theta
    u_TIMES_sin_theta = u * sin_theta
    u_outer = torch.ger(u, u)

    diag0 = torch.diagonal(u_outer) * one_MINUS_cos_theta + cos_theta

    upper = torch.stack([u_outer[1, 2], u_outer[0, 2], u_outer[0, 1]], 0) * one_MINUS_cos_theta

    diag1 = upper - u_TIMES_sin_theta
    diag2 = upper + u_TIMES_sin_theta

    rotMat = torch.stack([torch.stack([diag0[0], diag1[2], diag2[1]], 0),
                          torch.stack([diag2[2], diag0[1], diag1[0]], 0),
                          torch.stack([diag1[1], diag2[0], diag0[2]], 0)], 0)
    return rotMat

def adjust_all(model_torch,
               model_t_torch, model_r_torch):
    skel_coords = model_torch['skeleton_coords']
    skel_verts = model_torch['skeleton_vertices']
    skel_edges = model_torch['skeleton_edges']
    skel_verts_links = model_torch['skeleton_vertices_links']
    joint_marker_vec = model_torch['joint_marker_vectors']
    joint_marker_index = model_torch['joint_marker_index']

    skel_coords_new = model_torch['skeleton_coords_new']
    skel_coords_index = model_torch['skeleton_coords_index']
    skel_verts_new = model_torch['skeleton_vertices_new']
    joint_marker_vec_new = joint_marker_vec.clone()

    nBones = np.size(skel_edges, 0)
    surf_verts = model_torch['surface_vertices']
    surf_verts_weights = model_torch['surface_vertices_weights']
    surf_verts_new = torch.zeros_like(surf_verts)
    
    # inital translation
    # the position of the tail becomes equal to the translation vector
    skel_verts_new = torch.cat([torch.reshape(model_t_torch, (1, 3)),
                                skel_verts_new[1:]], 0)
    # the for loop goes through the skeleton in a directed order starting from the tail
    for i_bone in range(nBones):
        # rotates all skeleton coordinate systems that are affected by the rotation of bone i_bone
        R = rodrigues2rotMat(model_r_torch[i_bone])
        mask = (skel_verts_links[i_bone] == True)
        skel_coords_changed = torch.einsum('ij,njk->nik',
                                           (R, skel_coords_new[mask]))
        skel_coords_unchanged = skel_coords_new[~mask] 
        skel_coords_new = torch.cat([skel_coords_changed,
                                     skel_coords_unchanged], 0)
        skel_coords_changed_index = skel_coords_index[mask]
        skel_coords_unchanged_index = skel_coords_index[~mask]
        skel_coords_index_new = torch.cat([skel_coords_changed_index,
                                           skel_coords_unchanged_index], 0)
        _, skel_coords_index_new = torch.sort(skel_coords_index_new)
        skel_coords_new = skel_coords_new[skel_coords_index_new]
        
        # rotation matrix
        M = torch.mm(skel_coords_new[i_bone], skel_coords[i_bone].transpose(1, 0))
        # indices
        index_bone_start = skel_edges[i_bone, 0]
        index_bone_end = skel_edges[i_bone, 1]
        # skeleton
        bone_pos_norm = skel_verts[index_bone_end] - skel_verts[index_bone_start]
        skel_verts_new[index_bone_end] = torch.mv(M, bone_pos_norm) + skel_verts_new[index_bone_start]
        
        # joint-marker vector
        mask = (joint_marker_index == index_bone_end)
        if torch.any(mask):
            joint_marker_vec_new[mask] = torch.einsum('ij,nj->ni',
                                                      (M, joint_marker_vec_new[mask]))
    # fix for first marker at the tip of the tail
    M = torch.mm(skel_coords_new[0], skel_coords[0].transpose(1, 0))
    joint_marker_vec_new[[0]] = torch.einsum('ij,nj->ni',
                                             (M, joint_marker_vec_new[[0]]))
    # add joint position to get final marker positon
    marker_pos_new = skel_verts_new[joint_marker_index] + joint_marker_vec_new
    
    # surface
    for i_bone in range(nBones):
        index_bone_start = skel_edges[i_bone, 0]
        index_bone_end = skel_edges[i_bone, 1]
        M = torch.mm(skel_coords_new[i_bone], skel_coords[i_bone].transpose(1, 0))
        
        mask = (surf_verts_weights[:, index_bone_end] != 0)
        skin_pos_norm = surf_verts[mask] - skel_verts[index_bone_start]
        skin_pos_new = torch.einsum('ij,nj->ni', (M, skin_pos_norm)) + skel_verts_new[index_bone_start]
        w = surf_verts_weights[mask, index_bone_end].repeat(3, 1).transpose(1, 0)
        surf_verts_new[mask] = torch.addcmul(surf_verts_new[mask], 1, w, skin_pos_new)
    
    return skel_coords_new, skel_verts_new, surf_verts_new, marker_pos_new