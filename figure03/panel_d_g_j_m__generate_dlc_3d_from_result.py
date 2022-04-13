#!/usr/bin/env python3
import numpy as np
import os
import sys

resultfolder = sys.argv[1]
print(os.path.abspath(resultfolder+'/configuration'))
sys.path.append(os.path.abspath(resultfolder+'/configuration'))
import configuration as cfg

sys.path.append(os.path.abspath('../ACM'))
import data
import helper
import interp_3d
nFrames = cfg.index_frame_end-cfg.index_frame_start
file_dlc = resultfolder+'/configuration/file_labelsDLC.npy'
labels_dlc_dict = np.load(file_dlc, allow_pickle=True).item()
frame_list = labels_dlc_dict['frame_list']
labels_dlc = labels_dlc_dict['labels_all']
labels_dlc_shape = np.shape(labels_dlc)

nCameras = labels_dlc_shape[1]
nMarkers = labels_dlc_shape[2]
#
file_origin_coord = resultfolder+'/configuration/file_origin_coord.npy'
file_calibration = resultfolder+'/configuration/file_calibration.npy'
calibration = helper.get_calibration(file_origin_coord, file_calibration, cfg.scale_factor)

A_entries = calibration['A_fit'].cpu().numpy()
A = np.zeros((nCameras, 3, 3), dtype=np.float64)
A[:, 0, 0] = A_entries[:, 0]
A[:, 0, 2] = A_entries[:, 1]
A[:, 1, 1] = A_entries[:, 2]
A[:, 1, 2] = A_entries[:, 3]
A[:, 2, 2] = 1.0
k = calibration['k_fit'].cpu().numpy()
rX1 = calibration['rX1_fit'].cpu().numpy()
tX1 = calibration['tX1_fit'].cpu().numpy()

labels_dlc_3d = {
    'labels_all': np.full((nFrames, nMarkers, 3), np.nan, dtype=np.float64),
    'frame_list': np.asarray(list(range(cfg.index_frame_start,cfg.index_frame_end)))
}
for i_frame, frame_idx in enumerate(labels_dlc_3d['frame_list']):
    #
    print('\r\t\tframe:\t{:06d}/{:06d}'.format(i_frame+1, nFrames), end='', flush=True)
    #
    for i_marker in range(nMarkers):
        labels_2d = labels_dlc[np.where(frame_list==frame_idx)[0][0], :, i_marker]
        #print()
        #print(labels_2d)
        labels_2d_pcutoff = labels_2d[:, 2]
        labels_2d_pcutoff[np.isnan(labels_2d_pcutoff)] = 0.0
        # Only use 2 most certain detections
        index_max_1 = np.argmax(labels_2d_pcutoff)
        labels_2d_pcutoff[index_max_1] = -np.inf
        index_max_2 = np.argmax(labels_2d_pcutoff)
        mask_nan = np.ones(nCameras, dtype=bool)
        mask_nan[index_max_1] = False
        mask_nan[index_max_2] = False
        #
        labels_2d[mask_nan] = np.nan
        #print()
        #print(labels_2d)
        labels_dlc_3d['labels_all'][i_frame, i_marker] = interp_3d.calc_3d_point(labels_2d, A, k, rX1, tX1)
        #print()
        #print(labels_dlc_3d['labels_all'][i_frame, i_marker])
#
print(labels_dlc_3d['labels_all'])
file_save = file_dlc[:-4]+'__3d.npy'
np.save(file_save, labels_dlc_3d)
