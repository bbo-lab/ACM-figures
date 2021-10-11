#!/usr/bin/env python3

import numpy as np
import os
import sys

save = True

sys.path.append(os.path.abspath('../ACM/config'))
import configuration as cfg
sys.path.append(os.path.abspath('../ACM'))
import data
import helper
import interp_3d

if __name__ == '__main__':
    date_list = list(['20200205', '20200207'])
    folder_reqFiles = data.path + '/required_files' 
    file_dlc_list = list([folder_reqFiles+'/20200205/arena/labels_dlc_001900_060500.npy',
                          folder_reqFiles+'/20200207/arena/labels_dlc_001300_064700.npy',])
    nDates = len(date_list)

    scale_factor = 3.5 # cm
    
    for i_date in range(nDates):
        #
        print()
        print('date:\t{:s}'.format(date_list[i_date]))
        #
        date = date_list[i_date]
        #
        file_dlc = file_dlc_list[i_date]
        labels_dlc_dict = np.load(file_dlc, allow_pickle=True).item()
        frame_list = labels_dlc_dict['frame_list']
        labels_dlc = labels_dlc_dict['labels_all']
        labels_dlc_shape = np.shape(labels_dlc)
        nFrames = labels_dlc_shape[0]
        nCameras = labels_dlc_shape[1]
        nMarkers = labels_dlc_shape[2]
        #
        file_origin_coord = folder_reqFiles+'/'+date+'/arena/origin_coord.npy'
        file_calibration = folder_reqFiles+'/'+date+'/arena/multicalibration.npy'
        calibration = helper.get_calibration(file_origin_coord, file_calibration, scale_factor)
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

        labels_dlc_3d = np.full((nFrames, nMarkers, 3), np.nan, dtype=np.float64)
        for i_frame in range(nFrames):
            #
            print('\r\t\tframe:\t{:06d}/{:06d}'.format(i_frame+1, nFrames), end='', flush=True)
            #
            frame = frame_list[i_frame]
            for i_marker in range(nMarkers):
                labels_2d = labels_dlc[i_frame, :, i_marker]
                labels_2d_pcutoff = labels_2d[:, 2]
                labels_2d_pcutoff[np.isnan(labels_2d_pcutoff)] = 0.0
                #
                index_max_1 = np.argmax(labels_2d_pcutoff)
                labels_2d_pcutoff[index_max_1] = -np.inf
                index_max_2 = np.argmax(labels_2d_pcutoff)
                mask_nan = np.ones(nCameras, dtype=bool)
                mask_nan[index_max_1] = False
                mask_nan[index_max_2] = False
                #
                labels_2d[mask_nan] = np.nan
                labels_dlc_3d[i_frame, i_marker] = interp_3d.calc_3d_point(labels_2d, A, k, rX1, tX1)
        #
        file_dlc_split = file_dlc.split('/')
        file_save = '/'.join(file_dlc_split[:-1])+'/'+file_dlc_split[-1].split('.')[0]+'__3d.npy'
        if save:
            np.save(file_save, labels_dlc_3d)
