#!/usr/bin/env python3

import copy
import importlib
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
# import configuration as cfg
import data
import helper
import kalman
import model
sys_path0 = np.copy(sys.path)

save = True

folder_save = os.path.abspath('panels')

folder_reconstruction = data.path+'/reconstruction'
folder_list1 = list(['20200205/table_20200205_020300_021800__pcutoff9e-01', # mode4
                     '20200205/table_20200205_023300_025800__pcutoff9e-01',
                     '20200205/table_20200205_048800_050200__pcutoff9e-01',
                     '20200205/table_20200205_020300_021800__mode3__pcutoff9e-01', # mode3
                     '20200205/table_20200205_023300_025800__mode3__pcutoff9e-01',
                     '20200205/table_20200205_048800_050200__mode3__pcutoff9e-01',
                     '20200205/table_20200205_020300_021800__mode2__pcutoff9e-01', # mode2
                     '20200205/table_20200205_023300_025800__mode2__pcutoff9e-01',
                     '20200205/table_20200205_048800_050200__mode2__pcutoff9e-01',
                     '20200205/table_20200205_020300_021800__mode1__pcutoff9e-01', # mode1
                     '20200205/table_20200205_023300_025800__mode1__pcutoff9e-01',
                     '20200205/table_20200205_048800_050200__mode1__pcutoff9e-01',])
folder_list2 = list(['20200207/table_20200207_012500_013700__pcutoff9e-01', # mode4
                     '20200207/table_20200207_028550_029050__pcutoff9e-01',
                     '20200207/table_20200207_044900_046000__pcutoff9e-01',
                     '20200207/table_20200207_056900_057350__pcutoff9e-01',
                     '20200207/table_20200207_082800_084000__pcutoff9e-01',
                     '20200207/table_20200207_012500_013700__mode3__pcutoff9e-01', # mode3
                     '20200207/table_20200207_028550_029050__mode3__pcutoff9e-01',
                     '20200207/table_20200207_044900_046000__mode3__pcutoff9e-01',
                     '20200207/table_20200207_056900_057350__mode3__pcutoff9e-01',
                     '20200207/table_20200207_082800_084000__mode3__pcutoff9e-01',
                     '20200207/table_20200207_012500_013700__mode2__pcutoff9e-01', # mode2
                     '20200207/table_20200207_028550_029050__mode2__pcutoff9e-01',
                     '20200207/table_20200207_044900_046000__mode2__pcutoff9e-01',
                     '20200207/table_20200207_056900_057350__mode2__pcutoff9e-01',
                     '20200207/table_20200207_082800_084000__mode2__pcutoff9e-01',
                     '20200207/table_20200207_012500_013700__mode1__pcutoff9e-01', # mode1
                     '20200207/table_20200207_028550_029050__mode1__pcutoff9e-01',
                     '20200207/table_20200207_044900_046000__mode1__pcutoff9e-01',
                     '20200207/table_20200207_056900_057350__mode1__pcutoff9e-01',
                     '20200207/table_20200207_082800_084000__mode1__pcutoff9e-01',])
folder_list3 = list(['20210511_1/table_1_20210511_064700_066700__mode4__pcutoff9e-01', # animal #1, mode4
                     '20210511_1/table_1_20210511_072400_073600__mode4__pcutoff9e-01',
                     '20210511_1/table_1_20210511_083000_085000__mode4__pcutoff9e-01',
                     '20210511_1/table_1_20210511_094200_095400__mode4__pcutoff9e-01',
                     '20210511_1/table_1_20210511_064700_066700__mode3__pcutoff9e-01', # animal #1, mode3
                     '20210511_1/table_1_20210511_072400_073600__mode3__pcutoff9e-01',
                     '20210511_1/table_1_20210511_083000_085000__mode3__pcutoff9e-01',
                     '20210511_1/table_1_20210511_094200_095400__mode3__pcutoff9e-01',
                     '20210511_1/table_1_20210511_064700_066700__mode2__pcutoff9e-01', # animal #1, mode2
                     '20210511_1/table_1_20210511_072400_073600__mode2__pcutoff9e-01',
                     '20210511_1/table_1_20210511_083000_085000__mode2__pcutoff9e-01',
                     '20210511_1/table_1_20210511_094200_095400__mode2__pcutoff9e-01',
                     '20210511_1/table_1_20210511_064700_066700__mode1__pcutoff9e-01', # animal #1, mode1
                     '20210511_1/table_1_20210511_072400_073600__mode1__pcutoff9e-01',
                     '20210511_1/table_1_20210511_083000_085000__mode1__pcutoff9e-01',
                     '20210511_1/table_1_20210511_094200_095400__mode1__pcutoff9e-01',])
folder_list4 = list(['20210511_2/table_2_20210511_073000_075000__mode4__pcutoff9e-01', # animal #2, mode4
                     '20210511_2/table_2_20210511_078000_079700__mode4__pcutoff9e-01',
                     '20210511_2/table_2_20210511_081600_082400__mode4__pcutoff9e-01',
                     '20210511_2/table_2_20210511_112400_113600__mode4__pcutoff9e-01',
                     '20210511_2/table_2_20210511_114800_115700__mode4__pcutoff9e-01',
                     '20210511_2/table_2_20210511_117900_119500__mode4__pcutoff9e-01',
                     '20210511_2/table_2_20210511_073000_075000__mode3__pcutoff9e-01', # animal #2, mode3
                     '20210511_2/table_2_20210511_078000_079700__mode3__pcutoff9e-01',
                     '20210511_2/table_2_20210511_081600_082400__mode3__pcutoff9e-01',
                     '20210511_2/table_2_20210511_112400_113600__mode3__pcutoff9e-01',
                     '20210511_2/table_2_20210511_114800_115700__mode3__pcutoff9e-01',
                     '20210511_2/table_2_20210511_117900_119500__mode3__pcutoff9e-01',
                     '20210511_2/table_2_20210511_073000_075000__mode2__pcutoff9e-01', # animal #2, mode2
                     '20210511_2/table_2_20210511_078000_079700__mode2__pcutoff9e-01',
                     '20210511_2/table_2_20210511_081600_082400__mode2__pcutoff9e-01',
                     '20210511_2/table_2_20210511_112400_113600__mode2__pcutoff9e-01',
                     '20210511_2/table_2_20210511_114800_115700__mode2__pcutoff9e-01',
                     '20210511_2/table_2_20210511_117900_119500__mode2__pcutoff9e-01',
                     '20210511_2/table_2_20210511_073000_075000__mode1__pcutoff9e-01', # animal #2, mode1
                     '20210511_2/table_2_20210511_078000_079700__mode1__pcutoff9e-01',
                     '20210511_2/table_2_20210511_081600_082400__mode1__pcutoff9e-01',
                     '20210511_2/table_2_20210511_112400_113600__mode1__pcutoff9e-01',
                     '20210511_2/table_2_20210511_114800_115700__mode1__pcutoff9e-01',
                     '20210511_2/table_2_20210511_117900_119500__mode1__pcutoff9e-01',])
folder_list5 = list(['20210511_3/table_3_20210511_076900_078800__mode4__pcutoff9e-01', # animal #3, mode4
                     '20210511_3/table_3_20210511_088900_089600__mode4__pcutoff9e-01',
                     '20210511_3/table_3_20210511_093900_094700__mode4__pcutoff9e-01',
                     '20210511_3/table_3_20210511_098100_099100__mode4__pcutoff9e-01',
                     '20210511_3/table_3_20210511_106000_106600__mode4__pcutoff9e-01',
                     '20210511_3/table_3_20210511_128700_129800__mode4__pcutoff9e-01',
                     '20210511_3/table_3_20210511_076900_078800__mode3__pcutoff9e-01', # animal #3, mode3
                     '20210511_3/table_3_20210511_088900_089600__mode3__pcutoff9e-01',
                     '20210511_3/table_3_20210511_093900_094700__mode3__pcutoff9e-01',
                     '20210511_3/table_3_20210511_098100_099100__mode3__pcutoff9e-01',
                     '20210511_3/table_3_20210511_106000_106600__mode3__pcutoff9e-01',
                     '20210511_3/table_3_20210511_128700_129800__mode3__pcutoff9e-01',
                     '20210511_3/table_3_20210511_076900_078800__mode2__pcutoff9e-01', # animal #3, mode2
                     '20210511_3/table_3_20210511_088900_089600__mode2__pcutoff9e-01',
                     '20210511_3/table_3_20210511_093900_094700__mode2__pcutoff9e-01',
                     '20210511_3/table_3_20210511_098100_099100__mode2__pcutoff9e-01',
                     '20210511_3/table_3_20210511_106000_106600__mode2__pcutoff9e-01',
                     '20210511_3/table_3_20210511_128700_129800__mode2__pcutoff9e-01',
                     '20210511_3/table_3_20210511_076900_078800__mode1__pcutoff9e-01', # animal #3, mode1
                     '20210511_3/table_3_20210511_088900_089600__mode1__pcutoff9e-01',
                     '20210511_3/table_3_20210511_093900_094700__mode1__pcutoff9e-01',
                     '20210511_3/table_3_20210511_098100_099100__mode1__pcutoff9e-01',
                     '20210511_3/table_3_20210511_106000_106600__mode1__pcutoff9e-01',
                     '20210511_3/table_3_20210511_128700_129800__mode1__pcutoff9e-01',])
folder_list6 = list(['20210511_4/table_4_20210511_093600_094800__mode4__pcutoff9e-01', # animal #4, mode4
                     '20210511_4/table_4_20210511_102200_103400__mode4__pcutoff9e-01',
                     '20210511_4/table_4_20210511_122400_123400__mode4__pcutoff9e-01',
                     '20210511_4/table_4_20210511_127800_128400__mode4__pcutoff9e-01',
                     '20210511_4/table_4_20210511_135500_137200__mode4__pcutoff9e-01',
                     '20210511_4/table_4_20210511_093600_094800__mode3__pcutoff9e-01', # animal #4, mode3
                     '20210511_4/table_4_20210511_102200_103400__mode3__pcutoff9e-01',
                     '20210511_4/table_4_20210511_122400_123400__mode3__pcutoff9e-01',
                     '20210511_4/table_4_20210511_127800_128400__mode3__pcutoff9e-01',
                     '20210511_4/table_4_20210511_135500_137200__mode3__pcutoff9e-01',
                     '20210511_4/table_4_20210511_093600_094800__mode2__pcutoff9e-01', # animal #4, mode2
                     '20210511_4/table_4_20210511_102200_103400__mode2__pcutoff9e-01',
                     '20210511_4/table_4_20210511_122400_123400__mode2__pcutoff9e-01',
                     '20210511_4/table_4_20210511_127800_128400__mode2__pcutoff9e-01',
                     '20210511_4/table_4_20210511_135500_137200__mode2__pcutoff9e-01',
                     '20210511_4/table_4_20210511_093600_094800__mode1__pcutoff9e-01', # animal #4, mode1
                     '20210511_4/table_4_20210511_102200_103400__mode1__pcutoff9e-01',
                     '20210511_4/table_4_20210511_122400_123400__mode1__pcutoff9e-01',
                     '20210511_4/table_4_20210511_127800_128400__mode1__pcutoff9e-01',
                     '20210511_4/table_4_20210511_135500_137200__mode1__pcutoff9e-01',])
folder_list1 = list([folder_reconstruction + '/' + i for i in folder_list1])
folder_list2 = list([folder_reconstruction + '/' + i for i in folder_list2])
folder_list3 = list([folder_reconstruction + '/' + i for i in folder_list3])
folder_list4 = list([folder_reconstruction + '/' + i for i in folder_list4])
folder_list5 = list([folder_reconstruction + '/' + i for i in folder_list5])
folder_list6 = list([folder_reconstruction + '/' + i for i in folder_list6])

list_is_large_animal1 = list([0, 0, 0,
                              0, 0, 0,
                              0, 0, 0,
                              0, 0, 0,])
list_is_large_animal2 = list([0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0,])
list_is_large_animal3 = list([0, 0, 0, 0,
                              0, 0, 0, 0,
                              0, 0, 0, 0,
                              0, 0, 0, 0,])
list_is_large_animal4 = list([0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0,])
list_is_large_animal5 = list([1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,])
list_is_large_animal6 = list([1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1,])

i_cam_down_list1 = list([0, 0, 0,
                         0, 0, 0,
                         0, 0, 0,
                         0, 0, 0,])
i_cam_down_list2 = list([1, 0, 0, 0, 1,
                         1, 0, 0, 0, 1,
                         1, 0, 0, 0, 1,
                         1, 0, 0, 0, 1,])
i_cam_down_list3 = list([1, 0, 0, 1,
                         1, 0, 0, 1,
                         1, 0, 0, 1,
                         1, 0, 0, 1,])
i_cam_down_list4 = list([0, 1, 1, 0, 0, 0,
                         0, 1, 1, 0, 0, 0,
                         0, 1, 1, 0, 0, 0,
                         0, 1, 1, 0, 0, 0,])
i_cam_down_list5 = list([0, 1, 1, 1, 1, 1,
                         0, 1, 1, 1, 1, 1,
                         0, 1, 1, 1, 1, 1,
                         0, 1, 1, 1, 1, 1,])
i_cam_down_list6 = list([1, 1, 1, 0, 0,
                         1, 1, 1, 0, 0,
                         1, 1, 1, 0, 0,
                         1, 1, 1, 0, 0,])

date_list1 = list(['20200205', '20200205', '20200205', 
                   '20200205', '20200205', '20200205',
                   '20200205', '20200205', '20200205',
                   '20200205', '20200205', '20200205',])
date_list2 = list(['20200207', '20200207', '20200207', '20200207', '20200207',
                   '20200207', '20200207', '20200207', '20200207', '20200207',
                   '20200207', '20200207', '20200207', '20200207', '20200207',
                   '20200207', '20200207', '20200207', '20200207', '20200207',])
date_list3 = list(['20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511',])
date_list4 = list(['20210511', '20210511', '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511', '20210511', '20210511',])
date_list5 = list(['20210511', '20210511', '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511', '20210511', '20210511',])
date_list6 = list(['20210511', '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511', '20210511',
                   '20210511', '20210511', '20210511', '20210511', '20210511',])

task_list1 = list(['table', 'table', 'table',
                   'table', 'table', 'table',
                   'table', 'table', 'table',
                   'table', 'table', 'table',])
task_list2 = list(['table', 'table', 'table', 'table', 'table',
                   'table', 'table', 'table', 'table', 'table',
                   'table', 'table', 'table', 'table', 'table',
                   'table', 'table', 'table', 'table', 'table',])
task_list3 = list(['table_1', 'table_1', 'table_1', 'table_1',
                   'table_1', 'table_1', 'table_1', 'table_1',
                   'table_1', 'table_1', 'table_1', 'table_1',
                   'table_1', 'table_1', 'table_1', 'table_1',])
task_list4 = list(['table_2', 'table_2', 'table_2', 'table_2', 'table_2', 'table_2',
                   'table_2', 'table_2', 'table_2', 'table_2', 'table_2', 'table_2',
                   'table_2', 'table_2', 'table_2', 'table_2', 'table_2', 'table_2',
                   'table_2', 'table_2', 'table_2', 'table_2', 'table_2', 'table_2',])
task_list5 = list(['table_3', 'table_3', 'table_3', 'table_3', 'table_3', 'table_3',
                   'table_3', 'table_3', 'table_3', 'table_3', 'table_3', 'table_3',
                   'table_3', 'table_3', 'table_3', 'table_3', 'table_3', 'table_3',
                   'table_3', 'table_3', 'table_3', 'table_3', 'table_3', 'table_3',])
task_list6 = list(['table_4', 'table_4', 'table_4', 'table_4', 'table_4',
                   'table_4', 'table_4', 'table_4', 'table_4', 'table_4',
                   'table_4', 'table_4', 'table_4', 'table_4', 'table_4',
                   'table_4', 'table_4', 'table_4', 'table_4', 'table_4',])
                        
folder_list = folder_list1 + folder_list2 + folder_list3 + folder_list4 + folder_list5 + folder_list6
list_is_large_animal = list_is_large_animal1 + list_is_large_animal2 + list_is_large_animal3 + list_is_large_animal4 + list_is_large_animal5 + list_is_large_animal6
i_cam_down_list = i_cam_down_list1 + i_cam_down_list2 + i_cam_down_list3 + i_cam_down_list4 + i_cam_down_list5 + i_cam_down_list6
date_list = date_list1 + date_list2 + date_list3 + date_list4 + date_list5 + date_list6
task_list = task_list1 + task_list2 + task_list3 + task_list4 + task_list5 + task_list6

nCameras_up = 4

label_type = 'full'

# look at: Rational Radial Distortion Models with Analytical Undistortion Formulae, Lili Ma et al.
# source: https://arxiv.org/pdf/cs/0307047.pdf
# only works for k = [k1, k2, 0, 0, 0]
def calc_udst(m_dst, k):
    assert np.all(k[2:] == 0.0), 'ERROR: Undistortion only valid for up to two radial distortion coefficients.'
    
    x_2 = m_dst[:, 0]
    y_2 = m_dst[:, 1]
    
    # use r directly instead of c
    nPoints = np.size(m_dst, 0)
    p = np.zeros(6, dtype=np.float64)
    p[4] = 1.0
    sol = np.zeros(3, dtype=np.float64)
    x_1 = np.zeros(nPoints, dtype=np.float64)
    y_1 = np.zeros(nPoints, dtype=np.float64)
    for i_point in range(nPoints):
        cond = (np.abs(x_2[i_point]) > np.abs(y_2[i_point]))
        if (cond):
            c = y_2[i_point] / x_2[i_point]
            p[5] = -x_2[i_point]
        else:
            c = x_2[i_point] / y_2[i_point]
            p[5] = -y_2[i_point]
#        p[4] = 1
        p[2] = k[0] * (1.0 + c**2)
        p[0] = k[1] * (1.0 + c**2)**2
        sol = np.real(np.roots(p))
        # use min(abs(x)) to make your model as accurate as possible
        sol_abs = np.abs(sol)
        if (cond):
            x_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            y_1[i_point] = c * x_1[i_point]
        else:
            y_1[i_point] = sol[sol_abs == np.min(sol_abs)][0]
            x_1[i_point] = c * y_1[i_point]
    m_udst = np.concatenate([[x_1], [y_1], [m_dst[:, 2]]], 0).T
    return m_udst

def map_m(M,
          RX1, tX1, A, k): # nLabels, 3
    # RX1 * m + tX1
    m_cam = np.einsum('ij,lj->li', RX1, M) + tX1[None, :] # nLabels, 3
    # m / m[2]
    m = m_cam[:, :2] / m_cam[:, 2][:, None] # nLabels, 2
    # distort & A * m
    r_2 = m[:, 0]**2 + m[:, 1]**2 # nLabels
    sum_term = 1.0 + \
               k[0] * r_2 + \
               k[1] * r_2**2 + \
               k[4] * r_2**3 # nLabels
    x_times_y_times_2 = m[:, 0] * m[:, 1] * 2.0  # nLabels 
    m = np.stack([A[1] + A[0] * \
                  (m[:, 0] * sum_term + \
                   k[2] * x_times_y_times_2 + \
                   k[3] * (r_2 + 2.0 * m[:, 0]**2)),
                  A[3] + A[2] * \
                  (m[:, 1] * sum_term + \
                  k[2] * (r_2 + 2.0 * m[:, 1]**2) + \
                  k[3] * x_times_y_times_2)], 1) # nLabels, 2
    return m # nLabels, 2

if __name__ == '__main__':
    label_names = list(['spot_paw_front_left', # front left
                        'spot_finger_left_001',
                        'spot_finger_left_002',
                        'spot_finger_left_003',
                        'spot_paw_front_right', # front right
                        'spot_finger_right_001',
                        'spot_finger_right_002',
                        'spot_finger_right_003',
                        'spot_paw_hind_left', # hind left
                        'spot_toe_left_001',
                        'spot_toe_left_002',
                        'spot_toe_left_003',
                        'spot_paw_hind_right', # hind_right
                        'spot_toe_right_001',
                        'spot_toe_right_002',
                        'spot_toe_right_003'])
    nLabels_check = len(label_names)
    
    position0_all = list()
    angle0_all = list()
    position_all = dict()
    for i_mode in range(4):
        position_all[i_mode+1] = list()
    angle_all = dict()
    for i_mode in range(4):
        angle_all[i_mode+1] = list()
    for i_folder in range(len(folder_list)):
        print()
        print(folder_list[i_folder])
        
        folder = folder_list[i_folder]
        i_cam_down = i_cam_down_list[i_folder]
        date = date_list[i_folder]
        task = task_list[i_folder]
        
        sys.path = list(np.copy(sys_path0))
        sys.path.append(folder)
        importlib.reload(cfg)
        cfg.animal_is_large = list_is_large_animal[i_folder]
        importlib.reload(anatomy)
        
        print('mode:\t{:01d}'.format(cfg.mode))

        folder_reqFiles = data.path + '/required_files' 
        file_origin_coord = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/origin_coord.npy'
        file_calibration = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/multicalibration.npy'
        file_model = folder_reqFiles + '/model.npy'
        file_labelsDLC = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_dlc_{:06d}_{:06d}.npy'.format(cfg.index_frame_start, cfg.index_frame_end)

        file_labels_down = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/labels_down_use.npz'
        file_calibration_all = folder_reqFiles + '/' + cfg.date + '/' + cfg.task + '/calibration_all/multicalibration.npy'
        
        labels_down = np.load(file_labels_down, allow_pickle=True)['arr_0'].item()
        frame_list_fit = np.arange(cfg.index_frame_ini,
                                   cfg.index_frame_ini + cfg.nT * cfg.dt,
                                   cfg.dt, dtype=np.int64)
        frame_list_manual = np.array(sorted(list(labels_down.keys())), dtype=np.int64)
        frame_list_manual = frame_list_manual[np.array(list([i in frame_list_fit for i in frame_list_manual]))]
        
        mu_ini = np.load(folder + '/x_ini.npy', allow_pickle=True)
        save_dict = np.load(folder + '/save_dict.npy', allow_pickle=True).item()
        if ('mu_uks' in save_dict):
            mu_uks = save_dict['mu_uks'][1:]
            print(save_dict['message'])
        else:
            mu_uks = save_dict['mu'][1:]

        calib = np.load(file_calibration_all, allow_pickle=True).item()
        A = calib['A_fit'][nCameras_up + i_cam_down]
        k = calib['k_fit'][nCameras_up + i_cam_down]
        tX1 = calib['tX1_fit'][nCameras_up + i_cam_down] * cfg.scale_factor
        RX1 = calib['RX1_fit'][nCameras_up + i_cam_down]

        args_torch = helper.get_arguments(file_origin_coord, file_calibration, file_model, file_labelsDLC,
                                          cfg.scale_factor, cfg.pcutoff)
        if ((cfg.mode == 1) or (cfg.mode == 2)):
            args_torch['use_custom_clip'] = False
        elif ((cfg.mode == 3) or (cfg.mode == 4)):
            args_torch['use_custom_clip'] = True
        joint_marker_order = args_torch['model']['joint_marker_order']
        free_para_bones = args_torch['free_para_bones'].cpu().numpy()
        free_para_markers = args_torch['free_para_markers'].cpu().numpy()
        free_para_pose = args_torch['free_para_pose'].cpu().numpy()
        free_para_bones = np.zeros_like(free_para_bones, dtype=bool)
        free_para_markers = np.zeros_like(free_para_markers, dtype=bool)
        nFree_bones = int(0)
        nFree_markers = int(0)
        free_para = np.concatenate([free_para_bones,
                                    free_para_markers,
                                    free_para_pose], 0)    
        args_torch['x_torch'] = torch.from_numpy(mu_ini).type(model.float_type)
        args_torch['x_free_torch'] = torch.from_numpy(mu_ini[free_para]).type(model.float_type)
        args_torch['free_para_bones'] = torch.from_numpy(free_para_bones)
        args_torch['free_para_markers'] = torch.from_numpy(free_para_markers)
        args_torch['nFree_bones'] = nFree_bones
        args_torch['nFree_markers'] = nFree_markers    
        args_torch['plot'] = True
        del(args_torch['model']['surface_vertices'])
        
        #
        index = list()
        for i_label in range(nLabels_check):
            label_name = label_names[i_label]
            marker_name = 'marker_' + '_'.join(label_name.split('_')[1:]) + '_start'
            index.append(joint_marker_order.index(marker_name))
        index = np.array(index)

        # fit
        paws_fit = np.full((cfg.nT, nLabels_check, 3), np.nan, dtype=np.float64)
        velo_fit = np.full((cfg.nT, nLabels_check, 3), np.nan, dtype=np.float64)
        acc_fit = np.full((cfg.nT, nLabels_check, 3), np.nan, dtype=np.float64)
        ang_fit = np.full((cfg.nT, nLabels_check-4), np.nan, dtype=np.float64)
        #
        mu_t = torch.from_numpy(mu_uks)
        _, markers3d, _ = model.fcn_emission_free(mu_t, args_torch)
        markers3d = markers3d.cpu().numpy()
        #
        paws_fit = markers3d[:, index]
        for i_paw in range(4):
            paw = paws_fit[:, i_paw*4]
            fingers = paws_fit[:, i_paw*4+1:i_paw*4+3+1]
            vec = fingers - paw[:, None, :]
            vec_use = np.copy(vec)
            vec_use = vec_use[:, :, :2] # project on table
            vec_use_len = np.sqrt(np.sum(vec_use**2, 2))
            vec_use = vec_use / vec_use_len[:, :, None]
            vec_use = vec_use.reshape(cfg.nT*3, 2)
            ang_paw = np.arctan2(vec_use[:, 1], vec_use[:, 0]) * 180.0/np.pi
            ang_paw = ang_paw.reshape(cfg.nT, 3)
            ang_fit[:, i_paw*3:i_paw*3+3] = np.copy(ang_paw)
        #
        position_all[cfg.mode].append(paws_fit)
        angle_all[cfg.mode].append(ang_fit)
        # table
        paws_table = np.full((cfg.nT, nLabels_check, 3), np.nan, dtype=np.float64)
        ang_table = np.full((cfg.nT, nLabels_check-4), np.nan, dtype=np.float64)
        if (cfg.mode == 4):
            paws_table = np.full((cfg.nT, nLabels_check, 3), np.nan, dtype=np.float64)
            nT_check = len(frame_list_manual)
            for i_frame in range(nT_check):
                frame = frame_list_manual[i_frame]
                i_t = frame - frame_list_manual[0]
                #
                points_dst = np.full((nLabels_check, 3), np.nan, dtype=np.float64)
                points_udst = np.full((nLabels_check, 3), np.nan, dtype=np.float64)
                for i_label in range(nLabels_check):
                    label_name = label_names[i_label]
                    if label_name in labels_down[frame]:
                        points_dst[i_label] = np.array([(labels_down[frame][label_name][i_cam_down][0] - A[1]) / A[0],
                                                        (labels_down[frame][label_name][i_cam_down][1] - A[3]) / A[2],
                                                        1.0], dtype=np.float64)
                mask = ~np.any(np.isnan(points_dst), 1)
                points_udst[mask] = calc_udst(points_dst[mask], k)
                line0 = np.einsum('ij,xj->xi', RX1.T, -tX1[None, :])
                line1 = np.einsum('ij,xj->xi', RX1.T, points_udst - tX1[None, :])
                n = line0
                m = line1 - line0
                lambda_val = -n[:, 2] / m[:, 2]
                paws_table[i_t] = m * lambda_val[:, None] + n
            for i_paw in range(4):
                paw = paws_table[:, i_paw*4]
                fingers = paws_table[:, i_paw*4+1:i_paw*4+3+1]
                vec = fingers - paw[:, None, :]
                vec_use = np.copy(vec)
                vec_use = vec_use[:, :, :2] # project on table
                vec_use = vec_use / np.sqrt(np.sum(vec_use**2, 2))[:, :, None]
                vec_use = vec_use.reshape(cfg.nT*3, 2)
                ang_paw = np.arctan2(vec_use[:, 1], vec_use[:, 0]) * 180.0/np.pi
                ang_paw = ang_paw.reshape(cfg.nT, 3)
                ang_table[:, i_paw*3:i_paw*3+3] = np.copy(ang_paw)
            #
            position0_all.append(paws_table)
            angle0_all.append(ang_table)
            
#             # PLOT (useful to check for left/right errors in manual underneath labeling)
#             import matplotlib.pyplot as plt
#             if True:
#                 errors_pos = np.sqrt(np.sum((paws_fit[:, :, :2] - paws_table[:, :, :2])**2, 2)) # xy
#                 errors_ang = ang_fit - ang_table
                
#                 cmap = plt.cm.tab10
#                 colors = list([cmap(0.0), cmap(0.3), cmap(0.6), cmap(0.9)])
                
#                 fig = plt.figure(1, figsize=(9, 8))
#                 fig.clear()
#                 fig.canvas.manager.window.move(0, 0)
#                 ax_pos = fig.add_subplot(211)
#                 ax_pos.clear()
#                 ax_pos.spines["top"].set_visible(False)
#                 ax_pos.spines["right"].set_visible(False)
#                 ax_ang = fig.add_subplot(212)
#                 ax_ang.clear()
#                 ax_ang.spines["top"].set_visible(False)
#                 ax_ang.spines["right"].set_visible(False)
                                
#                 for i_paw in range(4):
#                     errors_pos_single = errors_pos[:, 4*i_paw]
#                     errors_ang_single = errors_ang[:, 3*i_paw]
#                     mask_pos = ~np.isnan(errors_pos_single)
#                     mask_ang = ~np.isnan(errors_ang_single)
#                     ax_pos.plot(frame_list_fit[mask_pos], errors_pos_single[mask_pos],
#                             linestyle='-', marker='.', color=colors[i_paw])
#                     ax_ang.plot(frame_list_fit[mask_ang], errors_ang_single[mask_ang],
#                             linestyle='-', marker='.', color=colors[i_paw])
#                 ax_pos.legend(list(['paw_front_left',
#                                 'paw_front_right',
#                                 'paw_hind_left',
#                                 'paw_hind_right']),
#                           loc='upper right', frameon=False,
#                           fontsize=16)
#                 ax_pos.set_xlabel('time (s)', fontsize=16)
#                 ax_pos.set_ylabel('error (cm)', fontsize=16)
#                 ax_ang.legend(list(['paw_front_left',
#                                 'paw_front_right',
#                                 'paw_hind_left',
#                                 'paw_hind_right']),
#                           loc='upper right', frameon=False,
#                           fontsize=16)
#                 ax_ang.set_xlabel('time (s)', fontsize=16)
#                 ax_ang.set_ylabel('error (deg)', fontsize=16)
                
#                 fig.canvas.draw()
#                 plt.show(block=False)
#                 input()

    # save calculations
    if save:
        np.save(os.path.abspath(folder_save + '/position0_all_{:s}.npy'.format(label_type)), position0_all)
        np.save(os.path.abspath(folder_save + '/angle0_all_{:s}.npy'.format(label_type)), angle0_all)
        #
        np.save(os.path.abspath(folder_save + '/position_all.npy'), position_all)
        np.save(os.path.abspath(folder_save + '/angle_all.npy'), angle_all)
