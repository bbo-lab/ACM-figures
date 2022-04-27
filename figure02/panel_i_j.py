#!/usr/bin/env python3

import copy
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import minimize
import scipy.stats
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
import kalman
import model
sys_path0 = np.copy(sys.path)

save = False

folder_save = os.path.abspath('panels')

# data_req_folder = 'data'
folder_reconstruction = data.path+'/datasets_figures/reconstruction'
data_folder1 = folder_reconstruction+'/20200205/'
data_folder2 = folder_reconstruction+'/20200207/'
data_folder3 = folder_reconstruction+'/20210511_1/'
data_folder4 = folder_reconstruction+'/20210511_2/'
data_folder5 = folder_reconstruction+'/20210511_3/'
data_folder6 = folder_reconstruction+'/20210511_4/'
folder_list = list([
                    [ # mode4
                    data_folder1+'table_20200205_020300_021800__pcutoff9e-01', # 20200205
                    data_folder1+'table_20200205_023300_025800__pcutoff9e-01',
                    data_folder1+'table_20200205_048800_050200__pcutoff9e-01',
                    data_folder2+'table_20200207_012500_013700__pcutoff9e-01', # 20200207
                    data_folder2+'table_20200207_028550_029050__pcutoff9e-01',
                    data_folder2+'table_20200207_044900_046000__pcutoff9e-01',
                    data_folder2+'table_20200207_056900_057350__pcutoff9e-01',
                    data_folder2+'table_20200207_082800_084000__pcutoff9e-01',
                    data_folder3+'table_1_20210511_064700_066700__mode4__pcutoff9e-01', # animal #1
                    data_folder3+'table_1_20210511_072400_073600__mode4__pcutoff9e-01',
                    data_folder3+'table_1_20210511_083000_085000__mode4__pcutoff9e-01',
                    data_folder3+'table_1_20210511_094200_095400__mode4__pcutoff9e-01',
                    data_folder4+'table_2_20210511_073000_075000__mode4__pcutoff9e-01', # animal #2
                    data_folder4+'table_2_20210511_078000_079700__mode4__pcutoff9e-01',
                    data_folder4+'table_2_20210511_081600_082400__mode4__pcutoff9e-01',
                    data_folder4+'table_2_20210511_112400_113600__mode4__pcutoff9e-01',
                    data_folder4+'table_2_20210511_114800_115700__mode4__pcutoff9e-01',
                    data_folder4+'table_2_20210511_117900_119500__mode4__pcutoff9e-01',
                    data_folder5+'table_3_20210511_076900_078800__mode4__pcutoff9e-01', # animal #3
                    data_folder5+'table_3_20210511_088900_089600__mode4__pcutoff9e-01',
                    data_folder5+'table_3_20210511_093900_094700__mode4__pcutoff9e-01',
                    data_folder5+'table_3_20210511_098100_099100__mode4__pcutoff9e-01',
                    data_folder5+'table_3_20210511_106000_106600__mode4__pcutoff9e-01',
                    data_folder5+'table_3_20210511_128700_129800__mode4__pcutoff9e-01',
                    data_folder6+'table_4_20210511_093600_094800__mode4__pcutoff9e-01', # animal #4
                    data_folder6+'table_4_20210511_102200_103400__mode4__pcutoff9e-01',
                    data_folder6+'table_4_20210511_122400_123400__mode4__pcutoff9e-01',
                    data_folder6+'table_4_20210511_127800_128400__mode4__pcutoff9e-01',
                    data_folder6+'table_4_20210511_135500_137200__mode4__pcutoff9e-01',
                    ],
                    [ # mode3
                    data_folder1+'table_20200205_020300_021800__mode3__pcutoff9e-01',
                    data_folder1+'table_20200205_023300_025800__mode3__pcutoff9e-01',
                    data_folder1+'table_20200205_048800_050200__mode3__pcutoff9e-01',
                    data_folder2+'table_20200207_012500_013700__mode3__pcutoff9e-01',
                    data_folder2+'table_20200207_028550_029050__mode3__pcutoff9e-01',
                    data_folder2+'table_20200207_044900_046000__mode3__pcutoff9e-01',
                    data_folder2+'table_20200207_056900_057350__mode3__pcutoff9e-01',
                    data_folder2+'table_20200207_082800_084000__mode3__pcutoff9e-01',
                    data_folder3+'table_1_20210511_064700_066700__mode3__pcutoff9e-01', # animal #1
                    data_folder3+'table_1_20210511_072400_073600__mode3__pcutoff9e-01',
                    data_folder3+'table_1_20210511_083000_085000__mode3__pcutoff9e-01',
                    data_folder3+'table_1_20210511_094200_095400__mode3__pcutoff9e-01',
                    data_folder4+'table_2_20210511_073000_075000__mode3__pcutoff9e-01', # animal #2
                    data_folder4+'table_2_20210511_078000_079700__mode3__pcutoff9e-01',
                    data_folder4+'table_2_20210511_081600_082400__mode3__pcutoff9e-01',
                    data_folder4+'table_2_20210511_112400_113600__mode3__pcutoff9e-01',
                    data_folder4+'table_2_20210511_114800_115700__mode3__pcutoff9e-01',
                    data_folder4+'table_2_20210511_117900_119500__mode3__pcutoff9e-01',
                    data_folder5+'table_3_20210511_076900_078800__mode3__pcutoff9e-01', # animal #3
                    data_folder5+'table_3_20210511_088900_089600__mode3__pcutoff9e-01',
                    data_folder5+'table_3_20210511_093900_094700__mode3__pcutoff9e-01',
                    data_folder5+'table_3_20210511_098100_099100__mode3__pcutoff9e-01',
                    data_folder5+'table_3_20210511_106000_106600__mode3__pcutoff9e-01',
                    data_folder5+'table_3_20210511_128700_129800__mode3__pcutoff9e-01',
                    data_folder6+'table_4_20210511_093600_094800__mode3__pcutoff9e-01', # animal #4
                    data_folder6+'table_4_20210511_102200_103400__mode3__pcutoff9e-01',
                    data_folder6+'table_4_20210511_122400_123400__mode3__pcutoff9e-01',
                    data_folder6+'table_4_20210511_127800_128400__mode3__pcutoff9e-01',
                    data_folder6+'table_4_20210511_135500_137200__mode3__pcutoff9e-01',
                    ],
                    [ # mode2
                    data_folder1+'table_20200205_020300_021800__mode2__pcutoff9e-01',
                    data_folder1+'table_20200205_023300_025800__mode2__pcutoff9e-01',
                    data_folder1+'table_20200205_048800_050200__mode2__pcutoff9e-01',
                    data_folder2+'table_20200207_012500_013700__mode2__pcutoff9e-01',
                    data_folder2+'table_20200207_028550_029050__mode2__pcutoff9e-01',
                    data_folder2+'table_20200207_044900_046000__mode2__pcutoff9e-01',
                    data_folder2+'table_20200207_056900_057350__mode2__pcutoff9e-01',
                    data_folder2+'table_20200207_082800_084000__mode2__pcutoff9e-01',
                    data_folder3+'table_1_20210511_064700_066700__mode2__pcutoff9e-01', # animal #1
                    data_folder3+'table_1_20210511_072400_073600__mode2__pcutoff9e-01',
                    data_folder3+'table_1_20210511_083000_085000__mode2__pcutoff9e-01',
                    data_folder3+'table_1_20210511_094200_095400__mode2__pcutoff9e-01',
                    data_folder4+'table_2_20210511_073000_075000__mode2__pcutoff9e-01', # animal #2
                    data_folder4+'table_2_20210511_078000_079700__mode2__pcutoff9e-01',
                    data_folder4+'table_2_20210511_081600_082400__mode2__pcutoff9e-01',
                    data_folder4+'table_2_20210511_112400_113600__mode2__pcutoff9e-01',
                    data_folder4+'table_2_20210511_114800_115700__mode2__pcutoff9e-01',
                    data_folder4+'table_2_20210511_117900_119500__mode2__pcutoff9e-01',
                    data_folder5+'table_3_20210511_076900_078800__mode2__pcutoff9e-01', # animal #3
                    data_folder5+'table_3_20210511_088900_089600__mode2__pcutoff9e-01',
                    data_folder5+'table_3_20210511_093900_094700__mode2__pcutoff9e-01',
                    data_folder5+'table_3_20210511_098100_099100__mode2__pcutoff9e-01',
                    data_folder5+'table_3_20210511_106000_106600__mode2__pcutoff9e-01',
                    data_folder5+'table_3_20210511_128700_129800__mode2__pcutoff9e-01',
                    data_folder6+'table_4_20210511_093600_094800__mode2__pcutoff9e-01', # animal #4
                    data_folder6+'table_4_20210511_102200_103400__mode2__pcutoff9e-01',
                    data_folder6+'table_4_20210511_122400_123400__mode2__pcutoff9e-01',
                    data_folder6+'table_4_20210511_127800_128400__mode2__pcutoff9e-01',
                    data_folder6+'table_4_20210511_135500_137200__mode2__pcutoff9e-01',
                    ],
                    [ # mode1
                    data_folder1+'table_20200205_020300_021800__mode1__pcutoff9e-01',
                    data_folder1+'table_20200205_023300_025800__mode1__pcutoff9e-01',
                    data_folder1+'table_20200205_048800_050200__mode1__pcutoff9e-01',
                    data_folder2+'table_20200207_012500_013700__mode1__pcutoff9e-01',
                    data_folder2+'table_20200207_028550_029050__mode1__pcutoff9e-01',
                    data_folder2+'table_20200207_044900_046000__mode1__pcutoff9e-01',
                    data_folder2+'table_20200207_056900_057350__mode1__pcutoff9e-01',
                    data_folder2+'table_20200207_082800_084000__mode1__pcutoff9e-01',
                    data_folder3+'table_1_20210511_064700_066700__mode1__pcutoff9e-01', # animal #1
                    data_folder3+'table_1_20210511_072400_073600__mode1__pcutoff9e-01',
                    data_folder3+'table_1_20210511_083000_085000__mode1__pcutoff9e-01',
                    data_folder3+'table_1_20210511_094200_095400__mode1__pcutoff9e-01',
                    data_folder4+'table_2_20210511_073000_075000__mode1__pcutoff9e-01', # animal #2
                    data_folder4+'table_2_20210511_078000_079700__mode1__pcutoff9e-01',
                    data_folder4+'table_2_20210511_081600_082400__mode1__pcutoff9e-01',
                    data_folder4+'table_2_20210511_112400_113600__mode1__pcutoff9e-01',
                    data_folder4+'table_2_20210511_114800_115700__mode1__pcutoff9e-01',
                    data_folder4+'table_2_20210511_117900_119500__mode1__pcutoff9e-01',
                    data_folder5+'table_3_20210511_076900_078800__mode1__pcutoff9e-01', # animal #3
                    data_folder5+'table_3_20210511_088900_089600__mode1__pcutoff9e-01',
                    data_folder5+'table_3_20210511_093900_094700__mode1__pcutoff9e-01',
                    data_folder5+'table_3_20210511_098100_099100__mode1__pcutoff9e-01',
                    data_folder5+'table_3_20210511_106000_106600__mode1__pcutoff9e-01',
                    data_folder5+'table_3_20210511_128700_129800__mode1__pcutoff9e-01',
                    data_folder6+'table_4_20210511_093600_094800__mode1__pcutoff9e-01', # animal #4
                    data_folder6+'table_4_20210511_102200_103400__mode1__pcutoff9e-01',
                    data_folder6+'table_4_20210511_122400_123400__mode1__pcutoff9e-01',
                    data_folder6+'table_4_20210511_127800_128400__mode1__pcutoff9e-01',
                    data_folder6+'table_4_20210511_135500_137200__mode1__pcutoff9e-01',
                    ],
                   ])

list_is_large_animal = list([0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0,
                             1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1,])
i_cam_down_list = list([0, 0, 0,
                        1, 0, 0, 0, 1,
                        1, 0, 0, 1,
                        0, 1, 1, 0, 0, 0,
                        0, 1, 1, 1, 1, 1,
                        1, 1, 1, 0, 0,])
date_list = list(['20200205', '20200205', '20200205',
                  '20200207', '20200207', '20200207', '20200207', '20200207',
                  '20210511', '20210511', '20210511', '20210511',
                  '20210511', '20210511', '20210511', '20210511', '20210511', '20210511',
                  '20210511', '20210511', '20210511', '20210511', '20210511', '20210511',
                  '20210511', '20210511', '20210511', '20210511', '20210511',])
task_list = list(['table', 'table', 'table',
                  'table', 'table', 'table', 'table', 'table',
                  'table_1', 'table_1', 'table_1', 'table_1',
                  'table_2', 'table_2', 'table_2', 'table_2', 'table_2', 'table_2',
                  'table_3', 'table_3', 'table_3', 'table_3', 'table_3', 'table_3',
                  'table_4', 'table_4', 'table_4', 'table_4', 'table_4',])
nCameras_up = 4

label_type = 'full'

mm_in_inch = 5.0/127.0
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

bottom_margin_x = 0.05 # inch
left_margin_x = 0.05 # inch
left_margin  = 0.4 # inch
right_margin = 0.05 # inch
bottom_margin = 0.375 # inch
top_margin = 0.05 # inch
between_margin_h = 0.1 # inch

fontsize = 6
linewidth_hist = 0.5
fontname = "Arial"

cmap = plt.cm.tab10
color_mode1 = cmap(5/9)
color_mode2 = cmap(1/9)
color_mode3 = cmap(0/9)
color_mode4 = cmap(2/9)
colors = list([color_mode1, color_mode2, color_mode3, color_mode4])

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


def find_closest_3d_point(m, n):
    def obj_func(x):
        estimated_points = x[:-3, None] * m + n
        res = 0.5 * np.sum((estimated_points - x[None, -3:])**2)

        jac = np.zeros(len(x), dtype=np.float64)
        jac[:-3] = np.sum(m**2, 1) * x[:-3] + \
                   np.sum(m * n, 1) - \
                   np.sum(m * x[None, -3:], 1)
        jac[-3:] = (np.float64(len(x)) - 3.0) * x[-3:] - np.sum(estimated_points, 0)
        return res, jac
    
    nPoints = np.size(m, 0)
    x0 = np.zeros(nPoints + 3)
    tolerance = np.finfo(np.float32).eps # using np.float64 can lead to the optimization not converging (results are the same though)
    min_result = minimize(obj_func,
                          x0,
                          method='l-bfgs-b',
                          jac=True,
                          tol=tolerance,
                          options={'disp':False,
                                   'maxcor':20,
                                   'maxfun':np.inf,
                                   'maxiter':np.inf,
                                   'maxls':40})
    if not(min_result.success):
        print('WARNING: 3D point interpolation did not converge')
        print('\tnPoints\t', nPoints)
        print('\tsuccess:\t', min_result.success)
        print('\tstatus:\t', min_result.status)
        print('\tmessage:\t',min_result.message)
        print('\tnit:\t', min_result.nit) 
    return min_result.x

def rodrigues2rotMat_single(r):
    sqrt_arg = np.sum(r**2)
    if (sqrt_arg < 2**-23):
        rotMat = np.identity(3, dtype=np.float64)
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

def calc_dst(m_udst, k):
    x_1 = m_udst[:, 0] / m_udst[:, 2]
    y_1 = m_udst[:, 1] / m_udst[:, 2]
    
    r2 = x_1**2 + y_1**2
    
    x_2 = x_1 * (1 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3) + \
          2 * k[2] * x_1 * y_1 + \
          k[3] * (r2 + 2 * x_1**2)
          
    y_2 = y_1 * (1 + k[0] * r2 + k[1] * r2**2 + k[4] * r2**3) + \
          k[2] * (r2 + 2 * y_1**2) + \
          2 * k[3] * x_1 * y_1
    
    nPoints = np.size(m_udst, 0)
    ones = np.ones(nPoints, dtype=np.float64)
    m_dst = np.concatenate([[x_2], [y_2], [ones]], 0).T
    return m_dst

def calc_3d_point(points_2d, A, k, rX1, tX1):
    mask_nan = ~np.any(np.isnan(points_2d), 1)
    mask_zero = ~np.any((points_2d == 0.0), 1)
    mask = np.logical_and(mask_nan, mask_zero)
    nValidPoints = np.sum(mask)
    if (nValidPoints < 2):
        print("WARNING: Less than 2 valid 2D locations for 3D point interpolation.")
    n = np.zeros((nValidPoints, 3))
    m = np.zeros((nValidPoints, 3))
    nCameras = np.size(points_2d, 0)
    
    index = 0
    for i_cam in range(nCameras):
        if (mask[i_cam]):
            point = np.array([[(points_2d[i_cam, 0] - A[i_cam][0, 2]) / A[i_cam][0, 0],
                               (points_2d[i_cam, 1] - A[i_cam][1, 2]) / A[i_cam][1, 1],
                               1.0]], dtype=np.float64)
            point = calc_udst(point, k[i_cam]).T
            line = point * np.linspace(0, 1, 2)
            RX1 = rodrigues2rotMat_single(rX1[i_cam])
            line = np.dot(RX1.T, line - tX1[i_cam].reshape(3, 1))
            n[index] = line[:, 0]
            m[index] = line[:, 1] - line[:, 0]
            index = index + 1
    x = find_closest_3d_point(m, n)
    return x[-3:]

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
    
    fig_w = np.round(mm_in_inch * 88.0*1/2, decimals=2)
    fig_h = np.round(mm_in_inch * 88.0*1/2, decimals=2)
    #
    fig2d_w = fig_w
    fig2d_h = fig_h
    fig2d = plt.figure(1, figsize=(fig2d_w, fig2d_h))
    fig2d.canvas.manager.window.move(0, 0)
    fig2d.clear()
    ax2d_x = left_margin/fig2d_w
    ax2d_y = bottom_margin/fig2d_h
    ax2d_w = 1.0 - (left_margin/fig2d_w + right_margin/fig2d_w)
    ax2d_h = 2/3 * ((1.0-top_margin/fig2d_h-bottom_margin/fig2d_h) - 1.0*between_margin_h/fig2d_h)
    ax2d = fig2d.add_axes([ax2d_x, ax2d_y, ax2d_w, ax2d_h])
    ax2d.clear()
    ax2d.spines["top"].set_visible(False)
    ax2d.spines["right"].set_visible(False)
    ax2d_n_x = left_margin/fig2d_w
    ax2d_n_y = ax2d_y + ax2d_h + between_margin_h/fig2d_h
    ax2d_n_w = 1.0 - (left_margin/fig2d_w + right_margin/fig2d_w)
    ax2d_n_h = 1/3 * ((1.0-top_margin/fig2d_h-bottom_margin/fig2d_h) - 1.0*between_margin_h/fig2d_h)
    ax2d_n = fig2d.add_axes([ax2d_n_x, ax2d_n_y, ax2d_n_w, ax2d_n_h])
    ax2d_n.clear()
    ax2d_n.spines["top"].set_visible(False)
    ax2d_n.spines["right"].set_visible(False)
    ax2d.set_xlabel('time since last / until next detection (ms)', va='bottom', ha='center', fontsize=fontsize)
    ax2d.set_ylabel('position error of\noccluded markers (cm)', va='top', ha='center', fontsize=fontsize)
#     ax2d_n.set_xlabel('time since last / until next detection (ms)', va='bottom', ha='center', fontsize=fontsize)
    ax2d_n.set_ylabel('sample size', va='top', ha='center', fontsize=fontsize)
    #
#     fig2d_w = fig_w
#     fig2d_h = fig_h*0.5
#     fig2d = plt.figure(1, figsize=(fig2d_w, fig2d_h))
#     fig2d.canvas.manager.window.move(0, 0)
#     fig2d.clear()
#     ax2d_x = left_margin/fig2d_w
#     ax2d_y = bottom_margin/fig2d_h
#     ax2d_w = 1.0 - (left_margin/fig2d_w + right_margin/fig2d_w)
#     ax2d_h = 1.0 - (bottom_margin/fig2d_h + top_margin/fig2d_h)
#     ax2d = fig2d.add_axes([ax2d_x, ax2d_y, ax2d_w, ax2d_h])
#     ax2d.clear()
#     ax2d.spines["top"].set_visible(False)
#     ax2d.spines["right"].set_visible(False)
#     ax2d.set_xlabel('time since last / until next detection (ms)', fontsize=fontsize)
#     ax2d.set_ylabel('position error of occluded markers (cm)', fontsize=fontsize)
#     #
#     fig2d_n_w = fig_w
#     fig2d_n_h = fig_h*0.5 
#     fig2d_n = plt.figure(2, figsize=(fig2d_n_w, fig2d_n_h))
#     fig2d_n.canvas.manager.window.move(0, 0)
#     fig2d_n.clear()
#     ax2d_n_x = left_margin/fig2d_n_w
#     ax2d_n_y = bottom_margin/fig2d_n_h
#     ax2d_n_w = 1.0 - (left_margin/fig2d_n_w + right_margin/fig2d_n_w)
#     ax2d_n_h = 1.0 - (bottom_margin/fig2d_n_h + top_margin/fig2d_n_h)
#     ax2d_n = fig2d_n.add_axes([ax2d_n_x, ax2d_n_y, ax2d_n_w, ax2d_n_h])
#     ax2d_n.clear()
#     ax2d_n.spines["top"].set_visible(False)
#     ax2d_n.spines["right"].set_visible(False)
#     ax2d_n.set_xlabel('time since last / until next detection (ms)', fontsize=fontsize)
#     ax2d_n.set_ylabel('number of samples', fontsize=fontsize)
    #
    fig_hist_w = fig_w
    fig_hist_h = fig_h   
    fig_hist = plt.figure(3, figsize=(fig_hist_w, fig_hist_h))
    fig_hist.canvas.manager.window.move(0, 0)
    fig_hist.clear()
    ax_hist_x = left_margin/fig_hist_w
    ax_hist_y = bottom_margin/fig_hist_h
    ax_hist_w = 1.0 - (left_margin/fig_hist_w + right_margin/fig_hist_w)
    ax_hist_h = 1.0 - (bottom_margin/fig_hist_h + top_margin/fig_hist_h)
    ax_hist = fig_hist.add_axes([ax_hist_x, ax_hist_y, ax_hist_w, ax_hist_h])
    ax_hist.clear()
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    ax_hist.set_xlabel('position error of occluded markers (cm)', va='bottom', ha='center', fontsize=fontsize)
    ax_hist.set_ylabel('probability', va='top', ha='center', fontsize=fontsize)
    #
    fig_detec_w = fig_w
    fig_detec_h = fig_h   
    fig_detec = plt.figure(4, figsize=(fig_detec_w, fig_detec_h))
    fig_detec.canvas.manager.window.move(0, 0)
    fig_detec.clear()
    ax_detec_x = left_margin/fig_detec_w
    ax_detec_y = bottom_margin/fig_detec_h
    ax_detec_w = 1.0 - (left_margin/fig_detec_w + right_margin/fig_detec_w)
    ax_detec_h = 1.0 - (bottom_margin/fig_detec_h + top_margin/fig_detec_h)
    ax_detec = fig_detec.add_axes([ax_detec_x, ax_detec_y, ax_detec_w, ax_detec_h])
    ax_detec.clear()
    ax_detec.spines["top"].set_visible(False)
    ax_detec.spines["right"].set_visible(False)
    ax_detec.set_xlabel('number of detections', va='bottom', ha='center', fontsize=fontsize)
    ax_detec.set_ylabel('probability', va='top', ha='center', fontsize=fontsize)
    #
    errors_modes = list()
    errors_modes_sorted = list()
    for i_mode in range(4):
        errors_all = list()
        #
        nFrames_errors = int(3000) # hard coded
        errors_power_1 = np.zeros(nFrames_errors, dtype=np.float64)
        errors_power_2 = np.zeros(nFrames_errors, dtype=np.float64)
        errors_n = np.zeros(nFrames_errors, dtype=np.float64)
        #
        nDetections_all = np.zeros(5, dtype=np.float64)
        #
        for i_folder in range(len(folder_list[i_mode])):
            print()
            print(folder_list[i_mode][i_folder])

            folder = folder_list[i_mode][i_folder]
            i_cam_down = i_cam_down_list[i_folder]
            date = date_list[i_folder]
            task = task_list[i_folder]

            sys.path = list(np.copy(sys_path0))
            sys.path.append(folder)
            importlib.reload(cfg)
            cfg.animal_is_large = list_is_large_animal[i_folder]
            importlib.reload(anatomy)

            folder_reqFiles = data.path + '/datasets_figures/required_files'
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
            nMarkers = args_torch['numbers']['nMarkers']
            nCameras = args_torch['numbers']['nCameras']
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
            
            # index of all 4x4 paw labels
            index_paws = list()
            for i_label in range(nLabels_check):
                label_name = label_names[i_label]
                marker_name = 'marker_' + '_'.join(label_name.split('_')[1:]) + '_start'
                index_paws.append(joint_marker_order.index(marker_name))
            index_paws = np.array(index_paws, dtype=np.int64)
            nMarkers_paws = len(index_paws)
            
            # fit
            paws_fit = np.full((cfg.nT, nLabels_check, 3), np.nan, dtype=np.float64)
            mu_t = torch.from_numpy(mu_uks)
            _, markers3d, _ = model.fcn_emission_free(mu_t, args_torch)
            markers3d = markers3d.cpu().numpy()
            paws_fit = markers3d[:, index_paws]
            # table
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

            # annotated 2d labels
            labels_dlc = np.load(file_labelsDLC, allow_pickle=True).item()
            labels_index1 = cfg.index_frame_start - labels_dlc['frame_list'][0]
            labels_index2 = cfg.index_frame_end - labels_dlc['frame_list'][0] + 1
            labels_mask = args_torch['labels_mask'].cpu().numpy().astype(np.float64)
            labels_mask = labels_mask[labels_index1:labels_index2]
            labels_mask_sum = np.sum(labels_mask, 1)
            detection_single = labels_mask_sum[:-1, index_paws]
            #
#             frame_list = np.arange(cfg.index_frame_start, cfg.index_frame_end, cfg.dt, dtype=np.int64)
#             labels_manual = np.load(cfg.file_labelsManual, allow_pickle=True)['arr_0'].item()
#             #
#             markers2d_manual_single = np.full((cfg.nT, nCameras, nMarkers, 2), np.nan, dtype=np.float64)
#             for i_frame in range(cfg.nT):
#                 frame = frame_list[i_frame]
#                 if frame in labels_manual:
#                     print(frame)
#                     for label_name in list(labels_manual[frame].keys()):
#                         marker_name = 'marker_'+'_'.join(label_name.split('_')[1:])+'_start'
#                         index = joint_marker_order.index(marker_name)
#                         markers2d_manual_single[i_frame, :, index] = np.copy(labels_manual[frame][label_name])
                        
            # get occlusion times
            fair_analysis = True
            nDetections = 0
            frames_occluded_forward = np.zeros_like(detection_single, dtype=np.int64)
            for i in range(nMarkers_paws):
                counter = 0
                for t in range(cfg.nT):
                    if (detection_single[t, i] <= nDetections):
                        counter += 1
                        frames_occluded_forward[t, i] = np.copy(counter)
                    else:
                        counter = 0
            frames_occluded_backward = np.zeros_like(detection_single, dtype=np.int64)
            for i in range(nMarkers_paws):
                counter = 0
                for t in range(cfg.nT)[::-1]:
                    if (detection_single[t, i] <= nDetections):
                        counter += 1
                        frames_occluded_backward[t, i] = np.copy(counter)
                    else:
                        counter = 0
            frames_occluded = np.minimum(frames_occluded_forward, frames_occluded_backward)
            for i in range(nMarkers_paws):
                if np.any(frames_occluded_forward[:, i] == 0):
                    index_forward = np.where(frames_occluded_forward[:, i] == 0)[0][0]
                    frames_occluded[:index_forward, i] = frames_occluded_backward[:index_forward, i]
                if np.any(frames_occluded_backward[:, i] == 0):
                    index_backward = np.where(frames_occluded_backward[:, i] == 0)[0][-1]
                    frames_occluded[index_backward:, i] = frames_occluded_forward[index_backward:, i]
            if (fair_analysis):
                if ((i_mode == 2) or (i_mode == 3)):
                    frames_occluded = np.copy(frames_occluded_forward)    
                    # to exlcude labels that where not detected in first frame
                    for i in range(nMarkers_paws):
                        if (frames_occluded_forward[0, i] == 1):
                            index = np.where(frames_occluded[:, i] == 0)[0][0]
                            frames_occluded[:index, i] = 0 # set to zero to exclude from error calculation (should be inf in principle but dtype is int, this has the same effect though)
                
            # update arrays for error calculation
            errors = np.sqrt(np.sum((paws_fit - paws_table)[:, :, :2]**2, 2))
            errors_all = errors_all + list(errors[detection_single <= nDetections].ravel()) 
            for t in range(cfg.nT):
                for i in range(nMarkers_paws):
                    if ((frames_occluded[t, i] > 0) and (~np.isnan(errors[t, i]))):
                        errors_power_1[frames_occluded[t, i]] = errors_power_1[frames_occluded[t, i]] + errors[t, i]
                        errors_power_2[frames_occluded[t, i]] = errors_power_2[frames_occluded[t, i]] + errors[t, i]**2
                        errors_n[frames_occluded[t, i]] = errors_n[frames_occluded[t, i]] + 1
            for i_detec in range(5):
                nDetections_all[i_detec] = nDetections_all[i_detec] + np.sum(detection_single[~np.isnan(errors)] == i_detec)

        # bin the data
        dFrame = 3 # uneven, nFrames_errors/dFrame \in N
        errors_power_1 = np.reshape(errors_power_1, (int(nFrames_errors/dFrame), dFrame))
        errors_power_1 = np.sum(errors_power_1, 1)
        errors_power_2 = np.reshape(errors_power_2, (int(nFrames_errors/dFrame), dFrame))
        errors_power_2 = np.sum(errors_power_2, 1)
        errors_n = np.reshape(errors_n, (int(nFrames_errors/dFrame), dFrame))
        errors_n = np.sum(errors_n, 1)
        errors_n_time = np.arange(nFrames_errors, dtype=np.float64).reshape(int(nFrames_errors/dFrame), dFrame)
        errors_n_time = errors_n_time[:, int((dFrame-1)/2)]
        
        # calculate errors
        errors_avg = errors_power_1 / errors_n
        errors_std = np.sqrt((errors_power_2 - (errors_power_1**2 / errors_n)) / errors_n)
        # mask for plotting
        mask_n = (errors_n >= 1) # minimum: 1
        mask_n = (errors_n >= 10)
        mask_time = np.ones_like(errors_n, dtype=bool)
        # for binned data
        mask_time[(errors_n_time.astype(np.float64) * 1e3 / 200.0) > 500.0] = False
        mask_use = np.ones_like(errors_n, dtype=bool)
        mask_use = np.logical_and(mask_use, mask_n)
        mask_use = np.logical_and(mask_use, mask_time)
                
        # plot errors
        x = errors_n_time[mask_use].astype(np.float64) * 1e3 / 200.0 # ms # when data is binned
        y = errors_avg[mask_use]
        errors_modes_sorted.append(y)
        if i_mode in list([0, 3]):
            ax2d.plot(x, y, color=colors[3-i_mode], zorder=3-i_mode, linewidth=linewidth_hist)
            ax2d.fill_between(x=x, y1=y-errors_std[mask_use], y2=y+errors_std[mask_use],
                              color=colors[3-i_mode], linewidth=linewidth_hist,
                              alpha=0.2, zorder=3-i_mode,
                              where=None, interpolate=False, step=None, data=None)
        # f(x) = sum((m*x_i+n - y_i)**2) [sum over all i]
        # solve for d/dm f(x) = 0 and d/dn f(x) = 0 
        N = len(x)
        n_term1 = np.sum(x*y)/np.sum(x**2) - np.sum(y)/np.sum(x)
        n_term2 = np.sum(x)/np.sum(x**2) - N/np.sum(x)
        n = n_term1 / n_term2
        m = (np.sum(y) - N*n) / np.sum(x)
        print('mode:\t\t{:01d}'.format(4-i_mode))
        print('slope:\t\t{:08e}'.format(m))
        print('intercept:\t{:08e}'.format(n))
        if i_mode in list([0, 3]):
            ax2d.plot(x, m*x+n, linestyle='--', marker='',
                      color=colors[3-i_mode], zorder=3-i_mode, linewidth=linewidth_hist, alpha=0.8)
        
        # plot number of samples (as histogramm when data is binned)
        if (fair_analysis):
            y_use = np.zeros(2+2*len(errors_n[mask_use]), dtype=np.float64)
            y_use[0] = 0.0
            y_use[1:-1:2] = np.copy(errors_n[mask_use])
            y_use[2:-1:2] = np.copy(errors_n[mask_use])
            y_use[-1] = 0.0
            #
            errors_n_time_use = np.full(len(errors_n_time[mask_use])+1, np.nan, dtype=np.float64)
            errors_n_time_use[:-1] = errors_n_time[mask_use] - int((dFrame-1)/2)
            errors_n_time_use[-1] = errors_n_time[mask_use][-1] + int((dFrame-1)/2)
            errors_n_time_use = errors_n_time_use.astype(np.float64) * 1e3 / 200.0 # ms
            x_use = np.zeros(len(y_use), dtype=np.float64)
            x_use[0] = np.copy(errors_n_time_use[0])
            x_use[1:-1:2] = np.copy(errors_n_time_use[:-1])
            x_use[2:-1:2] = np.copy(errors_n_time_use[1:])
            x_use[-1] = np.copy(errors_n_time_use[-1])
            #
            if (i_mode == 0):
                color = color_mode4
                ax2d_n.semilogy(x_use, y_use, color=color, linestyle='-', marker='', zorder=1, linewidth=linewidth_hist, clip_on=True)
            elif (i_mode == 3):
                color = color_mode1
                ax2d_n.semilogy(x_use, y_use, color=color, linestyle='-', marker='', zorder=0, linewidth=linewidth_hist, clip_on=True)
            
        # calculate and plot histogram
        inf_fac = 0.25
        nBins = 25
        # position
        pos_unit = 'cm'
        conversion_fac_pos = 1.0 # cm -> pos_unit
        maxBin_pos0 = 4e0 # cm
        dBin_pos = maxBin_pos0/nBins # cm
        maxBin_pos = maxBin_pos0 * (1.0 + inf_fac)
        bin_range_pos = np.arange(0.0, maxBin_pos+dBin_pos, dBin_pos, dtype=np.float64)
        nBin_pos = len(bin_range_pos) - 1
        #
        errors_all = np.array(errors_all, dtype=np.float64)
        errors_all = errors_all[~np.isnan(errors_all)]
        errors_modes.append(errors_all)
        n_hist = np.float64(len(errors_all))
        hist = np.histogram(errors_all, bins=nBin_pos, range=[0.0, maxBin_pos0], normed=None, weights=None, density=False)
        cumfun_pos = np.cumsum(hist[0] / n_hist)        
        print('mode: {:0d}'.format(4-i_mode))
        print('position ({:s}):'.format('cm'))
        print('n:\t\t{:0d}'.format(int(n_hist)))
        print('slope:\t\t{:0.8f}'.format(m))
        print('intercept:\t{:0.8f}'.format(n))
        print('avg.:\t\t{:0.8f}'.format(np.mean(errors_all)))
        print('sd.:\t\t{:0.8f}'.format(np.std(errors_all)))
        print('median:\t\t{:0.8f}'.format(np.median(errors_all)))
        print('max.:\t\t{:0.8f}'.format(np.max(errors_all)))
        print('min.:\t\t{:0.8f}'.format(np.min(errors_all)))
        print('cumfun:\t\t{:0.8f}'.format(cumfun_pos[-1]))
        print('1-cumfun:\t{:0.8f}'.format(1.0-cumfun_pos[-1]))
        errors_all[errors_all > maxBin_pos0] = maxBin_pos
        hist = np.histogram(errors_all, bins=nBin_pos, range=[bin_range_pos[0], bin_range_pos[-1]], normed=None, weights=None, density=False)
        hist[0][bin_range_pos[1:] > maxBin_pos0] = hist[0][-1]
        #
        y = np.zeros(2+2*len(hist[0]), dtype=np.float64)
        y[0] = 0.0
        y[1:-1:2] = np.copy(hist[0] / n_hist)
        y[2:-1:2] = np.copy(hist[0] / n_hist)
        y[-1] = 0.0
        x = np.zeros(2+2*(len(hist[1])-1), dtype=np.float64)
        x[0] = np.copy(hist[1][0])
        x[1:-1:2] = np.copy(hist[1][:-1])
        x[2:-1:2] = np.copy(hist[1][1:])        
        x[-2] = maxBin_pos
        x[-1] = maxBin_pos
        #
        ax_hist.plot(x, y, color=colors[3-i_mode], linestyle='-', marker='', zorder=0-i_mode, linewidth=linewidth_hist, clip_on=False)
    n_hist_detec = np.sum(nDetections_all)
    x = np.zeros(2+2*len(nDetections_all), dtype=np.float64)
    y = np.zeros(2+2*len(nDetections_all), dtype=np.float64)
    y[0] = 0.0
    y[1:-1:2] = np.copy(nDetections_all / n_hist_detec)
    y[2:-1:2] = np.copy(nDetections_all / n_hist_detec)
    y[-1] = 0.0
    hist_detec1 = np.arange(6, dtype=np.float64) - 0.5
    x[0] = np.copy(hist_detec1[0])
    x[1:-1:2] = np.copy(hist_detec1[:-1])
    x[2:-1:2] = np.copy(hist_detec1[1:])
    x[-1] = np.copy(hist_detec1[-1])
    print('nDetections_all:')
    print(n_hist_detec)
    print(nDetections_all)
    print(nDetections_all/n_hist_detec)
    #
    ax_detec.plot(x, y, color='black', linestyle='-', marker='', zorder=0-i_mode, linewidth=linewidth_hist, clip_on=False)   
    #
    for tick in ax_detec.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax_detec.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    ax_detec.set_xlim([-0.5, 4.5])
    ax_detec.set_xticks([0, 2, 4])
    ax_detec.set_ylim([0.0, 0.4])
    ax_detec.set_yticks([0, 0.2, 0.4])
    ax_detec.set_yticklabels([0, 0.2, 0.4])
    ax_detec.xaxis.set_label_coords(x=ax_detec_x+0.5*ax_detec_w, y=bottom_margin_x/fig_detec_h, transform=fig_detec.transFigure)
    ax_detec.yaxis.set_label_coords(x=left_margin_x/fig_detec_w, y=ax_detec_y+0.5*ax_detec_h, transform=fig_detec.transFigure)
    #
    for tick in ax2d.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax2d.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    ax2d.set_xlim([0.0, 500])
    ax2d.set_xlim([0.0, 500])
    ax2d.set_xticks([0, 250, 500])
    ax2d.set_xticklabels([0, 250, 500])
    ax2d.set_ylim([0.0, 8.0])
    ax2d.set_yticks([0, 4, 8])
    ax2d.set_yticklabels([0, 4, 8])
    ax2d.xaxis.set_label_coords(x=ax2d_x+0.5*ax2d_w, y=bottom_margin_x/fig2d_h, transform=fig2d.transFigure)
    ax2d.yaxis.set_label_coords(x=left_margin_x/fig2d_w, y=ax2d_y+0.5*ax2d_h, transform=fig2d.transFigure)
    
    for tick in ax2d_n.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax2d_n.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
        tick.set_fontsize(fontsize)
    ax2d_n.set_xlim([0.0, 500])
    ax2d_n.set_xlim([0.0, 500])
    ax2d_n.set_xticks([0, 250, 500])
    ax2d_n.set_xticklabels([])
    ax2d_n.set_ylim([1, 1000.0]) # semilogy
    ax2d_n.set_yticks([1, 10, 100, 1000]) # semilogy
    # ATTENTION: remember to manually set the first y-tick to 0 later!
    ax2d_n.xaxis.set_label_coords(x=ax2d_n_x+0.5*ax2d_n_w, y=bottom_margin_x/fig2d_h, transform=fig2d.transFigure)
    ax2d_n.yaxis.set_label_coords(x=left_margin_x/fig2d_w, y=ax2d_n_y+0.5*ax2d_n_h, transform=fig2d.transFigure)
    ax2d_n.minorticks_off()
    #
    position_box_and_lines = 10**0.5 # hard coded
    line_fac = 2.5e-2
    box_w_fac = 1.0 + 0.01
    line_w = np.diff(ax2d_n.get_xlim()) * line_fac * 1/3 * 2
    line_h = np.diff(np.log10(ax2d_n.get_ylim())) * line_fac * 1/3 # last 1/3 factor as sub-plot is only 1/3 of whole figure
    box_w = abs(np.diff(ax2d_n.get_xlim())) * box_w_fac
    box_h = (10**1 - 10**0) * 0.2 # hard coded
    ax2d_n.plot(np.array([-line_w*0.5, +line_w*0.5], dtype=np.float64) + ax2d_n.get_xlim()[0],
                np.array([10**(np.log10(position_box_and_lines - box_h*0.5) -line_h*0.5),
                          10**(np.log10(position_box_and_lines - box_h*0.5) +line_h*0.5)], dtype=np.float64),
                linewidth=ax2d_n.spines['left'].get_linewidth(), linestyle='-', color='black', zorder=101, clip_on=False)
    ax2d_n.plot(np.array([-line_w*0.5, +line_w*0.5], dtype=np.float64) + ax2d_n.get_xlim()[0],
                np.array([10**(np.log10(position_box_and_lines + box_h*0.5) -line_h*0.5),
                          10**(np.log10(position_box_and_lines + box_h*0.5) +line_h*0.5)], dtype=np.float64),
                linewidth=ax2d_n.spines['left'].get_linewidth(), linestyle='-', color='black', zorder=101, clip_on=False)
    box = plt.Rectangle((ax2d_n.get_xlim()[0]+abs(np.diff(ax2d_n.get_xlim()))*0.5-box_w*0.5,
                         position_box_and_lines - box_h*0.5),
                         box_w, box_h,
                         color='white', zorder=100, clip_on=False)
    ax2d_n.add_patch(box)
    
    
    offset_x_fac = 0.0
    for tick in ax_hist.get_xticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(fontname)
    for tick in ax_hist.get_yticklabels():
        tick.set_fontname(fontname)
        tick.set_fontsize(fontsize)
    ax_hist.set_xlim([(bin_range_pos[0]-offset_x_fac*dBin_pos)*conversion_fac_pos, (bin_range_pos[-1]+offset_x_fac*dBin_pos)*conversion_fac_pos])
    ax_hist.set_xticks([0.0, maxBin_pos0*conversion_fac_pos*0.5, maxBin_pos0*conversion_fac_pos, maxBin_pos*conversion_fac_pos])
    ax_hist.set_ylim([0.0, 0.24])
    ax_hist.set_yticks([0.0, 0.12, 0.24])
    ax_hist.set_yticklabels([0, 0.12, 0.24])
    labels = list(['{:0.0f}'.format(i) for i in ax_hist.get_xticks()])
    labels[-1] = 'inf'
    ax_hist.set_xticklabels(labels)
    ax_hist.xaxis.get_offset_text().set_fontsize(fontsize)
    ax_hist.yaxis.get_offset_text().set_fontsize(fontsize)
    ax_hist.xaxis.set_label_coords(x=ax_hist_x+0.5*ax_hist_w, y=bottom_margin_x/fig_hist_h, transform=fig_hist.transFigure)
    ax_hist.yaxis.set_label_coords(x=left_margin_x/fig_hist_w, y=ax_hist_y+0.5*ax_hist_h, transform=fig_hist.transFigure)
    #
    line_fac = 2.5e-2
    box_h_fac = 1.0 + 0.01
    line_w = np.diff(ax_hist.get_xlim()) * line_fac * 1/3
    line_h = np.diff(ax_hist.get_ylim()) * line_fac
    box_w = (maxBin_pos - maxBin_pos0) * conversion_fac_pos * 0.2
    box_h = abs(np.diff(ax_hist.get_ylim())) * box_h_fac
    ax_hist.plot(np.array([-line_w*0.5, +line_w*0.5], dtype=np.float64) + maxBin_pos0*conversion_fac_pos*(1.0+inf_fac*0.5) - box_w*0.5,
                 np.array([-line_h*0.5, +line_h*0.5], dtype=np.float64) + ax_hist.get_ylim()[0],
                 linewidth=ax_hist.spines['bottom'].get_linewidth(), linestyle='-', color='black', zorder=101, clip_on=False)
    ax_hist.plot(np.array([-line_w*0.5, +line_w*0.5], dtype=np.float64) + maxBin_pos0*conversion_fac_pos*(1.0+inf_fac*0.5) + box_w*0.5,
                 np.array([-line_h*0.5, +line_h*0.5], dtype=np.float64) + ax_hist.get_ylim()[0],
                 linewidth=ax_hist.spines['bottom'].get_linewidth(), linestyle='-', color='black', zorder=101, clip_on=False)
    box = plt.Rectangle((maxBin_pos0*conversion_fac_pos*(1.0+inf_fac*0.5) - box_w*0.5, ax_hist.get_ylim()[0]+abs(np.diff(ax_hist.get_ylim()))*0.5-box_h*0.5),
                         box_w, box_h,
                         color='white', zorder=100, clip_on=False)
    ax_hist.add_patch(box)
    
    
    fig2d.canvas.draw()
    fig_hist.canvas.draw()
        
    # STATS
    print('Kolmogorov-Smirnov test:')
    print()
    for i_mode in list([1, 2, 3]):
        s = 'mode'
        stat_pos, p_pos = scipy.stats.ks_2samp(errors_modes[0], errors_modes[i_mode], alternative='greater', mode='auto')
        print('mode pair:')
        print('mode4 / mode{:01d}'.format(-i_mode+4))
        print('stat={:0.8f}, p={:0.15e}'.format(stat_pos, p_pos))
        if p_pos > 0.05:
            print('Probably the same distribution')
        else:
            print('Probably different distributions')
        print()
    #
    print('Mann-Whitney rank test:')
    print()
    for i_mode in list([1, 2, 3]):
        stat_pos, p_pos = scipy.stats.mannwhitneyu(errors_modes_sorted[0], errors_modes_sorted[i_mode], alternative='less')
        print('mode pair:')
        print('mode4 / mode{:01d}'.format(-i_mode+4))
        print('stat={:0.8f}, p={:0.15e}'.format(stat_pos, p_pos))
        print()

    print('Wilcoxon signed-rank test:')
    print()
    for i_mode in list([1, 2, 3]):
        print(errors_modes_sorted[0].shape)
        print(errors_modes_sorted[i_mode].shape)
        stat_pos, p_pos = scipy.stats.wilcoxon(errors_modes_sorted[0], errors_modes_sorted[i_mode],
                                                   alternative='less')
        print('mode pair:')
        print('mode4 / mode{:01d}'.format(-i_mode + 4))
        print('stat={:0.8f}, p={:0.15e}'.format(stat_pos, p_pos))
        print()

    if save:
        fig2d.savefig(folder_save+'/undetected_marker_error__vs__time.svg',
    #                             bbox_inches="tight",
                                dpi=300,
                                transparent=True,
                                format='svg',
                                pad_inches=0)
#         fig2d_n.savefig(folder_save+'/sample_number__vs__time.svg',
#     #                             bbox_inches="tight",
#                                 dpi=300,
#                                 transparent=True,
#                                 format='svg',
#                                 pad_inches=0)
#         fig_hist.savefig(folder_save+'/undetected_marker_error_hist.svg',
#     #                             bbox_inches="tight",
#                                 dpi=300,
#                                 transparent=True,
#                                 format='svg',
#                                 pad_inches=0)
    
    plt.show()
