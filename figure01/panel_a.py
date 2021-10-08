#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../ccv'))
import ccv

sys.path.append(os.path.abspath('../ACM'))
import data
sys_path0 = np.copy(sys.path)

save = False
verbose = True

folder_save = os.path.abspath('panels')

file_ccv = data.path_ccv_fig1

bg_threshold = 16
dxy = 150

dt = 35
t_start = 85
t_end = t_start + 140
i_img_start = 34400 + t_start
dFrame = dt
nFrames = 4
    
if __name__ == '__main__':
    if not(file_ccv == ''):
        img0 = ccv.get_frame(file_ccv, 1).astype(np.float64)
    else:
        img0 = np.zeros((1024, 1280), dtype=np.float64)
        
    fig = plt.figure(1, frameon=False, figsize=(5, 5))
    fig.clear()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.clear()
    fig.add_axes(ax)
    img_dummy = np.zeros((2*dxy, 2*dxy), dtype=np.float64)
    h_img = ax.imshow(img_dummy,
                      'gray',
                      vmin=0, vmax=95)
    
    i_img_list = np.arange(i_img_start, i_img_start+dFrame*nFrames, dFrame, dtype=np.int64)
    for i_img in i_img_list:
        if not(file_ccv == ''):
            img = ccv.get_frame(file_ccv, i_img+1).astype(np.float64)
        else:
            img = np.full((1280, 1024), 48, dtype=np.float64)
        diff = abs(img - img0)
        diff[diff < bg_threshold] = 0.0

        index_y, index_x = np.where(diff > 0.0)
        center_x = np.median(index_x).astype(np.int64)
        center_y = np.median(index_y).astype(np.int64)

        yRes, xRes = np.shape(img)
        xlim_min = center_x - dxy
        xlim_max = center_x + dxy
        ylim_min = center_y - dxy
        ylim_max = center_y + dxy
        xlim_min_use = np.max([xlim_min, 0])
        xlim_max_use = np.min([xlim_max, xRes])
        ylim_min_use = np.max([ylim_min, 0])
        ylim_max_use = np.min([ylim_max, yRes])
        dx = np.int64(xlim_max_use - xlim_min_use)
        dy = np.int64(ylim_max_use - ylim_min_use)
        #
        dx_add = dxy - (center_x - xlim_min_use)
        dy_add = dxy - (center_y - ylim_min_use)
        dx_add_int = np.int64(dx_add)
        dy_add_int = np.int64(dy_add)
        #
        img_dummy.fill(0)
        img_dummy[dy_add_int:dy+dy_add_int, dx_add_int:dx+dx_add_int] = \
            img[ylim_min_use:ylim_max_use, xlim_min_use:xlim_max_use]
        
        h_img.set_array(img_dummy)

        fig.canvas.draw()
        plt.pause(2**-23)

        if save:
            fig.savefig(folder_save+'/cam{:01d}_img{:06d}.svg'.format(i_cam, i_img),
                        format='svg',
                        pad_inches=0)
            fig.savefig(folder_save+'/cam{:01d}_img{:06d}.tiff'.format(i_cam, i_img),
                        format='tiff',
                        pad_inches=0)
        
        if verbose:
            plt.show(block=False)
            print('Press any key to continue')
            input()