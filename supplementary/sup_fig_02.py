#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../ACM'))
import data

save = False
verbose = True

folder_save = os.path.abspath('panels')

folder_supFiles = data.path + '/supplementary' 
files_list = list([folder_supFiles+'/20200205/sketch.npy',
                   folder_supFiles+'/20200207/sketch.npy',
                   folder_supFiles+'/20210511_1/sketch.npy',
                   folder_supFiles+'/20210511_2/sketch.npy',
                   folder_supFiles+'/20210511_3/sketch.npy',
                   folder_supFiles+'/20210511_4/sketch.npy',])
dates = list(['20200205', '20200207', '20210511_1', '20210511_2', '20210511_3', '20210511_4'])

if __name__ == '__main__':     
    fig = plt.figure(1, figsize=(8, 8))
    fig.canvas.manager.window.move(0, 0)
    ax = fig.add_subplot(111)
    
    for i_date in range(len(dates)):
        date = dates[i_date]
        file = files_list[i_date]
    
        data = np.load(file, allow_pickle=True).item()
        sketch = data['sketch']
        locations = data['sketch_label_locations']
        
        ax.clear()
        ax.set_axis_off()
        ax.imshow(sketch)
        
        marker_list = sorted(list(locations.keys()))
        nMarkers = len(marker_list)
        loc_max_x = 0.0
        loc_max_y = 0.0
        for i_marker in range(nMarkers):
            marker_name = marker_list[i_marker]
            loc = locations[marker_name]
            ax.text(loc[0], loc[1], i_marker,
                    va='center', ha='center', color='red', alpha=1.0,
                    rotation=90,
                    fontsize=3)
            #
            loc_max_x = max(loc_max_x, loc[0])
            loc_max_y = max(loc_max_y, loc[1])
        
        list_replace = list()
        nReplace = np.shape(list_replace)[0]
        marker_list0 = list(np.copy(marker_list))
        for i_replace in range(nReplace):
            marker_list[marker_list0.index(list_replace[i_replace][0])] = list_replace[i_replace][1]
        # dipslay marker list
        marker_list_legend = list()
        for i_marker in range(nMarkers):
            marker_name = marker_list[i_marker]
            marker_name_split = marker_name.split('_')
            marker_name_split = marker_name_split[1:]
            if ('left' in marker_name_split):
                marker_name_split[marker_name_split.index('left')] = '(left)'
            elif ('right' in marker_name_split):
                marker_name_split[marker_name_split.index('right')] = '(right)'
            marker_name_split = ' '.join(marker_name_split)
            marker_name_use = '{:02d}. {:s}'.format(i_marker, marker_name_split)
            marker_list_legend.append(marker_name_use)
        marker_list_legend = '\n'.join(marker_list_legend)
        offset_x = 100
        offset_y = 0
        ax.text(loc_max_x + offset_x, 0.0 + offset_y, marker_list_legend,
                va='top', ha='left', color='black', alpha=1.0,
                fontsize=9)
        
        plt.pause(2**-23)
        fig.canvas.draw()
        
        if verbose:
            plt.show(block=False)
            print('Press any key to continue')
            input()
        
        if save:
            fig.savefig(saveFolder+'/sketch_surface_marker_{:s}.svg'.format(date),
                             bbox_inches="tight",
                             dpi=300,
                             transparent=True,
                             format='svg',
                             pad_inches=0) 
            
    plt.show()