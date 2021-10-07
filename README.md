# ACM-figures

## How to:
### 1. Clone repository
### 2. Download data folder: https://www.dropbox.com/sh/mtjr1g99khl4m81/AABXdk9Dv5gGjOzaN_UnXWV2a?dl=0
### 3. Change the 'path' variable in '/ACM-figures/ACM/data.py' to the absolute location of the downloaded data folder
### 4. Allow figure-scipts to execute files as progams (i.e. 'sudo chmod +x figure_script.py' on Linux)
### 5. Run the figure-scripts (i.e. './figure_script.py' on Linux)

## Additional information:
### - Most scripts have a 'save' variable after the package imports, which needs to be set to 'True' in order to save figures (default: 'False')
### - Some scripts have a 'verbose' variable after the package imports, which needs to be set to 'True' in order to display figures (default: 'True') 
### - Some scripts have a 'mode' variable after the package imports, which needs to be set to 'mode1' in order to perform calculations for the naive skeleton model (default: 'mode4')  
### - Some scripts require precalculations (e.g. in '/ACM-figures/figure02' panel_g_h_i_j__precalculations.py has to be exectued before running panel_g_h.py or panel_i_j.py)