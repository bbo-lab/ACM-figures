# ACM-figures

## How to:
1. Clone this repository ('git clone https://github.com/bbo-lab/ACM-figures.git' on Linux)
2. Install Anaconda: https://www.anaconda.com/products/individual#Downloads
3. Create a new Anaconda environment ('conda env create -f environment.yml' on Linux)
4. Activate the new Anaconda environment ('conda activate ACM-fig' on Linux)
5. Download the data folder: https://www.dropbox.com/sh/mtjr1g99khl4m81/AABXdk9Dv5gGjOzaN_UnXWV2a?dl=0
6. Change the 'path' variable in '/ACM-figures/ACM/data.py' to the full path of the downloaded data folder
7. Allow figure-scipts to be executed as progams ('sudo chmod +x figure_script.py' on Linux)
8. Run the figure-scripts ('./figure_script.py' on Linux)

## Additional information:
- Most scripts have a 'save' variable after the package imports, which needs to be set to 'True' to save calculations or figures (default: 'False')
- Some scripts have a 'verbose' variable after the package imports, which needs to be set to 'True' to display figures (default: 'True') 
- Some scripts have a 'mode' variable after the package imports, which needs to be set to 'mode1' to perform calculations for the naive skeleton model (default: 'mode4')  
- Some scripts require precalculations (i.e. in /ACM-figures/figure02 panel_g_h_i_j__precalculations.py has to be exectued with the 'save' variable set to 'True' before running panel_g_h.py or panel_i_j.py)
- To generate panels which require direct access to the raw video data one also needs to change the respective path variables in /ACM-figures/ACM/data.py to point at the correct file locations
- To generate videos FFmpeg is required
- Supplementary Figures 6-9 can be generated with panel_d_g_j_m.py in /ACM-figures/figure03