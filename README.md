# Homology Guided Binding Pocket Prediction

## Inputs:
When on the Dashboard scripted by `app.py` the user must enter a session name and at least 2 structure files (`.cif`, `.pdb`). The user may change many parameters that are inportant in the calculation of the final scores. Upon running the program will share visualizations of the results as color mapping on the 3D structures of the proteins and a visualization of the MSA generated during the workflow. 

## How to Run:
If you are only wishing to run the workflow, within the `main_workflow/` directory there is the `calc_stats.py` script which does all of the statistics and calculations required for the program. The user can start the program in command line if they also specify the weights and necessary directories as detailed by `./calc_stats.py --help`.

If you are wanting to run the dashboard platform, the proper containers have already been created. All you need to do is navigate to the repo directory and use the make command `make compose`, which shall build the container with the proper files, permissions, and required modules. It will then start the `app.py` script which runs out of port 8050 locally. To use the staging feature, use the make command `make compose-staging`, which will run the script out of port 8051.
