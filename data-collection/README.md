# Data Collection Pipeline for CARLA
This directory contains the source code for autopilot, radar data generation from SHENRON simulator and caption data generation.

## Requirements Installation
All the requirements are mentioned in the `requirements.txt` file, and should be installed by the following command:
```bash
python3 -m pip install -r requirements.txt
```

## Code for Autopilot
The code for autopilot is present in the `team_code` directory which has been inspired from the [carla-garage](https://github.com/autonomousvision/carla_garage) github repository. The code has been modified to suit the requirements of the project. For more information on how the autopilot works, please refer to the linked repository.

## Ground-Truth Data Collection
To extract ground-truth data from CARLA and store in JSON files, the functions were implemented in `/team_code/ground_truth_data.py`. This function is called in the data collection pipeline to generate the real-time actor information for every data frame collected.

## Caption Generation
For converting the data from JSON files to captions, a rule-based converter was implemented in `/team_code/generate_caption.py`. For more information on how the captions are generated, please refer to the code and the final report.

## LiDAR BEV Data Generation
The code for generating LiDAR BEV data from 3D point-clouds is implemented in `team_code/generate_lidar_bev.py`.

## Radar Data Generation
The code for integrating Shenron radar into CARLA was directly picked form [carla-radarimaging](https://github.com/ucsdwcsng/carla-radarimaging/) repository. Please refer to the codebase here for more details.

## Scripts Directory
This directory contains all the scripts that will be accessed by each Kubernetes pod to run the data collection pipeline. For every task, there is a python script that generates the bash script required to run the task. Each of the bash scripts are then executed by
the Kubernetes pod.