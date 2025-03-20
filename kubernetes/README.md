# Kubernetes Pipelines for Data-Collection in CARLA
This directory contains the Kubernetes pipelines for data-collection in CARLA simulator. The pipelines are used to collect data for training the models in the project.

The entire data collection process is split into multiple fragments. Each fragment collects data from a unique Scenario-Town combination in the CARLA simulator. Each of these fragments will run simultaneously on the Nautilus cluster to parallelize the data collection process.

## Contents in every directory
Every directory contains the following files:
- `generate_jobs.py`: This script generates the pod files that will run on the Nautilus cluster. In this file, you can specify the number of jobs you want to run and the parameters for each job.
- `create_jobs.sh`: This script uses the `kubectl` command to create the jobs in the Nautilus cluster.
- `kill_jobs.sh`: This script uses the `kubectl` command to delete the jobs in the Nautilus cluster.
- `clear_jobs.sh`: This script uses the `kubectl` command to clear the jobs in the Nautilus cluster.
- `data-collection`: This directory contains the yaml files for the pods that will be deployed.

## Instructions on running the pipelines
- Ensure that you have `kubectl` installed on your machine.
- First generate the yaml files by running:
```bash
python generate_jobs.py
```
- Then create the jobs by running:
```bash
bash create_jobs.sh
```
- Once the jobs are completed, you can either kill or delete the jobs by running either of the following commands:
```bash
bash clear_jobs.sh
```

```bash
bash kill_jobs.sh
```

## Description of the directories
To generate your own carla data, follow the order mentioned below by running the scripts in every folder:
- `Run Autopilot`: This directory will run the autopilot in CARLA. The autopilot will drive the car around the town and collect sensor and ground-truth location data.
- `Run Caption Gen`: This directory will run the caption generation model in CARLA. The model will generate captions for the images collected by the sensors.
- `Run Radar Sim`: This directory will run the SHENRON radar simulation in CARLA. The radar simulation will generate radar data from camera and 3D LiDAR point clouds.
- `Run LiDAR BEV Gen`: This directory will convert the LiDAR 3D point cloud into BEV images. These images will be used to train the LiDAR based CLIP model.
- `Run Unique Captions`: This directory will run through the generated captions to determine the unique captions set. The data from this directory will be used to train the LiDAR and Radar based CLIP model.