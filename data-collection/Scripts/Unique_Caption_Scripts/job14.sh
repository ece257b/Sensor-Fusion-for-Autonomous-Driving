#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

cd /radar-imaging-dataset/P2SIF/data-collection/
echo 'indide /radar-imaging-dataset/P2SIF/data-collection/...'
echo 'installing requirements...'
bash install_requirements_captions.sh
echo 'requirements installed...'
 
sleep 3s

echo 'running caption generation...'

cd /radar-imaging-dataset/P2SIF/lidar-radar-clip/

touch /radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/job14.log
touch /radar-imaging-dataset/P2SIF/lidar-radar-clip/caption/job14.json

echo 'running for /radar-imaging-dataset/P2SIF/carla_data/s10_2025-03-05/s10_Town07_Rep2...'

python3 count_unique_captions.py --town_folder /radar-imaging-dataset/P2SIF/carla_data/s10_2025-03-05/s10_Town07_Rep2 --save_file_path /radar-imaging-dataset/P2SIF/lidar-radar-clip/caption/job14.json > /radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/job14.log 2>&1

echo 'caption generation finished...'
