#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

cd /radar-imaging-dataset/P2SIF/data-collection/
echo 'indide /radar-imaging-dataset/P2SIF/data-collection/...'
echo 'installing requirements...'
bash install_requirements.sh
echo 'requirements installed...'
 
sleep 3s

echo 'running radar simulation...'

cd /radar-imaging-dataset/P2SIF/data-collection/team_code/e2e_agent_sem_lidar2shenron_package/

touch /radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/job108.log

python3 main.py --town_folder /radar-imaging-dataset/P2SIF/carla_data/s7_2025-03-05/s7_Town07_Rep0 > /radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/job108.log 2>&1

echo 'radar simulation finished...'
