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

cd /radar-imaging-dataset/P2SIF/data-collection/team_code/

touch /radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/job151.log

echo 'running for /radar-imaging-dataset/P2SIF/carla_data/s9_2025-03-05/s9_Town05_Rep1...'

python3 generate_caption.py --town_folder /radar-imaging-dataset/P2SIF/carla_data/s9_2025-03-05/s9_Town05_Rep1 > /radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/job151.log 2>&1

echo 'caption generation finished...'
