#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

echo 'running data collection for run_s4_Town01_repetition_1.sh...'

cd /radar-imaging-dataset/P2SIF/data-collection/
echo 'indide /radar-imaging-dataset/P2SIF/data-collection/...'
echo 'installing requirements...'
bash install_requirements.sh
echo 'requirements installed...'
 
sleep 3s

echo 'starting carla...'
CARLA_PORT=4321
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 /radar-imaging-dataset/carla_garage_radar/carla/CarlaUE4.sh --carla-world-port=15180 -opengl -nosound -carla-streaming-port=$CARLA_PORT -quality-level=Epic & 

CARLA_UP=0

# Check if CARLA is running
while [ $CARLA_UP -eq 0 ]
do
    # Check if the port is open
    if netstat -tuln | grep ":$CARLA_PORT" > /dev/null; then
        echo "Carla client is up and listening on port $CARLA_PORT"
        CARLA_UP=1
    else
        echo "Carla client is still setting up to listen on port $CARLA_PORT"
        sleep 1m
    fi
done

echo "Loop finished"

echo 'carla started...'
sleep 3s

echo 'going to leaderboard evaluator bash...'
touch /radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/job49.log
chmod u+x /radar-imaging-dataset/P2SIF/data-collection/Scripts/Job_Files/run_s4_Town01_repetition_1.sh

bash /radar-imaging-dataset/P2SIF/data-collection/Scripts/Job_Files/run_s4_Town01_repetition_1.sh > /radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/job49.log 2>&1

echo 'job finished...'
