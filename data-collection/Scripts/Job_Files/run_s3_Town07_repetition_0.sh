#!/bin/bash
export CARLA_ROOT=${1:-/radar-imaging-dataset/carla_garage_radar/carla/}
export WORK_DIR=${2:-/radar-imaging-dataset/P2SIF/data-collection}
export SAVE_DIR=${3:-/radar-imaging-dataset/P2SIF/carla_data}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# Server Ports
export PORT=15180 # same as the carla server port
export TM_PORT=25183 # port for traffic manager, required when spawning multiple servers/clients

# Evaluation Setup
export ROUTES=/radar-imaging-dataset/P2SIF/data-collection/leaderboard/data/training/routes/s3/Town07_Scenario3.xml
export SCENARIOS=/radar-imaging-dataset/P2SIF/data-collection/leaderboard/data/training/scenarios/s3/Town07_Scenario3.json
export DEBUG_CHALLENGE=0 # visualization of waypoints and forecasting
export RESUME=1
export REPETITIONS=1
export DATAGEN=1
export BENCHMARK=collection
export GPU=0
export CHALLENGE_TRACK_CODENAME=MAP
# Agent Paths
export TEAM_AGENT="${WORK_DIR}/team_code/data_agent.py" # agent
export CHECKPOINT_ENDPOINT=${SAVE_DIR}/s3_2025-03-05/s3_Town07_Rep0/Results.json # output results file
export SAVE_PATH=${SAVE_DIR}/s3_2025-03-05/s3_Town07_Rep0 # path for saving episodes (comment to disable)
echo 'creating checkpoint and save_path directory...'
sleep 2s
mkdir -p ${SAVE_PATH}
touch ${SAVE_DIR}/s3_2025-03-05/s3_Town07_Rep0/Results.json
echo 'running leaderboard_evaluator_local.py now...'
sleep 3s

cp /radar-imaging-dataset/P2SIF/data-collection/Scripts/Job_Files//run_s3_Town07_repetition_0.sh ${SAVE_DIR}/s3_2025-03-05/s3_Town07_Rep0/
cd ${LEADERBOARD_ROOT}

CUDA_VISIBLE_DEVICES=${GPU} python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py --scenarios=${SCENARIOS}  --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=${DEBUG_CHALLENGE} --record=${RECORD_PATH} --resume=${RESUME} --port=${PORT} --trafficManagerPort=${TM_PORT} --timeout=600.0
        