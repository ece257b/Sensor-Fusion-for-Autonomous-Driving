import os
import json
from datetime import datetime

def carla_job_script(directories,
                     file_path,
                     carla_world_port,
                     num_gpu,
                     scenario_name,
                     date,
                     town,
                     repetition,
                     job_filename):
    carla_root = directories["carla_root"]
    work_dir = directories["work_dir"]
    save_dir = directories["save_dir"]
    job_file_save_dir = directories["job_file_save_dir"]
    
    route_path = file_path["route_path"]
    scenario_path = file_path["scenario_path"]
    
    carla_root = "{1:-" + carla_root + "}"
    work_dir = "{2:-" + work_dir + "}"
    save_dir = "{3:-" + save_dir + "}"
    gpu = str(num_gpu)
    
    file_path = f'{job_file_save_dir}/{job_filename}'
    with open(file_path, 'w', encoding = 'utf-8') as f:
        f.write(f"""\
#!/bin/bash
export CARLA_ROOT=${carla_root}
export WORK_DIR=${work_dir}
export SAVE_DIR=${save_dir}

export CARLA_SERVER=${{CARLA_ROOT}}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

export SCENARIO_RUNNER_ROOT=${{WORK_DIR}}/scenario_runner
export LEADERBOARD_ROOT=${{WORK_DIR}}/leaderboard
export PYTHONPATH="${{CARLA_ROOT}}/PythonAPI/carla/":"${{SCENARIO_RUNNER_ROOT}}":"${{LEADERBOARD_ROOT}}":${{PYTHONPATH}}

# Server Ports
export PORT={carla_world_port} # same as the carla server port
export TM_PORT=25183 # port for traffic manager, required when spawning multiple servers/clients

# Evaluation Setup
export ROUTES={route_path}
export SCENARIOS={scenario_path}
export DEBUG_CHALLENGE=0 # visualization of waypoints and forecasting
export RESUME=1
export REPETITIONS=1
export DATAGEN=1
export BENCHMARK=collection
export GPU={gpu}
export CHALLENGE_TRACK_CODENAME=MAP
# Agent Paths
export TEAM_AGENT="${{WORK_DIR}}/team_code/data_agent.py" # agent
export CHECKPOINT_ENDPOINT=${{SAVE_DIR}}/{scenario_name}_{date}/{scenario_name}_{town}_Rep{repetition}/Results.json # output results file
export SAVE_PATH=${{SAVE_DIR}}/{scenario_name}_{date}/{scenario_name}_{town}_Rep{repetition} # path for saving episodes (comment to disable)
echo 'creating checkpoint and save_path directory...'
sleep 2s
mkdir -p ${{SAVE_PATH}}
touch ${{SAVE_DIR}}/{scenario_name}_{date}/{scenario_name}_{town}_Rep{repetition}/Results.json
echo 'running leaderboard_evaluator_local.py now...'
sleep 3s

cp {file_path} ${{SAVE_DIR}}/{scenario_name}_{date}/{scenario_name}_{town}_Rep{repetition}/
cd ${{LEADERBOARD_ROOT}}

CUDA_VISIBLE_DEVICES=${{GPU}} python3 ${{LEADERBOARD_ROOT}}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${{SCENARIOS}}  \
--routes=${{ROUTES}} \
--repetitions=${{REPETITIONS}} \
--track=${{CHALLENGE_TRACK_CODENAME}} \
--checkpoint=${{CHECKPOINT_ENDPOINT}} \
--agent=${{TEAM_AGENT}} \
--agent-config=${{TEAM_CONFIG}} \
--debug=${{DEBUG_CHALLENGE}} \
--record=${{RECORD_PATH}} \
--resume=${{RESUME}} \
--port=${{PORT}} \
--trafficManagerPort=${{TM_PORT}} \
--timeout=600.0
        """)

def main():
    carla_root = "/radar-imaging-dataset/carla_garage_radar/carla/"
    work_dir = "/radar-imaging-dataset/P2SIF/data-collection"
    save_dir = "/radar-imaging-dataset/P2SIF/carla_data"
    job_file_save_dir = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/Job_Files/"
    directories = {
        "carla_root": carla_root,
        "work_dir": work_dir,
        "save_dir": save_dir,
        "job_file_save_dir": job_file_save_dir
    }
        
    carla_world_port = 15180
    num_repititions = 3
    
    route_scenario_data_json = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/route_scenario_data.json"
    route_scenario_data_dir = "/radar-imaging-dataset/P2SIF/data-collection/leaderboard/data/training/"
    
    with open(route_scenario_data_json, 'r') as f:
        data = json.load(f)
    
    scenarios = data['scenarios']
    sc_allowed = ["s1", "s3", "s4", "s7", "s8", "s9", "s10"]
    
    # extract today's date
    date = datetime.today().strftime('%Y-%m-%d')
    
    for sc in scenarios:
        if sc not in sc_allowed:
            continue
        town_list = data[sc]['towns']
        scenario_list = data[sc]['scenarios']
        
        for j, town_filename in enumerate(town_list):
            town = town_filename.split("_")[0]
            
            scenario_filename = scenario_list[j]
            route_data_path = os.path.join(route_scenario_data_dir, "routes", sc, town_filename)
            scenario_data_path = os.path.join(route_scenario_data_dir, "scenarios", sc, scenario_filename)
            
            file_path = {
                "route_path": route_data_path,
                "scenario_path": scenario_data_path
            }
            
            for k in range(num_repititions):
                # job_filename
                job_filename = f"run_{sc}_{town}_repetition_{k}.sh"
                
                carla_job_script(directories = directories,
                                file_path = file_path,
                                carla_world_port = carla_world_port,
                                num_gpu = 0,
                                scenario_name = sc,
                                date = date,
                                town = town,
                                repetition = k,
                                job_filename = job_filename
                                )
                
                print(f"Job file created: {job_filename}")
                
if __name__ == "__main__":
    main()