import os
import json

# Starting the CARLA simulator
def start_carla_bash(job_number, 
                     directories, 
                     carla_streaming_port,
                     carla_world_port, 
                     job_filename):
    requirement_directory = directories['requirement_directory']
    carla_file = os.path.join(directories['carla_root'], "CarlaUE4.sh")
    log_dir = directories['log_dir']
    job_file_save_dir = directories['job_file_save_dir']
    carla_start_save_dir = directories['carla_start_save_dir']
    
    log_file_path = os.path.join(log_dir, f'job{job_number}.log')
    job_file_path = os.path.join(job_file_save_dir, job_filename)
    
    carla_start_file = os.path.join(carla_start_save_dir, f'job{job_number}.sh')
    qsub_template = f"""#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

echo 'running data collection for {job_filename}...'

cd {requirement_directory}
echo 'indide {requirement_directory}...'
echo 'installing requirements...'
bash install_requirements.sh
echo 'requirements installed...'
 
sleep 3s

echo 'starting carla...'
CARLA_PORT={carla_streaming_port}
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 {carla_file} --carla-world-port={carla_world_port} -opengl -nosound -carla-streaming-port=$CARLA_PORT -quality-level=Epic & 

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
touch {log_file_path}
chmod u+x {job_file_path}

bash {job_file_path} > {log_file_path} 2>&1

echo 'job finished...'
"""
    with open(carla_start_file, 'w', encoding='utf-8') as f:
        f.write(qsub_template)


def main():
    requirement_directory = "/radar-imaging-dataset/P2SIF/data-collection/"
    carla_root = "/radar-imaging-dataset/carla_garage_radar/carla/"
    log_dir = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/"
    job_file_save_dir = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/Job_Files"
    carla_start_save_dir = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/Start_Carla_Job_Scripts/"
    
    directories = {
        "requirement_directory": requirement_directory,
        "carla_root": carla_root,
        "log_dir": log_dir,
        "job_file_save_dir": job_file_save_dir,
        "carla_start_save_dir": carla_start_save_dir
    }
        
    carla_streaming_port = 4321
    carla_world_port = 15180
    num_repititions = 3
    
    route_scenario_data_json = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/route_scenario_data.json"
    
    with open(route_scenario_data_json, 'r') as f:
        data = json.load(f)
    
    scenarios = data['scenarios']
    sc_allowed = ["s1", "s3", "s4", "s7", "s8", "s9", "s10"]
    
    job_number = 0
    for sc in scenarios:
        if sc not in sc_allowed:
            continue
        town_list = data[sc]['towns']
        
        for j, town_filename in enumerate(town_list):
            town = town_filename.split("_")[0]
            
            for k in range(num_repititions):
                # job_filename
                job_filename = f"run_{sc}_{town}_repetition_{k}.sh"
                start_carla_bash(job_number, directories, carla_streaming_port, carla_world_port, job_filename)
                print(f"Job file created: {job_filename}")
                job_number += 1

if __name__ == "__main__":
    main()