import os

def generate_radar_sim_script(directories,
                              job_number):
    requirement_directory = directories['requirement_directory']
    main_file_dir_path = directories['main_file_dir_path']
    town_folder_path = directories['town_folder_path']
    log_file_path = os.path.join(directories['log_file_path'], f"job{job_number}.log")
    save_file_path = directories['save_file_path']
    
    template = f"""#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

cd {requirement_directory}
echo 'indide {requirement_directory}...'
echo 'installing requirements...'
bash install_requirements.sh
echo 'requirements installed...'
 
sleep 3s

echo 'running radar simulation...'

cd {main_file_dir_path}

touch {log_file_path}

python3 main.py --town_folder {town_folder_path} > {log_file_path} 2>&1

echo 'radar simulation finished...'
"""
    radar_sim_file_path = os.path.join(save_file_path, f'job{job_number}.sh')
    with open(radar_sim_file_path, 'w', encoding='utf-8') as f:
        f.write(template)
    return

data_dir = "/radar-imaging-dataset/P2SIF/carla_data"

# Get all the scenarios
scenario_paths = [f.path for f in os.scandir(data_dir) if f.is_dir()]
scenario_paths.sort()

requirement_directory = "/radar-imaging-dataset/P2SIF/data-collection/"
main_file_dir_path = "/radar-imaging-dataset/P2SIF/data-collection/team_code/e2e_agent_sem_lidar2shenron_package/"
log_file_path = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/logs"
save_file_path = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/Radar_Sim_Scripts"

job_number = 0
for scenario_path in scenario_paths:
    # Get all the towns
    town_paths = [f.path for f in os.scandir(scenario_path) if f.is_dir()]
    town_paths.sort()
    
    for town_path in town_paths:
        town_folder_path = town_path
        directories = {'requirement_directory': requirement_directory,
                        'main_file_dir_path': main_file_dir_path,
                        'town_folder_path': town_folder_path,
                        'log_file_path': log_file_path,
                        'save_file_path': save_file_path
                        }
        generate_radar_sim_script(directories, job_number)
        print(f"job{job_number}.sh created")
        job_number += 1
print("All radar simulation scripts generated")