import os

def count_caption_script(directories,
                              job_number):
    requirement_directory = directories['requirement_directory']
    unique_cap_file_dir_path = directories['unique_cap_file_dir_path']
    town_folder_path = directories['town_folder_path']
    log_file_path = os.path.join(directories['log_file_path'], f"job{job_number}.log")
    caption_save_file_path = os.path.join(directories['caption_save_file_path'], f"job{job_number}.json")
    save_file_path = directories['save_file_path']
    
    template = f"""#!/bin/bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "Job started: $dt"

cd {requirement_directory}
echo 'indide {requirement_directory}...'
echo 'installing requirements...'
bash install_requirements_captions.sh
echo 'requirements installed...'
 
sleep 3s

echo 'running caption generation...'

cd {unique_cap_file_dir_path}

touch {log_file_path}
touch {caption_save_file_path}

echo 'running for {town_folder_path}...'

python3 count_unique_captions.py --town_folder {town_folder_path} --save_file_path {caption_save_file_path} > {log_file_path} 2>&1

echo 'caption generation finished...'
"""
    gen_caption_file_path = os.path.join(save_file_path, f'job{job_number}.sh')
    with open(gen_caption_file_path, 'w', encoding='utf-8') as f:
        f.write(template)
    return

data_dir = "/radar-imaging-dataset/P2SIF/carla_data"

# Get all the scenarios
scenario_paths = [f.path for f in os.scandir(data_dir) if f.is_dir()]
scenario_paths.sort()

requirement_directory = "/radar-imaging-dataset/P2SIF/data-collection/"
unique_cap_file_dir_path = "/radar-imaging-dataset/P2SIF/lidar-radar-clip/"
log_file_path = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/logs/"
caption_save_file_path = "/radar-imaging-dataset/P2SIF/lidar-radar-clip/caption/"
save_file_path = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/Unique_Caption_Scripts/"

job_number = 0
for scenario_path in scenario_paths:
    # Get all the towns
    town_paths = [f.path for f in os.scandir(scenario_path) if f.is_dir()]
    town_paths.sort()
    
    for town_path in town_paths:
        town_folder_path = town_path
        directories = {'requirement_directory': requirement_directory,
                        'unique_cap_file_dir_path': unique_cap_file_dir_path,
                        'town_folder_path': town_folder_path,
                        'log_file_path': log_file_path,
                        'caption_save_file_path': caption_save_file_path,
                        'save_file_path': save_file_path
                        }
        count_caption_script(directories, job_number)
        print(f"job{job_number}.sh created")
        job_number += 1
print("All generate caption scripts generated")