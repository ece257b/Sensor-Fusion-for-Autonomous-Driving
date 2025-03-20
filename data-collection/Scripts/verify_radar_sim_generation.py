import os

data_dir = "/radar-imaging-dataset/P2SIF/carla_data"
scenario_paths = [f.path for f in os.scandir(data_dir) if f.is_dir()]
scenario_paths.sort()

radar_file_folders = ["sim_radar_front", "sim_radar_right", "sim_radar_back", "sim_radar_left"]

print("Incomplete radar route conversions are:")

# Printing the incomplete radar route conversions
for scenario_path in scenario_paths:
    # Get all the towns
    town_paths = [f.path for f in os.scandir(scenario_path) if f.is_dir()]
    town_paths.sort()
    
    for town_path in town_paths:
        town_folder_path = town_path
        route_paths = [f.path for f in os.scandir(town_folder_path) if f.is_dir()]
        route_paths.sort()
        
        err_routes = []
        
        for route_path in route_paths:
            flag = 0
            num_lidar = len(os.listdir(os.path.join(route_path, "lidar")))
            
            radar_folders_paths = [os.path.join(route_path, radar_file_folder) for radar_file_folder in radar_file_folders]
            exist_bool = [os.path.exists(radar_folder_path) for radar_folder_path in radar_folders_paths]
            if not all(exist_bool):
                flag = 1
            elif num_lidar != len(os.listdir(radar_folders_paths[0])):
                flag = 1
            elif num_lidar != len(os.listdir(radar_folders_paths[1])):
                flag = 1
            elif num_lidar != len(os.listdir(radar_folders_paths[2])):
                flag = 1
            elif num_lidar != len(os.listdir(radar_folders_paths[3])):
                flag = 1
            
            if flag:
                err_routes.append(route_path)
            
        if flag:
            print(f"{scenario_path.split('/')[-1]} - {town_folder_path.split('/')[-1]} - {len(err_routes)}")
            for err in err_routes:
                print(err)
            print("\n")