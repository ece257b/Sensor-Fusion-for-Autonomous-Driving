import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from num2words import num2words

def print_json(json_data):
    print(json.dumps(json_data, indent = 4, sort_keys = True))

def get_xyz(data):
    return np.array([data['x'], -data['y'], data['z']])

def get_transform(data):
    return np.array([data['pitch'], -data['yaw'], data['roll']])

def measure_distance(a, b):
    return np.linalg.norm(a - b)

def get_relative_vector(a, b):
    return b - a / measure_distance(a, b)

def get_forward_vector(transform):
    pitch = np.radians(transform[0])
    yaw = np.radians(transform[1])

    return np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch)
    ])

attributes = {
    "vehicle": [
        "id", 
        "location", 
        "transform", 
        "bounding_box", 
        "velocity", 
        "acceleration", 
        "is_at_traffic_light", 
        "traffic_light_state", 
        "vehicle_type"
    ], 
    
    "walker": [
        "id",
        "location",
        "transform",
        "bounding_box",
        "velocity",
        "acceleration"
    ],
    
    "traffic": [
        "id",
        "location",
        "transform",
        "bounding_box",
        "type",
        "state",
        "elapsed_time",
        "affected_lane_waypoints",
        "stop_waypoints",
        "light_boxes",
        "speed_limit"
    ]
}

def count_vehicles(data, ego_data, distance_threshold, bike_type_list):
    ego_id = ego_data["id"]
    ego_location = get_xyz(ego_data["location"])
    
    bike_list, car_list = [], []
    
    for vehicle in data["vehicle"]:
        if vehicle["id"] == ego_id:
            continue
        
        vehicle_loc = get_xyz(vehicle["location"])
        if measure_distance(ego_location, vehicle_loc) > distance_threshold:
            continue
        
        vehicle_desc = vehicle["vehicle_description"]
        if vehicle_desc[1] in bike_type_list or vehicle_desc[2] in bike_type_list:
            bike_list.append(vehicle)
        else:
            car_list.append(vehicle)
    
    return car_list, bike_list
        
bike_type_list = [
    "yamaha", 
    "crossbike", 
    "kawasaki", 
    "harley-davidson", 
    "omafiets", 
    "diamondback", 
    "vespa"
    ]

def generate_caption(town_folder,
                     distance_threshold = 40, 
                     modality = "lidar"):
    
    route_folders = os.listdir(town_folder)
    route_folders = [route_folder.strip() for route_folder in route_folders if "." not in route_folder]
    route_folders.sort()
    
    count = 0
    for route_folder in route_folders:
        route_path = os.path.join(town_folder, route_folder)
        gt_data_path = os.path.join(route_path, "ground_truth_data")
        gt_file_paths = [f.path for f in os.scandir(gt_data_path) if f.name.endswith(".json")]
        
        count += len(gt_file_paths)
    
    pbar = tqdm(total = count, desc = "Generating captions", position = 0, leave = True)
    for route_folder in route_folders:
        route_path = os.path.join(town_folder, route_folder)
        gt_data_path = os.path.join(route_path, "ground_truth_data")
        gt_file_paths = [f.path for f in os.scandir(gt_data_path) if f.name.endswith(".json")]
        
        ego_id_txt = os.path.join(route_path, "ego_vehicle_id.txt")
        
        try:
            with open(ego_id_txt, "r") as ego_file:
                ego_id = int(ego_file.read().strip())
        except:
            pbar.update(len(gt_file_paths))
            print(f"Error reading {ego_id_txt}")
            continue
        
        for gt_file_path in gt_file_paths:
            try:
                with open(gt_file_path, "r") as gt_file:
                    gt_data = json.load(gt_file)
            except:
                print(f"Error reading {gt_file_path}")
                continue
            
            if "caption_data" in gt_data.keys():
                if gt_data["caption_data"] is not None:
                    pbar.update(1)
                    continue
            
            for veh in gt_data["vehicle"]:
                if veh["id"] == ego_id:
                    ego_data = veh
                    break
            
            car_list, bike_list = count_vehicles(gt_data, ego_data, distance_threshold, bike_type_list)
            
            num_cars, num_bikes = len(car_list), len(bike_list)
            num_vehicles = num_cars + num_bikes
            
            if num_cars == 0 and num_bikes == 0:
                caption = "No vehicles nearby"
            else:
                caption = ""
                if num_cars > 0:
                    caption += f"There are {num2words(len(car_list))} car"
                    if num_cars > 1:
                        caption += "s"
                    if num_bikes > 0:
                        caption += " and "
                    else:
                        caption += " nearby"
                if num_bikes > 0:
                    if num_cars > 0:
                        caption += f"{num2words(len(bike_list))} bike"
                        if num_bikes > 1:
                            caption += "s"
                        caption += " nearby"
                    else:
                        caption += f"There are {num2words(len(bike_list))} bike"
                        if num_bikes > 1:
                            caption += "s"
                        caption += " nearby"
            
            gt_data["caption_data"] = {
                "num_cars": num_cars,
                "num_bikes": num_bikes,
                "num_vehicles": num_vehicles,
                "caption": caption
            }
            
            # write back to the same file
            with open(gt_file_path, "w") as gt_file:
                json.dump(gt_data, gt_file, indent = 4)
            pbar.update(1)
        print(f"Completed: {route_folder}")
    pbar.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generate captions for the ground truth data")
    parser.add_argument("--town_folder", default = " ", type = str, 
                        help = "Path to the town folder")
    parser.add_argument("--distance_threshold", default = 40, type = int,
                        help = "Distance threshold to consider vehicles nearby")
    parser.add_argument("--modality", default = "lidar", type = str,
                        help = "Modality used to generate the ground truth data")
    args = parser.parse_args()
    
    if args.town_folder == " ":
        print("Please provide the path to the town folder")
    else:
        generate_caption(
            town_folder = args.town_folder,
            distance_threshold = args.distance_threshold,
            modality = args.modality
        )