import os
import json
import argparse
from tqdm import tqdm
from num2words import num2words

def modify_captions(num_cars, num_bikes, flag_cars, flag_bikes):
    caption = ""
    if num_cars == 0 and num_bikes == 0:
        caption += "No vehicles nearby"
    else:
        if num_cars == 0:
            caption += "There is no car"
        
        elif num_cars > 0:
            caption += f"There"
            
            if num_cars == 1:
                caption += " is"
            else:
                caption += " are"
            
            if flag_cars:
                caption += " more than"
            
            caption += f" {num2words(num_cars)} car"
            
            if num_cars > 1:
                caption += "s"
            
        caption += " and"
        
        if num_bikes == 0:
            caption += " no bike nearby"
        
        elif num_bikes > 0:
            if flag_bikes:
                caption += " more than"
            
            caption += f" {num2words(num_bikes)} bike"
            
            if num_bikes > 1:
                caption += "s"
            
            caption += " nearby"
            
    return caption

def count_unique_captions(town_folder, save_file_path):
    route_folders = os.listdir(town_folder)
    route_folders = [route_folder.strip() for route_folder in route_folders if "." not in route_folder]
    route_folders.sort()
    
    count = 0
    for route_folder in route_folders:
        route_path = os.path.join(town_folder, route_folder)
        gt_data_path = os.path.join(route_path, "ground_truth_data")
        gt_file_paths = [f.path for f in os.scandir(gt_data_path) if f.name.endswith(".json")]
        
        count += len(gt_file_paths)
    
    pbar = tqdm(total = count, desc = "Counting Unique Captions", position = 0, leave = True)
    unique_captions = []
    unique_car_bike = []
    
    for route_folder in route_folders:
        route_path = os.path.join(town_folder, route_folder)
        gt_data_path = os.path.join(route_path, "ground_truth_data")
        gt_file_paths = [f.path for f in os.scandir(gt_data_path) if f.name.endswith(".json")]
        
        for gt_file_path in gt_file_paths:
            with open(gt_file_path, "r") as gt_file:
                gt_data = json.load(gt_file)
            
            caption = gt_data["caption_data"]["caption"]
            car_bike = (gt_data["caption_data"]["num_cars"], gt_data["caption_data"]["num_bikes"])
            
            flag_cars, flag_bikes = False, False
            
            if car_bike[0] > 10:
                car_bike = (10, car_bike[1])
                flag_cars = True
            
            if car_bike[1] > 5:
                car_bike = (car_bike[0], 5)
                flag_bikes = True
            
            updated_caption = modify_captions(car_bike[0], car_bike[1], flag_cars, flag_bikes)
            
            gt_data["caption_data"]["clipped_count"] = car_bike
            gt_data["caption_data"]["clipped_caption"] = updated_caption
            
            if updated_caption not in unique_captions:
                unique_captions.append(updated_caption)
                unique_car_bike.append(car_bike)
            
            # write to the file
            with open(gt_file_path, "w") as gt_file:
                json.dump(gt_data, gt_file, indent = 4)
            
            pbar.update(1)
    pbar.close()
    
    unique_data = {
        "captions": unique_captions,
        "car_bike": unique_car_bike
    }
    with open(save_file_path, "w") as save_file:
        json.dump(unique_data, save_file, indent = 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generate captions for the ground truth data")
    parser.add_argument("--town_folder", default = " ", type = str, 
                        help = "Path to the town folder")
    parser.add_argument("--save_file_path", default = " ", type = str,
                        help = "Path to save the unique captions and the number of cars and bikes")
    args = parser.parse_args()
    
    count_unique_captions(
        town_folder = args.town_folder, 
        save_file_path = args.save_file_path
    )