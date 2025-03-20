import os
import cv2
import json
import laspy
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from num2words import num2words

import sys
sys.path.append("./sim_radar_utils")
sys.path.append("./e2e_agent_sem_lidar2shenron_package")
from sim_radar_utils.convert2D_img import convert2D_img_func

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

def convert_angle_degree_to_pixel(angle_degrees, in_pixels, angle = None):
    if angle == "radian":
        return (np.degrees(angle_degrees) / 180) * (in_pixels / 2) + in_pixels / 2
    return int((angle_degrees / 180) * (in_pixels / 2) + in_pixels / 2)

def cart2polar_for_mask(x, y, in_pixels):
    # Don't worry about range because all are going to be one
    r = np.sqrt(x ** 2 + y ** 2)
    
    # Assuming uniform theta resolution
    theta = np.arctan2(y, x)
    theta_px = convert_angle_degree_to_pixel(theta, in_pixels, angle = "radian")
    return r, theta_px

def generate_mask(shape, start_angle, fov_degrees, overlap_mag = 0.5, end_mag = 0.5):
    # Origin at the center
    X, Y = np.meshgrid(np.linspace(-shape / 2, shape / 2, shape), np.linspace(-shape / 2, shape / 2, shape))
    # Rotating the axes to make the plane point upwards
    X = np.rot90(X)
    Y = np.rot90(Y)
    
    R, theta = cart2polar_for_mask(X, Y, shape)
    
    R = R.astype(int)
    theta = theta.astype(int)
    
    mask_polar = np.zeros((shape, shape))
    
    a = convert_angle_degree_to_pixel(-start_angle, shape)
    b = convert_angle_degree_to_pixel(start_angle, shape)
    
    mask_polar[:, a : b] = 1
    
    fov_pixels_a = convert_angle_degree_to_pixel(-fov_degrees / 2, shape)
    fov_pixels_b = convert_angle_degree_to_pixel(fov_degrees / 2, shape)
    
    mask_polar[:, fov_pixels_a : a] = np.linspace(end_mag, 1, a - fov_pixels_a).reshape(1, a - fov_pixels_a)
    mask_polar[:, b : fov_pixels_b] = np.linspace(1, end_mag, fov_pixels_b - b).reshape(1, fov_pixels_b - b)
    
    mask_cartesian = mask_polar[R, theta]
    
    return mask_cartesian

class CARLA_Dataset(Dataset):
    def __init__(
        self, 
        route_paths,
        radar_cat = 2,
        shared_cache = None,
        use_cache = False
    ):
        self.route_paths = route_paths
        self.radar_cat = radar_cat
        self.data_cache = shared_cache
        self.use_cache = use_cache
        
        if self.radar_cat == 1:        
            self.mask = generate_mask(shape = 256, 
                                      start_angle = 90, 
                                      fov_degrees = 180, 
                                      end_mag = 0)
        else:
            self.mask = generate_mask(shape = 256, 
                                      start_angle = 35, 
                                      fov_degrees = 110, 
                                      end_mag = 0)

        self.camera_paths = []
        self.lidar_paths = []
        self.lidar_pcd_paths = []
        self.radar_paths = []
        self.ground_truth_data_paths = []
        
        radar_view_folders = ["sim_radar_front", "sim_radar_back", "sim_radar_left", "sim_radar_right"]
        
        for route_path in self.route_paths:
            self.camera_paths += [f.path for f in os.scandir(os.path.join(route_path, "rgb"))]
            self.lidar_pcd_paths += [f.path for f in os.scandir(os.path.join(route_path, "lidar"))]
            self.lidar_paths += [f.path for f in os.scandir(os.path.join(route_path, "lidar_bev_img"))]
            self.ground_truth_data_paths += [f.path for f in os.scandir(os.path.join(route_path, "ground_truth_data"))]
            
            tmp_radar = []
            for view in radar_view_folders:
                radar_path = [f.path for f in os.scandir(os.path.join(route_path, view))]
                radar_path.sort()
                tmp_radar.append(radar_path)
            
            # Need to create a tuple of four entries, one for each view and finally have a list of these tuples
            self.radar_paths += list(zip(*tmp_radar))
        
    def __len__(self):
        return len(self.camera_paths)
    
    def __getitem__(self, index):
        # print(index)
        cv2.setNumThreads(0)
        image_path = self.camera_paths[index]
        lidar_path = self.lidar_paths[index]
        # lidar_pcd_path = self.lidar_pcd_paths[index]
        radar_path = self.radar_paths[index]
        ground_truth_data_path = self.ground_truth_data_paths[index]
        
        cache_key = image_path
        # If data is already in cache, return it
        if self.data_cache is not None and cache_key in self.data_cache and self.use_cache:
            image, lidar_bev_image, radar, car_bike, captions = self.data_cache[cache_key] 
            
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            image_tensor = torch.tensor(image).permute(2, 0, 1)
            
            lidar_bev_image = cv2.imdecode(lidar_bev_image, cv2.IMREAD_UNCHANGED)
            lidar_bev_image_tensor = torch.tensor(lidar_bev_image).permute(2, 0, 1)
            
            data = {
                "image": image_tensor,
                "lidar_bev": lidar_bev_image_tensor,
                "radar": radar,
                "car_bike": car_bike,
                "caption": captions
            }
            return data
        
        # Otherwise, load the data and save in cache
        else:
            image = cv2.imread(image_path)
            image_tensor = torch.tensor(image).permute(2, 0, 1)
            
            lidar_bev_image = cv2.imread(lidar_path)
            lidar_bev_image_tensor = torch.tensor(lidar_bev_image).permute(2, 0, 1)
            
            radar_front = convert2D_img_func(np.load(radar_path[0])) * self.mask
            radar_back = convert2D_img_func(np.load(radar_path[1])) * self.mask
            radar_left = convert2D_img_func(np.load(radar_path[2])) * self.mask
            radar_right = convert2D_img_func(np.load(radar_path[3])) * self.mask
            
            radar_left = np.rot90(radar_left)
            radar_back = np.rot90(np.rot90(radar_back))
            radar_right = np.rot90(np.rot90(np.rot90(radar_right)))
            
            if self.radar_cat == 1:
                radar = radar_front + radar_back
            elif self.radar_cat == 2:
                radar = radar_front + radar_back + radar_left + radar_right
            else:
                raise ValueError("Invalid radar concatenation")
            
            radar = np.expand_dims(radar, axis = 2)
            radar = np.log(np.transpose(radar, (2, 0, 1)) + 1e-12)
            
            # try:
            with open(ground_truth_data_path) as f:
                ground_truth_data = json.load(f)
            caption_data = ground_truth_data["caption_data"]
            num_cars, num_bikes = caption_data["num_cars"], caption_data["num_bikes"]
            captions = caption_data["caption"]
            flag_cars, flag_bikes = False, False
            if num_cars > 10:
                num_cars = 10
                flag_cars = True
            if num_bikes > 5:
                flag_bikes = True
                num_bikes = 5
            
            captions = modify_captions(num_cars, num_bikes, flag_cars, flag_bikes)
            car_bike = (num_cars, num_bikes)
                
            data = {
                "image": image_tensor,
                "lidar_bev": lidar_bev_image_tensor,
                "radar": radar,
                "car_bike": car_bike,
                "caption": captions
            }
            if self.use_cache:
                image_cache = cv2.imencode(".jpg", image)
                lidar_bev_image_cache = cv2.imencode(".jpg", lidar_bev_image)
                
                self.data_cache[cache_key] = (image_cache, lidar_bev_image_cache, radar, car_bike, captions)
        return data


def generate_carla_dataset(data_root, test_size = 0.3, num_repetitions = 1, shared_cache = None, load_only_train = False):
    route_dirs = []
    scenario_paths = [f.path for f in os.scandir(data_root)]
    scenario_paths.sort()
    
    for scenario_path in scenario_paths:
        if "zip" in scenario_path:
            continue
        town_paths = [f.path for f in os.scandir(scenario_path)]
        town_paths.sort()
        
        for town_path in town_paths:
            repetition_num = int(town_path.split("_")[-1][3 : ])
            if repetition_num >= num_repetitions:
                continue
            
            routes = os.listdir(town_path)
            route_paths = [os.path.join(town_path, route) for route in routes if "." not in route]
            route_paths.sort()
            
            route_dirs += route_paths
    
    test_len = int(len(route_dirs) * test_size)
    
    # Shuffle the data
    indices = np.arange(len(route_dirs))
    np.random.shuffle(indices)
    
    train_indices = indices[test_len : ]
    test_indices = indices[ : test_len]
    
    test_routes = [route_dirs[i] for i in test_indices]
    train_routes = [route_dirs[i] for i in train_indices]
    
    train_dataset, test_dataset = None, None
    train_dataset = CARLA_Dataset(train_routes, shared_cache = shared_cache)
    
    if not load_only_train:
        test_dataset = CARLA_Dataset(test_routes, shared_cache = shared_cache)
    
    return train_dataset, test_dataset