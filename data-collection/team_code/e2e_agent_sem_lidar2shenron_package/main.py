# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:00:26 2021

@author: ksban
"""

from lidar import run_lidar
import os
import yaml
import argparse

def run_shenron(sim_config, town_folder_path):
    town_folder_path = town_folder_path.strip()
    
    route_folders = os.listdir(town_folder_path)
    route_folders = [route_folder.strip() for route_folder in route_folders if "." not in route_folder]
    route_folders.sort()
    
    radar_out_folder = sim_config["RADAR_PATH_SIMULATED"]
    invert_angle_list = [0, 90, 180, 270]
    name = ["_front", "_right", "_back", "_left"]
    
    cnt = 0
    total = len(route_folders) * len(invert_angle_list)
    for i, invert_angle in enumerate(invert_angle_list):
        radar_out_folder = sim_config["RADAR_PATH_SIMULATED"] + name[i]
    
        for route_folder in route_folders:
            sim_config["INVERT_ANGLE"] = invert_angle
            print(f"Iteration {cnt + 1} / {total}: Invert Angle - {invert_angle} for {route_folder}")
            
            exec_path = os.path.join(town_folder_path, route_folder)
            out_path = os.path.join(exec_path, radar_out_folder)
            os.makedirs(out_path, exist_ok = True)
            
            if sim_config["INPUT"] == "lidar":
                run_lidar(sim_config, exec_path, out_path)
            else:
                print("Incorrect input in config")
            cnt += 1
        
def main():
    parser = argparse.ArgumentParser(description='process the base folder to run shenron on')
    parser.add_argument('--town_folder', default=" ", type=str,
                    help='base folder to run shenron on')
    args = parser.parse_args()
    
    if args.town_folder == " ":
        print("please provide the base folder to run shenron on")
        return
    else:
        with open('simulator_configs.yaml', 'r') as f:
            sim_config = yaml.safe_load(f)
    
        town_folder_path = args.town_folder
        
        run_shenron(sim_config, town_folder_path)
    
if __name__ == '__main__':
    main()