import sys
import os
sys.path.append("../")
import numpy as np
# from path_config import *
from ConfigureRadar import radar
from shenron.Sceneset import *
from shenron.heatmap_gen_fast import *
import scipy.io as sio
from lidar_utils import *
import shutil
from tqdm import tqdm
import laspy

def file_not_found(route_folder, err_lidar_file):
    folders = os.listdir(route_folder)
    folder_paths = [os.path.join(route_folder, f) for f in folders if "." not in f]
    file_number = int(err_lidar_file.split(".")[0])
    
    txt_file = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/rm.txt"
    for folder_path in folder_paths:
        # remove the files after the missing file
        files = os.listdir(folder_path)
        files.sort()
        for file in files:
            if int(file.split(".")[0]) >= file_number:
                os.remove(os.path.join(folder_path, file))
        
        with open(txt_file, "a") as f:
            f.write(f"{folder_path}\n")
    return

def map_carla_semantic_lidar_latest(carla_sem_lidar_data):
    '''
    Function to map material column in the collected carla ray_cast_shenron to shenron input 
    '''
    carla_sem_lidar_data_crop = carla_sem_lidar_data[:, (0, 1, 2, 5)]
    temp_list = np.array([0, 4, 2, 0, 11, 5, 0, 0, 1, 8, 12, 3, 7, 10, 0, 1, 0, 12, 6, 0, 0, 0, 0])
    
    col = temp_list[(carla_sem_lidar_data_crop[:, 3].astype(int))]
    carla_sem_lidar_data_crop[:, 3] = col
    
    return carla_sem_lidar_data_crop

def check_save_path(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return

def rotate_points(points, angle):
    rotMatrix = np.array([[np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0]
        , [- np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0]
        , [0, 0, 1]])
    return np.matmul(points, rotMatrix)

def Cropped_forRadar(pc, veh_coord, veh_angle, radarobj):
    """
    Removes Occlusions and calculates loss for each point
    """

    skew_pc = rotate_points(pc[:, 0:3] , veh_angle )
    skew_pc = np.vstack(((skew_pc ).T, pc[:, 3], pc[:, 5],pc[:,6])).T  #x,y,z,speed,material, cosines

    rowy = np.where((skew_pc[:, 1] > 0.8))
    new_pc = skew_pc[rowy, :].squeeze(0)

    new_pc = new_pc[new_pc[:,4]!=0]

    new_pc = new_pc[(new_pc[:,0]<50)*(new_pc[:,0]>-50)]
    new_pc = new_pc[(new_pc[:,1]<100)]
    new_pc = new_pc[(new_pc[:,2]<2)]

    simobj = Sceneset(new_pc)

    [rho, theta, loss, speed, angles] = simobj.specularpoints(radarobj)
    return rho, theta, loss, speed, angles

def run_lidar(sim_config, route_folder, out_path):

    #restructed lidar.py code
    lidar_path = f'{route_folder}/{sim_config["CARLA_SHENRON_SEM_LIDAR"]}'
    
    # setting the sem lidar inversion angle here
    veh_angle = sim_config['INVERT_ANGLE']

    shutil.copyfile('ConfigureRadar.py',f'{route_folder}/radar_params.py')

    lidar_files = os.listdir(lidar_path)
    lidar_files.sort()
    
    #Lidar specific settings
    radarobj = radar(sim_config["RADAR_TYPE"])
    radarobj.center = np.array([0.0, 0.0])  # center of radar
    radarobj.elv = np.array([0.0])
    
    pbar = tqdm(total = len(lidar_files), desc = 'Processing', position = 0, leave = True)
    for lidar_file in lidar_files:
        lidar_file_path = os.path.join(lidar_path, lidar_file)
        
        # load las file 
        try:
            las = laspy.read(lidar_file_path)
        except:
            file_not_found(route_folder, lidar_file)
            continue
        
        # Grab a numpy dataset of our clustering dimensions:
        dataset = np.vstack((las.x, las.y, las.z, las.cosine, las.index, las.sem_tag)).transpose()
        cosines = dataset[:,3]
        load_pc = dataset
        
        load_pc = map_carla_semantic_lidar_latest(load_pc)
        test = new_map_material(load_pc)
        
        points = np.zeros((np.shape(test)[0], 7))
        points[:, [0, 1, 2]] = test[:, [1, 0, 2]]

        """
        points mapping
        +ve ind 0 == right
        +ve ind 1 == +ve depth
        +ve ind 2 == +ve height
        """
        
        points[:, 5] = test[:, 3]
        points[:, 6] = cosines
        
        Crop_rho, Crop_theta, Crop_loss, Crop_speed, Crop_angles = Cropped_forRadar(points, np.array([0, 0, 0]), veh_angle, radarobj)        

        adc_data = heatmap_gen(Crop_rho, Crop_theta, Crop_loss, Crop_speed, radarobj, 1, 0)
        np.save(f'{out_path}/{lidar_file[:-4]}', adc_data)
        pbar.update(1)
    pbar.close()
    
    return

if __name__ == '__main__':

    points = np.zeros((100,6))

    points[:,5] = 4
    
    points[:,0] = 1
    points[:,1] = np.linspace(0,15,100)
    
    points[:,3] = -0.5*np.cos(np.arctan2(points[:,0],points[:,1]))
    radarobj = radar('radarbook')
    # radarobj.chirps = 128
    radarobj.center = np.array([0.0, 0.0])  # center of radar
    radarobj.elv = np.array([0.0])

    Crop_rho, Crop_theta, Crop_loss, Crop_speed = Cropped_forRadar(points, np.array([0, 0, 0]), 0, radarobj)
    Crop_loss = np.ones_like(Crop_loss)
    adc_data = heatmap_gen(Crop_rho, Crop_theta, Crop_loss, Crop_speed, radarobj, 1, 0)
    diction = {"adc_data": adc_data}
    sio.savemat(f"test_pc.mat", diction)
