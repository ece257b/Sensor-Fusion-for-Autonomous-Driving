import os
# import gzip
import laspy
from imgaug import augmenters as ia
# import ujson
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

def generate_and_save_lidar_bev(town_folder, lidar_folder_name, save_folder_name):
    print(f"Generating LiDAR BEV images for {town_folder}")
    route_folders = os.listdir(town_folder)
    route_folders = [route_folder.strip() for route_folder in route_folders if "." not in route_folder]
    route_folders.sort()
    
    count = 0
    for route_folder in route_folders:
        route_path = os.path.join(town_folder, route_folder)
        lidar_data_path = os.path.join(route_path, "lidar")
        lidar_file_paths = [f.path for f in os.scandir(lidar_data_path) if f.name.endswith(".laz")]
        
        count += len(lidar_file_paths)
    
    pbar = tqdm(total = count, desc = "Generating captions", position = 0, leave = True)    
    for route_folder in route_folders:
        route_path = os.path.join(town_folder, route_folder)
    
        lidar_dir = os.path.join(route_path, lidar_folder_name)
        
        lidar_files = [f.path for f in os.scandir(lidar_dir) if f.is_file()]
        lidar_files.sort()
        
        save_folder_path = os.path.join(route_path, save_folder_name)
        if os.path.exists(save_folder_path):
            shutil.rmtree(save_folder_path)
        
        os.makedirs(save_folder_path, exist_ok = True)
        
        for lidar_file in lidar_files:
            idx = int(lidar_file.split("/")[-1].split(".")[0])
            
            read_lidar = laspy.read(lidar_file)
            lidar_points = np.vstack([read_lidar.x, read_lidar.y, read_lidar.z]).T
            
            plt.figure(figsize=(10,10))
            plt.scatter(lidar_points[:, 1], lidar_points[:, 0], s=0.1, c=lidar_points[:, 2])
            plt.xlim(-50, 50)
            plt.ylim(-50, 50)
            plt.axis('off')

            img_path = os.path.join(save_folder_path, f"{idx:04}.png")
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi = 100)
            plt.close()
            
            pbar.update(1)
    pbar.close()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generate captions for the ground truth data")
    parser.add_argument("--town_folder", default = " ", type = str, 
                        help = "Path to the town folder")
    parser.add_argument("--lidar_folder_name", default = "lidar", type = str,
                        help = "Folder where LiDAR data is stored")
    parser.add_argument("--save_folder_name", default = "lidar_bev_img", type = str,
                        help = "Folder where LiDAR data is to be saved")
    args = parser.parse_args()
    
    generate_and_save_lidar_bev(
        town_folder = args.town_folder,
        lidar_folder_name = args.lidar_folder_name,
        save_folder_name = args.save_folder_name
    )


# min_x, max_x, pixels_per_meter, hist_max_per_pixel = -32, 32, 4.0, 5
# max_height_lidar, lidar_split_height = 100.0, 0.2
# min_y, max_y = -32, 32

# def get_lidar_bev_from_path(lidar_path, measurement_path):
#     def normalize_angle(x):
#         x = x % (2 * np.pi)  # force in range [0, 2 pi)
#         if x > np.pi:  # move to [-pi, pi)
#             x -= 2 * np.pi
#         return x
#     def lidar_augmenter(prob=0.2, cutout=False):
#         augmentations = []
#         if cutout:
#             augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False, cval=0.0)))
#         augmenter = ia.Sequential(augmentations, random_order=True)
#         return augmenter
#     def algin_lidar(lidar, translation, yaw):
#         rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
#         aligned_lidar = (rotation_matrix.T @ (lidar - translation).T).T
#         return aligned_lidar
#     def align(lidar_0, measurements_0, measurements_1, y_augmentation=0.0, yaw_augmentation=0):
#         pos_1 = np.array([measurements_1['pos_global'][0], measurements_1['pos_global'][1], 0.0])
#         pos_0 = np.array([measurements_0['pos_global'][0], measurements_0['pos_global'][1], 0.0])
#         pos_diff = pos_1 - pos_0
#         rot_diff = normalize_angle(measurements_1['theta'] - measurements_0['theta'])

#         # Rotate difference vector from global to local coordinate system.
#         rotation_matrix = np.array([[np.cos(measurements_1['theta']), -np.sin(measurements_1['theta']), 0.0],
#                                     [np.sin(measurements_1['theta']),
#                                     np.cos(measurements_1['theta']), 0.0], [0.0, 0.0, 1.0]])
#         pos_diff = rotation_matrix.T @ pos_diff

#         lidar_1 = algin_lidar(lidar_0, pos_diff, rot_diff)

#         pos_diff_aug = np.array([0.0, y_augmentation, 0.0])
#         rot_diff_aug = np.deg2rad(yaw_augmentation)

#         lidar_1_aug = algin_lidar(lidar_1, pos_diff_aug, rot_diff_aug)

#         return lidar_1_aug

#     def lidar_to_histogram_features(lidar, use_ground_plane):
#         def splat_points(point_cloud):
#             # 256 x 256 grid
#             xbins = np.linspace(min_x, max_x,
#                                 (max_x - min_x) * int(pixels_per_meter) + 1)
#             ybins = np.linspace(min_y, max_y,
#                                 (max_y - min_y) * int(pixels_per_meter) + 1)
#             hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
#             hist[hist > hist_max_per_pixel] = hist_max_per_pixel
#             overhead_splat = hist / hist_max_per_pixel
#             return overhead_splat.T

#         # Remove points above the vehicle
#         lidar = lidar[lidar[..., 2] < max_height_lidar]
#         below = lidar[lidar[..., 2] <= lidar_split_height]
#         above = lidar[lidar[..., 2] > lidar_split_height]
#         below_features = splat_points(below)
#         above_features = splat_points(above)
#         if use_ground_plane:
#             features = np.stack([below_features, above_features], axis=-1)
#         else:
#             features = np.stack([above_features], axis=-1)
#         features = np.transpose(features, (2, 0, 1)).astype(np.float32)
#         return features
    
#     func = lidar_augmenter(1.0, False)
#     las_object = laspy.read(lidar_path)
#     lidar = las_object.xyz
#     with gzip.open(measurement_path, 'rt', encoding='utf-8') as f1:
#         measurements = ujson.load(f1)
#     lidar = align(lidar, measurements, measurements)
#     lidar_bev = lidar_to_histogram_features(lidar, use_ground_plane = False)
#     lidar_bev = func(image=np.transpose(lidar_bev, (1, 2, 0)))
#     return np.transpose(lidar_bev, (2, 0, 1))