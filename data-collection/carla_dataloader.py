import os
import ujson
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch 

import cv2
import gzip
import laspy
from imgaug import augmenters as ia

import sys
sys.path.append("./team_code/")
sys.path.append("./team_code/sim_radar_utils/")
sys.path.append("./team_code/e2e_agent_sem_lidar2shenron_package/")
from sim_radar_utils.convert2D_img import convert2D_img_func


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



def algin_lidar(lidar, translation, yaw):
  """
  Translates and rotates a LiDAR into a new coordinate system.
  Rotation is inverse to translation and yaw
  :param lidar: numpy LiDAR point cloud (N,3)
  :param translation: translations in meters
  :param yaw: yaw angle in radians
  :return: numpy LiDAR point cloud in the new coordinate system.
  """

  rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

  aligned_lidar = (rotation_matrix.T @ (lidar - translation).T).T

  return aligned_lidar

def lidar_augmenter(prob=0.2, cutout=False):
  augmentations = []

  if cutout:
    augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False, cval=0.0)))

  augmenter = ia.Sequential(augmentations, random_order=True)

  return augmenter


class CARLA_Data(Dataset):  # pylint: disable=locally-disabled, invalid-name
  """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """


  def __init__(self,
               root,
               radar_cat = 2,
               shared_dict=None,
               rank=0):

    self.data_cache = shared_dict
    
    self.seq_len = 1
    self.images = []
    self.images_augmented = []
    self.semantics = []
    self.semantics_augmented = []
    self.bev_semantics = []
    self.bev_semantics_augmented = []
    self.depth = []
    self.depth_augmented = []
    self.lidars = []
    self.radars = []
    self.radars_back = []
    self.radars_right = []
    self.radars_left = []
    self.boxes = []
    self.future_boxes = []
    self.measurements = []
    self.sample_start = []

    self.temporal_lidars = []
    self.temporal_measurements = []

    self.radar_channels = 1
    self.radar_cat = radar_cat
    
    total_routes = 0
    crashed_routes = 0
    
    self.min_x = -32
    self.max_x = 32
    self.min_y = -32
    self.max_y = 32
    self.pixels_per_meter = 4.0
    self.hist_max_per_pixel = 5
    self.max_height_lidar = 100.0
    self.lidar_split_height = 0.2
    self.lidar_aug_prob = 1.0
    self.use_cutout = False
    self.use_ground_plane = False
    self.lidar_augmenter_func = lidar_augmenter(self.lidar_aug_prob, cutout=self.use_cutout)
    
    self.mask_for_radar = generate_mask(shape = 256, 
                               start_angle = 35, 
                               fov_degrees = 110, 
                               end_mag = 0)
    
    for sub_root in tqdm(root):
    
      # list subdirectories in root
      routes = next(os.walk(sub_root))[1]

      for route in routes:
        route_dir = sub_root + '/' + route

        # if not os.path.isfile(route_dir + '/results.json.gz'):
        #   total_routes += 1
        #   crashed_routes += 1
        #   continue

        # with gzip.open(route_dir + '/results.json.gz', 'rt', encoding='utf-8') as f:
        #   total_routes += 1

        # We skip data where the expert did not achieve perfect driving score
        # if results_route['scores']['score_composed'] < 100.0:
        #   continue

        # perfect_routes += 1

        num_seq = len(os.listdir(route_dir + '/lidar'))

        self.seq_len = 1
        self.carla_fps = 20
        self.data_save_freq = 5
        self.skip_first = int(2.5 * self.carla_fps) // self.data_save_freq
        self.pred_len = int(2.0 * self.carla_fps) // self.data_save_freq
        
        for seq in range(self.skip_first, num_seq - self.pred_len - self.seq_len):
          # load input seq and pred seq jointly
          image = []
          image_augmented = []
          semantic = []
          semantic_augmented = []
          bev_semantic = []
          bev_semantic_augmented = []
          depth = []
          depth_augmented = []
          lidar = []
          radar = [] #this is basically radar front, okay
          radar_back = []
          radar_left = []
          radar_right = []
          box = []
          future_box = []
          measurement = []

          # Loads the current (and past) frames (if seq_len > 1)
          for idx in range(self.seq_len):
            image.append(route_dir + '/rgb' + (f'/{(seq + idx):04}.jpg'))
            image_augmented.append(route_dir + '/rgb_augmented' + (f'/{(seq + idx):04}.jpg'))
            semantic.append(route_dir + '/semantics' + (f'/{(seq + idx):04}.png'))
            semantic_augmented.append(route_dir + '/semantics_augmented' + (f'/{(seq + idx):04}.png'))
            bev_semantic.append(route_dir + '/bev_semantics' + (f'/{(seq + idx):04}.png'))
            bev_semantic_augmented.append(route_dir + '/bev_semantics_augmented' + (f'/{(seq + idx):04}.png'))
            depth.append(route_dir + '/depth' + (f'/{(seq + idx):04}.png'))
            depth_augmented.append(route_dir + '/depth_augmented' + (f'/{(seq + idx):04}.png'))
            lidar.append(route_dir + '/lidar' + (f'/{(seq + idx):04}.laz'))
            radar.append(route_dir + '/radar_data_front_86' + (f'/{(seq + idx):04}.npy'))
            radar_back.append(route_dir + '/radar_data_rear_86' + (f'/{(seq + idx):04}.npy'))
            radar_left.append(route_dir + '/radar_data_left_86' + (f'/{(seq + idx):04}.npy'))
            radar_right.append(route_dir + '/radar_data_right_86' + (f'/{(seq + idx):04}.npy'))
              
            box.append(route_dir + '/boxes' + (f'/{(seq + idx):04}.json.gz'))
          # we only store the root and compute the file name when loading,
          # because storing 40 * long string per sample can go out of memory.

          measurement.append(route_dir + '/measurements')

          self.images.append(image)
          self.images_augmented.append(image_augmented)
          self.semantics.append(semantic)
          self.semantics_augmented.append(semantic_augmented)
          self.bev_semantics.append(bev_semantic)
          self.bev_semantics_augmented.append(bev_semantic_augmented)
          self.depth.append(depth)
          self.depth_augmented.append(depth_augmented)
          self.lidars.append(lidar)
          self.radars.append(radar)
          self.radars_back.append(radar_back)
          self.radars_right.append(radar_right)
          self.radars_left.append(radar_left)
          self.boxes.append(box)
          self.future_boxes.append(future_box)
          self.measurements.append(measurement)
          self.sample_start.append(seq)

    self.images = np.array(self.images).astype(np.bytes_)
    self.images_augmented = np.array(self.images_augmented).astype(np.bytes_)
    self.semantics = np.array(self.semantics).astype(np.bytes_)
    self.semantics_augmented = np.array(self.semantics_augmented).astype(np.bytes_)
    self.bev_semantics = np.array(self.bev_semantics).astype(np.bytes_)
    self.bev_semantics_augmented = np.array(self.bev_semantics_augmented).astype(np.bytes_)
    self.depth = np.array(self.depth).astype(np.bytes_)
    self.depth_augmented = np.array(self.depth_augmented).astype(np.bytes_)
    self.lidars = np.array(self.lidars).astype(np.bytes_)
    self.radars = np.array(self.radars).astype(np.bytes_)
    self.radars_back = np.array(self.radars_back).astype(np.bytes_)
    self.radars_left = np.array(self.radars_left).astype(np.bytes_)
    self.radars_right = np.array(self.radars_right).astype(np.bytes_)
    
    self.boxes = np.array(self.boxes).astype(np.bytes_)
    self.future_boxes = np.array(self.future_boxes).astype(np.bytes_)
    self.measurements = np.array(self.measurements).astype(np.bytes_)

    self.temporal_lidars = np.array(self.temporal_lidars).astype(np.bytes_)
    self.temporal_measurements = np.array(self.temporal_measurements).astype(np.bytes_)
    self.sample_start = np.array(self.sample_start)

  def __len__(self):
    """Returns the length of the dataset. """
    return self.lidars.shape[0]

  def __getitem__(self, index):
    """Returns the item at index idx. """
    # Disable threading because the data loader will already split in threads.
    cv2.setNumThreads(0)
    i = 0
    data = {}

    images = self.images[index]
    images_augmented = self.images_augmented[index]
    semantics = self.semantics[index]
    semantics_augmented = self.semantics_augmented[index]
    bev_semantics = self.bev_semantics[index]
    bev_semantics_augmented = self.bev_semantics_augmented[index]
    depth = self.depth[index]
    depth_augmented = self.depth_augmented[index]
    lidars = self.lidars[index]
    radars = self.radars[index]
    radars_back = self.radars_back[index]
    radars_left = self.radars_left[index]
    radars_right = self.radars_right[index]
    boxes = self.boxes[index]
    future_boxes = self.future_boxes[index]
    measurements = self.measurements[index]
    sample_start = self.sample_start[index]

    # load measurements
    loaded_images = []
    loaded_images_augmented = []
    loaded_semantics = []
    loaded_semantics_augmented = []
    loaded_bev_semantics = []
    loaded_bev_semantics_augmented = []
    loaded_depth = []
    loaded_depth_augmented = []
    loaded_lidars = []
    loaded_radars = []
    loaded_boxes = []
    loaded_future_boxes = []
    loaded_measurements = []

    # Because the strings are stored as numpy byte objects we need to
    # convert them back to utf-8 strings

    # Since we load measurements for future time steps, we load and store them separately
    for i in range(self.seq_len):
      measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')
      if (not self.data_cache is None) and (measurement_file in self.data_cache):
        measurements_i = self.data_cache[measurement_file]
      else:
        with gzip.open(measurement_file, 'rt', encoding='utf-8') as f1:
          measurements_i = ujson.load(f1)

        if not self.data_cache is None:
          self.data_cache[measurement_file] = measurements_i

      loaded_measurements.append(measurements_i)

    end = 0
    start = 0
    for i in range(start, end, 1):
      measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')
      if (not self.data_cache is None) and (measurement_file in self.data_cache):
        measurements_i = self.data_cache[measurement_file]
      else:
        with gzip.open(measurement_file, 'rt', encoding='utf-8') as f1:
          measurements_i = ujson.load(f1)

        if not self.data_cache is None:
          self.data_cache[measurement_file] = measurements_i

      loaded_measurements.append(measurements_i)

    for i in range(self.seq_len):
      cache_key = str(images[i], encoding='utf-8')

      # Retrieve data from the disc cache
      if not self.data_cache is None and cache_key in self.data_cache:
        boxes_i, future_boxes_i, images_i, images_augmented_i, semantics_i, semantics_augmented_i, bev_semantics_i,\
        bev_semantics_augmented_i, depth_i, depth_augmented_i, lidars_i, radars_i = self.data_cache[cache_key]
        images_i = cv2.imdecode(images_i, cv2.IMREAD_UNCHANGED)
        
      # Load data from the disc
      else:
        semantics_i = None
        semantics_augmented_i = None
        bev_semantics_i = None
        bev_semantics_augmented_i = None
        depth_i = None
        depth_augmented_i = None
        images_i = None
        images_augmented_i = None
        lidars_i = None
        radars_i = None
        future_boxes_i = None
        boxes_i = None

        # Load bounding boxes
        las_object = laspy.read(str(lidars[i], encoding='utf-8'))
        lidars_i = las_object.xyz
        # radars_i = np.load(str(radars[i]), encoding='utf-8')
        if self.radar_channels == 2:
          radar_front = convert2D_img_func(np.load(str(radars[i], encoding='utf-8')))
          radar_back = convert2D_img_func(np.load(str(radars_back[i], encoding='utf-8')))
          radars_i = np.stack((radar_front, radar_back), axis=0) #adding channels to the front
        elif self.radar_cat == 1:
          radar_front = convert2D_img_func(np.load(str(radars[i], encoding='utf-8')))
          radar_back = convert2D_img_func(np.load(str(radars_back[i], encoding='utf-8')))
          radar_back = np.rot90(np.rot90(radar_back))
          radar_cat = np.concatenate((radar_front, radar_back), axis=0) 
          
          center_x, center_y = radar_cat.shape[1] // 2, radar_cat.shape[0] // 2
          crop_size = 256
          radars_i = radar_cat[center_y - crop_size // 2:center_y + crop_size // 2,
                          center_x - crop_size // 2:center_x + crop_size // 2]
        elif self.radar_cat == 2:
          radar_front = convert2D_img_func(np.load(str(radars[i], encoding='utf-8')))
          radar_back = convert2D_img_func(np.load(str(radars_back[i], encoding='utf-8')))
          radar_left = convert2D_img_func(np.load(str(radars_left[i], encoding='utf-8')))
          radar_right = convert2D_img_func(np.load(str(radars_right[i], encoding='utf-8')))
          
          # Applying mask to all views
          radar_front = radar_front * self.mask_for_radar
          radar_back = radar_back * self.mask_for_radar
          radar_left = radar_left * self.mask_for_radar
          radar_right = radar_right * self.mask_for_radar
          
          # Rotating to correct orientation
          radar_left = np.rot90(radar_left)
          radar_back = np.rot90(np.rot90(radar_back))
          radar_right = np.rot90(np.rot90(np.rot90(radar_right)))
          
          # Adding all views
          radar_cat = radar_front + radar_left + radar_back + radar_right
          radars_i = radar_cat
          
        else: 
          radars_i = convert2D_img_func(np.load(str(radars[i], encoding='utf-8')))
        # radars_i = np.log(radars_i+1e-10)
        images_i = cv2.imread(str(images[i], encoding='utf-8'), cv2.IMREAD_COLOR)
        images_i = cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB)

        # # Store data inside disc cache
        # if not self.data_cache is None:
        #   # We want to cache the images in jpg format instead of uncompressed, to reduce memory usage
        #   compressed_image_i = None
        #   compressed_image_augmented_i = None
        #   compressed_semantic_i = None
        #   compressed_semantic_augmented_i = None
        #   compressed_bev_semantic_i = None
        #   compressed_bev_semantic_augmented_i = None
        #   compressed_depth_i = None
        #   compressed_depth_augmented_i = None
        #   compressed_lidar_i = None
        #   compressed_radar_i = None

        #   if not self.config.use_plant:
        #     _, compressed_image_i = cv2.imencode('.jpg', images_i)
        #     # _, compressed_radar_i = cv2.imencode('.jpg', radars_i)
        #     compressed_radar_i = radars_i
            
        #     if self.config.use_semantic:
        #       _, compressed_semantic_i = cv2.imencode('.png', semantics_i)
        #     if self.config.use_bev_semantic:
        #       _, compressed_bev_semantic_i = cv2.imencode('.png', bev_semantics_i)
        #     if self.config.use_depth:
        #       _, compressed_depth_i = cv2.imencode('.png', depth_i)
        #     if self.config.augment:
        #       _, compressed_image_augmented_i = cv2.imencode('.jpg', images_augmented_i)
        #       if self.config.use_semantic:
        #         _, compressed_semantic_augmented_i = cv2.imencode('.png', semantics_augmented_i)
        #       if self.config.use_bev_semantic:
        #         _, compressed_bev_semantic_augmented_i = cv2.imencode('.png', bev_semantics_augmented_i)
        #       if self.config.use_depth:
        #         _, compressed_depth_augmented_i = cv2.imencode('.png', depth_augmented_i)

        #     # LiDAR is hard to compress so we use a special purpose format.
        #     header = laspy.LasHeader(point_format=self.config.point_format)
        #     header.offsets = np.min(lidars_i, axis=0)
        #     header.scales = np.array(
        #         [self.config.point_precision, self.config.point_precision, self.config.point_precision])
        #     compressed_lidar_i = io.BytesIO()
        #     with laspy.open(compressed_lidar_i, mode='w', header=header, do_compress=True, closefd=False) as writer:
        #       point_record = laspy.ScaleAwarePointRecord.zeros(lidars_i.shape[0], header=header)
        #       point_record.x = lidars_i[:, 0]
        #       point_record.y = lidars_i[:, 1]
        #       point_record.z = lidars_i[:, 2]
        #       writer.write_points(point_record)

        #     compressed_lidar_i.seek(0)  # Resets file handle to the start

        #   self.data_cache[cache_key] = (boxes_i, future_boxes_i, compressed_image_i, compressed_image_augmented_i,
        #                                 compressed_semantic_i, compressed_semantic_augmented_i,
        #                                 compressed_bev_semantic_i, compressed_bev_semantic_augmented_i,
        #                                 compressed_depth_i, compressed_depth_augmented_i, compressed_lidar_i, compressed_radar_i)

      loaded_images.append(images_i)
      loaded_lidars.append(lidars_i)
      loaded_radars.append(radars_i)
      
      loaded_boxes.append(boxes_i)
      loaded_future_boxes.append(future_boxes_i)

    loaded_temporal_lidars = []
    loaded_temporal_measurements = []
    processed_image = loaded_images[self.seq_len - 1]
    current_measurement = loaded_measurements[self.seq_len - 1]
    aug_rotation = 0.0
    aug_translation = 0.0
    
    data['rgb'] = np.transpose(processed_image, (2, 0, 1))
    
    #unsqueeze the radar to add one more dimension for the channel
    processed_radar = loaded_radars[self.seq_len - 1]
    
    # pdb.set_trace()
    if self.radar_channels > 1:
      data['radar'] = np.log(processed_radar + 1e-10)
    else:
      processed_radar = np.expand_dims(processed_radar, axis=2)  
      data['radar'] = np.log(np.transpose(processed_radar, (2, 0, 1)) + 1e-10)
    
    lidars = []
    for i in range(self.seq_len):
      lidar = loaded_lidars[i]
      # transform lidar to lidar seq-1
      lidar = self.align(lidar,
                         loaded_measurements[i],
                         current_measurement,
                         y_augmentation=aug_translation,
                         yaw_augmentation=aug_rotation)
      lidar_bev = self.lidar_to_histogram_features(lidar, use_ground_plane=self.use_ground_plane)
      lidars.append(lidar_bev)
      
    lidar_bev = np.concatenate(lidars, axis=0)
    lidar_bev = self.lidar_augmenter_func(image=np.transpose(lidar_bev, (1, 2, 0)))
    data['lidar'] = np.transpose(lidar_bev, (2, 0, 1))

    return data

  def align(self, lidar_0, measurements_0, measurements_1, y_augmentation=0.0, yaw_augmentation=0):
    pos_1 = np.array([measurements_1['pos_global'][0], measurements_1['pos_global'][1], 0.0])
    pos_0 = np.array([measurements_0['pos_global'][0], measurements_0['pos_global'][1], 0.0])
    pos_diff = pos_1 - pos_0
    rot_diff = normalize_angle(measurements_1['theta'] - measurements_0['theta'])

    # Rotate difference vector from global to local coordinate system.
    rotation_matrix = np.array([[np.cos(measurements_1['theta']), -np.sin(measurements_1['theta']), 0.0],
                                [np.sin(measurements_1['theta']),
                                 np.cos(measurements_1['theta']), 0.0], [0.0, 0.0, 1.0]])
    pos_diff = rotation_matrix.T @ pos_diff

    lidar_1 = algin_lidar(lidar_0, pos_diff, rot_diff)

    pos_diff_aug = np.array([0.0, y_augmentation, 0.0])
    rot_diff_aug = np.deg2rad(yaw_augmentation)

    lidar_1_aug = algin_lidar(lidar_1, pos_diff_aug, rot_diff_aug)

    return lidar_1_aug

  def lidar_to_histogram_features(self, lidar, use_ground_plane):
    """
    Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
    :param lidar: (N,3) numpy, LiDAR point cloud
    :param use_ground_plane, whether to use the ground plane
    :return: (2, H, W) numpy, LiDAR as sparse image
    """

    def splat_points(point_cloud):
      # 256 x 256 grid
      xbins = np.linspace(self.min_x, self.max_x,
                          (self.max_x - self.min_x) * int(self.pixels_per_meter) + 1)
      ybins = np.linspace(self.min_y, self.max_y,
                          (self.max_y - self.min_y) * int(self.pixels_per_meter) + 1)
      hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
      hist[hist > self.hist_max_per_pixel] = self.hist_max_per_pixel
      overhead_splat = hist / self.hist_max_per_pixel
      # The transpose here is an efficient axis swap.
      # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
      # (x height channel, y width channel)
      return overhead_splat.T

    # Remove points above the vehicle
    lidar = lidar[lidar[..., 2] < self.max_height_lidar]
    below = lidar[lidar[..., 2] <= self.lidar_split_height]
    above = lidar[lidar[..., 2] > self.lidar_split_height]
    below_features = splat_points(below)
    above_features = splat_points(above)
    if use_ground_plane:
      features = np.stack([below_features, above_features], axis=-1)
    else:
      features = np.stack([above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

def get_radar_from_path(route_path, data_idx):
    mask_for_radar = generate_mask(shape = 256,
                                    start_angle = 35,
                                    fov_degrees = 110,
                                    end_mag = 0)
    
    radar_front = convert2D_img_func(np.load(route_path + '/sim_radar_front' + (f'/{data_idx:04}.npy')))
    radar_back = convert2D_img_func(np.load(route_path + '/sim_radar_back' + (f'/{data_idx:04}.npy')))
    radar_left = convert2D_img_func(np.load(route_path + '/sim_radar_left' + (f'/{data_idx:04}.npy')))
    radar_right = convert2D_img_func(np.load(route_path + '/sim_radar_right' + (f'/{data_idx:04}.npy')))
    
    # Applying mask to all views
    radar_front = radar_front * mask_for_radar
    radar_back = radar_back * mask_for_radar
    radar_left = radar_left * mask_for_radar
    radar_right = radar_right * mask_for_radar
    
    # Rotating to correct orientation
    radar_left = np.rot90(radar_left)
    radar_back = np.rot90(np.rot90(radar_back))
    radar_right = np.rot90(np.rot90(np.rot90(radar_right)))
    
    # Adding all views
    radar_cat = radar_front + radar_left + radar_back + radar_right
    return radar_cat

min_x, max_x, pixels_per_meter, hist_max_per_pixel = -32, 32, 4.0, 5
max_height_lidar, lidar_split_height = 100.0, 0.2
min_y, max_y = -32, 32

def get_lidar_bev_from_path(lidar_path, measurement_path):
    func = lidar_augmenter(1.0, False)
    def align(lidar_0, measurements_0, measurements_1, y_augmentation=0.0, yaw_augmentation=0):
        pos_1 = np.array([measurements_1['pos_global'][0], measurements_1['pos_global'][1], 0.0])
        pos_0 = np.array([measurements_0['pos_global'][0], measurements_0['pos_global'][1], 0.0])
        pos_diff = pos_1 - pos_0
        rot_diff = normalize_angle(measurements_1['theta'] - measurements_0['theta'])

        # Rotate difference vector from global to local coordinate system.
        rotation_matrix = np.array([[np.cos(measurements_1['theta']), -np.sin(measurements_1['theta']), 0.0],
                                    [np.sin(measurements_1['theta']),
                                    np.cos(measurements_1['theta']), 0.0], [0.0, 0.0, 1.0]])
        pos_diff = rotation_matrix.T @ pos_diff

        lidar_1 = algin_lidar(lidar_0, pos_diff, rot_diff)

        pos_diff_aug = np.array([0.0, y_augmentation, 0.0])
        rot_diff_aug = np.deg2rad(yaw_augmentation)

        lidar_1_aug = algin_lidar(lidar_1, pos_diff_aug, rot_diff_aug)

        return lidar_1_aug

    def lidar_to_histogram_features(lidar, use_ground_plane):
        """
        Convert LiDAR point cloud into 2-bin histogram over a fixed size grid
        :param lidar: (N,3) numpy, LiDAR point cloud
        :param use_ground_plane, whether to use the ground plane
        :return: (2, H, W) numpy, LiDAR as sparse image
        """

        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(min_x, max_x,
                                (max_x - min_x) * int(pixels_per_meter) + 1)
            ybins = np.linspace(min_y, max_y,
                                (max_y - min_y) * int(pixels_per_meter) + 1)
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > hist_max_per_pixel] = hist_max_per_pixel
            overhead_splat = hist / hist_max_per_pixel
            # The transpose here is an efficient axis swap.
            # Comes from the fact that carla is x front, y right, whereas the image is y front, x right
            # (x height channel, y width channel)
            return overhead_splat.T

        # Remove points above the vehicle
        lidar = lidar[lidar[..., 2] < max_height_lidar]
        below = lidar[lidar[..., 2] <= lidar_split_height]
        above = lidar[lidar[..., 2] > lidar_split_height]
        below_features = splat_points(below)
        above_features = splat_points(above)
        if use_ground_plane:
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)
        return features
    
    las_object = laspy.read(lidar_path)
    lidar = las_object.xyz
    with gzip.open(measurement_path, 'rt', encoding='utf-8') as f1:
        measurements = ujson.load(f1)
    lidar = align(lidar, measurements, measurements)
    lidar_bev = lidar_to_histogram_features(lidar, use_ground_plane = False)
    lidar_bev = func(image=np.transpose(lidar_bev, (1, 2, 0)))
    return np.transpose(lidar_bev, (2, 0, 1))