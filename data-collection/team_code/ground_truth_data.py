def convert_waypoints(waypoints):
    waypoints = [wp.transform for wp in waypoints]
    waypoints = [{'x': wp.location.x, 'y': wp.location.y, 'z': wp.location.z, 'pitch': wp.rotation.pitch, 'yaw': wp.rotation.yaw, 'roll': wp.rotation.roll} for wp in waypoints]
    return waypoints

def convert_bbox(bbox):
    bbox = [{'loc_x': bb.location.x, 'loc_y': bb.location.y, 'loc_z': bb.location.z, 'ext_x': bb.extent.x, 'ext_y': bb.extent.y, 'ext_z': bb.extent.z, 'pitch': bb.rotation.pitch, 'yaw': bb.rotation.yaw, 'roll': bb.rotation.roll} for bb in bbox]
    return bbox

'''
Structure of log_data:
- vehicle:
    - id
    - location
    - transform
    - bounding_box
    - velocity
    - acceleration
    - is_at_traffic_light
    - traffic_light_state
    - vehicle_type
- walker:
    - id
    - location
    - transform
    - bounding_box
    - velocity
    - acceleration
- traffic:
    - id
    - location
    - transform
    - bounding_box
    - type
    - state
    - elapsed_time
    - affected_lane_waypoints
    - stop_waypoints
    - light_boxes
    - speed_limit
'''

# log_data_struct = {'vehicle': ['id', 'location', 'transform', 'bounding_box', 'velocity', 'acceleration', 'is_at_traffic_light', 'traffic_light_state', 'vehicle_type'], 
#                    'walker': ['id', 'location', 'transform', 'bounding_box', 'velocity', 'acceleration'],
#                    'traffic': ['id', 'location', 'transform', 'bounding_box', 'type', 'state', 'elapsed_time', 'affected_lane_waypoints', 'stop_waypoints', 'light_boxes', 'speed_limit']}

def get_ground_truth_data(actors):
    log_data = {'vehicle': [], 'walker': [], 'traffic': []}
    avoid_logging = ['sensor', 'spectator', 'controller', 'static']

    for actor in actors:
        # attrs = actor.__dir__()
        # attrs = [ax for ax in attrs if "__" not in ax]

        data = {}
        data['id'] = actor.id
        actor_id_type = actor.type_id.split('.')
        
        actor_class = actor_id_type[0]
        if actor_class in avoid_logging:
            continue
        
        loc = actor.get_location()
        transform = actor.get_transform()
        data['location'] = {'x': loc.x, 'y': loc.y, 'z': loc.z}
        data['transform'] = {'pitch': transform.rotation.pitch, 'yaw': transform.rotation.yaw, 'roll': transform.rotation.roll}
        
        # if not actor.is_alive:
        #     print(f"Actor not alive: {actor_id_type}")

        if actor_class in ['vehicle', 'walker']:
            vel = actor.get_velocity()
            acc = actor.get_acceleration()
            data['velocity'] = {'x': vel.x, 'y': vel.y, 'z': vel.z}
            data['acceleration'] = {'x': acc.x, 'y': acc.y, 'z': acc.z}
            
            if actor_class == 'vehicle':
                data['is_at_traffic_light'] = actor.is_at_traffic_light()
                data['traffic_light_state'] = str(actor.get_traffic_light_state()).split('.')[-1]
                bbox = convert_bbox([actor.bounding_box])
                data['bounding_box'] = bbox
                data['vehicle_description'] = actor_id_type
            log_data[actor_class].append(data)
            
        elif actor_class == 'traffic':
            actor_type = actor_id_type[1]
            if actor_type == 'unknown':
                continue
            data['type'] = actor_type
            if actor_type == 'traffic_light':
                data['state'] = actor.get_state()
                data['elapsed_time'] = actor.get_elapsed_time()
                # data['affected_lane_waypoints'] = convert_waypoints(actor.get_affected_lane_waypoints())
                # data['stop_waypoints'] = convert_waypoints(actor.get_stop_waypoints())
                # data['light_boxes'] = convert_bbox(actor.get_light_boxes())
            elif actor_type == 'speed_limit':
                data['speed_limit'] = actor_id_type[-1]
            # else:
            #     print(actor_type)
            log_data['traffic'].append(data)

        else:
            print("Outsider: ", actor_class)
            print(actor_class, actor_id_type)
    return log_data