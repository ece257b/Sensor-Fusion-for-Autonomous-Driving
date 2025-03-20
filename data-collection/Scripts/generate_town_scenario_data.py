import os
import json
import numpy as np

save_filename = "/radar-imaging-dataset/P2SIF/data-collection/Scripts/route_scenario_data.json"
with open(save_filename, 'r') as f:
    route_scenario_data = json.load(f)

# Note that this is only specific to training routes from NEAT paper
# Other routes may have different town and scenario names
data_path = "/radar-imaging-dataset/P2SIF/data-collection/leaderboard/data/training/"
for sc in route_scenario_data['scenarios']:
    route_data_path = os.path.join(data_path, "routes", sc)
    scenario_data_path = os.path.join(data_path, "scenarios", sc)

    towns, scenarios = [], []
    # check for existence of directory
    if os.path.exists(route_data_path):
        towns = [x for x in np.sort(os.listdir(route_data_path))]
    
    if os.path.exists(scenario_data_path):
        scenarios = [x for x in np.sort(os.listdir(scenario_data_path))]
    
    route_scenario_data[sc] = {'towns': towns, 'scenarios': scenarios}

with open(save_filename, 'w') as f:
    json.dump(route_scenario_data, f, indent = 4)