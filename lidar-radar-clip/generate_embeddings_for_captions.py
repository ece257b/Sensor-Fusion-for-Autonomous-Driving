import os
import json
import torch
import numpy as np
from tqdm import tqdm
from CLIP.clip import clip
from num2words import num2words

unique_captions_root = "/radar-imaging-dataset/P2SIF/lidar-radar-clip/caption/"
caption_files = [f.path for f in os.scandir(unique_captions_root) if f.is_file()]

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

unique_captions = []
unique_numbers = []

for i, file_path in enumerate(caption_files):
    with open(file_path, 'r') as f:
        data = json.load(f)
    captions = data['captions']
    numbers = data['car_bike']
    
    for j, cap in enumerate(captions):
        car_bike = numbers[j]
        flag_car, flag_bike = False, False
        if car_bike[0] > 10:
            flag_car = True
            car_bike[0] = 10
        if car_bike[1] > 5:
            flag_bike = True
            car_bike[1] = 5
            
        cap = modify_captions(car_bike[0], car_bike[1], flag_car, flag_bike)
        if cap not in unique_captions:
            unique_captions.append(cap)
            unique_numbers.append(car_bike)


unique_numbers = np.array(unique_numbers)
print(unique_numbers.shape)
print(len(unique_captions))

data = {
    "captions": unique_captions,
    "car_bike": unique_numbers.tolist()
}

models = ["ViT-L/14", "ViT-B/32"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name in models:
    model_embeddings = []
    save_name = model_name.replace("/", "_")
    model = torch.load("./models/" + save_name + ".pth", map_location = device)
    
    print(f"Generating for {model_name}")
    
    text_enoder = model.encode_text
    
    with torch.no_grad():
        for cap in tqdm(unique_captions):
            text = clip.tokenize([cap]).to(device)
            text_features = text_enoder(text)
            model_embeddings.append(text_features.cpu().numpy().tolist())

    data[model_name] = model_embeddings

save_file_path = "/radar-imaging-dataset/P2SIF/lidar-radar-clip/caption/caption_embeddings.json"
with open(save_file_path, 'w') as f:
    json.dump(data, f, indent = 4)