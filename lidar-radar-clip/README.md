# CLIP based models for LiDAR and Radar Captioning
This directory contains the code for the CLIP based models for LiDAR and Radar captioning. All the required python libraries are mentioned in the `requirements.txt` file. 

A corresponding bash file is there to install the python requirements and other dependencies. Please run the following command to install the requirements:
```bash
bash install_requirements.sh
```

The `lidarclip` and `CLIP` folders contain the actual source code from the [OpenAI CLIP repository](https://github.com/openai/CLIP) and the [LiDARCLIP repository](https://github.com/atonderski/lidarclip). For more details, please refer to the respective repositories for instructions.

## Data Loader for CARLA dataset
The data loader with caching is implemented in the `carla_dataset.py` file. The data loader is used to load the CARLA dataset and cache the data for faster access on the Nautilus cluster.

## Generating Unique Captions
The CLIP model implemented for this project requires unique captions for the entire dataset. Therefore the functions in `count_unique_captions.py` are used to generate unique captions for the dataset.

## Generating CLIP Embeddings
The CLIP based textual embeddings are generated for each unqiue caption in the dataset. The corresponding script is in `generate_embeddings_for_captions.py`. The script generates the embeddings for the entire dataset and saves them in a JSON file along with the actual captions. To run this, use the following command:

```bash
python generate_embeddings_for_captions.py
```

## Training the CLIP model
The corresponding scripts for training the LiDAR and Radar based CLIP models are in `train_lidar_ViT_cos_loss.py` and `train_radar_ViT_cos_loss.py` respectively. The models are trained using the cosine similarity loss function.

These files require settings that can be modified in their bash scripts. To train the models, run the following commands:
```bash
bash train_lidar.sh # Training LiDAR based CLIP model
bash train_radar.sh # Training Radar based CLIP model
```

## Evaluation of the CLIP model
Due to time constraints, the model was evaluated qualitatively, and the corresponding code can be found in `test_caption_gen.ipynb` notebook. The notebook loads the trained model and generates captions from the test dataset.
