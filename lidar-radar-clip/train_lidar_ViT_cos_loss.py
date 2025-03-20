import os
import torch
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from diskcache import Cache

from CLIP.clip import clip
import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode, ConvertImageDtype
BICUBIC = InterpolationMode.BICUBIC

from carla_dataset import generate_carla_dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def transform_image(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ConvertImageDtype(torch.float),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def main(args):
    shared_cache = None
    if args.use_disk_cache:
        tmp_folder = str(os.environ.get('SCRATCH', '/tmp'))
        tmp_folder = tmp_folder + '/dataset_cache' + args.image_model
        shared_cache = Cache(directory = tmp_folder, size_limit = int(768 * 1024 ** 3))
    
    # Rank of the current process
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Total number of processes available
    world_size = int(os.environ['WORLD_SIZE'])
    
    print(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
    
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize the backend process for multu-gpu training
    init_process_group(
        backend = 'nccl', 
        init_method = 'env://',
        world_size = world_size,
        rank = rank,
        timeout = datetime.timedelta(minutes = 10)
    )
    
    # Computing the number of workers for data loading
    num_gpus = torch.cuda.device_count()
    num_cpu_cores = args.num_cpu_cores
    num_workers = int(num_cpu_cores / num_gpus)
    
    # Setting the GPU device
    torch.cuda.device(device)
    
    # Enabling the cudnn backend
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    
    data_root = args.data_dir
    num_repetitions = args.num_repetitions
    test_size = args.test_size
    
    print(f"Rank {local_rank}: Generating dataset")
    train_set, test_set = generate_carla_dataset(
        data_root = data_root, 
        test_size = test_size, 
        num_repetitions = num_repetitions,
        shared_cache = shared_cache
    )
    
    print(f"Rank {local_rank}: Dataset generated")
    # To maintain different data distribution for each GPU
    train_sampler = DistributedSampler(
        train_set,
        num_replicas = world_size,
        rank = rank,
        shuffle = True
    )
    
    test_sampler = DistributedSampler(
        test_set,
        num_replicas = world_size,
        rank = rank,
        shuffle = False
    )
    
    # Randomizers for the dataloaders
    rand_gen = torch.Generator(device = 'cpu')
    rand_gen.manual_seed(torch.initial_seed())
    
    # Dataloaders for training and testing
    # No need to pin memory as we are caching on disk
    train_loader = DataLoader(
        train_set,
        batch_size = args.batch_size,
        sampler = train_sampler,
        num_workers = num_workers,
        pin_memory = False,
        worker_init_fn = seed_worker,
        generator = rand_gen
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size = args.batch_size,
        sampler = test_sampler,
        num_workers = num_workers,
        pin_memory = False,
        worker_init_fn = seed_worker,
        generator = rand_gen
    )

    # Load the CLIP model
    model_list = os.listdir("./models")
    save_name = args.image_model.replace("/", "_")
    if save_name + ".pth" in model_list:
        model = torch.load("./models/" + save_name + ".pth", map_location = device)
    else:
        model, _ = clip.load(args.image_model, device = device)
        torch.save(model, f"./models/{save_name}.pth")
    
    # convert model parameters to fp32
    model.float()
    
    preprocess = transform_image(model.visual.input_resolution)
    
    # Data-parallelism
    model = DDP(
        model, 
        device_ids = None, 
        output_device = None,
        broadcast_buffers = False
    )
    
    # Optimizer
    optimizer = Adam(
        model.module.visual.parameters(),
        lr = args.lr
    )
    
    loss_fn = lambda x, y: -F.cosine_similarity(x, y).mean()
    
    len_train_loader = len(train_loader)
    loss_list = []
    
    print("Starting training")
    # Training loop
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        pbar = tqdm(total = len(train_loader), disable = local_rank != 0, desc = f"Epoch {epoch + 1}")
        for data in train_loader:
            lidar_bev_image_tensor = data["lidar_bev"]
            captions = data["caption"]
            curr_loss = 0
            
            with torch.no_grad():
                lidar_processed = preprocess(lidar_bev_image_tensor).to(device)
                caption = clip.tokenize(captions).to(device)
                caption_features = model.module.encode_text(caption)

            image_features = model.module.encode_image(lidar_processed)

            loss = loss_fn(image_features, caption_features)
            curr_loss += loss.item()

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            if rank == 0:
                pbar.update(1)
            
            curr_loss /= len_train_loader
            pbar.set_postfix(loss = curr_loss)
            loss_list.append(curr_loss)
        pbar.close()
        
        torch.cuda.empty_cache()

        if epoch % 2 == 0:
            torch.save(model, f"./models/{save_name}_train_lidar_epoch{epoch}.pth")
        
    # Save the model
    if rank == 0:
        torch.save(model, f"./models/{save_name}_trained_lidar.pth")
        np.save(f"./models/{save_name}_train_lidar_loss.npy", np.array(loss_list))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model on the Carla dataset')
    parser.add_argument('--batch_size', 
                        type = int, 
                        default = 2, 
                        help = 'Batch size')
    parser.add_argument('--epochs', 
                        type = int, 
                        default = 10, 
                        help = 'Number of epochs')
    parser.add_argument('--lr', 
                        type = float, 
                        default = 1e-4, 
                        help = 'Learning rate')
    parser.add_argument('--test_size', 
                        type = float, 
                        default = 0.992, 
                        help = 'Weight decay')
    parser.add_argument('--image_model', 
                        type = str, 
                        default = 'ViT-L/14', 
                        help = 'CLIP model')
    parser.add_argument('--data_dir', 
                        type = str, 
                        default = '/radar-imaging-dataset/P2SIF/carla_data', 
                        help = 'Data directory')
    parser.add_argument('--num_repetitions',
                        type = int,
                        default = 1,
                        help = 'Number of times to include route repetitions')
    parser.add_argument('--use_disk_cache',
                        type = bool,
                        default = True,
                        help = 'Use disk cache in case of slow storage system')
    parser.add_argument('--num_cpu_cores',
                        type = int,
                        default = 1,
                        help = 'Number of CPU cores to use for data loading')
    args = parser.parse_args()
    main(args)