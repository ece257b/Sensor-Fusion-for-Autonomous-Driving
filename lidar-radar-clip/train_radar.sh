export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export WORLD_SIZE=6

torchrun --nnodes=1 --nproc_per_node=gpu --max_restarts=1 --rdzv_id=42353467 --rdzv_backend=c10d train_radar_ViT_cos_loss.py \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-4 \
    --test_size 0.3 \
    --image_model ViT-L/14 \
    --data_dir /radar-imaging-dataset/P2SIF/carla_data \
    --num_repetitions 1 \
    --use_disk_cache 1 \
    --num_cpu_cores 32