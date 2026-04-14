ulimit -n 65536

WANDB_MODE=offline python train.py \
  --config config_liosam.yaml \
  --gpus -1 \
  --batch_size 4 \
  --lr 0.0002 \
  --max_epochs 200
