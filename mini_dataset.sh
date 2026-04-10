WANDB_MODE=offline python train.py \
  --config config_liosam.yaml \
  --gpus 0 \
  --batch_size 4 \
  --lr 0.002 \
  --max_epochs 200
