WANDB_MODE=offline python train.py \
  --config config_liosam.yaml \
  --gpus 2 \
  --batch_size 1 \
  --lr 0.02 \
  --max_epochs 200 \
  --debug_fixed_batch
