# Clean wandb cache
wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp


export PROJECT_ROOT=$(pwd)
# mode={stage1,stage2}
#python train.py model=stage2
CUDA_LAUNCH_BLOCKING=1 python train.py
