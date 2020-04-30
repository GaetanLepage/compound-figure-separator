PATH_TO_TF_MODELS="/home/gaetan/hevs/implementations/object_detection/models/"
MODEL_DIR="panel_seg/panel_split/models"

python3 ${PATH_TO_TF_MODELS}official/vision/detection/main.py \
  --strategy_type=mirrored \
  --num_gpus=2 \
  --model_dir="${MODEL_DIR}" \
  --mode=train \
  --config_file="panel_seg/panel_split/panel_split_image_clef.yaml"
