python compfigsep/panel_segmentation/train_net.py \
    --num-gpus "${NUM_GPUS:-$(nvidia-smi -L | wc -l)}" \
    --config-file compfigsep/panel_segmentation/config.yaml \
    --resume \
