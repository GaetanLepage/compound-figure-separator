python compfigsep/panel_splitting/train_net.py \
    --num-gpus "${NUM_GPUS:-$(nvidia-smi -L | wc -l)}" \
    --config-file compfigsep/panel_splitting/config_imageclef.yaml \
    --resume \
    --eval-only
