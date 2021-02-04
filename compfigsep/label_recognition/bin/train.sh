python compfigsep/label_recognition/train_net.py \
    --num-gpus "${NUM_GPUS:-$(nvidia-smi -L | wc -l)}" \
    --config-file compfigsep/label_recognition/config.yaml \
    --resume \
