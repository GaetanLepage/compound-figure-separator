python compfigsep/label_recognition/train_net.py \
    --num-gpus 1 \
    --config-file compfigsep/label_recognition/config.yaml \
    --resume \
    # --num-gpus "${NUM_GPUS:-$(nvidia-smi -L | wc -l)}" \
