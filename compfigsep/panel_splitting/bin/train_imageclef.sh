# python compfigsep/panel_splitting/train_net.py \
#     --num-gpus "${NUM_GPUS:-$(nvidia-smi -L | wc -l)}" \
#     --config-file compfigsep/panel_splitting/config_imageclef.yaml \
#     --resume \

python -m yolox.tools.train \
    -f compfigsep/panel_splitting/exp.py \
    --devices 1 \
    --batch-size 8 \
    --fp16 \
    --occupy \
    -c weights/yolox_x.pth \
    --cache
