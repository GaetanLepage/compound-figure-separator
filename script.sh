#!/bin/sh

export PYTHONPATH=.
export NUM_GPUS=1

SCRIPT=panel_splitting/bin/train_panel_seg.sh

bash compfigsep/$SCRIPT

# python compfigsep/$SCRIPT
