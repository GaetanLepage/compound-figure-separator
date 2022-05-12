#!/bin/sh

export PYTHONPATH=.
export NUM_GPUS=1


# TASK="panel_splitting"
TASK="label_recognition"
# TASK="panel_segmentation"

# SCRIPT_NAME="test_imageclef.sh"
# SCRIPT_NAME="train_imageclef.sh"
# SCRIPT_NAME="train_panel_seg.sh"
SCRIPT_NAME="train.sh"

SCRIPT_PATH=$TASK/bin/$SCRIPT_NAME
# SCRIPT="panel_splitting/bin/train_panel_seg.sh"
# SCRIPT="panel_splitting/bin/test_imageclef.sh"


COMMAND="bash compfigsep/${SCRIPT_PATH}"
# COMMAND="which python"

if [[ `hostname` =~ "bigfoot" ]]; then
    source /applis/environments/conda.sh
    conda activate compfigsep
fi
# \time -f "Command took %E" $COMMAND $@
$COMMAND $@
