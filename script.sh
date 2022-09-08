#!/bin/sh

export PYTHONPATH=.
export NUM_GPUS=1


########
# TASK #
########
# TASK="panel_splitting"
# TASK="label_recognition"
TASK="panel_segmentation"

##########
# SCRIPT #
##########
# SCRIPT_NAME="train_imageclef.sh"
# SCRIPT_NAME="test_imageclef.sh"

# SCRIPT_NAME="train_panel_seg.sh"
# SCRIPT_NAME="test_panel_seg.sh"

# SCRIPT_NAME="train.sh"
SCRIPT_NAME="test.sh"
SCRIPT_PATH=$TASK/bin/$SCRIPT_NAME

########
# DATA #
########

# SCRIPT_NAME="preview_json_data_set.py"

# SCRIPT_PATH="data/bin/"$SCRIPT_NAME

# SCRIPT_PATH="data/preview_imageclef_data_set.py"


COMMAND="bash compfigsep/${SCRIPT_PATH}"
# COMMAND="python compfigsep/${SCRIPT_PATH}"

# \time -f "Command took %E" $COMMAND $@
$COMMAND $@
