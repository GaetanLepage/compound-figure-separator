# CompFigSeg

Implementation of a complete pipeline for compound figures separation.\
The code for the panel segmentation task is heavily inspired from Zou & al. work ([paper](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.24334) and [implementation](https://github.com/JieZou1/PanelSeg/tree/master/PanelSeg_Keras)).

## Objective

**Compound figures** are numerous in scientific publications. They consist in figures containing multiple (more or less related) _sub figures_.
In the context of medical scientific publications, compound figures account for a significant amount of visual data.
To exploit the information from those compound figures, they need to be segmented in several sub figures as independent as possible.

The _compound figure separation_ task is composed of several subtasks:
*   Panel segmentation
    *   Panel splitting
    *   Label recognition
*   Caption splitting

## How to use

In order to be sure to fulfill the **software requirements**, it is best to work within a **Python virtual environment**.


```{bash}
# Create the virtual environment.
python3 -m venv venv

# activate it
. venv/bin/activate

# make sure pip is up to date
pip install --upgrade pip

# install the required packages
pip install -r requirements.txt
```


# TODO: explain how to train/test

It is possible to follow training using _TensorBoard_
```{bash}
ten
```

## Implementation details

### Pipeline

### Modules

*   `data`
*   `utils`
    *   `utils.detectron_utils`
    *   `utils.figure`
*   `panel_splitting`
*   `label_recognition`
*   `panel_segmentation`
*   `caption_splitting`
