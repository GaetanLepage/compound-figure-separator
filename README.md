# CompFigSeg

Implementation of a complete pipeline for compound figures separation.\
The code for the panel segmentation task is heavily inspired from Zou & al. work ([paper](https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/asi.24334) and [implementation](https://github.com/JieZou1/PanelSeg/tree/master/PanelSeg_Keras)).

![image](https://www.ontotext.com/wp-content/uploads/2019/02/cropped-ExaModeLogo_blacklines_TranspBackGround1.png =100x)

This work is part of the [ExaMode](https://www.examode.eu) (Extreme-scale Analytics via Multimodal Ontology Discovery & Enhancement) project and aims to enhance the manipulation of massive medical data.


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
```bash
tensorboard --logdir=compfigsep/<TASK_NAME>/output/ [--bind_all]
```

## Implementation details

### Pipeline

### Modules

*   `data`

The `data` module contains function dealing with the various data sources. Among other things, one can preview, load and export the different data sets.
*   `utils`

In `utils`, several functions are here to handle miscellaneous tasks.

    *   `utils.detectron_utils`
    *   `utils.figure`
*   `panel_splitting`
*   `label_recognition`
*   `panel_segmentation`
*   `caption_splitting`

## Data sets

Different data sets are involved in this project.

Learn more by reading this [README.md](data/README.md).


## Contact

I have been realizing this project from April to August 2020 within the [Medgift team](http://medgift.hevs.ch/wordpress/) from [HES-SO](https://www.hevs.ch/) for my Masters project. I worked under the supervision of [Henning Müller](http://medgift.hevs.ch/wordpress/team/henning-mueller/) and [Manfredo Atzori](http://medgift.hevs.ch/wordpress/team/manfredo-atzori/). [Niccolò Marini](http://medgift.hevs.ch/wordpress/team/niccolo-marini/) also offered his helpful contribution.
