# CompFigSep `data` module

The `data` module relies on an efficient factorization using figure generators.
Indeed, handling all supported data sets in a similar manner is made possible
by using the generic `utils.figure.figure.Figure` object.

## Input
The data module handles different input formats/annotations with a specific **generator**
for each. They all yield similar Figure objects.

## Output
The Figure generators can be used seamlessly to produce several outputs:
* **visualize** the different data sets by displaying images and annotations.
* **export** the data sets/annotations to several formats (`JSON`, `csv`, `tfrecord`, `dict`...) that can be used with different models.\
Scripts for exporting and previewing the different data sets are available in the `bin/` folder.

## One format to rule them all

With CompFigSep, I intoduced a JSON unique file format.\
Basically, a single JSON file can handle a full data set as a list of __figures__ which may contain:
* The reference to the image file (the path where it is stored).
* Base caption text for the compound figure.
* Ground truth information (bounding boxes for panels and labels, label texts, split captions...)
* Predicted information (bounding boxes for panels and labels, label texts, split captions...)

Basically, this format is a way of __serializing__ `Figure` objects.

**Motivations:**
* One format for all the different data sets.
* Capacity to store both ground truth **and** detections. It is then possible to evaluate detections afterwards, with no need for additional computations.
