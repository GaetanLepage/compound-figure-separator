# FigCompSep `data` module

The IO module relies on an efficient factorization using figure generators.
Indeed, handling all supported data sets in a similar manner is made possible
by using the generic Figure object.

## Input
The data module handles different input formats/annotations with a specific **generator**
for each. They all yield similar Figure objects.

## Output
The Figure generators can be used seamlessly to produce several outputs:
* **visualize** the different data sets by displaying images and annotations.
* **export** the data sets/annotations to several formats (`csv`, `tfrecord`, `dict`...) that
can be used with different models.\
Scripts for exporting and previewing the different data sets are available in the `bin/` folder.
