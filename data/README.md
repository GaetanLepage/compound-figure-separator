# Data

Two data sets are used to test this pipeline :


## ![](https://www.imageclef.org/files/tech_logo.png =100x) ImageCLEF 2016

The [_ImageCLEF 2016_ medical challenge](https://www.imageclef.org/2016/medical) includes different subtasks. One of them is _figure separation_. It consists in _panel splitting_. Hence, the data set provided for this challenge only contains annotations for panels. Neither labels nor captions are part of this data set.

We choose to use this reference data set to compare our results for the panel splitting task to other works.

**Download:** To be able to download the data set, it is needed to register [here](http://medgift.hevs.ch:8080/CLEF2016/faces/Register.jsp) for the _ImageCLEF2016:medical-task_. The data set can then be downloaded [here](http://fast.hevs.ch/imageclefmed/2016/).


## PanelSeg data set (by Zou et al.)

As the first people to tackle the _panel segmentation task_, Zou et al. manually annotated a compound figure data set with both panel and label annotations.

**Credits:** J. Zou S. Antani and G. Thoma, _"Unified Deep Neural Network for Segmentation and Labeling of Multi-Panel Biomedical Figures,"_ submitted to Journal of the Association for Information Science and Technology (JASIST), 2019

### What is it ?
The dataset is a freely-available repository of multi-panel medical journal figure images. The set has been collected for the purpose of design, training, and testing of algorithms for automatic image panel splitting and annotation. Both panels and their labels are annotated, and the data has been manually curated to establish its “ground truth”.

### Data set description

The dataset contains 10,642 images are randomly selected from the Open Access subset of biomedical articles available in the [National Library of Medicine’s (NLM)](https://www.nlm.nih.gov/) [Pubmed Central (PMC)](https://www.ncbi.nlm.nih.gov/pmc/) Repository. The images were obtained using the NLM’s [Open-i](https://lhncbc.nlm.nih.gov/project/open-i)® multimodal biomedical information retrieval system which indexes such articles and images. A human operator manually generated the panel and label annotations. Another human operator conducted a final review and confirmed the ground-truth annotation. We randomly partition the dataset into training and evaluation sets. Evaluation set contains 1,000 figures, and the remaining 9,642 are training samples.

The dataset is available as a single ZIP compressed file. Available [here](ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/NLM-Multi-Panel-Figure-Segmentation-Dataset/).

Inside the ZIP file, there are 8 folders and 4 files where the folders contains the actual figure images and ground-truth annotations. In each folder, there are JPG image files with associated XML annotation files. The XML file follows [iPhotoDraw](https://www.iphotodraw.com/) format.

The 4 files are `train.txt` which contains the list of 9,642 training images and `eval.txt` which contains the list of 1,000 evaluation images. Other two files are `train.csv` and `eval.csv` which are ground-truth annotations in CSV file format. The expected format of each line is:
```
path/to/image.jpg,p_x1,p_y1,p_x2,p_y2,panel,l_x1,l_y1,l_x2,l_y2,label
```
Note that some panels may not have panel labels. In this case, we set the label part empty, but still keep the commas, i.e.,
```
path/to/image.jpg,p_x1,p_y1,p_x2,p_y2,panel,,,,,
```

This data set can be used for the evaluation of _panel splitting_, _label recognition_ and _panel segmentation_.

10642 images:
* 9642 (training)
* 1000 (test)

### Additions

As one of the goal of **CompFigSep** is to handle the **caption splitting** task, this data set was partially augmented with caption data.\
The caption texts where downloaded thanks to the PubMed Central ID numbers.\
The script in `compfigsep.data.bin.download_captions.py` directly download the captions from PMC website.
