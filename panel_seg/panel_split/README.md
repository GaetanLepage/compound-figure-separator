# Panel splitting

**Panel splitting** is the base task achieved by the panel segmentation tool.
The current implementation relies on [**Facebook Detectron 2 API**](https://github.com/facebookresearch/detectron2).\
For now, the panel splitting task is achieved using [**Retinanet**](https://arxiv.org/abs/1708.02002) with **Resnet 50** as the backbone.

Panel splitting can be seen as a classical **object detection** problem.
Two metrics can be used to score the results :
* The **ImageCLEF** accuracy:
$$\text{Acc}_\text{ImageCLEF}=\frac{1}{N} \sum\limits_{i=1}^N \text{acc}_i $$
where
$$\text{acc}_i = \frac{\#\{\text{Correct panels}\}} {\max\Big( \#\{\text{predicted panels}\}, \#\{\text{ground truth panels}\}\Big)}$$

* The more conventional **mean average precision (mAP)**:
