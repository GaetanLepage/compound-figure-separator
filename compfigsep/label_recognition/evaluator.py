"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.org
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborator:   Niccolò Marini (niccolo.marini@hevs.ch)


#########################################
Evaluator for the label recognition task.
"""

from typing import List

from ..utils.figure import Figure, DetectedLabel, CLASS_LABEL_MAPPING
from ..utils.detectron_utils.evaluator import PanelSegAbstractEvaluator
from .evaluate import evaluate_detections


class LabelRecogEvaluator(PanelSegAbstractEvaluator):
    """
    Evaluator for the label recognition task.
    Perform the evaluation of label recognition metrics on a given test set.
    """

    def __init__(self, dataset_name: str, export: bool = False):
        """
        Init function.
        Call the init function of the parent function (PanelSegAbstractEvaluator).

        Args:
            dataset_name (str): The name of the data set to evaluate.
            export (bool):      Whether or not to export predictions as a JSON file.
        """
        super().__init__(dataset_name=dataset_name,
                         task_name='label_recog',
                         evaluation_function=evaluate_detections,
                         export=export)


    def process(self,
                inputs: List[dict],
                outputs: List[dict]):
        """
        Process pairs of inputs and outputs.

        Args:
            inputs (List[dict]):    The inputs that's used to call the model.
            outputs (List[dict]):   The return value of `model(inputs)`.
        """

        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)

            # Get prediction data.
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()

            # Store predictions.
            predicted_panels = []
            for box, score, cls in zip(boxes, scores, classes):
                prediction = {}
                prediction['cls'] = cls
                prediction['box'] = box
                prediction['score'] = score
                predicted_panels.append(prediction)

            self._predictions[image_id] = predicted_panels


    def _augmented_figure_generator(self, predictions: dict) -> Figure:
        """
        Loop over the Figure generator, fill the Figure objects with predictions and yield back
        the augmented Figure objects.

        Args:
            predictions (dict): The dict containing the predictions from the model.

        Yields:
            figure (Figure):    Figure objects augmented with predictions.
        """
        for figure in self._figure_generator:

            detected_labels = predictions[figure.index]

            detected_label_objects = []

            for detected_label in detected_labels:

                # Instanciate a DetectedLabel object.
                detected_label = DetectedLabel(text=CLASS_LABEL_MAPPING[detected_label['cls']],
                                               box=detected_label['box'],
                                               detection_score=detected_label['score'])

                detected_label_objects.append(detected_label)

                # TODO do some post processing here maybe

            figure.detected_labels = detected_label_objects

            yield figure
