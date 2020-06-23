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


##########################################
Evaluator for the panel segmentation task.
"""

from typing import List

from ..utils.figure import Figure, DetectedSubFigure, CLASS_LABEL_MAPPING
from ..utils.detectron_utils.evaluator import PanelSegAbstractEvaluator
from .evaluate import evaluate_detections

class PanelSegEvaluator(PanelSegAbstractEvaluator):
    """
    Evaluator for the panel segmentation task.
    Perform the evaluation of panel segmentation metrics on a given test set.
    """

    def __init__(self, dataset_name: str):
        """
        Init function.
        Call the init function of the parent class (PanelSegAbstractEvaluator).

        Args:
            dataset_name (str): The name of the data set to evaluate.
        """
        super().__init__(dataset_name=dataset_name,
                         task_name='panel_seg',
                         evaluation_function=evaluate_detections)

    def process(self,
                inputs: List[dict],
                outputs: List[dict]):
        """
        Process the pair of inputs and outputs.

        Args:
            inputs (List[dict]):    The inputs that's used to call the model.
            outputs (List[dict]):   The return value of `model(inputs)`.
        """

        for input, output in zip(inputs, outputs):
            image_id = input['image_id']

            self._predictions[image_id] = {}

            # Panels
            panel_instances = output["panels"].to(self._cpu_device)
            panel_boxes = panel_instances.pred_boxes.tensor.numpy()
            panel_scores = panel_instances.scores.tolist()

            predicted_panels = []
            for box, score in zip(panel_boxes, panel_scores):
                prediction = {
                    'box': box,
                    'score': score}

                predicted_panels.append(prediction)

            self._predictions[image_id]['panels'] = predicted_panels

            # Labels
            label_instances = output["labels"].to(self._cpu_device)
            label_boxes = label_instances.pred_boxes.tensor.numpy()
            label_scores = label_instances.scores.tolist()
            label_classes = label_instances.pred_classes.tolist()

            predicted_labels = []

            for box, score, cls in zip(label_boxes, label_scores, label_classes):
                prediction = {
                    'box': box,
                    'score': score,
                    'label': cls
                }
                predicted_labels.append(prediction)

            self._predictions[image_id]['labels'] = predicted_labels


    def _augmented_figure_generator(self, predictions: dict) -> Figure:
        """
        Loop over the Figure generator, fill the Figure objects with predictions and yield back
        the augmented Figure objects.

        Args:
            predictions (dict): The dict containing the predictions from the model.

        Yields:
            figure (Figure): Figure objects augmented with predictions.
        """
        for figure in self._figure_generator:

            try:
                detected_panels = predictions[figure.index]['panels']
                detected_labels = predictions[figure.index]['labels']
            except Exception:
                continue


            # Convert panels and labels from dict to DetectedPanel objects
            figure.raw_detected_panels = [DetectedSubFigure(panel_rect=panel['box'],
                                                        panel_detection_score=panel['score'])
                                          for panel in detected_panels]

            figure.raw_detected_labels = [DetectedSubFigure(label_rect=label['box'],
                                                        label_detection_score=label['score'],
                                                        label=CLASS_LABEL_MAPPING[label['label']])
                                          for label in detected_labels]

            # ==> Those two sets of "half" objects will be merged to give
            # complete DetectedPanel objects (with label, panel_rect and label_rect fields).

            # TODO remove !
            # figure.save_preview(folder='data/pubmed_extract/preview/',
                                # mode='pred')

            # figure.show_preview(mode='pred', delay=0)

            # TODO do some post processing here maybe


            yield figure
