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

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


##########################################
Evaluator for the panel segmentation task.
"""

from typing import List, Dict, Any

from ..utils.figure import (Figure,
                            DetectedPanel,
                            DetectedLabel)
from ..utils.figure.label import CLASS_LABEL_MAPPING
from ..utils.detectron_utils.evaluator import PanelSegAbstractEvaluator
from .evaluate import evaluate_detections


class PanelSegEvaluator(PanelSegAbstractEvaluator):
    """
    Evaluator for the panel segmentation task.
    Perform the evaluation of panel segmentation metrics on a given test set.
    """

    def __init__(self,
                 dataset_name: str,
                 export: bool = False,
                 export_dir: str = None) -> None:
        """
        Init function.
        Call the init function of the parent class (PanelSegAbstractEvaluator).

        Args:
            dataset_name (str): The name of the data set to evaluate.
            export (bool):      Whether or not to export predictions as a JSON file.
            export_dir (str):   Path to the directory where to store the inference results.
        """
        super().__init__(dataset_name=dataset_name,
                         task_name='panel_seg',
                         evaluation_function=evaluate_detections,
                         export=export,
                         export_dir=export_dir)


    def process(self,
                inputs: List[dict],
                outputs: List[dict]) -> None:
        """
        Process pairs of inputs and outputs.

        Args:
            inputs (List[dict]):    The inputs that's used to call the model.
            outputs (List[dict]):   The return value of `model(inputs)`.
        """

        for input_element, output_element in zip(inputs, outputs):
            image_id: int = input_element['image_id']

            self._predictions[image_id] = {}

            # Panels
            panel_instances = output_element["panels"].to(self._cpu_device)
            panel_boxes = panel_instances.pred_boxes.tensor.numpy()
            panel_scores = panel_instances.scores.tolist()

            predicted_panels: List[Dict[str, Any]] = []
            for box, score in zip(panel_boxes, panel_scores):
                prediction = {
                    'box': box,
                    'score': score}

                predicted_panels.append(prediction)

            self._predictions[image_id]['panels'] = predicted_panels

            # Labels
            label_instances = output_element["labels"].to(self._cpu_device)
            label_boxes = label_instances.pred_boxes.tensor.numpy()
            label_scores = label_instances.scores.tolist()
            label_classes = label_instances.pred_classes.tolist()

            predicted_labels: List[Dict[str, Any]] = []

            for box, score, l_cls in zip(label_boxes, label_scores, label_classes):
                prediction = {
                    'box': box,
                    'score': score,
                    'label': l_cls
                }
                predicted_labels.append(prediction)

            self._predictions[image_id]['labels'] = predicted_labels


    def _predict(self, figure: Figure) -> None:
        """
        TODO

        Args:
            figure (Figure):    TODO.
        """

        try:
            detected_panels = self._predictions[figure.index]['panels']
            detected_labels = self._predictions[figure.index]['labels']
        except AttributeError:
            return


        # Convert panels and labels from dict to DetectedPanel objects
        figure.detected_panels = [DetectedPanel(box=panel['box'],
                                                detection_score=panel['score'])
                                  for panel in detected_panels]

        figure.detected_labels = [DetectedLabel(text=CLASS_LABEL_MAPPING[label['label']],
                                                box=label['box'],
                                                detection_score=label['score'])
                                  for label in detected_labels]
