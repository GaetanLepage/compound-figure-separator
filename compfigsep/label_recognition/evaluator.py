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


#########################################
Evaluator for the label recognition task.
"""

from typing import List, Dict, Any

from ..utils.figure import Figure, DetectedLabel
from ..utils.figure.label import CLASS_LABEL_MAPPING
from ..utils.detectron_utils.evaluator import PanelSegAbstractEvaluator
from .evaluate import evaluate_detections


class LabelRecogEvaluator(PanelSegAbstractEvaluator):
    """
    Evaluator for the label recognition task.
    Perform the evaluation of label recognition metrics on a given test set.
    """

    def __init__(self,
                 dataset_name: str,
                 export: bool = False,
                 export_dir : str = None):
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

        for input_element, output_element in zip(inputs, outputs):
            image_id = input_element["image_id"]
            instances = output_element["instances"].to(self._cpu_device)

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


    def _predict(self, figure: Figure) -> None:
        """
        Write the predictions (stored in the `_predictions` attribute) in the appropriate
        attributes of the given figure object.
        The latter is modified by side effet.

        Args:
            figure (Figure):    A Figure object to augment with prediction data.
        """

        detected_labels_dicts: List[Dict[str, Any]] = self._predictions[figure.index]

        detected_labels: List[DetectedLabel] = []

        for detected_label_dict in detected_labels_dicts:

            # Instanciate a DetectedLabel object.
            detected_label: DetectedLabel = DetectedLabel(
                text=CLASS_LABEL_MAPPING[detected_label_dict['cls']],
                box=detected_label_dict['box'],
                detection_score=detected_label_dict['score'])

            detected_labels.append(detected_label)

        figure.detected_labels = detected_labels
