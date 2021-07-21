"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.fr
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborators:  Niccolò Marini (niccolo.marini@hevs.ch)
                Stefano Marchesin (stefano.marchesin@unipd.it)


#######################################
Evaluator for the panel splitting task.
"""

from typing import Any

from ..utils.figure import Figure, DetectedPanel
from ..utils.detectron_utils.evaluator import PanelSegAbstractEvaluator
from .evaluate import evaluate_detections


class PanelSplitEvaluator(PanelSegAbstractEvaluator):
    """
    Evaluator for the panel splitting task.
    Perform the evaluation of panel splitting metrics on a given test set.
    """

    def __init__(self,
                 dataset_name: str,
                 export: bool = False,
                 export_dir: str = None):
        """
        Init function.
        Call the init function of the parent class (PanelSegAbstractEvaluator).

        Args:
            dataset_name (str): The name of the data set to evaluate.
            export (bool):      Whether or not to export predictions as a JSON file.
            export_dir (str):   Path to the directory where to store the inference
                                    results.
        """
        super().__init__(dataset_name=dataset_name,
                         task_name='panel_splitting',
                         evaluation_function=evaluate_detections,
                         export=export,
                         export_dir=export_dir)

    def process(self,
                inputs: list[dict],
                outputs: list[dict]):
        """
        Process pairs of inputs and outputs.

        Args:
            inputs (list[dict]):    The inputs that's used to call the model.
            outputs (list[dict]):   The return value of `model(inputs)`.
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
                assert cls == 0, "class should be 0 as we are only detecting panels."
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
        predicted_panels: list[dict[str, Any]] = self._predictions[figure.index]

        predicted_panel_objects: list[DetectedPanel] = []

        for prediction in predicted_panels:

            # Instanciate a Panel object.
            panel = DetectedPanel(box=prediction['box'],
                                  detection_score=prediction['score'])

            predicted_panel_objects.append(panel)

            # TODO do some post processing here maybe

        figure.detected_panels = predicted_panel_objects
