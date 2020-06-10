"""
TODO
"""

from typing import List

from panel_seg.utils.figure.panel import DetectedPanel

from panel_seg.utils.detectron_utils.evaluator import PanelSegAbstractEvaluator
from panel_seg.panel_splitting.evaluate import evaluate_detections

from panel_seg.utils.figure.figure import Figure


class PanelSplitEvaluator(PanelSegAbstractEvaluator):
    """
    TODO
    Perform the evaluation of panel splitting metrics on a given test set.
    """

    def __init__(self, dataset_name: str):
        """
        Init function.
        Call the init function of the parent class (PanelSegAbstractEvaluator).

        Args:
            dataset_name (str): The name of the data set to evaluate.
        """
        super().__init__(dataset_name=dataset_name,
                         task_name='panel_splitting',
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
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            # print("boxes =", boxes)
            scores = instances.scores.tolist()
            # print("scores =", scores)
            classes = instances.pred_classes.tolist()
            # print("classes =", classes)
            predicted_panels = []
            for box, score, cls in zip(boxes, scores, classes):
                prediction = {}
                assert cls == 0, "class should be 0 as we are only detecting panels."
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
            figure (Figure): Figure objects augmented with predictions.
        """
        for figure in self._figure_generator:

            predicted_panels = predictions[figure.index]

            predicted_panel_objects = []

            for prediction in predicted_panels:

                # Instanciate a Panel object.
                panel = DetectedPanel(panel_rect=prediction['box'],
                                      panel_detection_score=prediction['score'])

                predicted_panel_objects.append(panel)

                # TODO do some post processing here maybe

            figure.detected_panels = predicted_panel_objects

            figure.save_preview(mode='pred')

            yield figure
