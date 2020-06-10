"""
TODO
"""

from typing import List

from panel_seg.utils.figure.panel import DetectedPanel

from panel_seg.utils.detectron_utils.evaluator import PanelSegAbstractEvaluator

from panel_seg.label_recognition.evaluate import evaluate_detections
from panel_seg.utils.figure.label_class import CLASS_LABEL_MAPPING
from panel_seg.utils.figure.figure import Figure


class LabelRecogEvaluator(PanelSegAbstractEvaluator):
    """
    Perform the evaluation of label recognition metrics on a given test set.
    """

    def __init__(self, dataset_name: str):
        """
        Init function.
        Call the init function of the parent function (PanelSegAbstractEvaluator).
        """
        super().__init__(dataset_name=dataset_name,
                         task_name='label_recog',
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
            figure (Figure): Figure objects augmented with predictions.
        """
        for figure in self._figure_generator:

            predicted_panels = predictions[figure.index]

            predicted_panel_objects = []

            for prediction in predicted_panels:

                # Instanciate a Panel object.
                panel = DetectedPanel(label=CLASS_LABEL_MAPPING[prediction['cls']],
                                      label_rect=prediction['box'],
                                      label_detection_score=prediction['score'])

                predicted_panel_objects.append(panel)

                # TODO do some post processing here maybe

            figure.detected_panels = predicted_panel_objects

            yield figure
