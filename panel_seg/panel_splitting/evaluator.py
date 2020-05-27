"""
        # TODO inverser les boucles
TODO
"""

from panel_seg.utils.figure.panel import DetectedPanel
from panel_seg.utils.detectron_utils.evaluator import PanelSegAbstractEvaluator
from panel_seg.panel_splitting.evaluate import evaluate_detections

class PanelSplitEvaluator(PanelSegAbstractEvaluator):
    """
    TODO
    Perform the evaluation of panel splitting metrics on a given test set.
    """

    def __init__(self, dataset_name):
        """
        TODO
        """
        super().__init__(dataset_name=dataset_name,
                         task_name='panel_splitting',
                         evaluation_function=evaluate_detections)

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
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


    def _augmented_figure_generator(self, predictions):
        """
        Iterate over a Figure generator, make predictions and yield back the augmented
        Figure objects.
        """
        for figure in self._figure_generator:

            predicted_panels = predictions[figure.index]

            predicted_panel_objects = []

            for prediction in predicted_panels:
                panel = DetectedPanel(panel_rect=prediction['box'],
                                      panel_detection_score=prediction['score'])

                predicted_panel_objects.append(panel)

                # TODO do some post processing here maybe

            figure.detected_panels = predicted_panel_objects

            yield figure
