"""
# TODO inverser les boucles
TODO
"""

from panel_seg.utils.figure.panel import DetectedPanel

from panel_seg.utils.detectron_utils.evaluator import PanelSegAbstractEvaluator

from panel_seg.panel_segmentation.evaluate import evaluate_detections
from panel_seg.utils.figure.label_class import CLASS_LABEL_MAPPING

import panel_seg.utils.figure.beam_search as beam_search


class PanelSegEvaluator(PanelSegAbstractEvaluator):
    """
    TODO
    Perform the evaluation of panel segmentation metrics on a given test set.
    """

    def __init__(self, dataset_name):
        """
        TODO
        """
        super().__init__(dataset_name=dataset_name,
                         task_name='panel_seg',
                         evaluation_function=evaluate_detections)

    @staticmethod
    def _beam_search_mapping(panels, labels):
        """
        TODO
        """
        panels = [DetectedPanel(panel_rect=panel['box'],
                                panel_detection_score=panel['score'])
                  for panel in panels]

        labels = [DetectedPanel(label_rect=label['box'],
                                label_detection_score=label['score'],
                                label=CLASS_LABEL_MAPPING[label['label']])
                  for label in labels]

        beam_search.assign_labels_to_panels(panels, labels)


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

            self._predictions[image_id] = {}

            panel_instances = output["panels"].to(self._cpu_device)
            panel_boxes = panel_instances.pred_boxes.tensor.numpy()
            panel_scores = panel_instances.scores.tolist()
            # TODO we don't need the panel classes (only one class)
            # panel_classes = panel_instances.pred_classes.tolist()

            predicted_panels = []
            for box, score in zip(panel_boxes, panel_scores):
                prediction = {
                    'box': box,
                    'score': score}

                predicted_panels.append(prediction)

            self._predictions[image_id]['panels'] = predicted_panels

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


    def _augmented_figure_generator(self, predictions):
        """
        Iterate over a Figure generator, make predictions and yield back the augmented
        Figure objects.
        """
        for figure in self._figure_generator:

            try:
                predicted_panels = predictions[figure.index]['panels']
                predicted_labels = predictions[figure.index]['labels']
            except Exception:
                continue


            panels = [DetectedPanel(panel_rect=panel['box'],
                                    panel_detection_score=panel['score'])
                      for panel in predicted_panels]

            labels = [DetectedPanel(label_rect=label['box'],
                                    label_detection_score=label['score'],
                                    label=CLASS_LABEL_MAPPING[label['label']])
                      for label in predicted_labels]

            figure.detected_panels = labels
            # figure.detected_panels.extend(labels)

            figure.show_preview(mode='pred')


            # self._beam_search_mapping(panels=predicted_panels,
                                      # labels=predicted_labels)

                # TODO do some post processing here maybe

            # figure.detected_panels = predicted_panels

            yield figure
