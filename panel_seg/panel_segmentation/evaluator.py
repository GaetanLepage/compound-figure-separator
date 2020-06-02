"""
TODO
"""

from panel_seg.utils.figure.panel import DetectedPanel

from panel_seg.utils.detectron_utils.evaluator import PanelSegAbstractEvaluator

from panel_seg.panel_segmentation.evaluate import evaluate_detections
from panel_seg.utils.figure.label_class import CLASS_LABEL_MAPPING
from panel_seg.utils.figure.figure import Figure

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

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.

        Args:
            inputs (List[dict]):    The inputs that's used to call the model.
            outputs (List[dict]):   The return value of `model(inputs)`.
        """

        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]

            self._predictions[image_id] = {}

            # Panels
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
            detected_panels = [DetectedPanel(panel_rect=panel['box'],
                                             panel_detection_score=panel['score'])
                               for panel in detected_panels]

            detected_labels = [DetectedPanel(label_rect=label['box'],
                                             label_detection_score=label['score'],
                                             label=CLASS_LABEL_MAPPING[label['label']])
                               for label in detected_labels]

            # ==> Those two sets of "half" objects will be merged to give
            # complete DetectedPanel objects (with label, panel_rect and label_rect fields).

            # TODO remove
            figure.detected_panels = detected_panels
            figure.detected_panels.extend(detected_labels)

            figure.show_preview(mode='pred', delay=0)


            # self._beam_search_mapping(panels=predicted_panels,
                                      # labels=predicted_labels)

            # beam_search.assign_labels_to_panels(detected_panels,
                                                # detected_labels)

            # figure.detected_panels = detected_panels

            # TODO do some post processing here maybe

            # figure.detected_panels = predicted_panels

            yield figure
