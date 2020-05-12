"""
TODO
"""

import logging
from collections import defaultdict, OrderedDict

import torch

from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils import comm

from panel_seg.io.figure_generators import image_clef_xml_figure_generator
from panel_seg.utils.figure.panel import Panel
from panel_seg.panel_split.evaluate import evaluate_predictions


class PanelSplitEvaluator(DatasetEvaluator):
    """
    Class subclassing Detectron2's `DatasetEvaluator`.
    Perform the evaluation of panel splitting metrics on a given test set.
    """


    def __init__(self, dataset_name, output_dir):
        """
        init function.

        Args:
            dataset_name (str): name of the dataset
            output_dir (str): output directory to save results for evaluation
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predictions = dict()

        # TODO assert we are dealing with ImageCLEF
        self._xml_annotation_file_path = meta.xml_annotation_file_path
        self._image_directory_path = meta.image_directory_path


    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predictions = dict()


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


    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        # Gathering
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        # TODO make cases {Zou, ImageCLEF}

        figure_generator = image_clef_xml_figure_generator(
            xml_annotation_file_path=self._xml_annotation_file_path,
            image_directory_path=self._image_directory_path)

        def augmented_figure_generator():

            for figure in figure_generator:

                predicted_panels = predictions[figure.index]

                predicted_panel_objects = []

                for prediction in predicted_panels:
                    panel = Panel(panel_rect=prediction['box'])

                    predicted_panel_objects.append(panel)

                figure.pred_panels = predicted_panel_objects

                yield figure

        metrics_dict = evaluate_predictions(figure_generator=augmented_figure_generator())

        # Respect the expected result for a DatasetEvaluator
        return OrderedDict({'bbox': metrics_dict})
