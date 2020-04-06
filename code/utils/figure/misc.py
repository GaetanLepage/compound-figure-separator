"""
Miscellaneous functions for figures.
"""

def extract_bbox_from_iphotodraw_node(item, image_width, image_height):
    """
    Extract bounding box information from Element item (ElementTree).
    It also makes sure that the bounding box is within the image.

    Args:
        item:
        image_width:
        image_height:

    Returns:
        return: x_min, y_min, x_max, y_max
    """
    extent_item = item.find('./Data/Extent')
    height = extent_item.get('Height')
    width = extent_item.get('Width')
    x = extent_item.get('X')
    y = extent_item.get('Y')
    x_min = round(float(x))
    y_min = round(float(y))
    x_max = x_min + round(float(width))
    y_max = y_min + round(float(height))
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > image_width:
        x_max = image_width
    if y_max > image_height:
        y_max = image_height

    return x_min, y_min, x_max, y_max


def assign_labels_to_panels(panels, labels):
    """
    Use beam search to assign labels to panels according to the overall distance
    Assign labels.label_rect to panels.label_rect
    panels and labels must have the same length

    Args:
        panels: panels having the same label character
        labels: labels having the same label character
    """
    # calculate distance from panels to labels
    distances = []
    for panel_i, panel in enumerate(panels):
        dists = []
        panel_rect = panel.panel_rect
        panel_c = [(panel_rect[0] + panel_rect[2])/2.0, (panel_rect[1] + panel_rect[3])/2.0]
        for label_i, label in enumerate(labels):
            label_rect = label.label_rect
            label_c = [(label_rect[0] + label_rect[2])/2.0, (label_rect[1] + label_rect[3])/2.0]

            d = math.hypot(panel_c[0] - label_c[0], panel_c[1] - label_c[1])
            dists.append(d)
        distances.append(dists)

    # Beam search
    beam_length = 100
    all_item_pairs = []  # in the format (overall_distance, label_indexes)
    for panel_i, panel in enumerate(panels):
        item_pairs = []
        if panel_i == 0:
            for label_i, label in enumerate(labels):
                dist = distances[panel_i][label_i]
                label_indexes = [label_i]
                item_pair = [dist, label_indexes]
                item_pairs.append(item_pair)
        else:
            prev_item_pairs = all_item_pairs[panel_i - 1]
            for pair_i, prev_item_pair in enumerate(prev_item_pairs):
                prev_label_indexes = prev_item_pair[1]
                prev_dist = prev_item_pair[0]
                for label_i, label in enumerate(labels):
                    if label_i in prev_label_indexes:
                        # We allow a label assigned to one panel only
                        continue
                    dist = distances[panel_i][label_i] + prev_dist
                    label_indexes = list(prev_item_pair[1])
                    label_indexes.append(label_i)
                    item_pair = [dist, label_indexes]
                    item_pairs.append(item_pair)

        # sort item_pairs
        item_pairs.sort(key=lambda pair: pair[0])
        # keep only at most beam_length item pairs
        if len(item_pairs) > 100:
            item_pairs = item_pairs[:beam_length]

        all_item_pairs.append(item_pairs)

    # check the last item_pairs
    best_path = all_item_pairs[-1][0][1]
    for i in range(len(panels)):
        panels[i].label_rect = labels[best_path[i]].label_rect


def read_sample_list(list_path):
    """
    TODO

    Parameters :
        * list_path (TODO): TODO

    Returns :
        * TODO (TODO): TODO
    """
    # TODO
    # with tf.gfile.GFile(list_path) as fid:
        # lines = fid.readlines()
    # return [line.strip().split(' ')[0] for line in lines]
    pass


