"""
Deal with annotations from ImageCLEF xml files.
"""

def convert_annotations_iphotdraw_to_csv(
        eval_list_csv: str,
        output_csv_file: str = None):
    """
    Convert annotations from a xml file (ImageCLEF format) to a CSV annotations file

    Args:
        eval_list_csv: the path of the list of figures which annotations
            have to be loaded
        output_csv_file: the path where to store the newly created csv file
    """
    


def extract_bbox_from_iphotodraw_node(item, image_width, image_height):
    """
    Extract bounding box information from Element item (ElementTree).
    It also makes sure that the bounding box is within the image.

    Args:
        item: TODO
        image_width: TODO
        image_height: TODO

    Returns:
        return: x_min, y_min, x_max, y_max
    """
    extent_item = item.find('./Data/Extent')

    # Get data from the xml item
    height_string = extent_item.get('Height')
    width_string = extent_item.get('Width')

    x_string = extent_item.get('X')
    y_string = extent_item.get('Y')

    # Compute coordinates of the bounding box
    x_min = round(float(x_string))
    y_min = round(float(y_string))
    x_max = x_min + round(float(width_string))
    y_max = y_min + round(float(height_string))

    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max > image_width:
        x_max = image_width
    if y_max > image_height:
        y_max = image_height

    return x_min, y_min, x_max, y_max

