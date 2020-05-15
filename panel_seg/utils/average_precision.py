"""
TODO
"""

import numpy as np

def compute_average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    TODO

    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));

    Args:
        rec (np.array[float]):  TODO
        prec (np.array[float]): TODO

    Returns:
        AP (float): the resulting average precision.
    """

    # insert 0.0 at begining of list
    np.insert(rec, 0, 0.0)

    # insert 1.0 at end of list
    np.append(rec, 1.0)

    # Copy the array
    mrec = rec.copy()

    # insert 0.0 at begining of list
    np.insert(prec, 0, 0.0)

    # insert 1.0 at end of list
    np.append(prec, 1.0)

    # Copy the array
    mpre = prec.copy()

    # Make the precision monotonically decreasing (goes from the end to the beginning)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    # Create a list of indexes where the recall changes
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    # The Average Precision (AP) is the area under the curve
    #    (numerical integration)

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    return ap, mrec, mpre
