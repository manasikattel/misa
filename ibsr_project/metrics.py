import numpy as np
import SimpleITK as sitk


def hausdorff_distance(in1, in2, label="all"):
    """
    Compute hausdorff distance

    Parameters
    ----------
    in1 : sitkImage
    in2 : sitkImage
    label : str, optional
        label to be considered foreground, by default "all"

    Returns
    -------
    float
        hausdorff distance value
    """
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    if label == "all":
        hausdorff_distance_filter.Execute(in1, in2)
    else:

        in1_array = sitk.GetArrayFromImage(in1)
        in1_array = (in1_array == label) * 1
        in1_array = in1_array.astype("uint16")
        img1 = sitk.GetImageFromArray(in1_array)

        in2_array = sitk.GetArrayFromImage(in2)
        in2_array = (in2_array == label) * 1
        in2_array = in2_array.astype("uint16")
        img2 = sitk.GetImageFromArray(in2_array)
        hausdorff_distance_filter.Execute(img1, img2)
    return hausdorff_distance_filter.GetHausdorffDistance()


def volumetric_difference(in1, in2, label="all"):
    """
    Compute volumetric difference

    Parameters
    ----------
    in1 : ndarray
    in2 : ndarray
    label : str, optional
        label to be considered foreground, by default "all"

    Returns
    -------
    float
        volumetric distance
    """
    if label == "all":
        return np.sum((in1 != in2)) / ((np.sum(in1 > 0) + np.sum(in2 > 0)))

    else:
        in1 = (in1 == label) * 1
        in2 = (in2 == label) * 1
        return np.sum((in1 != in2)) / ((np.sum(in1 > 0) + np.sum(in2 > 0)))


def dice_score(in1, in2, label="all"):
    """
    Compute dice score

    Parameters
    ----------
    in1 : ndarray
    in2 : ndarray
    label : str, optional
        label to be considered foreground, by default "all"

    Returns
    -------
    float
        dice score
    """
    if label == "all":
        return (
            2
            * np.sum((in1 > 0) & (in2 > 0) & (in1 == in2))
            / (np.sum(in1 > 0) + np.sum(in2 > 0))
        )
    else:
        return (
            2
            * np.sum((in1 == label) & (in2 == label))
            / (np.sum(in1 == label) + np.sum(in2 == label))
        )
