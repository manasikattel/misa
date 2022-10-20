from pathlib import Path
import nibabel as nib
from scipy.cluster.vq import kmeans
import numpy as np
import matplotlib.pyplot as plt
import cv2
from monai.metrics import DiceMetric
import torch
from metrics import dice, dice_coef_multilabel

t1_image_filenames = [i for i in Path("data").rglob("*/*.nii") if "T1" in str(i)]


def cluster(vectorized):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 4
    attempts = 10
    ret, label, center = cv2.kmeans(
        vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    # centroids,distortion = kmeans(image,6)
    # print(ret, label,center)
    return ret, label, center


for image_filename in t1_image_filenames:
    image = nib.load(image_filename)
    gt = nib.load(image_filename.parent / Path("LabelsForTesting.nii"))
    gt_array = np.array(gt.get_fdata())
    data = np.array(image.get_fdata())
    # plt.hist(data.reshape((-1,1)))
    # plt.show()
    data_reshaped = data.reshape((-1, 1)).astype("float32")
    ret, label, center = cluster(data_reshaped)
    res = center[label.flatten()]
    result_image = res.reshape((data.shape))

    for label, val in enumerate(np.unique(result_image)):
        result_image[result_image == val] = label
    nifti_image = nib.Nifti1Image(result_image, affine=np.eye(4))
    print(
        f"Dice for {image_filename}",
        dice_coef_multilabel(
            torch.tensor(result_image).flatten(), torch.tensor(gt_array).flatten()
        ),
    )

    save_dir = Path("results") / image_filename.parent.stem
    save_dir.mkdir(exist_ok=True, parents=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    # print(dice_metric(y_pred=torch.tensor(result_image), y=torch.tensor(gt_array)))

    nib.save(nifti_image, save_dir / Path(image_filename.name))
