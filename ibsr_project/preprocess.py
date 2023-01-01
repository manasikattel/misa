import nibabel as nib
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm
import SimpleITK as sitk
from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.fcm import FCMNormalize
from scipy.ndimage._filters import gaussian_filter


datadir = Path("data/project_data/")

image_files = [
    Path(str(i) + f"/{str(i.stem)}.nii.gz")
    for i in (datadir / Path("Training_Set")).iterdir()
    if "IBSR" in i.stem
]


def preprocess(image, blur_sigma=0.5):
    image = sitk.Cast(image, sitk.sitkFloat32)
    bias_corrected = sitk.N4BiasFieldCorrection(image)
    bias_corrected_arr = sitk.GetArrayFromImage(bias_corrected)
    fcm_norm = FCMNormalize(tissue_type=TissueType.WM)
    normalized = fcm_norm(bias_corrected_arr)
    gaussian_applied_arr = gaussian_filter(normalized, blur_sigma)
    gaussian_applied = sitk.GetImageFromArray(gaussian_applied_arr)
    gaussian_applied.CopyInformation(image)
    return gaussian_applied


def histogram_matching(image, reference_image):
    matching = sitk.HistogramMatchingImageFilter()
    matching.ThresholdAtMeanIntensityOn()
    matched_image = matching.Execute(image, reference_image)
    return matched_image


def save_hist_matched(image_dir, reference_image_path):
    image_files = [
        i for i in image_dir.rglob("*_preprocessed.nii.gz") if "seg" not in str(i)
    ]
    images_prep = [sitk.ReadImage(i) for i in image_files]
    reference_image = sitk.ReadImage(reference_image_path)
    i = 0
    for image in tqdm(images_prep):
        matched = histogram_matching(image, reference_image)
        save_path = image_files[i].parent / image_files[i].name.replace(
            ".nii.gz", "_histmatched.nii.gz"
        )
        sitk.WriteImage(matched, save_path)
        i = i + 1


if __name__ == "__main__":
    images = [sitk.ReadImage(i) for i in image_files]
    print("preprocessing...")
    # for i in tqdm(range(len(images))):
    #     preprocessed = preprocess(images[i])
    #     save_path = image_files[i].parent / image_files[i].name.replace(
    #         ".nii.gz", "_preprocessed.nii.gz"
    #     )
    #     sitk.WriteImage(preprocessed, save_path)
    #     print(f"Preprocessed image saved to {save_path}")

    reference_image_path = (
        datadir / "Training_Set/IBSR_12" / "IBSR_12_preprocessed.nii.gz"
    )
    print("Histogram Matching...")

    save_hist_matched(datadir / "Training_Set", reference_image_path)
