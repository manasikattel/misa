from pathlib import Path
from sys import breakpointhook
from tkinter import N
import numpy as np
import nibabel as nib
import numpy_indexed as npi
from .tissue_map import calc_prob_sum1
import pandas as pd
import matplotlib.pyplot as plt

image_path = Path("atlas/data/test-set/testing-images")


def segment_intensity_only(image, tissue_maps):
    intensity, wm, gm, csf = tissue_maps[
        ["intensity", "p_intensity_wm", "p_intensity_gm", "p_intensity_csf"]
    ].T.values
    # breakpoint()
    image_wm = image.copy()
    image_gm = image.copy()
    image_csf = image.copy()
    image_bg = (image == 0).astype(int)

    for intensity, prob in enumerate(wm):
        image_wm[image_wm == intensity] = prob
    for intensity, prob in enumerate(gm):
        image_gm[image_gm == intensity] = prob
    for intensity, prob in enumerate(csf):
        image_csf[image_csf == intensity] = prob

    stacked = np.stack((image_bg, image_csf, image_wm, image_gm))
    labels = np.argmax(stacked, axis=0)

    return stacked, labels


# def segment_position_only(image, tissue_maps):
# image


def normalize(image, max_val):
    image *= float(max_val) / image.max()
    image = image.astype(int)
    image = image.astype(np.float64)

    return image


def strip_skull(image, mask):
    return np.multiply(image, mask)


if __name__ == "__main__":
    image_files = list(image_path.rglob("*nii.gz"))
    mask_files = [
        str(i)
        .replace("testing-images", "testing-mask")
        .replace(".nii.gz", "_1C.nii.gz")
        for i in image_files
    ]
    images = [nib.load(i).get_fdata() for i in image_files]

    masks = [nib.load(i).get_fdata() for i in mask_files]
    nif_image = nib.load(image_files[0])

    skull_stripped = [strip_skull(image, mask) for image, mask in zip(images, masks)]
    skull_stripped_normalized = [normalize(s, 255) for s in skull_stripped]

    tissue_maps = pd.read_csv("prob_df_no1_255.csv")

    output_folder = Path("data/output/intensity")
    save_path_probs = output_folder / Path("probs")
    save_path_labels = output_folder / Path("labels")
    save_path_probs.mkdir(exist_ok=True, parents=True)
    save_path_labels.mkdir(exist_ok=True, parents=True)

    for img_path, img in zip(image_files, skull_stripped_normalized):
        stacked, labels = segment_intensity_only(img, tissue_maps)
        stacked_nifti = nib.Nifti1Image(stacked, nif_image.affine, nif_image.header)
        labels_nifti = nib.Nifti1Image(labels, nif_image.affine, nif_image.header)

        nib.save(stacked_nifti, save_path_probs / Path(img_path.name))
        nib.save(labels_nifti, save_path_labels / Path(img_path.name))
