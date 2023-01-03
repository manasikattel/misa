import nibabel as nib
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm
import SimpleITK as sitk

output_folder = Path("training_ours_reg/raw")

image_files = [i for i in (output_folder / Path("training-images")).rglob("*.nii.gz")]
label_files = [
    Path(str(i.parent).replace("images", "labels"))
    / Path(Path(i.stem).stem + "_3C.nii.gz")
    for i in image_files
]

CSFLabel = 1
GMLabel = 2
WMLabel = 3
BGLabel = 0


def visualize(image, label, filename):

    filename.parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(1, 3)
    print(np.unique(label))
    ax[0].imshow(image[:, :, 128])
    ax[1].imshow(label[:, :, 128])
    ax[2].hist(image.flatten())

    fig.savefig(filename.parent / Path(filename.stem + ".png"))


def calc_prob_sum1(image, label, visualize=True):
    # image *= 255.0 / image.max()
    image = image.astype(int)

    bg = image * (label == BGLabel)
    csf = image * (label == CSFLabel)
    gm = image * (label == GMLabel)
    wm = image * (label == WMLabel)

    # csf = csf[csf > 0]
    # mask = label >= 1
    image = image.flatten()
    label = label.flatten()

    # p_tissue_csf = sum(label == 1) / len(image)
    # p_tissue_gm = sum(label == 3) / len(image)
    # p_tissue_wm = sum(label == 2) / len(image)

    onlybrain = image[label != 0]
    mask = label[label != 0]
    bg = image[label == 0]
    csf = onlybrain[mask == CSFLabel]
    gm = onlybrain[mask == GMLabel]
    wm = onlybrain[mask == WMLabel]
    intensity_hist = np.bincount(image, minlength=np.amax(onlybrain))

    p_intensity_bg = np.bincount(bg, minlength=np.amax(image) + 1)
    # / intensity_hist

    p_intensity_csf = (
        np.bincount(csf, minlength=np.amax(onlybrain) + 1) / intensity_hist
    )
    p_intensity_gm = np.bincount(gm, minlength=np.amax(onlybrain) + 1) / intensity_hist

    # / intensity_hist
    p_intensity_wm = np.bincount(wm, minlength=np.amax(onlybrain) + 1) / intensity_hist

    # / intensity_hist
    if visualize:
        plt.plot(range(len(p_intensity_csf)), p_intensity_csf, alpha=0.5, label="csf")
        plt.plot(range(len(p_intensity_gm)), p_intensity_gm, alpha=0.5, label="gm")
        plt.plot(range(len(p_intensity_wm)), p_intensity_wm, alpha=0.5, label="wm")
        plt.plot(range(len(p_intensity_bg)), p_intensity_bg, alpha=0.5, label="bg")

        plt.legend()
        plt.savefig(output_folder / Path("intensity_prob_bg.png"))
        plt.show()

    return p_intensity_wm, p_intensity_gm, p_intensity_csf


def calc_prob(image, label):

    image = image.astype(int)

    csf = image * (label == CSFLabel)
    gm = image * (label == GMLabel)
    wm = image * (label == WMLabel)
    bg = image * (label == BGLabel)
    # csf = csf[csf > 0]
    # mask = label >= 1
    image = image.flatten()
    label = label.flatten()

    # p_tissue_csf = sum(label == 1) / len(image)
    # p_tissue_gm = sum(label == 3) / len(image)
    # p_tissue_wm = sum(label == 2) / len(image)

    onlybrain = image[label != 0]
    mask = label[label != 0]
    csf = onlybrain[mask == CSFLabel]
    gm = onlybrain[mask == GMLabel]
    wm = onlybrain[mask == WMLabel]

    bins = tuple(range(0, np.amax(onlybrain) + 1))
    CSF_hist, _ = np.histogram(csf, bins, density=True)
    GM_hist, _ = np.histogram(gm, bins, density=True)
    WM_hist, _ = np.histogram(wm, bins, density=True)
    bins = bins[:-1]
    # plot histograms
    plt.plot(bins, CSF_hist, bins, GM_hist, bins, WM_hist)
    # plt.title("CSF Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.legend(["CSF", "GM", "WM"])
    plt.savefig(output_folder / Path("intensity_hist.png"))

    plt.show()
    return WM_hist, GM_hist, CSF_hist


def model_args_fn(x):
    return x[:, 0], x[:, 1:]


def normalize(image, max_val):
    image = ((image - np.amin(image)) / (np.amax(image) - np.amin(image))) * float(
        max_val
    )
    image = image.astype(int)
    image = image.astype(np.float64)
    return image


def main(image_files, label_files):
    images = [sitk.GetArrayFromImage(sitk.ReadImage(str(i))) for i in image_files]
    labels = [sitk.GetArrayFromImage(sitk.ReadImage(str(i))) for i in label_files]
    # masks = [nib.load(i).get_fdata() for i in mask_files]

    skull_stripped_normalized = [normalize(i, 255) for i in images]
    p_intensity_wm, p_intensity_gm, p_intensity_csf = calc_prob(
        np.stack(skull_stripped_normalized), np.stack(labels)
    )

    # p_intensity_wm, p_intensity_gm, p_intensity_csf = calc_prob_sum1(
    #     np.stack(skull_stripped_normalized), np.stack(labels)
    # )
    prob_df = pd.DataFrame(
        {
            "intensity": range(len(p_intensity_csf)),
            "p_intensity_wm": p_intensity_wm,
            "p_intensity_gm": p_intensity_gm,
            "p_intensity_csf": p_intensity_csf,
        }
    )
    prob_df.to_csv("tissue_map.csv")

    prob_df["p_intensity_gm"] = prob_df["p_intensity_gm"].bfill().ffill()
    prob_df["p_intensity_wm"] = prob_df["p_intensity_wm"].bfill().ffill()
    prob_df["p_intensity_csf"] = prob_df["p_intensity_csf"].bfill().ffill()

    prob_df.to_csv("tissue_map_imputed.csv")


if __name__ == "__main__":
    main(image_files, label_files)
