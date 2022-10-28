import nibabel as nib
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

output_folder = Path("../Elastix/training_reg/non_rigid_transform.txt/True/1000.nii.gz")

image_files = [i for i in (output_folder / Path("training_images")).rglob("*.nii.gz")]
label_files = [
    Path(str(i.parent).replace("images", "labels")) / Path(Path(i.stem).stem + "_3C.nii.gz")
    for i in image_files
]

def strip_skull(image, mask):
    gt_mask = copy.deepcopy(mask)
    gt_mask[gt_mask > 0] = 1
    return np.multiply(image, gt_mask)


def visualize(image, label, filename):

    filename.parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(1, 3)
    print(np.unique(label))
    ax[0].imshow(image[:, :, 128])
    ax[1].imshow(label[:, :, 128])
    ax[2].hist(image.flatten())

    fig.savefig(filename.parent / Path(filename.stem + ".png"))


def calc_prob_sum1(image, label):
    image = image.astype(int)

    csf = image * (label == 1)
    gm = image * (label == 3)
    wm = image * (label == 2)

    # csf = csf[csf > 0]
    # mask = label >= 1
    image = image.flatten()
    label = label.flatten()

    # p_tissue_csf = sum(label == 1) / len(image)
    # p_tissue_gm = sum(label == 3) / len(image)
    # p_tissue_wm = sum(label == 2) / len(image)

    onlybrain = image[label != 0]
    mask = label[label != 0]
    csf = onlybrain[mask == 1]
    gm = onlybrain[mask == 3]
    wm = onlybrain[mask == 2]

    intensity_hist = np.bincount(onlybrain, minlength=np.amax(onlybrain))

    p_intensity_csf = (
        np.bincount(csf, minlength=np.amax(onlybrain) + 1) / intensity_hist
    )
    p_intensity_gm = np.bincount(gm, minlength=np.amax(onlybrain) + 1) / intensity_hist

    # / intensity_hist
    p_intensity_wm = np.bincount(wm, minlength=np.amax(onlybrain) + 1) / intensity_hist

    # / intensity_hist
    plt.plot(range(len(p_intensity_csf)), p_intensity_csf, alpha=0.5, label="csf")
    plt.plot(range(len(p_intensity_gm)), p_intensity_gm, alpha=0.5, label="gm")
    plt.plot(range(len(p_intensity_wm)), p_intensity_wm, alpha=0.5, label="wm")
    plt.legend()
    plt.savefig(output_folder/Path('intensity_prob.png'))
    plt.show()



def calc_prob(image, label):

    image = image.astype(int)

    csf = image * (label == 1)
    gm = image * (label == 3)
    wm = image * (label == 2)

    # csf = csf[csf > 0]
    # mask = label >= 1
    image = image.flatten()
    label = label.flatten()

    # p_tissue_csf = sum(label == 1) / len(image)
    # p_tissue_gm = sum(label == 3) / len(image)
    # p_tissue_wm = sum(label == 2) / len(image)

    onlybrain = image[label != 0]
    mask = label[label != 0]
    csf = onlybrain[mask == 1]
    gm = onlybrain[mask == 3]
    wm = onlybrain[mask == 2]

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
    plt.savefig(output_folder/Path('intensity_hist.png'))

    plt.show()


def main(image_files, label_files):
    images = [nib.load(i).get_fdata() for i in image_files]
    labels = [nib.load(i).get_fdata() for i in label_files]
    skull_stripped = [strip_skull(images[i], labels[i]) for i in range(len(images))]

    calc_prob(np.stack(skull_stripped), np.stack(labels))
    calc_prob_sum1(np.stack(skull_stripped), np.stack(labels))


if __name__ == "__main__":
    main(image_files, label_files)
