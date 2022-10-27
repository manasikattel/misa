import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

image_files = [i for i in Path("atlas/data/registered/training_images").rglob("*.nii")]
label_files = [
    Path(str(i.parent).replace("images", "labels")) / Path(i.stem + "_3C.nii")
    for i in image_files
]

CSFLabel = 1
GMLabel = 3
WMLabel = 2


def get_atlas(images, labels):
    images_stacked = np.stack(images)
    labels_stacked = np.stack(labels)

    csf_stacked = labels_stacked * (labels_stacked == CSFLabel)
    gm_stacked = labels_stacked * (labels_stacked == GMLabel)
    wm_stacked = labels_stacked * (labels_stacked == WMLabel)

    prob_atlas_CSF = np.average(csf_stacked, axis=0)
    prob_atlas_GM = np.average(gm_stacked, axis=0)
    prob_atlas_WM = np.average(wm_stacked, axis=0)
    return prob_atlas_CSF / CSFLabel, prob_atlas_GM / GMLabel, prob_atlas_WM / WMLabel


def main(image_files, label_files):
    save_dir = Path("atlas/output")
    save_dir.mkdir(exist_ok=True, parents=True)
    images = [nib.load(i).get_fdata() for i in image_files]
    labels = [nib.load(i).get_fdata() for i in label_files]
    csf_atlas, gm_atlas, wm_atlas = get_atlas(images, labels)

    plt.imshow(csf_atlas[:, :, 128])
    plt.show()
    plt.imshow(gm_atlas[:, :, 128])
    # plt.imsave(gm_atlas[:, :, 128], "gmatlas.png")
    plt.show()
    plt.imshow(wm_atlas[:, :, 128])
    plt.show()
    # plt.imsave(wm_atlas[:, :, 128], "wmatlas.png")

    csf_atlas_nifti = nib.Nifti1Image(csf_atlas, affine=np.eye(4))
    gm_atlas_nifti = nib.Nifti1Image(gm_atlas, affine=np.eye(4))
    wm_atlas_nifti = nib.Nifti1Image(wm_atlas, affine=np.eye(4))

    nib.save(csf_atlas_nifti, save_dir / Path("atlasCSF.nii"))
    nib.save(gm_atlas_nifti, save_dir / Path("atlasGM.nii"))
    nib.save(wm_atlas_nifti, save_dir / Path("atlasWM.nii"))


if __name__ == "__main__":
    main(image_files, label_files)
