import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

output_folder = Path("../Elastix/training_ours_reg/processed")

image_files = [i for i in (output_folder / Path("training-images")).rglob("*.nii.gz")]
label_files = [
    Path(str(i.parent).replace("images", "labels")) / Path(Path(i.stem).stem + "_3C.nii.gz")
    for i in image_files
]

CSFLabel = 1
GMLabel = 3
WMLabel = 2
BGLabel = 0


def get_atlas(images, labels):
    # images_stacked = np.stack(images)
    labels_stacked = np.stack(labels)

    csf_stacked = labels_stacked * (labels_stacked == CSFLabel)
    gm_stacked = labels_stacked * (labels_stacked == GMLabel)
    wm_stacked = labels_stacked * (labels_stacked == WMLabel)
    bg_stacked = (labels_stacked == BGLabel).astype(np.uint8)

    prob_atlas_CSF = np.average(csf_stacked, axis=0)
    prob_atlas_GM = np.average(gm_stacked, axis=0)
    prob_atlas_WM = np.average(wm_stacked, axis=0)
    prob_atlas_BG = np.average(bg_stacked, axis=0)

    return prob_atlas_CSF / CSFLabel, prob_atlas_GM / GMLabel, prob_atlas_WM / WMLabel, prob_atlas_BG


def get_mean_image(images):
    images_stacked = np.stack(images)
    mean_image = np.average(images_stacked, axis=0)
    return mean_image


def main(image_files, label_files, output_folder):
    save_dir = output_folder
    save_dir.mkdir(exist_ok=True, parents=True)
    images = [nib.load(i).get_fdata() for i in image_files]
    labels = [nib.load(i).get_fdata() for i in label_files]

    csf_atlas, gm_atlas, wm_atlas, bg_atlas = get_atlas(images, labels)
    #
    # plt.imshow(csf_atlas[:, :, 128])
    # plt.show()
    # plt.imshow(gm_atlas[:, :, 128])
    # # plt.imsave(gm_atlas[:, :, 128], "gmatlas.png")
    # plt.show()
    # plt.imshow(wm_atlas[:, :, 128])
    # plt.show()
    #
    # plt.imshow(bg_atlas[:, :, 128])
    # plt.show()
    # # plt.imsave(wm_atlas[:, :, 128], "wmatlas.png")

    mean_image = get_mean_image(images)
    # plt.imshow(mean_image[:, :, 128])
    # plt.show()
    #
    affine = nib.load(image_files[0]).affine
    csf_atlas_nifti = nib.Nifti1Image(csf_atlas, affine=affine)
    gm_atlas_nifti = nib.Nifti1Image(gm_atlas, affine=affine)
    wm_atlas_nifti = nib.Nifti1Image(wm_atlas, affine=affine)
    bg_atlas_nifti = nib.Nifti1Image(bg_atlas, affine=affine)

    mean_image_nifti = nib.Nifti1Image(mean_image, affine=affine)

    nib.save(csf_atlas_nifti, save_dir / Path("atlasCSF.nii"))
    nib.save(gm_atlas_nifti, save_dir / Path("atlasGM.nii"))
    nib.save(wm_atlas_nifti, save_dir / Path("atlasWM.nii"))
    nib.save(bg_atlas_nifti, save_dir / Path("atlasBG.nii"))
    nib.save(mean_image_nifti, save_dir / Path("mean_image.nii"))


if __name__ == "__main__":
    main(image_files, label_files, output_folder)
