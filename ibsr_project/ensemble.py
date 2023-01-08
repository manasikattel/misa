import matplotlib.pyplot as plt
from scipy import stats
import SimpleITK as sitk
from pathlib import Path
import numpy as np

gt_dir = Path("data/project_data/Test_Set/")
datadir = Path("nnunet_infer")
sus = lambda x: "_preprocessed_histmatched.nii.gz" if "pre" in x else ".nii.gz"
dices = []
hausdorffs = []
volumetric_diff = []

test_set = ["02", "10", "15"]
val_set = ["13", "11", "17", "12", "14"]

for num in test_set:
    op_images_name = [
        i / f"IBSR_{num}{sus(i.stem)}"
        for i in datadir.iterdir()
        if ".DS_Store" not in str(i)
    ]

    gt_file = gt_dir / f"IBSR_{num}" / f"IBSR_{num}.nii.gz"
    gt_image = sitk.ReadImage(str(gt_file))
    op_arrs = np.array(
        [sitk.GetArrayFromImage(sitk.ReadImage(str(i))) for i in op_images_name]
    )
    # ensemble step
    output_array = np.expand_dims(stats.mode(op_arrs, axis=0).mode.squeeze(), axis=3)
    output_image = sitk.GetImageFromArray(output_array)
    output_image.CopyInformation(gt_image)

    # save the ensemble image
    sitk.WriteImage(
        output_image, str(gt_dir / f"IBSR_{num}" / f"IBSR_{num}_seg_nnunet_pred.nii.gz")
    )

    plt.imshow(output_array[145, :, :, 0])
    plt.show()
