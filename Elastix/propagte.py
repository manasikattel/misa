# import
import os
from pathlib import Path

import SimpleITK as sitk

images_path = Path('../training_set/training_images')
labels_path = Path('../training_set/training_labels')
masks_path = Path('../training_set/training_mask')

data_files = os.listdir(images_path)
labels_files = os.listdir(labels_path)
masks_files = os.listdir(masks_path)

data_files.sort(key=lambda x: x.split('.')[0])
labels_files.sort(key=lambda x: x.split('_')[0])
masks_files.sort(key=lambda x: x.split('_')[0])
use_mask = False

# data_files[0]

reg_methods = ['non_rigid_transform.txt', 'affine_transform.txt', 'rigid_transform.txt']

##put it in folders all labels and fixed

for reg_method in reg_methods:

    for data_fixed, fixed_label_file in zip(data_files, labels_files):
        fixed_image = sitk.ReadImage(os.path.join(images_path, data_fixed))
        fixed_label = sitk.ReadImage(os.path.join(labels_path, fixed_label_file))

        for data_moving, moving_label_file in zip(data_files, labels_files):

            in_folder = Path(f'./{use_mask}/{data_fixed}/{data_moving}')

            out_folder = Path(f'./training_reg/{reg_method}/{use_mask}/{data_fixed}/')
            training_images_path = out_folder / Path('training_images')
            training_labels_path = out_folder / Path('training_labels')

            training_images_path.mkdir(parents=True, exist_ok=True)
            training_labels_path.mkdir(parents=True, exist_ok=True)

            moving_image = sitk.ReadImage(os.path.join(images_path, data_moving))
            moving_label = sitk.ReadImage(os.path.join(labels_path, moving_label_file), sitk.sitkUInt8)

            if data_fixed == data_moving:
                # save the original labels and images if the image using for registration is the same
                sitk.WriteImage(fixed_image, str(training_images_path / Path(data_fixed)))
                sitk.WriteImage(fixed_label, str(training_labels_path / Path(fixed_label_file)))
                continue

            parameterMap0 = sitk.ReadParameterFile(str(in_folder / Path(reg_method)))

            # Transform label map using the deformation field from above BSplineInterpolator
            registered_image = sitk.Transformix(moving_image, parameterMap0)

            # labels should be interpolated by nearest neighbors
            parameterMap0["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

            registered_label = sitk.Transformix(moving_label, parameterMap0)

            sitk.WriteImage(registered_image, str(training_images_path / Path(data_moving)))

            sitk.WriteImage(registered_label, str(training_labels_path / Path(moving_label_file)))
