# import
import os
import sys
from pathlib import Path

import SimpleITK as sitk

images_path = Path('../training_set/training_images')
labels_path = Path('../training_set/training_labels')
masks_path = Path('../training_set/training_mask')

data_files = os.listdir(images_path)
labels_files = os.listdir(labels_path)
masks_files = os.listdir(masks_path)

sys.setrecursionlimit(4000)

data_files.sort(key=lambda x: x.split('.')[0])
labels_files.sort(key=lambda x: x.split('_')[0])
masks_files.sort(key=lambda x: x.split('_')[0])
use_mask = True

# data_files[0]

reg_methods = ['affine_transform.txt', 'non_rigid_transform.txt', 'rigid_transform.txt']
reg_param_texts = ['affine', 'bspline', 'rigid', ]
transformation_mask_path = '../reg_affine_bspline_rigid'
##put it in folders all labels and fixed

for reg_method, reg_param_text in zip(reg_methods[1:], reg_param_texts[1:]):

    for data_fixed, fixed_label_file in zip(data_files[:1], labels_files[:1]):
        fixed_image = sitk.ReadImage(os.path.join(images_path, data_fixed))
        fixed_label = sitk.ReadImage(os.path.join(labels_path, fixed_label_file))

        for data_moving, moving_label_file in zip(data_files, labels_files):
            print(reg_method, data_fixed, data_moving)

            in_folder = transformation_mask_path/Path(f'reg_{reg_param_text}_{data_moving[:-3]}')

            out_folder = Path(f'./training_reg/{reg_method}/{use_mask}/{data_fixed}/')

            training_images_path = out_folder / Path('training_images')
            training_labels_path = out_folder / Path('training_labels')

            training_images_path.mkdir(parents=True, exist_ok=True)
            training_labels_path.mkdir(parents=True, exist_ok=True)

            moving_image = sitk.ReadImage(os.path.join(images_path, data_moving))
            moving_label = sitk.ReadImage(os.path.join(labels_path, moving_label_file), sitk.sitkUInt8)

            parameterMap0 = sitk.ReadParameterFile(str(in_folder / Path('TransformParameters.0.txt')))

            # Transform label map using the deformation field from above BSplineInterpolator
            registered_image = sitk.Transformix(moving_image, parameterMap0)

            # labels should be interpolated by nearest neighbors
            parameterMap0["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

            registered_label = sitk.Transformix(moving_label, parameterMap0)

            sitk.WriteImage(registered_image, str(training_images_path / Path(data_moving)))

            sitk.WriteImage(registered_label, str(training_labels_path / Path(moving_label_file)))
