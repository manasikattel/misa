# import
import os
import sys
from pathlib import Path

import SimpleITK as sitk

#todo change this to original dataset
images_path = Path('../TrainingValidationTestSets/Training_Set')
labels_path = Path('../TrainingValidationTestSets/Training_Set')
# masks_path = Path('../test-set/testing-mask')

data_files = os.listdir(images_path)
labels_files = os.listdir(labels_path)
# masks_files = os.listdir(masks_path)

sys.setrecursionlimit(4000)

data_files.sort(key=lambda x: x.split('.')[0])
labels_files.sort(key=lambda x: x.split('_')[0])
# masks_files.sort(key=lambda x: x.split('_')[0])
use_mask = True
# data_files[0]
# reg_methods = ['affine_transform.txt', 'non_rigid_transform.txt', 'rigid_transform.txt']
# reg_param_texts = ['affine', 'bspline', 'rigid', ]

reg_methods = ['non_rigid_transform.txt']
reg_param_texts = ['bspline']

# todo change this to your transformation path file
transformation_mask_path = '../bspline_training/processed/'

#todo change to ur desired output folder
out_folder = Path(f'./training_ours_reg/processed')

training_images_path = out_folder / Path('training-images')
training_labels_path = out_folder / Path('training-labels')

##put it in folders all labels and fixed
for reg_method, reg_param_text in zip(reg_methods, reg_param_texts):

    for data_moving, moving_label_file in zip(data_files, labels_files):
        print(reg_method, data_moving)

        in_folder = transformation_mask_path / Path(f'reg_{data_moving}')

        training_images_path.mkdir(parents=True, exist_ok=True)
        training_labels_path.mkdir(parents=True, exist_ok=True)


        #todo change this to normal or preprocessed
        moving_image = sitk.ReadImage(os.path.join(images_path, data_moving, data_moving + '_preprocessed_histmatched.nii.gz'))
        moving_label = sitk.ReadImage(os.path.join(labels_path, moving_label_file, moving_label_file + '_seg.nii.gz'),
                                      sitk.sitkUInt8)

        # parameterMap0 = sitk.ReadParameterFile(str(in_folder / Path('TransformParameters.0.txt')))

        # Transform label map using the deformation field from above BSplineInterpolator
        # registered_image = sitk.Transformix(moving_image, parameterMap0)
        parameter0 = sitk.ReadParameterFile(str(in_folder / Path('TransformParameters.0.txt')))
        parameter1 = sitk.ReadParameterFile(str(in_folder / Path('TransformParameters.1.txt')))

        myfilter = sitk.TransformixImageFilter()

        myfilter.SetTransformParameterMap(parameter0)
        myfilter.AddTransformParameterMap(parameter1)

        myfilter.SetMovingImage(moving_image)
        registered_image = myfilter.Execute()

        # labels should be interpolated by nearest neighbors
        # parameterMap0["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

        parameter0["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        parameter1["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

        myfilter.SetMovingImage(moving_label)
        registered_label = myfilter.Execute()

        # registered_label = sitk.Transformix(moving_label, parameterMap0)

        sitk.WriteImage(registered_image, str(training_images_path / Path(data_moving + '.nii.gz')))

        sitk.WriteImage(registered_label, str(training_labels_path / Path(moving_label_file + '_3C.nii.gz')))
