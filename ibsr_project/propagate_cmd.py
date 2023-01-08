# import
import os
import sys
from pathlib import Path

import SimpleITK as sitk


#read data and param files parmaeter files
transformation_mask_path = '../bspline_training/processed/'
images_path = Path('../TrainingValidationTestSets/Training_Set')
labels_path = Path('../TrainingValidationTestSets/Training_Set')

#create outfolder for the registered images
out_folder = Path(f'./training_ours_reg/processed')
training_images_path = out_folder / Path('training-images')
training_labels_path = out_folder / Path('training-labels')
training_images_path.mkdir(parents=True, exist_ok=True)
training_labels_path.mkdir(parents=True, exist_ok=True)

data_files = os.listdir(images_path)
labels_files = os.listdir(labels_path)

sys.setrecursionlimit(4000)

data_files.sort(key=lambda x: x.split('.')[0])
labels_files.sort(key=lambda x: x.split('_')[0])


for data_moving, moving_label_file in zip(data_files, labels_files):

    #get the param file for the datamoving
    in_folder = transformation_mask_path / Path(f'reg_{data_moving}')


    #read the moving data and the moving label
    moving_image = sitk.ReadImage(os.path.join(images_path, data_moving, data_moving + '.nii.gz'))
    moving_label = sitk.ReadImage(os.path.join(labels_path, moving_label_file, moving_label_file + '_seg.nii.gz'),
                                  sitk.sitkUInt8)

    #read the affine and bspline param file
    parameter0 = sitk.ReadParameterFile(str(in_folder / Path('TransformParameters.0.txt')))
    parameter1 = sitk.ReadParameterFile(str(in_folder / Path('TransformParameters.1.txt')))

    #execute the param files seq.
    myfilter = sitk.TransformixImageFilter()

    #param of affine
    myfilter.SetTransformParameterMap(parameter0)
    #param of bspline
    myfilter.AddTransformParameterMap(parameter1)

    #set the moving image
    myfilter.SetMovingImage(moving_image)
    #execute the filter and retain the registered image
    registered_image = myfilter.Execute()

    #change interpolation to execute on the labels
    parameter0["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    parameter1["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

    #execute the filter on the labels
    myfilter = sitk.TransformixImageFilter()

    myfilter.SetTransformParameterMap(parameter0)
    myfilter.AddTransformParameterMap(parameter1)

    myfilter.SetMovingImage(moving_label)
    registered_label = myfilter.Execute()

    #write the registered image and label
    sitk.WriteImage(registered_image, str(training_images_path / Path(data_moving + '.nii.gz')))
    sitk.WriteImage(registered_label, str(training_labels_path / Path(moving_label_file + '_3C.nii.gz')))
