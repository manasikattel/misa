import os
from pathlib import Path

import SimpleITK as sitk
from SimpleITK.SimpleITK import ProcessObject

images_path = Path('../training_set/training_images')
labels_path = Path('../training_set/training_labels')
masks_path = Path('../training_set/training_mask')
use_mask = False

data_files = os.listdir(images_path)
labels_files = os.listdir(labels_path)
masks_files = os.listdir(masks_path)

data_files.sort(key=lambda x: x.split('.')[0])
labels_files.sort(key=lambda x: x.split('_')[0])
masks_files.sort(key=lambda x: x.split('_')[0])

vectorOfImages = sitk.VectorOfImage()

origin = sitk.ReadImage(os.path.join(images_path, data_files[0]))

for filename in data_files:
    image = sitk.ReadImage(os.path.join(images_path, filename))
    image = sitk.Resample(image, origin, sitk.Transform(), sitk.sitkBSpline, 0, origin.GetPixelID())
    vectorOfImages.push_back(image)

image = sitk.JoinSeries(vectorOfImages)

# Register
elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(image)
elastixImageFilter.SetMovingImage(image)
elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('groupwise'))
image_group_wise = elastixImageFilter.Execute()

sitk.WriteImage(image_group_wise, "group.nii.gz")

print(1)
