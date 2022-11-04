import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk

small_dataset = './trainingSmall/'

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

# fixed_img=data_files[0]
Reg_MI = sitk.ImageRegistrationMethod()
Reg_MI.SetMetricAsMattesMutualInformation()

Reg_MSE = sitk.ImageRegistrationMethod()
Reg_MSE.SetMetricAsMeanSquares()

# use each image as the fixed image and pair it and check the csv file for the best image
# use mask = True
# make group wise segmentation and check for all pairs MI and mean values and compare
# register with the image masks
df_values_list = []

#############Metric Value###interpolator########

# label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
# label_shape_filter.Execute(moving_mask)
# bounding_box = label_shape_filter.GetBoundingBox(1)

# roi_filter=sitk.RegionOfInterestImageFilter()

# fixed_image = sitk.ReadImage(os.path.join(images_path, data_files[0]))
# fixed_mask = sitk.ReadImage(os.path.join(labels_path, labels_files[0]), sitk.sitkUInt8)
for data_fixed, labels_fixed in zip(data_files, labels_files):
    fixed_image = sitk.ReadImage(os.path.join(images_path, data_fixed))
    fixed_mask = sitk.ReadImage(os.path.join(labels_path, labels_fixed), sitk.sitkUInt8)

    for data_moving, label_moving in zip(data_files, labels_files):
        if data_moving == data_fixed:
            continue
        print(f'fixed image {data_fixed}, moving image {data_moving}')
        out_folder = Path(f'./{use_mask}/{data_fixed}/{data_moving}')

        out_folder.mkdir(parents=True, exist_ok=True)

        df_values = {}

        df_values['fixed_image'] = data_fixed
        df_values['moving_image'] = data_moving
        df_values['use_mask'] = use_mask


        moving_image = sitk.ReadImage(os.path.join(images_path, data_moving))

        elastixImageFilter = sitk.ElastixImageFilter()

        elastixImageFilter.SetFixedImage(fixed_image)
        elastixImageFilter.SetMovingImage(moving_image)

        if use_mask:
            moving_image_mask = sitk.ReadImage(os.path.join(labels_path, label_moving), sitk.sitkUInt8)

            elastixImageFilter.SetFixedMask(fixed_mask)
            elastixImageFilter.SetMovingMask(moving_image_mask)

        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid"))
        start_time = time.time()
        rigid_out_img = elastixImageFilter.Execute()
        print(f'rigid took {(time.time() - start_time)}')
        rigid_mutual_info_metric_value = Reg_MI.MetricEvaluate(fixed_image, rigid_out_img)
        rigid_mse_value = Reg_MSE.MetricEvaluate(fixed_image, rigid_out_img)

        df_values['rigid_mi'] = rigid_mutual_info_metric_value
        df_values['rigid_mse'] = rigid_mse_value

        # sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0], str(
        #     out_folder / Path('rigid_transform.txt')))

        ###############################Affine######################
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_image)
        elastixImageFilter.SetMovingImage(moving_image)

        if use_mask:
            elastixImageFilter.SetFixedMask(fixed_mask)
            elastixImageFilter.SetMovingMask(moving_image_mask)



        parameterMap = sitk.GetDefaultParameterMap('affine')
        parameterMap["fMask"] = ["true"]
        parameterMap["ErodeMask"] = ["true"]

        elastixImageFilter.SetParameterMap(parameterMap)
        start_time = time.time()
        affine_out_img = elastixImageFilter.Execute()
        print(f'affine took {(time.time() - start_time)}')

        # sitk.WriteImage(elastixImageFilter.GetResultImage())
        affine_mutual_info_metric_value = Reg_MI.MetricEvaluate(fixed_image, affine_out_img)
        affine_mse_value = Reg_MSE.MetricEvaluate(fixed_image, affine_out_img)
        df_values['affine_mi'] = affine_mutual_info_metric_value
        df_values['affine_mse'] = affine_mse_value

        # sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0],
        #                         str(out_folder / Path('affine_transform.txt')))

        ################Non Rigid Registration###############
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(fixed_image)
        elastixImageFilter.SetMovingImage(moving_image)

        if use_mask:
            elastixImageFilter.SetFixedMask(fixed_mask)
            elastixImageFilter.SetMovingMask(moving_image_mask)

        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
        parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
        elastixImageFilter.SetParameterMap(parameterMapVector)

        start_time = time.time()
        non_rigid_out_img = elastixImageFilter.Execute()
        print(f'bspline took {(time.time() - start_time)}')

        # sitk.WriteImage(elastixImageFilter.GetResultImage())
        non_rigid_mutual_info_metric_value = Reg_MI.MetricEvaluate(fixed_image, non_rigid_out_img)
        non_rigid_mse_value = Reg_MSE.MetricEvaluate(fixed_image, non_rigid_out_img)
        df_values['non_rigid_mi'] = non_rigid_mutual_info_metric_value
        df_values['non_rigid_mse'] = non_rigid_mse_value
        # sitk.WriteParameterFile(elastixImageFilter.GetTransformParameterMap()[0],
        #                         str(out_folder / Path('non_rigid_transform.txt')))
        df_values_list.append(df_values)
        # exit()
# df = pd.DataFrame(df_values_list)
# df.to_csv('reg_results_2.csv')
print(1)
