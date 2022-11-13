import copy
import os
import pickle
from pathlib import Path
import pandas as pd
import pylab
import logging
import sys

from metrics import dice_coef_multilabel
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
import nibabel as nib
from EM import EM
import matplotlib.pyplot as plt
import numpy as np
from utils import read_img, flatten_img, get_features

# data_path = Path('./data')
# data_folders = os.listdir(data_path)

images_path = Path('../test-set/testing-images')
labels_path = Path('../test-set/testing-labels')
# masks_path = Path('../Elastix/training_reg/non_rigid_transform.txt/True/1000.nii.gz/training_mask')

data_files = os.listdir(images_path)
labels_files = os.listdir(labels_path)
# masks_files = os.listdir(masks_path)

data_files.sort(key=lambda x: x.split('.')[0])
labels_files.sort(key=lambda x: x.split('_')[0])
# masks_files.sort(key=lambda x: x.split('_')[0])

max_iter = 50
n_clusters = 3
error = 0.1
# to keep track for the results for ablation studies
results_list = []
output_path = Path('./test_results_3')
output_path.mkdir(parents=True, exist_ok=True)

use_tissue_model = False
use_atlas_model = False

visualize = False
# blur_sigmas=[None, 0.1, 0.3, 0.5, 0.7, 1]
blur_sigmas = [None]
# init_types = ['kmeans', 'random', 'atlas', 'tissue', 'atlas_tissue']
# mean_intensity, _ = read_img('../Elastix/training_reg/non_rigid_transform.txt/True/1000.nii.gz/mean_image.nii')
# init_types = ['atlas', 'kmeans']
init_types = ['tissue', 'atlas_tissue']

# kmeans init + mni + into
# use (atlas_ours * atlas tissue model) init into_EM True/False
# use (atlas tissue model) into_EM True/False / different atlas tissue model init
# use Kmeans atlas_init  + (atlas_ours * atlas tissue model) into_EM True

# atlas_type =both / tissue model alone
# init_type = both/tissue models/kmeans
# into_EM true/false if kmeans and into em==false do not run
# check mni experiment missing
for into_EM in [True, False]:
    for atlas_type in ['ours', 'mni']:
        atlas_testing_path = Path(f'../Elastix/test_reg_atlas_{atlas_type}')
        for blur_sigma in blur_sigmas:
            for use_T2 in [False]:
                for data_folder, labels_file in zip(data_files, labels_files):

                    # tm_subject_path = Path(f'../Elastix/test_reg_atlas_tm') / Path(data_folder)
                    # # load atlas for every test img
                    # tissue_model_CSF, _ = read_img(tm_subject_path / Path('atlasCSF.nii.gz'))
                    # tissue_model_GM, _ = read_img(tm_subject_path / Path('atlasGM.nii.gz'))
                    # tissue_model_WM, _ = read_img(tm_subject_path / Path('atlasWM.nii.gz'))

                    atlas_subject_path = atlas_testing_path / Path(data_folder)
                    # load atlas for every test img
                    atlas_model_BG, _ = read_img(atlas_subject_path / Path('atlasBG.nii.gz'))
                    atlas_model_CSF, _ = read_img(atlas_subject_path / Path('atlasCSF.nii.gz'))
                    atlas_model_GM, _ = read_img(atlas_subject_path / Path('atlasGM.nii.gz'))
                    atlas_model_WM, _ = read_img(atlas_subject_path / Path('atlasWM.nii.gz'))

                    atlas_model_BG = flatten_img(atlas_model_BG, mode='3d')
                    atlas_model_CSF = flatten_img(atlas_model_CSF, mode='3d')
                    atlas_model_GM = flatten_img(atlas_model_GM, mode='3d')
                    atlas_model_WM = flatten_img(atlas_model_WM, mode='3d')

                    # make sure
                    atlas_model = np.stack((atlas_model_BG, atlas_model_CSF, atlas_model_GM, atlas_model_WM),
                                           axis=1).squeeze()

                    tissue_model = np.stack((atlas_model_BG, atlas_model_CSF, atlas_model_GM, atlas_model_WM),
                                            axis=1).squeeze()

                    # tissue_model = np.stack((atlas_model_BG, tissue_model_CSF, tissue_model_WM, atlas_model_WM),
                    #                         axis=1).squeeze()

                    # del tissue_model_CSF, tissue_model_GM, tissue_model_WM

                    del atlas_model_BG, atlas_model_CSF, atlas_model_GM, atlas_model_WM

                    # reading patients files/gt
                    T1_fileName = images_path / data_folder
                    T2_fileName = images_path / data_folder
                    gt_fileName = labels_path / labels_file

                    T1, T1_affine = read_img(filename=T1_fileName, blur_sigma=blur_sigma)
                    if use_T2:
                        T2, _ = read_img(filename=T2_fileName, blur_sigma=blur_sigma)
                    else:
                        T2 = None

                    gt, gt_affine = read_img(filename=gt_fileName)

                    gt[(gt != 0) & (gt != 1) & (gt != 2) & (gt != 3)] = 0

                    for init_type in init_types:
                        if atlas_type == 'mni' and init_type == 'kmeans' and into_EM == False:
                            continue

                        if init_type == 'tissue':
                            atlas_model_init = tissue_model
                        elif init_type == 'atlas_tissue':
                            atlas_model_init = tissue_model * atlas_model
                        elif init_type == 'kmeans':
                            atlas_model_init = atlas_model
                        elif init_type == 'atlas':
                            atlas_model_init = atlas_model
                        else:
                            raise Exception("hey")
                        print('Use T2:{}, init_type:{}, data_folder:{}, atlas into {}'.format(use_T2, init_type,
                                                                                              data_folder, into_EM))
                        # keeping track of the results of every experiment
                        results_dict = {}
                        results_dict['use_T2'] = use_T2
                        results_dict['into_EM'] = into_EM
                        results_dict['init_type'] = init_type
                        results_dict['patient_id'] = data_folder
                        results_dict['blur_sigma'] = blur_sigma
                        results_dict['atlas_type'] = atlas_type

                        out_folder = output_path / data_folder / Path(
                            'init_' + init_type + '_atlas_' + atlas_type + '_blurred_' + str(
                                blur_sigma) + '_into_EM_' + str(into_EM))
                        out_folder.mkdir(parents=True, exist_ok=True)

                        # getting the features it can be T1 or (T1,T2)
                        stacked_features, T1_masked, T2_masked = get_features(T1, T2, gt, use_T2)

                        # saving the T1 masked image after removing skull
                        nib.save(nib.Nifti1Image(T1_masked, affine=T1_affine), out_folder / Path('T1_masked.nii'))
                        nib.save(nib.Nifti1Image(gt, affine=gt_affine), out_folder / Path('gt.nii'))

                        # initalizing the EM algorithm with the required segmentation
                        em = EM(stacked_features, atlas_model, init_type, n_clusters, T1.shape, into_EM=into_EM,
                                atlas_model_init=atlas_model_init)

                        # Executing the EM algorithm and return the segmented image
                        recovered_img, n_iter = em.execute(error, max_iter, visualize=visualize)

                        with open(out_folder / Path('log_likelihood.pkl'), 'wb') as f:
                            pickle.dump(em.log_likelihood_arr, f)

                        results_dict['converged_at'] = n_iter
                        results_dict['log_likelihood'] = em.log_likelihood
                        results_dict['em_time'] = em.em_time

                        results_dict['kmeans_dice_CSF'] = 0
                        results_dict['kmeans_dice_GM'] = 0
                        results_dict['kmeans_dice_WM'] = 0
                        results_dict['kmeans_time'] = 0

                        results_dict['atlas_dice_CSF'] = 0
                        results_dict['atlas_dice_GM'] = 0
                        results_dict['atlas_dice_WM'] = 0
                        results_dict['atlas_time'] = 0

                        seg_mask_em = em.mask_from_recovered(recovered_img)
                        seg_mask_atlas = em.atlas_model_mask

                        check_CSF = (seg_mask_atlas == 1).astype(np.uint8) * seg_mask_em
                        check_GM = (seg_mask_atlas == 2).astype(np.uint8) * seg_mask_em
                        check_WM = (seg_mask_atlas == 3).astype(np.uint8) * seg_mask_em
                        #
                        check_CSF, count_csf = np.unique(check_CSF, return_counts=True)
                        count_csf = np.argsort(-count_csf)

                        check_GM, count_gm = np.unique(check_GM, return_counts=True)
                        count_gm = np.argsort(-count_gm)
                        #
                        check_WM, count_wm = np.unique(check_WM, return_counts=True)
                        count_wm = np.argsort(-count_wm)
                        #
                        # # 0 is always the background
                        CSF_Label = count_csf[1]
                        GM_Label = count_gm[1]
                        WM_Label = count_wm[1]

                        CSF_Label = list(set([0, 1, 2, 3]).difference(set([0, GM_Label, WM_Label])))[0]

                        my_dict = {0: 0, CSF_Label: 1, GM_Label: 3, WM_Label: 2}
                        print(my_dict)
                        seg_mask_em = np.vectorize(my_dict.get)(seg_mask_em)

                        # print(my_dict)
                        if init_type == 'kmeans':
                            seg_mask_kmeans = em.kmeans_mask
                            # seg_mask_kmeans[np.isin(seg_mask_kmeans, list(my_dict.keys())) == 0] = 0
                            seg_mask_kmeans = np.vectorize(my_dict.get)(seg_mask_kmeans)

                            # saving the kmeans seg mask
                            nib.save(nib.Nifti1Image(seg_mask_kmeans, affine=gt_affine),
                                     out_folder / Path('kmeans_mask.nii'))
                            dice_list_kmeans = dice_coef_multilabel(gt, seg_mask_kmeans)

                            results_dict['kmeans_dice_CSF'] = dice_list_kmeans[1]
                            results_dict['kmeans_dice_GM'] = dice_list_kmeans[2]
                            results_dict['kmeans_dice_WM'] = dice_list_kmeans[3]
                            results_dict['kmeans_time'] = em.k_means_time
                            print('kmeans dice', dice_list_kmeans)
                        elif init_type == 'atlas' or init_type == 'tissue' or init_type == 'atlas_tissue':
                            seg_mask_atlas = em.atlas_model_mask
                            nib.save(nib.Nifti1Image(seg_mask_atlas, affine=gt_affine),
                                     out_folder / Path('atlas_mask.nii'))
                            my_dict_atlas = {0: 0, 1: 1, 2: 3, 3: 2}

                            dice_list_atlas = dice_coef_multilabel(gt, np.vectorize(my_dict_atlas.get)(seg_mask_atlas))
                            results_dict['atlas_dice_CSF'] = dice_list_atlas[1]
                            results_dict['atlas_dice_GM'] = dice_list_atlas[2]
                            results_dict['atlas_dice_WM'] = dice_list_atlas[3]
                            results_dict['atlas_time'] = em.atlas_time
                            print('atlas dice', dice_list_atlas)

                        results_dict['post_atlas_dice_CSF'] = 0
                        results_dict['post_atlas_dice_GM'] = 0
                        results_dict['post_atlas_dice_WM'] = 0

                        if atlas_type == 'ours':
                            my_dict_atlas = {0: 0, 1: 1, 2: 3, 3: 2}
                            if init_type == 'kmeans':
                                post_atlas_mask = em.get_post_atlas_mask(my_dict)
                            else:
                                post_atlas_mask = em.get_post_atlas_mask()

                            # post_atlas_mask[np.isin(post_atlas_mask, list(my_dict.keys())) == 0] = 0
                            post_atlas_mask = np.vectorize(my_dict_atlas.get)(post_atlas_mask)

                            nib.save(nib.Nifti1Image(post_atlas_mask, affine=gt_affine),
                                     out_folder / Path('post_atlas_mask.nii'))
                            dice_list_post_atlas = dice_coef_multilabel(gt, post_atlas_mask)
                            results_dict['post_atlas_dice_CSF'] = dice_list_post_atlas[1]
                            results_dict['post_atlas_dice_GM'] = dice_list_post_atlas[2]
                            results_dict['post_atlas_dice_WM'] = dice_list_post_atlas[3]
                            print('post atlas dice', dice_list_post_atlas)

                        results_dict['total_time'] = results_dict['kmeans_time'] + results_dict['em_time']

                        # seg_mask_em[np.isin(seg_mask_em, list(my_dict.keys())) == 0] = 0

                        # saving the seg mask of em
                        nib.save(nib.Nifti1Image(seg_mask_em, affine=gt_affine), out_folder / Path('em_mask.nii'))

                        dice_list_em = dice_coef_multilabel(gt, seg_mask_em)
                        results_dict['em_dice_CSF'] = dice_list_em[1]
                        results_dict['em_dice_GM'] = dice_list_em[2]
                        results_dict['em_dice_WM'] = dice_list_em[3]
                        print(dice_list_em)
                        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                        #
                        # ax1.imshow(T1_masked[:, :, 140])
                        # # plt.show()
                        # ax1.set_title('T1 image')
                        #
                        # ax2.imshow(gt[:, :, 140], cmap=pylab.cm.cool)
                        # # plt.show()
                        # ax2.set_title('Ground Truth')
                        # ax3.imshow(recovered_img[:, :, 140], cmap=pylab.cm.cool)
                        # ax3.set_title('Segmented Image')
                        # plt.title(T1_fileName)
                        # plt.tight_layout()
                        # plt.show()

                        results_list.append(results_dict)

# saving the results with the ablation to a dataframe
df = pd.DataFrame(results_list)
df.to_csv('results_kmeans_mni_ours_3.csv')
