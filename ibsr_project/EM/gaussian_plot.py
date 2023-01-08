import copy
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import read_img

data_path = Path('../data')
output_path = Path('../results')
use_T2 = True
init_type = 'kmeans'

blur_sigmas = [0.1, 0.3, 0.5, 0.7, 1]
data_folders = os.listdir(data_path)
fig, ax = plt.subplots()

for data_folder in data_folders:
    T1_fileName = data_path / data_folder / Path('T1.nii')
    gt_fileName = data_path / data_folder / Path('LabelsForTesting.nii')

    T1, T1_affine = read_img(filename=T1_fileName, blur_sigma=None)
    gt, gt_affine = read_img(filename=gt_fileName)

    gt_mask = copy.deepcopy(gt)
    gt_mask[gt_mask > 0] = 1
    T1_masked = np.multiply(T1, gt_mask)

    nz_indices = [i for i, x in enumerate(T1_masked) if x.any()]

    data = T1_masked[nz_indices]
    data = data.flatten()
    data_gt = gt[nz_indices]
    data_gt = data_gt.flatten()

    CSF_hist = data[data_gt == 1]
    GM_hist = data[data_gt == 2]
    WM_hist = data[data_gt == 3]
    sns.distplot(a=CSF_hist,label="CSF")
    sns.distplot(a=GM_hist,label="GM")
    sns.distplot(a=WM_hist,label="WM")
    plt.legend()
    plt.xlabel('Intensity')
    plt.xlim([0,300])
    plt.ylabel('Frequency')
    plt.title(f'Patient {data_folder}')
    plt.savefig(f'./density/Patient{data_folder}.pdf')

    plt.show()

color_arr = ['darkred', 'darkblue', 'darkgreen']
color_arr2 = ['lightcoral', 'cornflowerblue', 'lightgreen']

for idx, tissue in enumerate(['CSF', 'GM', 'WM']):
    pass
# plt.legend()
# plt.xlabel('Sigma $\sigma$')
# plt.ylabel('Mean-Dice (%)')
# plt.title(f'Patient{data_folder}')
# plt.savefig('sigma_ablation_fill.pdf')
plt.show()
