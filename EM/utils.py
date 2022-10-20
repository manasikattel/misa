import copy
import numpy as np
import nibabel as nib
import seaborn
from scipy.ndimage._filters import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def read_img(filename, blur_sigma=None):
    img_3d = nib.load(filename)
    affine = img_3d.affine
    img_3d = img_3d.get_fdata()
    if blur_sigma:
        img_3d = gaussian_filter(img_3d, blur_sigma)

    return img_3d, affine


def flatten_img(img_3d, mode):
    if mode == '3d':
        x, y, z = img_3d.shape
        img_2d = img_3d.reshape(x * y * z, 1)
        img_2d = np.array(img_2d, dtype=np.float)
    elif mode == '2d':
        x, y = img_3d.shape
        img_2d = img_3d.reshape(x * y, 1)
        img_2d = np.array(img_2d, dtype=np.float)
    return img_2d


def get_features(T1, T2, gt, use_T2):
    gt_mask = copy.deepcopy(gt)
    gt_mask[gt_mask > 0] = 1
    T1_masked = np.multiply(T1, gt_mask)
    # x, y, z = T1.shape
    T2_masked = None
    if use_T2:
        T2_masked = np.multiply(T2, gt_mask)
        stacked_features = np.stack((flatten_img(T1_masked, mode='3d'), flatten_img(T2_masked, mode='3d')),
                                    axis=1).squeeze()
    else:
        stacked_features = flatten_img(T1_masked, mode='3d')

    return stacked_features, T1_masked, T2_masked


def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 3):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(data, recover_img, mean_s, cov_s, title=None, labels_names=['CSF', 'GM', 'WM']):
    my_cmap = seaborn.color_palette("pastel", as_cmap=True)
    # recover_img = em.get_segm_mask(em.mean_s, em.seg_labels, em.orig_data, em.nz_indices)

    recovered_flatten = recover_img.flatten()
    u_labels = np.unique(recovered_flatten)[1:]

    for idx, (i, label_name) in enumerate(zip(u_labels, labels_names)):
        data_plot = data[recovered_flatten[recovered_flatten != 0] == i, :]
        plt.scatter(data_plot[:, 0],
                    data_plot[:, 1], c=my_cmap[idx], label=label_name, s=5, cmap=my_cmap)
    # plt.show()
    for pos, covar in zip(mean_s, cov_s):
        draw_ellipse(pos, covar, alpha=0.2)

    plt.title(title)
    plt.legend()
    plt.xlabel('T1-Intensity')
    plt.ylabel('T2-Intensity')
    # plt.show()
