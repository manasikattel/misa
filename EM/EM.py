import logging
import sys
import numpy as np
import pylab
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import multivariate_normal
from sklearn.cluster._kmeans import KMeans
import time

from utils import plot_gmm

logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


# which image to register to which /mean intensity to test or test to mean intensity


class EM:
    def __init__(self, img, atlas_model_EM, init_type, n_clusters, dim, into_EM=False, atlas_model_init=None):
        self.into_EM = into_EM
        self.orig_data = img

        # backgroud removal and not used while clustering (a dense cluster with all zeros)
        self.nz_indices = [i for i, x in enumerate(atlas_model_init[:, 1:] * img * atlas_model_EM[:, 1:]) if x.any()]
        self.dim = dim

        self.data = img[self.nz_indices]

        #atlas used for init is different than atlas used for EM
        self.atlas_model_init = atlas_model_init[self.nz_indices, :]
        self.atlas_model_EM = atlas_model_EM[self.nz_indices, :]

        #getting the atlas seg mask (without EM)
        atlas_labels = np.argmax(self.atlas_model_init, axis=1)
        out_labels = np.zeros(self.orig_data.shape[0])
        out_labels[self.nz_indices] = atlas_labels
        img_recovered = out_labels.reshape(self.dim)
        self.atlas_model_mask = img_recovered

        self.atlas_model_init = self.atlas_model_init[:, 1:]
        self.atlas_model_EM = self.atlas_model_EM[:, 1:]

        self.atlas_model_EM = self.atlas_model_EM / np.sum(self.atlas_model_EM, axis=1).reshape(
            (len(self.atlas_model_EM), 1))

        self.n_clusters = n_clusters
        self.log_likelihood_arr = []
        self.init_type = init_type
        self.k_means_time = 0
        self.atlas_time = 0
        self.init()
        self.responsibilities = None
        self.log_likelihood = 0
        self.seg_labels = None
        self.em_time = 0

        self.post_atlas_mask = None

    def init(self):
        if self.init_type == 'random':
            self.mean_s, self.cov_s, self.pi_s = self.init_random(self.data, self.n_clusters)
        elif self.init_type == 'kmeans':
            self.mean_s, self.cov_s, self.pi_s = self.init_kmeans(self.data, self.n_clusters)
        elif self.init_type == 'atlas':
            logger.info('using atlas init')
            self.mean_s, self.cov_s, self.pi_s = self.init_atlas_model(self.data, self.atlas_model_init,
                                                                       self.n_clusters)
        elif self.init_type == 'tissue':
            self.mean_s, self.cov_s, self.pi_s = self.init_atlas_model(self.data, self.atlas_model_init,
                                                                       self.n_clusters)
        elif self.init_type == 'atlas_tissue':
            self.mean_s, self.cov_s, self.pi_s = self.init_atlas_model(self.data, self.atlas_model_init,
                                                                       self.n_clusters)
        else:
            raise Exception("Init Type not defined")

    def init_random(self, data, n_clusters):
        mean_s = np.random.randint(low=np.min(data), high=np.max(data), size=(n_clusters, data.shape[1]))
        random_assign = np.random.randint(low=0, high=n_clusters, size=data.shape[0])
        cov_s = [np.cov(data[random_assign == i].T) for i in range(n_clusters)]
        pi_s = np.random.rand(n_clusters)
        return mean_s, cov_s, pi_s

    # based on the inituation that 0 is background and lowest inten. is CSF and highest intens. is WM
    def mask_from_recovered(self, recovered_img):
        mean_values = np.unique(recovered_img)
        seg_mask = np.zeros_like(recovered_img)
        for i in range(len(mean_values)):
            seg_mask[recovered_img == mean_values[i]] = i
        return seg_mask

    # initalization the centeriod with kmeans
    def init_kmeans(self, data, n_clusters):
        start_time = time.time()

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
        y = kmeans.fit_predict(data)
        mean_s = [np.mean(data[y == i], axis=0) for i in range(n_clusters)]
        cov_s = np.array([np.cov(data[y == i].T) for i in range(n_clusters)])
        ids = set(y)
        pi_s = np.array([np.sum([y == i]) / len(y) for i in ids])

        # keeping track of kmeans running time
        self.k_means_time = (time.time() - start_time)

        # mask from the first initalization (to check kmeans acc)
        recovered_img = self.get_segm_mask(mean_s, y, self.orig_data, self.nz_indices)
        self.kmeans_mask = self.mask_from_recovered(recovered_img)

        self.e_kmeans_dict = self.get_dict_map(self.atlas_model_mask, self.kmeans_mask)
        self.e_kmeans_sorted = {k: v for k, v in enumerate(np.argsort(mean_s, axis=0).squeeze())}

        return mean_s, cov_s, pi_s

    def init_tissue_model(self, data, tissue_model, n_clusters):
        # read from pickle file
        mean_s, cov_s, pi_s = self.m_step(data, tissue_model, n_clusters)
        # do same logic as manasi
        start_time = time.time()
        tissue_model_labels = np.argmax(tissue_model, axis=1)
        self.tissue_Time = (time.time() - start_time)

        tissue_model_mask = self.get_segm_mask(mean_s, tissue_model_labels, self.orig_data, self.nz_indices)
        self.tissue_model_mask = self.mask_from_recovered(tissue_model_mask)

        return mean_s, cov_s, pi_s

    def init_atlas_model(self, data, atlas_model, n_clusters):
        atlas_model = atlas_model / np.sum(atlas_model, axis=1).reshape(
            (len(atlas_model), 1))
        mean_s, cov_s, pi_s = self.m_step(data, atlas_model, n_clusters)

        start_time = time.time()

        atlas_labels = np.argmax(atlas_model, axis=1)
        self.atlas_time = (time.time() - start_time)

        return mean_s, cov_s, pi_s

    def get_post_atlas_mask(self, my_dict=None):
        #using the dict to map the responsibilities to the atlas
        if my_dict:
            new_responsiblities = np.zeros_like(self.responsibilities)
            sorted_resp_dict = {k: v for k, v in enumerate(np.argsort(self.mean_s, axis=0).squeeze())}
            for idx, key in enumerate(list(my_dict.keys())[1:]):
                new_responsiblities[:, idx] = self.responsibilities[:,
                                              sorted_resp_dict[key - 1]] * self.atlas_model_EM[:, idx]

        else:
            new_responsiblities = self.responsibilities * self.atlas_model_EM

        new_responsiblities = new_responsiblities / np.sum(new_responsiblities, axis=1).reshape(
            (len(new_responsiblities), 1))

        seg_labels = self.get_labels(new_responsiblities)
        mean_s, cov_s, pi_s = self.m_step(self.data, new_responsiblities, self.n_clusters)

        post_atlas_mask = self.get_segm_mask(mean_s, seg_labels, self.orig_data, self.nz_indices)
        post_atlas_mask = self.mask_from_recovered(post_atlas_mask)
        return post_atlas_mask

    def init_atlas_tissue_model(self, data):
        self.atlas_tissue_model_mask = None
        pass

    # expectation step for the EM algorithm
    def e_step(self, data, mu_s, cov_s, pi_s, n_clusters):
        posterior_probabilities = np.zeros((data.shape[0], n_clusters), dtype=np.float64)
        for k in range(n_clusters):
            posterior_probabilities[:, k] = pi_s[k] * multivariate_normal.pdf(data, mean=mu_s[k], cov=cov_s[k],
                                                                              allow_singular=True)
        posterior_probabilities = posterior_probabilities / np.sum(posterior_probabilities, axis=1).reshape(
            (len(posterior_probabilities), 1))

        if self.into_EM:
            if self.init_type == 'kmeans':
                #making sure that CSF atlas is multipled by CSF kmeans...etc
                new_responsiblities = np.zeros_like(posterior_probabilities)
                for idx, key in enumerate(list(self.e_kmeans_dict.keys())[1:]):
                    new_responsiblities[:, idx] = posterior_probabilities[:, idx] * self.atlas_model_EM[:,
                                                                                    self.e_kmeans_sorted[key - 1]]
                posterior_probabilities = new_responsiblities
            else:
                posterior_probabilities = posterior_probabilities * self.atlas_model_EM

        # normalize the posterior probabilities
        posterior_probabilities = posterior_probabilities / np.sum(posterior_probabilities, axis=1).reshape(
            (len(posterior_probabilities), 1))

        return posterior_probabilities

    # maximization step of the EM algorithm
    def m_step(self, img, posterior_probabilities, K):
        mu_s = np.zeros((K, img.shape[1]))
        cov_s = np.zeros((K, img.shape[1], img.shape[1]))
        for k in range(K):
            class_posterior = posterior_probabilities[:, k].reshape(1, len(posterior_probabilities[:, k]))
            mu_s[k] = np.matmul(class_posterior, img) / np.sum(class_posterior)
            class_posterior_norm = class_posterior / np.sum(class_posterior)
            cov_s[k] = (class_posterior_norm[0, :] * np.transpose(img - mu_s[k])) @ (img - mu_s[k])

        pi_s = np.sum(posterior_probabilities, axis=0) / posterior_probabilities.shape[0]

        return mu_s, cov_s, pi_s

    def get_labels(self, responsibilities):
        labels = np.argmax(responsibilities, axis=1)
        return labels

    def get_log_likelihood(self, data, mu_s, cov_s, pi_s, n_clusters):
        posterior_probabilities = np.zeros((data.shape[0], n_clusters))
        for k in range(n_clusters):
            posterior_probabilities[:, k] = pi_s[k] * multivariate_normal.pdf(data, mean=mu_s[k], cov=cov_s[k],
                                                                              allow_singular=True)

        log_likelihood = np.sum(np.log(np.sum(posterior_probabilities, axis=1)))
        return log_likelihood

    # retrieve a segmentation mask from the output of the EM algorithm
    def get_segm_mask(self, means, labels, orig_data, nz_indices):
        out_labels = np.zeros(orig_data.shape[0])
        data_mean_replaced = np.array([element[0] for element in means])
        # todo check
        em_img = data_mean_replaced[labels]
        out_labels[nz_indices] = em_img
        img_recovered = out_labels.reshape(self.dim)
        return img_recovered

    def execute(self, tol, max_iter, visualize=True):
        prev_log_likelihood = 100
        iter_n = 0

        self.responsibilities = self.e_step(self.data, self.mean_s, self.cov_s, self.pi_s, self.n_clusters)
        start_time = time.time()

        # run until convergence or max iteration is reached
        while (abs(prev_log_likelihood - self.log_likelihood) > tol) and (iter_n <= max_iter):
            prev_log_likelihood = self.log_likelihood

            # get inital segmentation labels based on the random init
            self.seg_labels = self.get_labels(self.responsibilities)

            # visualization of the gmm or the patient picture every 20 iterations
            if visualize and iter_n % 20 == 0:
                recover_img = self.get_segm_mask(self.mean_s, self.seg_labels, self.orig_data, self.nz_indices)
                title = f'EM-Iter-{iter_n}'
                # plot_gmm(self.data, recover_img, self.mean_s, self.cov_s, title)
                plt.imshow(recover_img[:, :, 140], cmap=pylab.cm.cool)

                plt.title('iter' + str(iter_n))
                plt.show()

            # E-step
            self.responsibilities = self.e_step(self.data, self.mean_s, self.cov_s, self.pi_s, self.n_clusters)

            # M-Step
            self.mean_s, self.cov_s, self.pi_s = self.m_step(self.data, self.responsibilities, self.n_clusters)

            # calculate the loglikelihood to check for convergence
            self.log_likelihood = self.get_log_likelihood(self.data, self.mean_s, self.cov_s, self.pi_s,
                                                          self.n_clusters)

            self.log_likelihood_arr.append(self.log_likelihood)

            logger.info("iter_n:{}, log_likelihood = {}".format(iter_n, self.log_likelihood))

            iter_n += 1

        self.em_time = (time.time() - start_time)

        logger.info('Converged at iter:{} with loglikelihood:{}'.format(iter_n, self.log_likelihood))

        return self.get_segm_mask(self.mean_s, self.seg_labels, self.orig_data, self.nz_indices), iter_n

    def get_dict_map(self, seg_mask_atlas, seg_mask_em):
        # check_CSF = (seg_mask_atlas == 1).astype(np.uint8) * seg_mask_em
        check_GM = (seg_mask_atlas == 2).astype(np.uint8) * seg_mask_em
        check_WM = (seg_mask_atlas == 3).astype(np.uint8) * seg_mask_em
        #
        # check_CSF, count_csf = np.unique(check_CSF, return_counts=True)
        # count_csf = np.argsort(-count_csf)

        check_GM, count_gm = np.unique(check_GM, return_counts=True)
        count_gm = np.argsort(-count_gm)
        #
        check_WM, count_wm = np.unique(check_WM, return_counts=True)
        count_wm = np.argsort(-count_wm)
        #
        # # 0 is always the background
        # CSF_Label = count_csf[1]
        GM_Label = count_gm[1]
        WM_Label = count_wm[1]

        CSF_Label = list(set([0, 1, 2, 3]).difference(set([0, GM_Label, WM_Label])))[0]

        # my_dict = {0: 0, CSF_Label: 1, GM_Label: 3, WM_Label: 2}

        my_dict = {0: 0, CSF_Label: 1, GM_Label: 3, WM_Label: 2}
        return my_dict
