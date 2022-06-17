#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from torch.utils.data import Dataset
from scipy.stats import gaussian_kde
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class gaussian_kde_approx(gaussian_kde):
    """
    Cloned from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    """

    def __init__(self, dataset, bw_method=None, weights=None):
        super(gaussian_kde, self).__init__()
        self.dataset = np.atleast_2d(np.asarray(dataset))
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape

        if weights is not None:
            self._weights = np.atleast_1d(weights).astype(float)
            self._weights /= sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1/sum(self._weights**2)

        self.set_bandwidth(bw_method=bw_method)

        self.dataset = torch.Tensor(self.dataset).to(device).unsqueeze(0)
        self.inv_cov = torch.Tensor(self.inv_cov).to(device).unsqueeze(0)
        self.weights_tensor = torch.Tensor(self.weights).to(device).unsqueeze(0)

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            if self.dataset.shape[1] == 1:
                self._data_covariance = np.eye(self.dataset.shape[0])
            else:
                self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                                          bias=False,
                                                          aweights=self.weights))
                self._data_covariance += np.eye(self._data_covariance.shape[0]) * 1e-8
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        L = linalg.cholesky(self.covariance*2*np.pi)
        self.log_det = 2*np.log(np.diag(L)).sum()

    def logpdf(self, x):
        """
        Evaluate the log of the estimated pdf on a provided set of points.
        Overwrite the original implementation
        """

        diff = self.dataset - x.unsqueeze(-1)
        tdiff = self.inv_cov @ diff
        energy = torch.sum(diff * tdiff, dim=1)
        log_to_sum = 2.0 * torch.log(self.weights_tensor) - self.log_det - energy
        result = torch.logsumexp(0.5 * log_to_sum, dim=1)

        return result


def estimate_gm(data, label, digit_list, method='scikit', tol=1e-5):
    not_none_idx = []
    est_mean_pack = []
    est_cov_pack = []
    feature = data
    for idx, digit in enumerate(digit_list):
        feature_selected = feature[torch.max(label, dim=1)[1] == digit]
        if feature_selected.size(0) != 0:

            not_none_idx.append(idx)

            if method == 'scikit':
                stat = EmpiricalCovariance().fit(feature_selected.cpu().detach())
                est_mean = torch.from_numpy(stat.location_).to(device).float()
                est_cov = torch.from_numpy(stat.covariance_).to(device).float()
                est_cov = est_cov + tol * torch.eye(est_cov.size(-1)).to(device)

            elif method == 'manual':
                raise NotImplementedError

            est_mean_pack.append(est_mean)
            est_cov_pack.append(est_cov)

    return est_mean_pack, est_cov_pack, not_none_idx


def estimate_kde(data, label, digit_list, bw=1):
    not_none_idx = []
    kernel_list = []
    feature = data
    for idx, digit in enumerate(digit_list):
        feature_selected = feature[torch.max(label, dim=1)[1] == digit]
        if feature_selected.size(0) != 0:

            not_none_idx.append(idx)

            feature_selected = feature_selected.T.cpu().numpy()
            kernel = gaussian_kde_approx(feature_selected, bw_method=bw)
            kernel_list.append(kernel)

    return kernel_list, not_none_idx


def compute_pdf(mid_feature, mean, cov_inv, log_det):
    return -0.5 * ((mid_feature - mean).unsqueeze(-2) @ cov_inv @ (mid_feature - mean).unsqueeze(-1)).squeeze() - 0.5 * log_det


class SoftCrossEntropy(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        return

    def forward(self, inputs, target):
        log_likelihood = - F.log_softmax(inputs, dim=1)
        batch_size = target.size()[0]

        if self.reduction == "mean":
            loss = torch.sum(torch.mul(log_likelihood, target)) / batch_size
        elif self.reduction == "none":
            loss = torch.sum(torch.mul(log_likelihood, target), dim=-1)
        else:
            print('Fill in here')
            exit()
        return loss


def get_lambda(alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def mixup_data(x, y, lam):

    batch_size = x.size()[0]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def generate_nn_label(mid_data, data, label):

    data_unsq = data.unsqueeze(0)
    mid_data_unsq = mid_data.unsqueeze(1)

    dist = torch.norm(mid_data_unsq - data_unsq, dim=-1)

    min_index = torch.argmin(dist, dim=-1)

    nn_label = label[min_index]

    return nn_label


def check_intrusion(mid_data, data, label, mix_label_1, mix_label_2):

    nn_label = generate_nn_label(mid_data, data, label)

    mix_label_1, mix_label_2 = mix_label_1.max(1)[1], mix_label_2.max(1)[1]
    label, nn_label = label.max(1)[1], nn_label.max(1)[1]

    non_intrusion = (nn_label == mix_label_1) + (nn_label == mix_label_2)

    return non_intrusion


def extract_dataset(dataset):
    datapack, labelpack = [], []

    for (data, label) in dataset:
        datapack.append(data)
        labelpack.append(label)

    datapack = torch.stack(datapack).to(device)
    labelpack = torch.stack(labelpack).to(device)

    return datapack, labelpack


def cov(tensor, mean, rowvar=True, bias=False):
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - mean.unsqueeze(-1)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * (tensor.unsqueeze(-2) @ tensor.unsqueeze(-1)).squeeze().to(device)


def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False