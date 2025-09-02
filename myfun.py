#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:05:31 2019

@author: zhen
"""
import numpy as np
import math
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C_ker
import random
import scipy as sp
from scipy.stats import multivariate_normal
import scipy.stats as stats
import pickle
from sklearn import mixture


def Sample_select(Xtrain, N_select):
    Ninput = len(Xtrain[0])  # Number of input variables
    N_points = len(Xtrain)  # Number of training points
    Ninput_normalize = np.zeros(shape=(N_points, Ninput))
    for i in range(Ninput):
        Input_temp = Xtrain[:, i]
        Input_max = np.amax(Input_temp)
        Input_min = np.amin(Input_temp)
        Input_normalize = (Input_temp - Input_min) / (Input_max - Input_min)
        Ninput_normalize[:, i] = Input_normalize
    Index_select = random.sample(range(1, N_points), 10)  # Initial training points;
    for i in range(N_select):
        X_selected = Ninput_normalize[Index_select, :]  # Selected training points
        dis_mat = sp.spatial.distance_matrix(X_selected, Ninput_normalize)  # Distance matrix
        dis_min = np.amin(dis_mat, 0)
        index_new = np.argmax(dis_min)
        Index_select.append(index_new)
    return Index_select


def GP_training(Xtrain, Y_train, Option):
    nInputs = len(Xtrain[0])  # Number of input variables
    if Option == 1:
        L0 = 10
        L_bounds = (1e-5, 1e5)
        ############### Constant Kernel##############
        kernel_Con = C_ker(L0, (1e-5, 1e5))
        ############### Matern Kernel##############
        Matern_length0 = [L0]
        Matern_length_bounds = [L_bounds]
        for i in range(nInputs - 1):
            Matern_length0.append(L0)
            Matern_length_bounds.append(L_bounds)
        kernel_Matern = Matern(Matern_length0, Matern_length_bounds, nu=1.5)
        kernal_prod = kernel_Con * kernel_Matern
        gp_GPML = GaussianProcessRegressor(kernel=kernal_prod, alpha=0, n_restarts_optimizer=20, normalize_y=True)
        gp_GPML.fit(Xtrain, Y_train)
    else:
        ################################## USE DACE ####################################
        gp_GPML = gaussian_process.GaussianProcess(regr='quadratic', corr='squared_exponential', \
                                                   theta0=10 * np.ones(shape=(nInputs)),
                                                   thetaL=1e-4 * np.ones(shape=(nInputs)), \
                                                   thetaU=1e3 * np.ones(shape=(nInputs)), optimizer='fmin_cobyla',
                                                   normalize=True, random_start=1)
        gp_GPML.fit(Xtrain, Y_train)
    return gp_GPML


def GP_predict(n_important, GP_SVD, Xtrain, Option):
    Y_latent = []
    Y_latent_std = []
    nsamples = len(Xtrain[:, 0])
    n_one = 5000
    for Y_index in range(n_important):
        if Option == 1:
            if nsamples < 5000:
                y_pred, y_std = GP_SVD[Y_index].predict(Xtrain, return_std=True)
            else:
                y_pred = np.zeros(nsamples)
                y_std = np.zeros(nsamples)
                for ite in range(int(nsamples / n_one) + 1):
                    if ite < nsamples // n_one:
                        y_pred_temp, y_std_temp = GP_SVD[Y_index].predict(Xtrain[ite * n_one:(ite + 1) * n_one, ],
                                                                          return_std=True)
                        y_pred[ite * n_one:(ite + 1) * n_one] = y_pred_temp
                        y_std[ite * n_one:(ite + 1) * n_one] = y_std_temp
                    else:
                        y_pred_temp, y_std_temp = GP_SVD[Y_index].predict(Xtrain[ite * n_one:, ], return_std=True)
                        y_pred[ite * n_one:] = y_pred_temp
                        y_std[ite * n_one:] = y_std_temp
        else:
            if nsamples < 5000:
                y_pred, y_std = GP_SVD[Y_index].predict(Xtrain, eval_MSE=True)
            else:
                y_pred = np.zeros(nsamples)
                y_std = np.zeros(nsamples)
                for ite in range(int(nsamples / n_one)):
                    y_pred_temp, y_std_temp = GP_SVD[Y_index].predict(Xtrain[ite * n_one:(ite + 1) * n_one, ],
                                                                      eval_MSE=True)
                    y_pred[ite * n_one:(ite + 1) * n_one] = y_pred_temp
                    y_std[ite * n_one:(ite + 1) * n_one] = y_std_temp
        Y_latent.append(y_pred)
        Y_latent_std.append(y_std)
    Y_latent = np.asarray(Y_latent)
    Y_latent_std = np.asarray(Y_latent_std)
    return Y_latent, Y_latent_std
