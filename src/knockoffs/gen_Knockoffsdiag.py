import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
from scipy.stats import hmean
from sklearn.covariance import (GraphicalLassoCV, empirical_covariance,
                                ledoit_wolf)
from sklearn.linear_model import (ElasticNetCV, Lasso, LassoCV, LassoLars,
                                  LassoLarsCV, LinearRegression,
                                  LogisticRegression, LogisticRegressionCV)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, shuffle
from sklearn.utils.validation import check_memory
from statsmodels.distributions.empirical_distribution import (
    ECDF, monotone_fn_inverter)
from tqdm import tqdm
from xgboost import XGBClassifier


def quantile_aggregation(pvals, gamma=0.5, gamma_min=0.05, adaptive=False, drop_gamma=False):
    if pvals.shape[0] == 1:
        return pvals[0]
    if adaptive:
        return _adaptive_quantile_aggregation(pvals, gamma_min)
    else:
        return _fixed_quantile_aggregation(pvals, gamma, drop_gamma=drop_gamma)


def fdr_threshold(pvals, fdr=0.1, method='bhq', reshaping_function=None):
    if method == 'bhq':
        return _bhq_threshold(pvals, fdr=fdr)
    elif method == 'bhy':
        return _bhy_threshold(
            pvals, fdr=fdr, reshaping_function=reshaping_function)
    elif method == 'ebh':
        return _ebh_threshold(pvals, fdr=fdr)
    else:
        raise ValueError(
            '{} is not support FDR control method'.format(method))


def _bhq_threshold(pvals, fdr=0.1):
    """Standard Benjamini-Hochberg for controlling False discovery rate
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if pvals_sorted[i] <= fdr * (i + 1) / n_features:
            selected_index = i
            break
    if selected_index <= n_features:
        return pvals_sorted[selected_index]
    else:
        return -1.0


def _ebh_threshold(evals, fdr=0.1):
    """e-BH procedure for FDR control (see Wang and Ramdas 2021)
    """
    n_features = len(evals)
    evals_sorted = -np.sort(-evals)  # sort in descending order
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if evals_sorted[i] >= n_features / (fdr * (i + 1)):
            selected_index = i
            break
    if selected_index <= n_features:
        return evals_sorted[selected_index]
    else:
        return np.infty


def _bhy_threshold(pvals, reshaping_function=None, fdr=0.1):
    """Benjamini-Hochberg-Yekutieli procedure for controlling FDR, with input
    shape function. Reference: Ramdas et al (2017)
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    # Default value for reshaping function -- defined in
    # Benjamini & Yekutieli (2001)
    if reshaping_function is None:
        temp = np.arange(n_features)
        sum_inverse = np.sum(1 / (temp + 1))
        return _bhq_threshold(pvals, fdr / sum_inverse)
    else:
        for i in range(n_features - 1, -1, -1):
            if pvals_sorted[i] <= fdr * reshaping_function(i + 1) / n_features:
                selected_index = i
                break
        if selected_index <= n_features:
            return pvals_sorted[selected_index]
        else:
            return -1.0


def _fixed_quantile_aggregation(pvals, gamma=0.5, drop_gamma=False):
    """Quantile aggregation function based on Meinshausen et al (2008)

    Parameters
    ----------
    pvals : 2D ndarray (n_bootstrap, n_test)
        p-value (adjusted)

    gamma : float
        Percentile value used for aggregation.

    Returns
    -------
    1D ndarray (n_tests, )
        Vector of aggregated p-value
    """
    if drop_gamma:
        converted_score = np.percentile(pvals, q=100*gamma, axis=0)
    
    else:
        converted_score = (1 / gamma) * (np.percentile(pvals, q=100*gamma, axis=0))

    return np.minimum(1, converted_score)


def _adaptive_quantile_aggregation(pvals, gamma_min=0.05):
    """adaptive version of the quantile aggregation method, Meinshausen et al.
    (2008)"""
    gammas = np.arange(gamma_min, 1.05, 0.05)
    list_Q = np.array([
        _fixed_quantile_aggregation(pvals, gamma) for gamma in gammas])

    return np.minimum(1, (1 - np.log(gamma_min)) * list_Q.min(0))


def simu_data(n, p, rho=0.25, snr=2.0, sparsity=0.06, effect=1.0, seed=None):
    """Function to simulate data follow an autoregressive structure with Toeplitz
    covariance matrix.
    Adapted from hidimstat: https://github.com/ja-che/hidimstat

    Parameters
    ----------
    n : int
        number of observations
    p : int
        number of variables
    sparsity : float, optional
        ratio of number of variables with non-zero coefficients over total
        coefficients
    rho : float, optional
        correlation parameter
    effect : float, optional
        signal magnitude, value of non-null coefficients
    seed : None or Int, optional
        random seed for generator

    Returns
    -------
    X : ndarray, shape (n, p)
        Design matrix resulted from simulation
    y : ndarray, shape (n, )
        Response vector resulted from simulation
    beta_true : ndarray, shape (n, )
        Vector of true coefficient value
    non_zero : ndarray, shape (n, )
        Vector of non zero coefficients index

    """
    # Setup seed generator
    rng = np.random.default_rng(seed)

    # Number of non-null
    k = int(sparsity * p)

    # Generate the variables from a multivariate normal distribution
    mu = np.zeros(p)
    Sigma = toeplitz(rho ** np.arange(0, p))  # covariance matrix of X
    # X = np.dot(np.random.normal(size=(n, p)), cholesky(Sigma))
    X = rng.multivariate_normal(mu, Sigma, size=(n))
    # Generate the response from a linear model
    blob_indexes = np.linspace(0, p - 6, int(k/5), dtype=int)
    non_zero = np.array([np.arange(i, i+5) for i in blob_indexes])
    # non_zero = rng.choice(p, k, replace=False)
    beta_true = np.zeros(p)
    beta_true[non_zero] = effect
    eps = rng.standard_normal(size=n)
    prod_temp = np.dot(X, beta_true)
    noise_mag = np.linalg.norm(prod_temp) / (snr * np.linalg.norm(eps))
    y = prod_temp + noise_mag * eps

    return X, y, beta_true, non_zero, Sigma


def get_null_pis(B, p):
    """
    Sample from the joint distribution of pi statistics
    under the null.

    Parameters
    ----------

    B: int
        Number of samples
    p: int
        Number of variables

    Returns
    -------
    pi0 : array-like of shape (B, p)
        Each row contains a vector of
        p \pi statistics
    
    """
    pi0 = np.zeros((B, p))
    for b in range(B):
        signs = (np.random.binomial(1, 0.5, size=p) * 2) - 1
        Z = 0
        for j in range(p):
            if signs[j] < 0:
                pi0[b][j] = 1
                Z += 1
            else:
                pi0[b][j] = (1 + Z) / p
    pi0 = np.sort(pi0, axis=1)
    return pi0

def get_pi_template(B, p):
    """
    Build a template by sampling from the joint distribution 
    of pi statistics under the null.

    Parameters
    ----------

    B: int
        Number of samples
    p: int
        Number of variables

    Returns
    -------
    template : array-like of shape (B, p)
        Sorted set of candidate threshold families
    
    """
    pi0 = get_null_pis(B, p)
    template = np.sort(pi0, axis=0) # extract quantile curves
    return template


def _estimate_distribution(X, shrink=True, cov_estimator='ledoit_wolf', n_jobs=1):
    """
    Adapted from hidimstat: https://github.com/ja-che/hidimstat
    """
    alphas = [1e-3, 1e-2, 1e-1, 1]

    mu = X.mean(axis=0)
    Sigma = empirical_covariance(X)

    if shrink or not _is_posdef(Sigma):

        if cov_estimator == 'ledoit_wolf':
            Sigma_shrink = ledoit_wolf(X, assume_centered=True)[0]

        elif cov_estimator == 'graph_lasso':
            model = GraphicalLassoCV(alphas=alphas, n_jobs=n_jobs)
            Sigma_shrink = model.fit(X).covariance_

        else:
            raise ValueError('{} is not a valid covariance estimated method'
                             .format(cov_estimator))

        return mu, Sigma_shrink

    return mu, Sigma


def _is_posdef(X, tol=1e-14):
    """Check a matrix is positive definite by calculating eigenvalue of the
    matrix. Adapted from hidimstat: https://github.com/ja-che/hidimstat

    Parameters
    ----------
    X : 2D ndarray, shape (n_samples x n_features)
        Matrix to check

    tol : float, optional
        minimum threshold for eigenvalue

    Returns
    -------
    True or False
    """
    eig_value = np.linalg.eigvalsh(X)
    return np.all(eig_value > tol)


def _get_single_clf_ko(X, j, method="lasso"):
    """
    Fit a single classifier to predict the j-th variable from all others.

    Args:
        X : input data
        j (int): variable index
        method (str, optional): Classifier used. Defaults to "lasso".

    Returns:
        pred: Predicted values for variable j from all others.
    """
    
    n, p = X.shape
    idc = np.array([i for i in np.arange(0, p) if i != j])

    if method == "lasso":
        lambda_max = np.max(np.abs(np.dot(X[:, idc].T, X[:, j]))) / (2 * (p - 1))
        alpha = (lambda_max / 100)
        clf = Lasso(alpha)
    
    if method == "logreg_cv":
        clf = LogisticRegressionCV(cv=5, max_iter=int(10e4), n_jobs=-1)

    if method == "xgb":
        clf = xgb.XGBRegressor(n_jobs=-1)

    clf.fit(X[:, idc], X[:, j])
    pred = clf.predict(X[:, idc])
    return pred

def _adjust_marginal(v, ref, discrete=False):
    """
    Make v follow the marginal of ref.
    """
    if discrete:
        sorter = np.argsort(v)
        sorter_ = np.argsort(sorter)
        return np.sort(ref)[sorter_]
    
    else:
        G = ECDF(ref)
        F = ECDF(v)

        unif_ = F(v)

        # G_inv = np.argsort(G(ref))
        G_inv = monotone_fn_inverter(G, ref)

        return G_inv(unif_)


def _get_samples_ko(X, pred, j, discrete=False, adjust_marg=True, seed=None):
    """

    Generate a Knockoff for variable j.

    Args:
        X : input data
        pred (array): Predicted Xj using all other variables
        j (int): variable index
        discrete (bool, optional): Indicates discrete or continuous data. Defaults to False.
        adjust_marg (bool, optional): Whether to adjust marginals or not. Defaults to True.
        seed (int, optional): seed. Defaults to None.

    Returns:
        sample: Knockoff for variable j.
    """
    np.random.seed(seed)
    n, p = X.shape

    residuals = X[:, j] - pred
    indices_ = np.arange(residuals.shape[0])
    np.random.shuffle(indices_)

    sample = pred + residuals[indices_]

    if adjust_marg:
        sample = _adjust_marginal(sample, X[:, j], discrete=discrete)

    return sample[np.newaxis].T


def conditional_sequential_gen_ko(X, discrete, n_jobs=1, method_ko_gen='lasso', adjust_marg=False, seed=None):
    """
    Generate Knockoffs for all variables in X.

    Args:
        X : input data
        discrete (array): Indicates discrete or continuous data.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        method_ko_gen (str, optional): Classifier used. Defaults to "lasso".
        adjust_marg (bool, optional): Whether to adjust marginals or not. Defaults to True.
        seed (int, optional): seed. Defaults to None.
    Returns:
        samples: Knockoffs for all variables in X.
    """
    
    rng = check_random_state(seed)
    n, p = X.shape
    preds = np.array(Parallel(n_jobs=n_jobs)(delayed(
        _get_single_clf_ko)(X, j, method_ko_gen) for j in tqdm(range(p))))
    samples = np.hstack(Parallel(n_jobs=n_jobs)(delayed(
        _get_samples_ko)(X, preds[j], j, discrete=discrete, adjust_marg=adjust_marg) for j, discrete in tqdm(enumerate(discrete))))
    
    return samples