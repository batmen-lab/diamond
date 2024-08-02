import os, argparse, random, time, csv;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import torch

from .DeepKnockoffs.machine import KnockoffMachine;
from .DeepKnockoffs.gaussian import GaussianKnockoffs

def main(args):
    logger.info('input_url={}'.format(args.input_url));
    assert os.path.exists(args.input_url);

    X = np.genfromtxt(args.input_url, delimiter=',', skip_header=0, encoding='ascii');
    X = np.asarray(X, dtype=float);
    logger.info('X={}'.format(X.shape, ));

    knockoff_url = args.input_url.replace('.csv', '_knockoff_DeepKnockoffs.csv');
    if os.path.isfile(knockoff_url): return;


    def get_model_cp_dir(epochs):
        return os.path.join(os.path.dirname(args.input_url), "DeepKnockoffs_checkpoints",
            "epoch{}_epochLen{}_batch{}_test{}_lr{}_gamma{}_lambda{}_delta{}".format( epochs, args.epoch_length, args.batch_size, args.test_size, args.learning_rate, args.Gamma, args.Lambda, args.Delta));

    def get_model_cp_url(epochs):
        return os.path.join(get_model_cp_dir(epochs),"DeepKnockoffs_checkpoint.pth.tar");

    model_cp_dir = get_model_cp_dir(args.epochs)
    for e in range(args.epochs, 0, -1):
        prev_model_cp_dir = get_model_cp_dir(e)
        prev_model_cp_url = get_model_cp_url(e)
        if os.path.exists(prev_model_cp_url):
            os.rename(prev_model_cp_dir, model_cp_dir)
            break

    model_plot_dir = os.path.join(
        model_cp_dir, "plots"
    )
    if not os.path.exists(model_cp_dir):
        os.makedirs(model_cp_dir)
    if not os.path.exists(model_plot_dir):
        os.makedirs(model_plot_dir)

    checkpoint_name = os.path.join(model_cp_dir, "DeepKnockoffs")
    logs_name = checkpoint_name+"_progress.txt"

    # Compute the empirical covariance matrix of the training data
    SigmaHat = np.cov(X, rowvar=False)

    # Initialize generator of second-order knockoffs
    second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X,0), method="sdp")

    # Measure pairwise second-order knockoff correlations
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

    print('Average absolute pairwise correlation: {}.'.format((np.mean(np.abs(corr_g)))))

    alphas = [1.,2.,4.,8.,16.,32.,64.,128.]
    # Set the parameters for training deep knockoffs
    pars = dict()
    # Number of epochs
    pars['epochs'] = args.epochs
    # Number of iterations over the full data per epoch
    pars['epoch_length'] = args.epoch_length
    # Data type, either "continuous" or "binary"
    pars['family'] = "continuous"
    # Dimensions of the data
    pars['p'] = p
    # Size of the test set
    pars['test_size']  = args.test_size
    # Batch size
    pars['batch_size'] = args.batch_size
    # Learning rate
    pars['lr'] = args.learning_rate
    # When to decrease learning rate (unused when equal to number of epochs)
    pars['lr_milestones'] = [pars['epochs']]
    # Width of the network (number of layers is fixed to 6)
    pars['dim_h'] = int(10*p)
    # Penalty for the MMD distance
    pars['GAMMA'] = args.Gamma
    # Penalty encouraging second-order knockoffs
    pars['LAMBDA'] = args.Lambda
    # Decorrelation penalty hyperparameter
    pars['DELTA'] = args.Delta
    # Target pairwise correlations between variables and knockoffs
    pars['target_corr'] = corr_g
    # Kernel widths for the MMD measure (uniform weights)
    pars['alphas'] = alphas

    machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)
    print("Fitting the knockoff machine...")
    machine.train(X, resume=True)

    # Generate deep knockoffs
    X_knockoff = machine.generate(X);
    assert X_knockoff.shape == X.shape;
    logger.info('X_knockoff={}'.format(X_knockoff.shape, ));
    np.savetxt(knockoff_url, X_knockoff, delimiter=",", fmt='%.6f');
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Optional app description')
#     # data paams
#     parser.add_argument('--input_url', type=str, help='input_url', required=True);
#
#     parser.add_argument('--epochs', type=int, help="total number of epochs to train", default=20);
#     parser.add_argument('--epoch_length', type=int, help="Number of iterations over the full data per epoch", default=100);
#     parser.add_argument('--batch_size', type=int, help="batch size", default=128);
#     parser.add_argument('--test_size', type=int, help="test size", default=128);
#     parser.add_argument("--learning_rate", type=int, help="neural net's learning rate", default=0.01)
#     parser.add_argument("--Gamma", type=float, help="Penalty for the MMD distance", default=1.0)
#     parser.add_argument("--Lambda", type=float, help="Penalty encouraging second-order knockoffs", default=1.0)
#     parser.add_argument("--Delta", type=float, help="Decorrelation penalty hyperparameter", default=1.0)
#
#     args = parser.parse_args()
#     main(args)


def DeepKnockoffs(X, lr=0.01, batch_size=256, test_size=10000, nepochs=10, epoch_length=50):
    n, p = X.shape
    assert test_size < n
    # Compute the empirical covariance matrix of the training data
    SigmaHat = np.cov(X, rowvar=False)
    # Initialize generator of second-order knockoffs
    second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X,0), method="sdp")
    # Measure pairwise second-order knockoff correlations
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
    alphas = [1.,2.,4.,8.,16.,32.,64.,128.]
    # Set the parameters for training deep knockoffs
    pars = dict()
    # Number of epochs
    pars['epochs'] = nepochs
    # Number of iterations over the full data per epoch
    pars['epoch_length'] = epoch_length
    # Data type, either "continuous" or "binary"
    pars['family'] = "continuous"
    # Dimensions of the data
    pars['p'] = p
    # Size of the test set
    pars['test_size']  = test_size
    # Batch size
    pars['batch_size'] = batch_size
    # Learning rate
    pars['lr'] = lr
    # When to decrease learning rate (unused when equal to number of epochs)
    pars['lr_milestones'] = [pars['epochs']]
    # Width of the network (number of layers is fixed to 6)
    pars['dim_h'] = int(10*p)
    # Penalty for the MMD distance
    pars['GAMMA'] = 1.0
    # Penalty encouraging second-order knockoffs
    pars['LAMBDA'] = 1.0
    # Decorrelation penalty hyperparameter
    pars['DELTA'] = 1.0
    # Target pairwise correlations between variables and knockoffs
    pars['target_corr'] = corr_g
    # Kernel widths for the MMD measure (uniform weights)
    pars['alphas'] = alphas

    machine = KnockoffMachine(pars, checkpoint_name=None, logs_name=None)
    print("Fitting the knockoff machine...")
    machine.train(X, resume=False)
    X_knockoff = machine.generate(X)
    assert X_knockoff.shape == X.shape

    return X_knockoff

