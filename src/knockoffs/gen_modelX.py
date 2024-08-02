import os, argparse, random, time, csv;

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from DeepKnockoffs.gaussian import GaussianKnockoffs


def main(args):
    logger.info('input_url={}'.format(args.input_url));
    assert os.path.exists(args.input_url);

    X = np.genfromtxt(args.input_url, delimiter=',', skip_header=0, encoding='ascii');
    X = np.asarray(X, dtype=float);
    logger.info('X={}'.format(X.shape, ));

    knockoff_url = args.input_url.replace('.csv', '_knockoff_ModelX.csv');
    if os.path.isfile(knockoff_url): return;


    # Compute the empirical covariance matrix of the training data
    SigmaHat = np.cov(X, rowvar=False)

    # Initialize generator of second-order knockoffs
    try:
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X,0), method=args.method)
    except:
        raise Exception("Model-X knockoff generation failed")

    X_knockoff = second_order.generate(X)
    assert X_knockoff.shape == X.shape;
    logger.info('X_knockoff={}'.format(X_knockoff.shape, ));
    np.savetxt(knockoff_url, X_knockoff, delimiter=",", fmt='%.6f');

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Options for generating model-X knockoff')
#
#     parser.add_argument('--input_url', type=str, help='input_url', required=True);
#     parser.add_argument('--method', type=str, help='model-X method (equi, sdp)', default="sdp")
#     args = parser.parse_args()
#     main(args)
