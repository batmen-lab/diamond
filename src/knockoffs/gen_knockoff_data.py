import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import gen_KnockoffGAN
from utils import viz_data

def main(args):
    print('input_url={}'.format(args.input_url))
    assert os.path.isfile(args.input_url)

    X = pd.read_csv(args.input_url, sep=',', header=None)
    X = X.to_numpy(dtype=float)
    print('X={}\t{}'.format(X.shape, X))

    assert args.knockoff == 'knockoffgan'
    knockoff_url = args.input_url.replace('.csv', '_{}.csv'.format(args.knockoff))
    print('knockoff_url={}'.format(knockoff_url))
    if os.path.isfile(knockoff_url):
        X_knockoff = pd.read_csv(knockoff_url, sep=',', header=None)
        X_knockoff = X_knockoff.to_numpy(dtype=float)
    else:
        knockoff_func = gen_KnockoffGAN.KnockoffGAN(X, "Uniform")
        X_knockoff = knockoff_func(X)
        assert X_knockoff.shape == X.shape
        np.savetxt(knockoff_url, X_knockoff, delimiter=",", fmt='%.6f')
    print('X_knockoff={}\t{}'.format(X_knockoff.shape, X_knockoff))

    plot_url = knockoff_url.replace('.csv', '.png')
    print('plot_url={}'.format(plot_url))
    # corr_arr = viz_data.knockoff_correlation(X, X_knockoff)
    # viz_data.plot_knockoff_correlation(corr_arr=corr_arr, plot_url=plot_url)

    fig = viz_data.ScatterCovariance(X, X_knockoff)
    fig.savefig(plot_url)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument("--knockoff", type=str, choices=["knockoffgan"], default="knockoffgan", help="choice for knockoff generator")
    parser.add_argument("--input_url", type=str, help="URL to input file")
    main(parser.parse_args())


### command
# python knockoffs/gen_knockoff_data.py --input_url ../data/simulation/X_n10000_p50_seed1.csv
