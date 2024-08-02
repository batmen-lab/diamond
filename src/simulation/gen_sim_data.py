import os, sys, random, argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import gen_X, gen_Y

def get_paramstr(args):
    return 'n{}_p{}_seed{}'.format(args.n, args.p, args.seed)

def main(args):
    print('output_dir={}'.format(args.output_dir))
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_x_url = os.path.join(args.output_dir, 'X_{}.csv'.format(get_paramstr(args)))
    print('output_x_url={}'.format(output_x_url))
    if os.path.isfile(output_x_url):
        X = pd.read_csv(output_x_url, sep=',', header=None)
        X = X.to_numpy(dtype=float)
    else:
        X = gen_X.UniformSampler(p=args.p, low_list=([0] * args.p), high_list=([1] * args.p), seed=args.seed).sample(n=args.n)
        np.savetxt(output_x_url, X, delimiter=",")
    print('X={}\t{}'.format(X.shape, X))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--n", type=int, default=10000, help="number of synthetic data points")
    parser.add_argument("--p", type=int, default=30, help="number of synthetic data features")
    parser.add_argument("--output_dir", type=str, help="directory to output file")
    main(parser.parse_args())


### command
# python simulation/gen_sim_data.py --output_dir ../data/simulation --seed 1 --n 10000 --p 30
