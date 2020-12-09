import argparse
import numpy as np
from os.path import splitext
from afs.afs import AFS
from utility.utiliy import init_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-path", metavar='PATH', default='./data/mnist_one_zero.mat',
                        nargs='?', help="The full path of the dataset folder")
    parser.add_argument("-label", metavar='TOTAL_LABELS', default=500,
                        type=int, nargs='?', help="The budget for labels")
    parser.add_argument("-k", metavar='Features', default=5,
                        type=int, nargs='?', help="The size of the output features set")
    parser.add_argument("-delta", metavar='DELTA', default=0.05,
                        type=float, nargs='?', help="delta")
    parser.add_argument("-Lambda", metavar='LAMBDA', default=30,
                        type=str, nargs='?', help="The number of iterations for testing change.")
    args = parser.parse_args()
    file = args.path
    Lambda = False if args.Lambda == 'inf' else int(args.Lambda)
    data_name = str(splitext(file)[0].split('/')[-1])
    X, y = init_data(file)
    prob_x = [{x: c / len(f_val) for x, c in zip(*np.unique(f_val, return_counts=True))} for f_val in X.T]
    afs = AFS(X, y, data_name, prob_x, k=args.k, delta=args.delta)
    afs.run(args.label, plato=Lambda)
    print(afs.top_k)
