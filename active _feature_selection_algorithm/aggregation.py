import argparse
import multiprocessing as mp
import os
import time
import torch

from afs.afs import AFS
from baselines.random.random import feature_ranking_random
from utility.plot_restuts import plot
from utility.utiliy import *
from os import makedirs
from os.path import splitext


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-path", metavar='PATH', default='data.mat',
                        nargs='?', help="The full path of the dataset folder")
    parser.add_argument("-label", metavar='TOTAL_LABELS', default=500,
                        type=int, nargs='?', help="The budget for labels")
    parser.add_argument("-k", metavar='Features', default=5,
                        type=int, nargs='?', help="The size of the output features set")
    parser.add_argument("-delta", metavar='DELTA', default=0.05,
                        type=float, nargs='?', help="delta")
    parser.add_argument("-Lambda", metavar='LAMBDA', default=30,
                        type=str, nargs='?', help="The number of iterations for testing change.")
    parser.add_argument("-n_threads", metavar='NUMBER_OF_THREADS', default=30,
                        type=int, nargs='?', help="ache threads will run the experiment respect to -n_exp parm")
    parser.add_argument("-t_times", metavar='TIMES_PEER_THREAD', default=1,
                        type=int, nargs='?', help="the number of time that ache that will run the experiment")

    args = parser.parse_args()
    print(args)
    time_start = time.time()
    lam = False if args.Lambda == 'inf' else int(args.Lambda)
    start(file=args.path, total_labels=args.label,
          k=args.k, delta=args.delta,
          n_threads=args.n_threads, t_times=args.t_times, lam=lam)

    end_start = time.time()
    hours, rem = divmod(end_start - time_start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("run time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def start(file, total_labels, k, delta, t_times, n_threads, lam):
    data_name = str(splitext(file)[0].split('/')[-1])
    X, y = init_data(file)
    print(data_name, 'shape:', X.shape)
    print(data_name + ': processing')
    run_all_experiment(X, y, total_labels, k, data_name, delta, n_threads, t_times, lam)


def shuffle_features(X):
    idx = np.arange(len(X.T))
    np.random.shuffle(idx)
    return X.T[idx].T


def run_experiment(X, y, pole, total_labels, k, d, output, t_times, lam):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    X = shuffle_features(X)
    prob_x = [{x: c / len(f_val) for x, c in zip(*np.unique(f_val, return_counts=True))} for f_val in X.T]
    afs = AFS(X, y, prob_x, k=k, delta=d)

    for _ in range(t_times):
        result = []
        for p in pole:
            if p == 'NORM1':
                result.append(afs.run(total_labels, agg_func=np.sum, plato=lam))
            elif p == 'NORM2':
                result.append(afs.run(total_labels, agg_func=np.linalg.norm, plato=lam))
            elif p == 'NORM-INF':
                result.append(afs.run(total_labels, agg_func=np.max, plato=lam))
            elif p == 'RANDOM':
                result.append(feature_ranking_random(X, y, k, prob_x, total_labels))

        output.append(result)


def run_all_experiment(X, y, total_labels, k, data_name, delta, n_threads, t_times, lam):
    pole = ['RANDOM', 'NORM1', 'NORM2', 'NORM-INF']
    process = []
    outputs = mp.Manager().list()

    X = k_means_quantization(X, 10)
    X = remove_single_val_feature(X)
    total_labels = min(total_labels, len(X))
    for j in range(n_threads):
        p = mp.Process(target=run_experiment, args=(X, y, pole, total_labels, k, delta, outputs, t_times, lam))
        process.append(p)
        p.start()

    for p in process:
        p.join()

    makedirs('results', exist_ok=True)
    file_name = 'results/{}_k={}.pth'.format(data_name, k)
    output = np.array(outputs)
    results = {p: output[:, i] for i, p in enumerate(pole)}
    torch.save(results, file_name)
    plot(results, data_name, k)


if __name__ == '__main__':
    main()
