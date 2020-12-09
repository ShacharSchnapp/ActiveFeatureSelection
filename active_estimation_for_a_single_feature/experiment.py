import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pickle

from typing import List
from multiprocessing import Process, Manager
from polices.entropy_b_policy import EntropyBPolicy
from polices.entropy_h_policy import EntropyHPolicy
from polices.inf_b_policy import INFBPolicy
from polices.inf_h_policy import INFHPolicy
from polices.entropy_cp_policy import EntropyCPPolicy
from polices.policy import Policy
from polices.var_b_policy import VARBPolicy
from polices.var_cp_policy import VARCPPolicy
from polices.proportional_policy import ProportionalPolicy
from mab_unit.mab import MAB
from os import makedirs
from os.path import join
from utility.utiliy import init_data
from polices.var_h_policy import VARHPolicy
from utility.utiliy_active import get_prob


def pross_run(police, output_polices, probs, prob_x, avg_time, seeds, T):
    for i in range(avg_time):
        mab = MAB(probs, prob_x, seeds[i])
        police.run_places(T, mab)
    output_polices.append(police)


def runner(polices: List[Policy], probs, prob_x, T, avg_time):
    k = len(probs)
    seeds = []
    for _ in range(avg_time):
        seed = []
        for x in range(k):
            seed.append(np.random.choice([0, 1], T + k*2, p=[1 - probs[x], probs[x]]))
        seeds.append(seed)

    process = []
    output_polices = Manager().list()

    for policy in polices:
        p = Process(target=pross_run, args=(policy, output_polices, probs, prob_x, avg_time, seeds, T))
        p.start()
        process.append(p)

    for p in process:
        p.join()

    output_polices = sorted(output_polices, key=lambda x: x.name)
    return output_polices


def auto_label(rects):
    for rect in rects:
        height = int(rect.get_height())
        plt.annotate('{}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=7)


def plot_results(polices: List[Policy], output_path, T):
    plt.figure(figsize=(8.0, 5.0))
    times = np.arange(T)

    # MSE Score
    all = plt.subplot(211)
    plt.title('Information Score')
    plt.xlabel('Times')
    plt.ylabel('Score')

    zoom_start = plt.subplot(223)
    # plt.title('MSE Score')
    plt.xlabel('Times')
    plt.ylabel('Score')

    zoo_end = plt.subplot(224)
    # plt.title('MSE Score')
    plt.xlabel('Times')
    plt.ylabel('Score')

    results = {'x': times}
    for i, policy in enumerate(polices):
        mean, lb, ub = policy.get_avg()
        results[policy.name] = mean
        results[policy.name + '_lb'] = lb
        results[policy.name + '_ub'] = ub
        plt.subplot(all)
        plt.plot(times, mean, linewidth=1, label=policy.name)
        plt.fill_between(times, ub, lb, alpha=.3)
        plt.subplot(zoom_start)
        plt.plot(times[50:200], mean[50:200], linewidth=1, label=policy.name)
        plt.fill_between(times[50:200], ub[50:200], lb[50:200], alpha=.3)
        plt.subplot(zoo_end)
        plt.plot(times[200:], mean[200:], linewidth=1, label=policy.name)
        plt.fill_between(times[200:], ub[200:], lb[200:], alpha=.3)

    scipy.io.savemat(output_path + '.mat', results)
    plt.subplot(all)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path + '.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def save_results(output_dir, data):
    with open(join(output_dir, 'file_results.pickle'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_exp(exp_name, list_probs, T):
    makedirs(exp_name, exist_ok=True)

    for probs in list_probs:
        run_polices = [ProportionalPolicy(),
                       EntropyCPPolicy(delta=delta),
                       EntropyBPolicy(delta=delta),
                       EntropyHPolicy(delta=delta),
                       VARCPPolicy(delta=delta),
                       VARHPolicy(delta=delta),
                       VARBPolicy(delta=delta),
                       INFHPolicy(delta=delta),
                       INFBPolicy(T, delta=delta)]

        prob_x = len(probs) * [1 / len(probs)]
        output_polices = runner(run_polices, probs, prob_x, T,  avg_time=avg_time)
        plot_results(output_polices, join(exp_name, str(probs)), T)


def get_prob_of_fachuer(data_name, f_id):
    X, y = init_data(data_name)
    prob_x, prob_x_geven_y = get_prob(X[:, f_id], y, np.arange(len(X)))
    probs = [prob_x_geven_y.get((x, 1), 0) for x in prob_x]
    return probs, list(prob_x.values())


delta = 0.05
k = 10
avg_time = 1000
T = 500

if __name__ == '__main__':
    # MMA
    print('start min max experiment')
    min_max_arms_probs = []
    for k in range(2, 11, 2):
        min_max_arms_probs += [[0.5]*k]
        for p_min in [0.001, 0.01, 0.1]:
            for n_p_min in range(k):
                min_max_arms_probs.append([p_min] * (k - n_p_min) + [0.5] * n_p_min)

    process = []
    p = Process(target=run_exp, args=('results/min_max_arms', min_max_arms_probs, T))
    p.start()
    process.append(p)

    # Uniform
    print('start uniform experiment')
    uniform_probs = []
    for k in range(2, 11, 2):
        for _ in range(5):
            uniform_probs.append(np.round(np.random.uniform(low=0, high=0.5, size=k), decimals=2).tolist())

    p = Process(target=run_exp, args=('results/uniform', uniform_probs, T))
    p.start()
    process.append(p)

    # Adult
    print('start adult experiment')
    for j in range(10):
        probs, prob_x = get_prob_of_fachuer('./data/adult', j)
        k = len(probs)
        run_polices = [ProportionalPolicy(),
                       EntropyCPPolicy(delta=delta),
                       EntropyBPolicy(delta=delta),
                       EntropyHPolicy(delta=delta),
                       VARCPPolicy(delta=delta),
                       VARHPolicy(delta=delta),
                       VARBPolicy(delta=delta),
                       INFHPolicy(delta=delta),
                       INFBPolicy(T, delta=delta)]

        output_polices = runner(run_polices, probs, prob_x, T, avg_time=avg_time)

        exp_name = 'results/adult'
        makedirs(exp_name, exist_ok=True)
        plot_results(output_polices, join(exp_name, str(j)), T)

    for p in process:
        p.join()
