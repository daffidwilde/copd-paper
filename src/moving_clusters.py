""" Functions for moving clusters experiment. """

import itertools as it
import sys

import ciw
import dask
import numpy as np
import pandas as pd
import tqdm
from ciw.dists import Exponential
from dask.diagnostics import ProgressBar

from util import (
    COPD,
    DATA_DIR,
    MAX_TIME,
    NUM_SERVERS,
    PROPS,
    get_queue_params,
    get_results,
)

OUT_DIR = DATA_DIR / "moving_clusters/"
OUT_DIR.mkdir(exist_ok=True)

NUM_CORES = int(sys.argv[1])
NUM_SEEDS = int(sys.argv[2])
MOVE_GRANULARITY = float(sys.argv[3])

PROP_TO_MOVE_RANGE = np.arange(0, 1, MOVE_GRANULARITY).round(2)

n_clusters = COPD["cluster"].nunique()


def get_combinations():
    return (
        labels
        for labels in it.product(range(n_clusters), repeat=2)
        if labels[0] != labels[1]
    )


def get_params():
    return it.product(get_combinations(), PROP_TO_MOVE_RANGE, range(NUM_SEEDS))


def update_arrival_params(all_queue_params, origin, destination, prop_to_move):

    origin_lambda = all_queue_params[origin]["arrival"]
    destination_lambda = all_queue_params[destination]["arrival"]

    destination_lambda += prop_to_move * origin_lambda
    origin_lambda *= 1 - prop_to_move

    all_queue_params[origin]["arrival"] = origin_lambda
    all_queue_params[destination]["arrival"] = destination_lambda

    return all_queue_params


@dask.delayed
def simulate_queue(
    data, props, num_servers, origin, destination, prop_to_move, seed, max_time
):
    """ Build and simulate a queue under the provided parameters. """

    ciw.seed(seed)

    all_queue_params = {}
    n_clusters = data["cluster"].nunique()
    for label, prop in zip(range(n_clusters), props):

        cluster = data[data["cluster"] == label]
        all_queue_params[label] = get_queue_params(cluster, prop)

    all_queue_params = update_arrival_params(
        all_queue_params, origin, destination, prop_to_move
    )

    N = ciw.create_network(
        arrival_distributions={
            f"Class {label}": [Exponential(params["arrival"])]
            for label, params in all_queue_params.items()
        },
        service_distributions={
            f"Class {label}": [Exponential(params["service"])]
            for label, params in all_queue_params.items()
        },
        number_of_servers=[num_servers],
    )

    Q = ciw.Simulation(N)
    Q.simulate_until_max_time(max_time)

    return Q


def main():

    tasks = (
        simulate_queue(
            COPD,
            PROPS,
            NUM_SERVERS,
            origin,
            destination,
            prop_to_move,
            seed,
            MAX_TIME,
        )
        for (origin, destination), prop_to_move, seed in get_params()
    )

    with ProgressBar():
        queues = dask.compute(
            *tasks, scheduler="processes", num_workers=NUM_CORES
        )

    util_dfs, time_dfs = [], []
    for ((orgn, dest), move, seed), queue in tqdm.tqdm(
        zip(get_params(), queues)
    ):
        utilisations, system_times = get_results(
            queue,
            MAX_TIME,
            origin=orgn,
            destination=dest,
            prop_to_move=move,
            seed=seed,
        )
        util_dfs.append(utilisations)
        time_dfs.append(system_times)

    utilisations = pd.concat(util_dfs)
    system_times = pd.concat(time_dfs)

    utilisations.to_csv(OUT_DIR / "utilisations.csv", index=False)
    system_times.to_csv(OUT_DIR / "system_times.csv", index=False)


if __name__ == "__main__":
    main()
