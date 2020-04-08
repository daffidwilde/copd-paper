""" Functions for moving clusters experiment. """

import ciw
import dask
import numpy as np
import pandas as pd

from .util import get_best_params, get_queue_params, get_simulation_results

DATA_DIR = Path("../../data/")
OUT_DIR = DATA_DIR / "moving_clusters/"

PROPS, NUM_SERVERS = get_best_params()
COPD = pd.read_csv(
    DATA_DIR / "copd_clustered.csv",
    parse_dates=["admission_date", "discharge_date"],
)

NUM_CORES = int(sys.argv[1])
NUM_SEEDS = int(sys.argv[2])
GRANULARITY = float(sys.argv[3])

PROP_TO_MOVE_RANGE = np.arange(0, 1, GRANULARITY)
MAX_TIME = 365 * 4


def update_arrival_params(all_queue_params, origin, destination, prop_to_move):

    origin_lambda = all_queue_params[origin]["arrival"]
    destination_lambda = all_queue_params[destination]["arrival"]

    destination_lambda += prop_to_move * origin_lambda
    origin_lambda *= 1 - prop_to_move

    all_queue_params[origin]["arrival"] = origin_lambda
    all_queue_params[destination]["arrival"] = destination_lambda

    return all_queue_params


@dask.delayed
def simulate_moving_clusters_queue(
    data, props, num_servers, origin, destination, prop_to_move, seed, max_time
):

    all_queue_params = {}
    for (label, cluster), service_prop in zip(data.groupby("cluster"), props):
        all_queue_params[label] = get_queue_params(cluster, service_prop)

    all_queue_params = update_arrival_params(
        all_queue_params, origin, destination, prop_to_move
    )

    ciw.seed(seed)
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

    results = get_simulation_results(Q, max_time)

    results["utilisation"] = Q.transitive_nodes[0].server_utilisation
    results["prop_to_move"] = prop_to_move
    results["origin"] = origin
    results["destination"] = destination
    results["seed"] = seed

    name = "_".join(
        map(str, [round(prop_to_move, 2), origin, destination, seed])
    )
    results.to_csv(OUT_DIR / f"{name}.csv", index=False)

    return results


def main(
    num_cores, props, num_servers, num_seeds, max_time, prop_to_move_range
):

    n_clusters = copd["cluster"].nunique()
    label_combinations = (
        labels
        for labels in it.product(range(n_clusters), repeat=2)
        if labels[0] != labels[1]
    )

    tasks = (
        simulate_moving_clusters_queue(
            COPD,
            props,
            num_servers,
            origin,
            destination,
            prop_to_move,
            seed,
            max_time,
        )
        for (origin, destination), prop_to_move, seed in it.product(
            label_combinations, prop_to_move_range, range(num_seeds)
        )
    )

    with ProgressBar():
        _ = dask.compute(*tasks, scheduler="processes", num_workers=num_cores)


if __name__ == "__main__":
    main(NUM_CORES, PROPS, NUM_SERVERS, NUM_SEEDS, MAX_TIME, PROP_TO_MOVE_RANGE)
