""" Functions to produce data for the what-if scenarios. """

from pathlib import Path

import ciw
import dask
import numpy as np
import pandas as pd
from ciw.dists import Exponential
from scipy import stats

DATA_DIR = Path("../data/")


def get_times(diff):

    times = diff.dt.total_seconds().div(24 * 60 * 60, fill_value=0)
    return times


def get_queue_params(data, prop=1, lambda_coeff=1, dist=stats.expon):
    """ Get the arrival and service parameters from `data` and the given `prop`. """

    inter_arrivals = (
        data.set_index("admission_date").sort_index().index.to_series().diff()
    )
    interarrival_times = get_times(inter_arrivals)
    lambda_ = lambda_coeff / np.mean(interarrival_times)

    mean_system_time = np.mean(data["true_los"])
    mu_estimate = mean_system_time * prop

    queue_params = {"arrival": lambda_, "service": 1 / mu_estimate}

    return queue_params


def get_simulation_results(Q, max_time):

    records = Q.get_all_records()
    results = pd.DataFrame(
        [
            r
            for r in records
            if max_time * 0.25 < r.arrival_date < max_time * 0.75
        ]
    )

    results["utilisation"] = Q.transitive_nodes[0].server_utilisation
    results["system_time"] = results["exit_date"] - results["arrival_date"]
    results["num_servers"] = num_servers
    results["lambda_coeff"] = round(lambda_coeff, 2)
    results["seed"] = seed

    return results[
        [
            "customer_class",
            "utilisation",
            "system_time",
            "num_servers",
            "lambda_coeff",
            "seed",
        ]
    ]


def get_best_params():

    with open(DATA_DIR / "wasserstein/best_params.txt", "r") as f:
        strings = f.read().split(" ")
        props, num_servers = list(map(float, strings[:-1])), int(strings[-1])

    return props, num_servers


@dask.delayed
def simulate_queue(
    data, column, props, num_servers, seed, max_time, lambda_coeff=1
):

    ciw.seed(seed)
    all_queue_params = {}
    for (label, subdata), service_prop in zip(data.groupby(column), props):
        all_queue_params[label] = get_queue_params(
            subdata, service_prop, lambda_coeff
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

    result = get_simulation_results(Q, max_time)
    return result
