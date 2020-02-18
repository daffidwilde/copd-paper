""" A script to simulate M|M|c queues with varying parameters. """

import itertools as it
import sys

import ciw
import dask
import numpy as np
import pandas as pd

from ciw.dists import Exponential
from dask.diagnostics import ProgressBar
from scipy import stats


PATH_TO_DATA = str(sys.argv[1])
PATH_TO_OUT = str(sys.argv[2])
NUM_SEEDS = int(sys.argv[3])
NUM_CORES = int(sys.argv[4])


def get_queue_params(data, prop, dist=stats.expon):
    """ Evaluate the empirical arrival and service parameters from `data` and
    the given `prop`. """

    inter_arrivals = (
        data.set_index("admission_date").sort_index().index.to_series().diff()
    )
    interarrival_times = inter_arrivals.dt.total_seconds().div(
        24 * 60 * 60, fill_value=0
    )
    lambda_ = np.mean(interarrival_times)

    mean_system_time = np.mean(data["true_los"])
    mu_estimate = mean_system_time * service_prop

    queue_params = {"arrival": 1 / lambda_, "service": 1 / mu_estimate}

    return queue_params


@dask.delayed
def run_multiple_class_trial(
    data, props, num_servers, max_time=365 * 30, queue_capacity=None, seed=0
):
    """ A function to run a multi-class simulation trial on an M|M|c queue.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to use for simulation. Assumed columns are: "intervention",
        "arrival_date", "discharge_date", "true_los".
    props : list
        A list of the service proportions for each class, :math:`p_i`.
    num_servers : int
        The number of servers in the system, :math:`c`.
    max_time : int
        The maximum time for the simulation. Time units in days; defaults to
        thirty years.
    queue_capacity : int, optional
        The capacity of the queue. If not specified, infinite capacity is
        assumed.
    seed : int
        A seed for Ciw's pseudo-random number generator.

    Returns
    -------
    results : pandas.DataFrame
        A dataset containing a detailed record of every individual that has been
        through the queuing system.
    """

    if queue_capacity is not None:
        queue_capacity = [queue_capacity]

    ciw.seed(seed)

    all_queue_params = defaultdict(dict)
    for (intervention, subdata), service_prop in zip(
        data.groupby("intervention"), props
    ):
        all_queue_params[intervention] = get_queue_params(subdata, service_prop)

    N = ciw.create_network(
        arrival_distributions={
            f"Class {i}": [Exponential(params["arrival"])]
            for i, params in enumerate(all_queue_params.values())
        },
        service_distributions={
            f"Class {i}": [Exponential(params["service"])]
            for i, params in enumerate(all_queue_params.values())
        },
        num_servers=[num_servers],
        queue_capacities=queue_capacity,
    )

    Q = ciw.Simulation(N)
    Q.simulate_until_max_time(max_time)

    records = Q.get_all_records()
    results = pd.DataFrame(
        [
            r
            for r in records
            if max_time * 0.25 < r.arrival_date < max_time * 0.75
        ]
    )

    results["service_prop"] = round(props[results["customer_class"] - 1], 2)
    results["num_servers"] = num_servers
    results["seed"] = seed

    return results


def main(path_to_data, path_to_out, num_seeds, num_cores):

    copd = pd.read_csv(
        path_to_data, parse_dates=["admission_date", "discharge_date"]
    )

    prop_lims, steps = (0.1, 1), 10
    server_lims = (10, 21)

    tasks = (
        run_multiple_class_trial(copd, props, num_servers, seed=seed)
        for props, num_servers, seed in it.product(
            it.product(np.linspace(*prop_lims, steps)),
            range(*server_lims),
            range(num_seeds),
        )
    )

    with ProgressBar():
        dfs = dask.compute(*tasks, scheduler="processes", num_workers=num_cores)

    df = pd.concat(dfs)
    df["total_time"] = df["exit_date"] - df["arrival_date"]
    df.to_csv(path_to_out, index=False)


if __name__ == "__main__":
    main(PATH_TO_DATA, PATH_TO_OUT, NUM_SEEDS, NUM_CORES)
