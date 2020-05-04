""" Functions relating to parameter estimation via Wasserstein distance. """

import itertools as it
import sys
from collections import defaultdict

import ciw
import dask
import numpy as np
import pandas as pd
from ciw.dists import Exponential
from dask.diagnostics import ProgressBar
from scipy import stats

from util import DATA_DIR, get_queue_params

OUT_DIR = DATA_DIR / "wasserstein/"
OUT_DIR.mkdir(exist_ok=True)

COPD = pd.read_csv(
    DATA_DIR / "copd_clustered.csv",
    parse_dates=["admission_date", "discharge_date"],
)

NUM_CORES = int(sys.argv[1])
NUM_SEEDS = int(sys.argv[2])

NUM_CLUSTERS = COPD["cluster"].nunique()
MAX_TIME = 365 * 4
PROP_LIMS = (0.5, 1, 11)
SERVER_LIMS = (40, 56, 1)


@dask.delayed
def run_multiple_class_trial(data, column, props, num_servers, seed, max_time):

    ciw.seed(seed)
    all_queue_params = defaultdict(dict)
    for (label, subdata), service_prop in zip(data.groupby(column), props):
        all_queue_params[label] = get_queue_params(subdata, service_prop)

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

    records = Q.get_all_records()
    results = pd.DataFrame(
        [
            r
            for r in records
            if max_time * 0.25 < r.arrival_date < max_time * 0.75
        ]
    )

    results["system_time"] = results["exit_date"] - results["arrival_date"]
    results["service_prop"] = results["customer_class"].apply(
        lambda x: props[x]
    )
    results["num_servers"] = num_servers
    results["seed"] = seed

    name = (
        "_".join([str(p) for p in props])
        + "_"
        + "_".join([str(num_servers), str(seed)])
    )
    results.to_csv(OUT_DIR / f"{name}.csv", index=False)

    distance = stats.wasserstein_distance(
        results["total_time"], copd["true_los"]
    )
    return (*props, num_servers, seed, distance)


def main(prop_lims, n_clusters, server_lims, seeds, cores):

    tasks = (
        run_multiple_class_trial(
            COPD, "cluster", props, num_servers, seed, MAX_TIME
        )
        for props, num_servers, seed in it.product(
            it.product(np.linspace(*prop_lims), repeat=n_clusters),
            range(*server_lims),
            range(seeds),
        )
    )

    with ProgressBar():
        results = dask.compute(*tasks, scheduler="processes", num_workers=cores)

    columns = [
        *(f"p_{i}" for i in range(n_clusters)),
        "num_servers",
        "seed",
        "distance",
    ]
    df = pd.DataFrame(results, columns=columns)

    df.to_csv(OUT_DIR / "main.csv", index=False)


if __name__ == "__main__":
    main(PROP_LIMS, NUM_CLUSTERS, SERVER_LIMS, NUM_SEEDS, NUM_CORES)
