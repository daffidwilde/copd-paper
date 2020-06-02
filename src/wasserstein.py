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

NUM_CORES = int(sys.argv[1])
NUM_SEEDS = int(sys.argv[2])

GRANULARITY = 0.05
if len(sys.argv) > 3:
    GRANULARITY = float(sys.argv[3])

if len(sys.argv) > 4:
    OUT_DIR = DATA_DIR / str(sys.argv[4])

OUT_DIR.mkdir(exist_ok=True)

COPD = pd.read_csv(
    DATA_DIR / "clusters/copd_clustered.csv",
    parse_dates=["admission_date", "discharge_date"],
)

COPD = COPD.dropna(subset=["cluster"])
COPD["cluster"] = COPD["cluster"].astype(int)

NUM_CLUSTERS = COPD["cluster"].nunique()
MAX_TIME = 365 * 4
PROP_LIMS = (0.5, 1.01, GRANULARITY)
SERVER_LIMS = (40, 56, 5)


@dask.delayed
def run_multiple_class_trial(
    data, column, props, num_servers, seed, max_time, write=None
):

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
    if write is not None:
        results.to_csv(OUT_DIR / write / f"{seed}.csv", index=False)

    distance = stats.wasserstein_distance(
        results["system_time"], data["true_los"]
    )

    return (*props, num_servers, seed, distance)


def get_case(data, case):

    maximal_distance = data.groupby(
        ["p_0", "p_1", "p_2", "p_3", "num_servers"]
    )["distance"].max()
    
    if case == "best":
        *ps, c = maximal_distance.idxmin()
        distance = maximal_distance.min()
    elif case == "worst":
        *ps, c = maximal_distance.idxmax()
        distance = maximal_distance.max()
    else:
        raise NotImplementedError("Case must be one of `'best'` or `'worst'`.")

    CASE_DIR = OUT_DIR / case
    CASE_DIR.mkdir(exist_ok=True)

    tasks = (
        run_multiple_class_trial(
            COPD, "cluster", ps, c, seed, MAX_TIME, write=case
        )
        for seed in range(NUM_SEEDS)
    )

    with ProgressBar():
        _ = dask.compute(*tasks, scheduler="processes", num_workers=NUM_CORES)

    dfs = (
        pd.read_csv(CASE_DIR / f"{seed}.csv") for seed in range(NUM_SEEDS)
    )

    df = pd.concat(dfs)
    df.to_csv(CASE_DIR / "main.csv", index=False)

    with open(CASE_DIR / "params.txt", "w") as f:
        string = " ".join(map(str, [*ps, c, distance]))
        f.write(string)


def main(prop_lims, n_clusters, server_lims, seeds, cores):

    tasks = (
        run_multiple_class_trial(
            COPD, "cluster", props, num_servers, seed, MAX_TIME
        )
        for props, num_servers, seed in it.product(
            it.product(np.arange(*prop_lims), repeat=n_clusters),
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

    for case in ["best", "worst"]:
        get_case(df, case)


if __name__ == "__main__":
    main(PROP_LIMS, NUM_CLUSTERS, SERVER_LIMS, NUM_SEEDS, NUM_CORES)
