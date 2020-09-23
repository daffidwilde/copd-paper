""" Functions relating to parameter estimation via Wasserstein distance. """

import itertools as it
import sys

import ciw
import dask
import numpy as np
import pandas as pd
from ciw.dists import Exponential
from dask.diagnostics import ProgressBar
from scipy import stats

from util import DATA_DIR, ShiftedExponential, get_queue_params

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
# SERVER_LIMS = (20, 61, 5)
SERVER_LIMS = (20, 41, 10)


@dask.delayed
def run_multiple_class_trial(
    data, props, num_servers, seed, max_time, write=None
):
    """Run a single trial with multiple customer classes (clusters). This means
    calculating the queuing parameters for our M/(S)M/c queue and simulating it
    for a fixed amount of time. Record everything if needed, return the model
    parameters and the Wasserstein distance."""

    ciw.seed(seed)

    all_queue_params = {}
    for label, prop in zip(range(NUM_CLUSTERS), props):

        cluster = data[data["cluster"] == label]
        all_queue_params[label] = get_queue_params(cluster, prop)

    N = ciw.create_network(
        arrival_distributions={
            f"Class {label}": [Exponential(params["arrival"])]
            for label, params in all_queue_params.items()
        },
        service_distributions={
            f"Class {label}": [ShiftedExponential(*params["service"])]
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

    distances = [
        stats.wasserstein_distance(
            results[results["customer_class"] == label]["system_time"],
            data[data["cluster"] == label]["true_los"],
        )
        for label in range(NUM_CLUSTERS)
    ]

    return (*props, num_servers, seed, *distances)


def get_case(data, case):
    """ Get the best, median or worst case from the data. """

    data["max_distance"] = data[
        [f"distance_{i}" for i in range(NUM_CLUSTERS)]
    ].max(axis=1)

    maximal_distance = data.groupby(
        [f"p_{i}" for i in range(NUM_CLUSTERS)] + ["num_servers"]
    )["max_distance"].max()

    if case == "best":
        *ps, c = maximal_distance.idxmin()
        distance = maximal_distance.min()
    elif case == "worst":
        *ps, c = maximal_distance.idxmax()
        distance = maximal_distance.max()
    elif case == "median":
        diffs = (maximal_distance - maximal_distance.median()).abs()
        *ps, c = diffs.idxmin()
        distance = maximal_distance.median()
    else:
        raise NotImplementedError(
            "Case must be one of `'best'`, `'median'` or `'worst'`."
        )

    CASE_DIR = OUT_DIR / case
    CASE_DIR.mkdir(exist_ok=True)

    tasks = (
        run_multiple_class_trial(COPD, ps, c, seed, MAX_TIME, write=case)
        for seed in range(NUM_SEEDS)
    )

    with ProgressBar():
        _ = dask.compute(*tasks, scheduler="processes", num_workers=NUM_CORES)

    dfs = (pd.read_csv(CASE_DIR / f"{seed}.csv") for seed in range(NUM_SEEDS))

    df = pd.concat(dfs)
    df.to_csv(CASE_DIR / "main.csv", index=False)

    with open(CASE_DIR / "params.txt", "w") as f:
        string = " ".join(map(str, [*ps, c, distance]))
        f.write(string)


def main():
    """ The main function for running and writing. """

    tasks = (
        run_multiple_class_trial(COPD, props, num_servers, seed, MAX_TIME)
        for props, num_servers, seed in it.product(
            it.product(np.arange(*PROP_LIMS), repeat=NUM_CLUSTERS),
            range(*SERVER_LIMS),
            range(NUM_SEEDS),
        )
    )

    with ProgressBar():
        results = dask.compute(
            *tasks, scheduler="processes", num_workers=NUM_CORES
        )

    columns = [
        *(f"p_{i}" for i in range(NUM_CLUSTERS)),
        "num_servers",
        "seed",
        *(f"distance_{i}" for i in range(NUM_CLUSTERS)),
    ]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(OUT_DIR / "main.csv", index=False)

    for case in ["best", "median", "worst"]:
        get_case(df, case)


if __name__ == "__main__":
    main()
