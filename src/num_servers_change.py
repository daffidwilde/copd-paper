""" Main function for number of servers experiment. """

import itertools as it
import sys

import dask
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

from util import DATA_DIR, get_best_params, simulate_queue

OUT_DIR = DATA_DIR / "num_servers_change/"
OUT_DIR.mkdir(exist_ok=True)

PROPS, _ = get_best_params()
COPD = pd.read_csv(
    DATA_DIR / "copd_clustered.csv",
    parse_dates=["admission_date", "discharge_date"],
)

NUM_CORES = int(sys.argv[1])
NUM_SEEDS = int(sys.argv[2])
MIN_SERVERS = int(sys.argv[3])
MAX_SERVERS = int(sys.argv[4])

SERVER_RANGE = range(MIN_SERVERS, MAX_SERVERS)
MAX_TIME = 365 * 4


def main(num_cores, props, num_servers_range, num_seeds, max_time=365 * 3):

    tasks = (
        simulate_queue(COPD, "cluster", props, num_servers, seed, max_time)
        for num_servers, seed in it.product(num_servers_range, range(num_seeds))
    )

    with ProgressBar():
        results = dask.compute(
            *tasks, scheduler="processes", num_workers=num_cores
        )

    for result in results:
        num_servers = result["num_servers"].iloc[0]
        seed = result["seed"].iloc[0]

        filename = OUT_DIR / f"{num_servers}_{seed}.csv"
        result.to_csv(filename, index=False)


if __name__ == "__main__":
    main(NUM_CORES, PROPS, SERVER_RANGE, NUM_SEEDS, MAX_TIME)
