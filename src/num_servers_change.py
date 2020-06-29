""" Main function for number of servers experiment. """

import itertools as it
import sys

import tqdm

import dask
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from util import (
    COPD,
    DATA_DIR,
    MAX_TIME,
    NUM_SERVERS,
    PROPS,
    get_results,
    simulate_queue,
)

OUT_DIR = DATA_DIR / "num_servers_change/"
OUT_DIR.mkdir(exist_ok=True)

NUM_CORES = int(sys.argv[1])
NUM_SEEDS = int(sys.argv[2])
MIN_SERVER_LEVEL = float(sys.argv[3])
MAX_SERVER_LEVEL = float(sys.argv[4])

SERVER_RANGE = np.arange(
    int(MIN_SERVER_LEVEL * NUM_SERVERS), int(MAX_SERVER_LEVEL * NUM_SERVERS) + 1
)

PARAMS = lambda: it.product(SERVER_RANGE, range(NUM_SEEDS))


def main():

    tasks = (
        simulate_queue(COPD, PROPS, num_servers, seed, MAX_TIME)
        for num_servers, seed in PARAMS()
    )

    with ProgressBar():
        queues = dask.compute(
            *tasks, scheduler="processes", num_workers=NUM_CORES
        )

    util_dfs, time_dfs = [], []
    for (num_servers, seed), queue in tqdm.tqdm(zip(PARAMS(), queues)):
        utilisations, system_times = get_results(
            queue, MAX_TIME, num_servers=num_servers, seed=seed
        )

        util_dfs.append(utilisations)
        time_dfs.append(system_times)

    utilisations = pd.concat(util_dfs)
    system_times = pd.concat(time_dfs)

    utilisations.to_csv(OUT_DIR / "utilisations.csv", index=False)
    system_times.to_csv(OUT_DIR / "system_times.csv", index=False)


if __name__ == "__main__":
    main()
