""" Main function for lambda scaling experiment. """

import itertools as it
import sys

import dask
import numpy as np
import pandas as pd
import tqdm
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

OUT_DIR = DATA_DIR / "lambda_scaling/"
OUT_DIR.mkdir(exist_ok=True)

NUM_CORES = int(sys.argv[1])
NUM_SEEDS = int(sys.argv[2])
SIGMA_GRANULARITY = float(sys.argv[3])

SIGMA_RANGE = np.arange(0.5, 2.01, SIGMA_GRANULARITY).round(2)


def get_params():
    return it.product(SIGMA_RANGE, range(NUM_SEEDS))


def main():

    tasks = (
        simulate_queue(COPD, PROPS, NUM_SERVERS, seed, MAX_TIME, sigma)
        for sigma, seed in get_params()
    )

    with ProgressBar():
        queues = dask.compute(
            *tasks, scheduler="processes", num_workers=NUM_CORES
        )

    util_dfs, time_dfs = [], []
    for (sigma, seed), queue in tqdm.tqdm(zip(get_params(), queues)):
        utilisations, system_times = get_results(
            queue, MAX_TIME, sigma=sigma, seed=seed
        )

        util_dfs.append(utilisations)
        time_dfs.append(system_times)

    utilisations = pd.concat(util_dfs)
    system_times = pd.concat(time_dfs)

    utilisations.to_csv(OUT_DIR / "utilisations.csv", index=False)
    system_times.to_csv(OUT_DIR / "system_times.csv", index=False)


if __name__ == "__main__":
    main()
