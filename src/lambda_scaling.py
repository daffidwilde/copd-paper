""" Main function for lambda scaling experiment. """

import itertools as it
import sys

import dask
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar

from util import DATA_DIR, get_best_params, simulate_queue

OUT_DIR = DATA_DIR / "lambda_scaling/"
OUT_DIR.mkdir(exist_ok=True)

PROPS, NUM_SERVERS = get_best_params()
COPD = pd.read_csv(
    DATA_DIR / "copd_clustered.csv",
    parse_dates=["admission_date", "discharge_date"],
)

NUM_CORES = int(sys.argv[1])
NUM_SEEDS = int(sys.argv[2])
LAMBDA_GRANULARITY = float(sys.argv[3])

LAMBDA_COEFF_RANGE = np.arange(0.5, 2.01, LAMBDA_GRANULARITY)
MAX_TIME = 365 * 4


def main(
    num_cores, props, num_servers, num_seeds, max_time, lambda_coeff_range
):

    tasks = (
        simulate_queue(
            COPD, "cluster", props, num_servers, seed, max_time, lambda_coeff
        )
        for lambda_coeff, seed in it.product(
            lambda_coeff_range, range(num_seeds)
        )
    )

    with ProgressBar():
        results = dask.compute(
            *tasks, scheduler="processes", num_workers=num_cores
        )

    for result in results:
        lambda_coeff = result["lambda_coeff"].iloc[0]
        seed = result["seed"].iloc[0]

        filename = OUT_DIR / f"{lambda_coeff}_{seed}.csv"
        result.to_csv(filename, index=False)


if __name__ == "__main__":
    main(NUM_CORES, PROPS, NUM_SERVERS, NUM_SEEDS, MAX_TIME, LAMBDA_COEFF_RANGE)
