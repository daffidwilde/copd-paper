""" Functions to produce data for the what-if scenarios. """

from pathlib import Path
from random import expovariate

import ciw
import dask
import numpy as np
import pandas as pd
from ciw.dists import Distribution, Exponential

DATA_DIR = Path("../data/")

COPD = pd.read_csv(
    DATA_DIR / "clusters/copd_clustered.csv", parse_dates=["admission_date"]
)
COPD = COPD.dropna(subset=["cluster"])
COPD["cluster"] = COPD["cluster"].astype(int)

MAX_TIME = 365 * 4

with open(DATA_DIR / "wasserstein/best/params.txt", "r") as f:
    parts = f.read().split(" ")
    PROPS = list(map(float, parts[:-2]))
    NUM_SERVERS = int(parts[-2])


class ShiftedExponential(Distribution):
    """An exponential distribution with a `rate` parameter shifted by some
    constant `shift`."""

    def __init__(self, rate, shift):

        self.rate = rate
        self.shift = shift

    def sample(self, t=None, ind=None):

        return self.shift + expovariate(self.rate)


def get_queue_params(data, prop=1, sigma=1):
    """ Get the arrival and service parameters from `data`. """

    inter_arrivals = (
        data.set_index("admission_date").sort_index().index.to_series().diff()
    )
    interarrival_times = inter_arrivals.dt.total_seconds().div(
        24 * 60 * 60, fill_value=0
    )
    arrival_rate = sigma / np.mean(interarrival_times)

    minimum_length = max(0, data["true_los"].min())
    shifted_lengths = data["true_los"] - minimum_length
    mean_shifted_length = shifted_lengths.mean()
    service_rate = 1 / (mean_shifted_length * prop)

    queue_params = {
        "arrival": arrival_rate,
        "service": [service_rate, minimum_length],
    }

    return queue_params


@dask.delayed
def simulate_queue(data, props, num_servers, seed, max_time, sigma=1):
    """ Build and simulate a queue under the provided parameters. """

    ciw.seed(seed)

    all_queue_params = {}
    n_clusters = data["cluster"].nunique()
    for label, prop in zip(range(n_clusters), props):

        cluster = data[data["cluster"] == label]
        all_queue_params[label] = get_queue_params(cluster, prop, sigma)

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

    return Q


def get_utilisations(queue, **kwargs):
    """ Get the utilisation for each server in the queue. """

    df = pd.DataFrame()
    node = queue.transitive_nodes[0]

    utilisations = []
    for server in node.servers:
        utilisations.append(server.busy_time / server.total_time)

    df["utilisation"] = utilisations

    for key, value in kwargs.items():
        df[key] = value

    return df


def get_system_times(queue, max_time, **kwargs):
    """Get the system times for every patient to pass through the queue within
    the centre 50% of the `max_time` period."""

    records = queue.get_all_records()
    results = pd.DataFrame(
        [
            r
            for r in records
            if max_time * 0.25 < r.arrival_date < max_time * 0.75
        ],
        columns=ciw.DataRecord._fields,
    )

    df = pd.DataFrame()
    df["system_time"] = results["exit_date"] - results["arrival_date"]

    for key, value in kwargs.items():
        df[key] = value

    return df


def get_results(queue, max_time, **kwargs):
    """ Get the utilisation and system time results back from a queue. """

    utilisations = get_utilisations(queue, **kwargs)
    system_times = get_system_times(queue, max_time, **kwargs)

    return utilisations, system_times
