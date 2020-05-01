""" Functions for finding the optimal number of clusters in the dataset. """

import sys

import ciw
import dask
import numpy as np
import pandas as pd
from ciw.dists import Exponential
from scipy import special, stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from yellowbrick.utils import KneeLocator

from kmodes.kprototypes import KPrototypes
from util import DATA_DIR

OUT_DIR = DATA_DIR / "clusters/"
OUT_DIR.mkdir(exist_ok=True)

PATH = str(sys.argv[1])
NUM_CORES = int(sys.argv[2])

CLUSTER_LIMS = (2, 9)
copd = pd.read_csv(PATH, parse_dates=["admission_date", "discharge_date"])

clinicals = [
    "n_spells",
    "n_wards",
    "n_consultants",
    "true_los",
    "n_pr_attendances",
    "n_sn_attendances",
    "n_copd_admissions_last_year",
    "charlson_gross",
    "n_icds",
    "intervention",
    "day_of_week",
    "gender",
]

codes = [
    "infectious",
    "neoplasms",
    "blood",
    "endocrine",
    "mental",
    "nervous",
    "eye",
    "ear",
    "circulatory",
    "respiratory",
    "digestive",
    "skin",
    "muscoloskeletal",
    "genitourinary",
    "perinatal",
    "congenital",
    "abnormal_findings",
    "injury",
    "external_causes",
    "contact_factors",
    "special_use",
]

conditions = [
    "ami",
    "cva",
    "chf",
    "ctd",
    "dementia",
    "diabetes",
    "liver_disease",
    "peptic_ulcer",
    "pvd",
    "pulmonary_disease",
    "cancer",
    "diabetic_complications",
    "paraplegia",
    "renal_disease",
    "metastatic_cancer",
    "sever_liver_disease",
    "hiv",
    "cdiff",
    "mrsa",
    "obese",
    "sepsis",
]

cols = clinicals + codes + conditions
DATA = copd[cols].copy()

CATEGORICAL = [
    i
    for i, (col, dtype) in enumerate(dict(DATA.dtypes).items())
    if dtype == "object"
]


def clean_data(data, missing_prop=0.25, max_stay=365):
    """ Get rid of the columns where enough data is missing, and remove records
    that last too long or have any missing data. """

    for col in data.columns:
        if data[col].isnull().sum() > missing_prop * len(data):
            data = data.drop(col, axis=1)

    data = data[data["true_los"] <= max_stay]
    data = data.dropna()

    return data


def get_knee_results(data, cluster_lims, cores, categorical):

    knee_results = []
    cluster_range = range(*cluster_lims)
    for n_clusters in tqdm(cluster_range):

        kp = KPrototypes(n_clusters, init="cao", random_state=0, n_jobs=cores)
        kp.fit(data, categorical=categorical)

        knee_results.append(kp.cost_)

    kl = KneeLocator(
        cluster_range,
        knee_results,
        curve_nature="convex",
        curve_direction="decreasing",
    )

    n_clusters = kl.knee

    with open(OUT_DIR / "n_clusters.txt", "w") as f:
        f.write(str(n_clusters))

    knee_results = pd.Series(index=cluster_range, data=knee_results)
    knee_results.to_csv(OUT_DIR / "knee_results.csv", header=False)

    return n_clusters


def assign_labels(data, n_clusters, cores, categorical):

    kp = KPrototypes(n_clusters, init="matching", random_state=0, n_jobs=cores)
    kp.fit(data, categorical=categorical)

    labels = kp.labels_
    copd["cluster"] = labels
    copd.to_csv(DATA_DIR / "copd_clustered.csv", index=False)


def main(data, cluster_lims, cores, categorical):

    data = clean_data(data)
    n_clusters = get_knee_results(data, cluster_lims, cores, categorical)
    assign_labels(data, n_clusters, cores, categorical)


if __name__ == "__main__":
    main(DATA, CLUSTER_LIMS, NUM_CORES, CATEGORICAL)
