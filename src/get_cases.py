""" Get the worst and best cases from the Wasserstein data. """

from pathlib import Path
import sys

import pandas as pd


SEEDS = int(sys.argv[1])

DATA_DIR = Path("../data/wasserstein/")


def get_case(data, case):

    maximal_distance = data.groupby(["p_0", "p_1", "p_2", "p_3", "num_servers"])["distance"].max()
    if case == "best":
        *ps, c = maximal_distance.idxmin()
        distance = maximal_distance.min()
    elif case == "worst":
        *ps, c = maximal_distance.idxmax()
        distance = maximal_distance.max()
    else:
        raise NotImplementedError("Case must be one of `'best'` or `'worst'`.")

    dfs = (
        pd.read_csv(DATA_DIR / f"{'_'.join(map(str, ps))}_{c}_{seed}.csv")
        for seed in range(SEEDS)
    )

    df = pd.concat(dfs)
    df.to_csv(DATA_DIR / f"{case}.csv", index=False)

    with open(DATA_DIR / f"{case}_params.txt", "w") as f:
        string = " ".join(map(str, [*ps, c, distance]))
        f.write(string)


def main():

    data = pd.read_csv(DATA_DIR / "main.csv")
    print("Data read in.")

    for case in ["best", "worst"]:
        get_case(data, case)
        print(case, "done.")

if __name__ == "__main__":
    main()
