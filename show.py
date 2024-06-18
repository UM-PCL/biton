from arbiter import info, sim_arbiter
from pandas import DataFrame
from grid import get_gridspecs
from tqdm import tqdm


def run_tests(n_runs: int, clear: bool = False):
    confs = [(2, 64), (4, 128), (8, 256)]
    for k, n in confs:
        sim_arbiter(k, n, n_runs, plot=False, clear=clear)


def get_data():
    data = [
        info(2, 64),
        info(4, 128),
        info(8, 256),
        info(8, 512)
    ]
    df = DataFrame.from_records(data)
    df["backwards_delay"] +=20
    df["total_delay"] +=20
    return df


def grid_data():
    data = [get_gridspecs(d, 100) for d in tqdm(range(3, 22, 2), desc="grid_data")]
    df = DataFrame.from_records(data)
    return df
