from arbiter import info, sim_arbiter
from pandas import DataFrame


def run_tests(n_runs):
    confs = [(2,64),(4,128),(8,256)]
    for k, n in confs:
        sim_arbiter(k,n,n_runs,False)


def get_data():
    data = [
        info(2, 64),
        info(4, 128),
        info(8, 256),
    ]
    df = DataFrame.from_records(data)
    return df
