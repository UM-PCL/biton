from arbiter import info, sim_arbiter
from pandas import DataFrame


def run_tests(n_runs: int, clear:bool = False):
    confs = [(2,64),(4,128),(8,256)]
    for k, n in confs:
        sim_arbiter(k,n,n_runs,plot=False,clear=clear)


def get_data():
    data = [
        info(2, 64),
        info(4, 128),
        info(8, 256),
    ]
    df = DataFrame.from_records(data)
    return df


def extra_jj(k, n):
    jj_s = 3
    jj_m = 5
    jjtl = 2
    jj_temp = 96
    # spliter tree for 3 addr bits and start signal
    cost_spl_temp = 4*(n-1)*jj_s
    # cost of temporal encoding
    cost_temp = jj_temp * n + cost_spl_temp
    cost_sorted_inps = (n-1)*(jj_s+3*jjtl) + n*jj_m
    cost_clear = (n-1)*jj_s + jj_m
    cost_reset = cost_sorted_inps + cost_clear
    return cost_temp, cost_reset
