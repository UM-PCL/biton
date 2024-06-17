from itertools import product
from math import ceil, log2
from pandas import DataFrame

from numpy import ndarray
from numpy.random import choice
from pylse import Simulation, Wire, inp_at, inspect, working_circuit
from tqdm import tqdm

from grid import gridx, late_est, quadrant_flags, sample_synd
from helpers import get_jj, sample_synd9, xcnt
from sfq_cells2 import dro, jtl_chain, m, s, split
from sortk import simple_sortk
from temp_prior import coarse_delay, monotone_del

min_dt = 10
dt, ratio = coarse_delay(min_dt)


def mtree(subset: list[Wire]) -> Wire:
    "merge signals in one"
    if len(subset) == 1:
        return subset[0]
    mglayer = list(map(m, subset[::2], subset[1::2])) + subset[-1:] * (len(subset) % 2)
    return mtree(mglayer)


def sync_mtree(subset: list[Wire], clk: Wire, name_m: None|str=None):
    "mergetree with dro at end to eat extra pulses"
    h = mtree(subset)
    if(name_m):
        inspect(h, name_m)
    return dro(h, clk)


def setscores(flags: list[Wire], clk: Wire) -> list[Wire]:
    "complex flags to disjoint set flags"
    subsets = [flags[i::4] for i in range(4)]
    clks = split(clk, n=4)
    return list(map(sync_mtree, subsets, clks))

def mtr_wait(n: int) -> int:
    x = est_mtreetime(n)
    return ceil(x/4.6)


def qubit_priority(cpx: list[Wire], clk: Wire, t: Wire, nt: Wire) -> Wire:
    clk_score, clk_start0 = s(clk)
    scores = setscores(cpx, clk_score)
    # Why did I have 17 here, 12 is the 
    jtl_wait_sort = 17
    clk_start = jtl_chain(clk_start0, jtl_wait_sort, names="clk_t")
    enc = temporal_encode(scores, anord, clk_start, t, nt)
    return enc


def temporal_encode(
    set_flags: list[Wire],
    ordering: tuple[list[int], list[int]],
    clk: Wire,
    t: Wire,
    nt: Wire,
) -> Wire:
    "temporaly encode priorities from quadrant outputs"
    start_t, start_nt = s(clk)
    ctrl_flags = simple_sortk(set_flags)
    for i, x in enumerate(set_flags):
        inspect(x, f"sets{i}")
    for i, x in enumerate(ctrl_flags):
        inspect(x, f"ctrl{i}")
    ctrls_t, ctrls_not = zip(*(map(s, ctrl_flags)))
    t_delays = [ratio * x for x in ordering[1]]
    nt_delays = [ratio * x for x in ordering[0]]
    t_enc = monotone_del(start_t, ctrls_t, t_delays)
    nt_enc = monotone_del(start_nt, ctrls_not, nt_delays)
    t_pass = dro(t, t_enc, name="Tout")
    nt_pass = dro(nt, nt_enc, name="NTout")
    enc = m(t_pass, nt_pass)
    return enc


def demo_priority(flags: list[bool], t: bool):
    "evaluate syndromes and tgate flag to temporal encoding"
    working_circuit().reset()
    ilist = [[0] * x for x in flags]
    insets = [inp_at(*x, name=f"cpx{i}") for i, x in enumerate(ilist)]
    nt = not t
    ti, nti = [10] * t, [10] * nt
    tx, ntx = inp_at(*ti, name="Tp"), inp_at(*nti, name="Tn")
    start_time = est_mtreetime(
        len(flags)
    )  # Yes I know mtree is n/4, left extra for safety
    clk = inp_at(start_time, name="start")
    enc = qubit_priority(insets, clk, tx, ntx)
    inspect(enc, "enc")
    sim = Simulation()
    events = sim.simulate()
    timer = events["enc"][0]
    return events, timer


def eval_prio(flags: list[bool], t: bool):
    n = len(flags)
    ev, time = demo_priority(flags, t)
    setflags = [len(ev[f"sets{i}"]) > 0 for i in range(4)]
    ctrlflags = [len(ev[f"ctrl{i}"]) > 0 for i in range(4)]
    shouldflags = [any(flags[i::4]) for i in range(4)]
    assert setflags == shouldflags
    assert ctrlflags == sorted(shouldflags)[::-1]
    n_quad = sum(setflags)
    code = anord[t][n_quad - 1] if n_quad > 0 else 0
    basetime = est_priodelay(n)
    extra_time = time - basetime
    encoded = extra_time / dt
    # print(f"{(n_quad, t, code,encod)=}")
    assert abs(code - encoded) < 1e-3


def quick_prio(n_runs: int = 100):
    for _ in tqdm(range(n_runs), desc="Prio_enc d=9"):
        t = bool(choice([False, True], p=[0.75, 0.25]))
        cpx = sample_synd9()
        eval_prio(cpx, t)


def wrap_embed(d: int, force_zero: bool):
    working_circuit().reset()
    t = bool(choice([False, True], p=[0.75, 0.25]))
    clkg = inp_at(20, name="power")
    bsynd = sample_synd(d)
    if force_zero:
        bsynd = bsynd * 0
    enc = demo_embed(d, t, bsynd, clkg)
    inspect(enc, "enc")
    sim = Simulation()
    events = sim.simulate()
    timer = events["enc"][0]
    return events, timer, bsynd, t


def demo_embed(d: int, t: bool, bsynd: ndarray, clkg: Wire):
    "evaluate syndromes and tgate flag to temporal encoding"
    nt = not t
    ti, nti = [0] * t, [0] * nt
    tx, ntx = inp_at(*ti, name="Tp"), inp_at(*nti, name="Tn")
    flags, prop_clk = gridx(d, bsynd, clkg)
    n = xcnt(d)
    start_time = est_mtreetime(ceil(n/4))  # Yes I know mtree is n/4, left extra for safety
    mtree_catchup = ceil(start_time / 4.6)
    if mtree_catchup >= 1:
        clk = jtl_chain(prop_clk, mtree_catchup, names="start")  
    else:
        clk = prop_clk
        inspect(clk, "start")
    enc = qubit_priority(flags, clk, tx, ntx)
    return enc


def chkembd(d, time, bsynd, t):
    n = xcnt(d)
    grid_del, _ = late_est(d)
    d0 = est_priodelay(n) + grid_del
    n_quad = quadrant_flags(d, bsynd)
    code = anord[t][n_quad - 1] if n_quad > 0 else 0
    basetime = d0
    extra_time = time - basetime
    encoded = extra_time / dt
    # print(f"{(n_quad, t, code,encod)=}")
    assert abs(code - encoded) < 1e-3


def hacky_integration_test(d: int):
    for _ in tqdm(range(1000)):
        _, dl, sin, t = wrap_embed(d=d, force_zero=False)
        chkembd(d, dl, sin, t)


def demo_tenc(
    set_flags: list[bool], ordering: tuple[list[int], list[int]], t: bool
) -> tuple[dict[str, list[float]], float]:
    "evaluate disjoint set inputs, t flag and configuration to delay/raw temp encoding"
    working_circuit().reset()
    ilist = [[10] * x for x in set_flags]
    insets = [inp_at(*x, name=f"set{i}") for i, x in enumerate(ilist)]
    nt = not t
    ti, nti = [10] * t, [10] * nt
    tx, ntx = inp_at(*ti, name="Tp"), inp_at(*nti, name="Tn")
    start_time, _ = calculate()
    clk = inp_at(start_time + 20, name="start")
    enc = temporal_encode(insets, ordering, clk, tx, ntx)
    inspect(enc, "enc")
    sim = Simulation()
    events = sim.simulate()
    timer = events["enc"][0] - events["start"][0]
    return events, timer


def predelay():
    z = [0] * 4
    ev, _ = demo_tenc([True] * 4, (z, z), False)
    keys = [f"ctrl{i}" for i in range(4)]
    whens = [max(ev[k], default=0) for k in keys]
    predelay = max(whens) - 10
    return predelay


def del_0():
    z = [0] * 4
    _, delay = demo_tenc([False] * 4, (z, z), False)
    return delay


def full_tenc(ordering: tuple[list[int], list[int]]):
    bulls = [False, True]
    bull4 = [list(x) for x in product(bulls, repeat=4)]
    configs = product(bull4, [False, True])
    outcomes = {
        (tuple(setf), t): demo_tenc(setf, ordering, t)[1] for setf, t in configs
    }
    return outcomes


def check_tenc(ordering: tuple[list[int], list[int]]):
    outcomes = full_tenc(ordering)
    baseline = calculate()[1]
    for (setf, t), enc in outcomes.items():
        v = sum(setf)
        ought = ordering[t][v - 1] if v > 0 else 0
        normalized = (enc - baseline) / dt
        print(f"{(ought, normalized)=}")
        assert abs(ought - normalized) < 1e-3
    sort_del, temp_del = calculate()
    enc_del = sort_del + temp_del + 10
    jjj = get_jj()
    return enc_del, jjj


def time_mtree(n: int):
    working_circuit().reset()
    inps = [inp_at(0, name=f"inp{i}") for i in range(n)]
    out = mtree(inps)
    inspect(out, "out")
    sim = Simulation()
    events = sim.simulate()
    return max(events["out"])


anord = ([1, 1, 4, 5], [2, 3, 6, 7])


def calculate():
    comp_del = 8.8 + 5.1
    sort_del = 3 * comp_del
    route_del = 9.5 + 6.3
    baseline = 5.1 + 3.6 + 6.3 + 4 * route_del
    # actual encoding "delay" is counted in arbiter delay
    # d = 5
    # xcount = ((d-1)**2 + d - 1) // 2
    # mtree_del = 6.3 * ceil(log2(ceil(xcount/4))) + 3.6
    return sort_del, baseline


def est_mtreetime(n: int):
    # est_jj = 5 * (n-1)
    return 6.3 * ceil(log2(n))


def est_priodelay(n: int):
    mt = est_mtreetime(ceil(n/4))
    norm_mtree = ceil(mt / 4.6) * 4.6
    temp_base = calculate()[1]
    jtl_wait_sort = 17
    start_temp = norm_mtree + 5.1 + jtl_wait_sort * 4.6
    est = start_temp + temp_base
    return est

def time_data(d: int):
    n = xcnt(d)
    x = {}
    x["d"] = d
    x["t_grid"] = late_est(d)[0]
    x["t_mtree"] = est_mtreetime(ceil(n/4))
    x["t_sort"], x["t_encode"] = calculate()
    t_prio = est_priodelay(n)
    hm =x["t_mtree"] + x["t_encode"] + x["t_sort"]
    x["t_intercon"] = t_prio - hm
    x["t_priority"] = t_prio + x["t_grid"]
    return x

def time_df():
    t_records = [time_data(d) for d in range(3,22,2)]
    df = DataFrame.from_records(t_records)
    return df
    

