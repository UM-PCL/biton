from helpers import get_jj, get_latency
from pylse import inp_at, inspect, working_circuit, Wire, Simulation
from sfq_cells2 import dro, m, s, split
from sortk import simple_sortk
from temp_prior import monotone_del, coarse_delay
from itertools import product

min_dt = 10
dt, ratio = coarse_delay(min_dt)

def mtree(subset: list[Wire]) -> Wire:
    "merge signals in one"
    if len(subset) == 1:
        return subset[0]
    mglayer = list(map(m, subset[::2], subset[1::2]))
    return mtree(mglayer)


def sync_mtree(subset: list[Wire], clk: Wire):
    "mergetree with dro at end to eat extra pulses"
    return dro(mtree(subset), clk)


def setscores(flags: list[Wire], clk: Wire) -> list[Wire]:
    "complex flags to disjoint set flags"
    subsets = [flags[i::4] for i in range(4)]
    clks = split(clk, n=4)
    return list(map(sync_mtree, subsets, clks))


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
    for i, x in enumerate(ctrl_flags):
        inspect(x,f"ctrl{i}")
    ctrls_t, ctrls_not = zip(*(map(s, ctrl_flags)))
    t_delays = [ratio * x for x in ordering[1]]
    nt_delays = [ratio * x for x in ordering[0]]
    t_enc = monotone_del(start_t, ctrls_t, t_delays)
    nt_enc = monotone_del(start_nt, ctrls_not, nt_delays)
    t_pass = dro(t, t_enc)
    nt_pass = dro(nt, nt_enc)
    enc = m(t_pass, nt_pass)
    return enc


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
        ought = ordering[t][v-1] if v>0 else 0
        normalized = (enc - baseline) / dt
        print(f"{(ought, normalized)=}")
        assert abs(ought - normalized) < 1e-3
    sort_del, temp_del = calculate()
    enc_del = sort_del + temp_del + 10
    jjj = get_jj()
    return enc_del, jjj


anord = ([1,1,4,5],[2,3,6,7])


def calculate():
    comp_del = 8.8 + 5.1
    sort_del = 3 * comp_del
    route_del = 9.5 + 6.3
    baseline = 5.1 + 3.6 + 6.3 + 4*route_del
    # actual encoding "delay" is counted in arbiter delay
    return sort_del, baseline
