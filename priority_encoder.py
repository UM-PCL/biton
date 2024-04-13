from helpers import get_jj, get_latency
from pylse import working_circuit, Wire, Simulation
from sfq_cells2 import dro, m, s, split
from sortk import simple_sortk
from temp_prior import monotone_del, coarse_delay


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
    ctrls_t, ctrls_not = zip(*(map(s, ctrl_flags)))
    min_dt = 10
    _, ratio = coarse_delay(min_dt)
    t_delays = [ratio * x for x in ordering[1]]
    nt_delays = [ratio * x for x in ordering[0]]
    t_enc = monotone_del(start_t, ctrls_t, t_delays)
    nt_enc = monotone_del(start_nt, ctrls_not, nt_delays)
    t_pass = dro(t, t_enc)
    nt_pass = dro(nt, nt_enc)
    enc = m(t_pass, nt_pass)
    return enc
