from sfq_cells2 import jtl_chain, m, dro_c, JTL
from pylse import working_circuit, Wire, inp_at, Simulation, inspect
from math import ceil
from operator import sub
from helpers import get_jj, get_latency


def coarse_delay(min_dt: float) -> tuple[float, int]:
    "Smallest delay above threshold via jtls and #jtl for it"
    djtl: float = JTL.firing_delay  # type: ignore
    ratio = ceil(min_dt / djtl)
    dt = djtl * ratio
    return dt, ratio


def temp_prior(addr: list[Wire], start: Wire, min_dt: float) -> tuple[float, Wire]:
    "Debug wrapper for binary_del"
    dt, ratio = coarse_delay(min_dt)
    print(f"{dt=}")
    nxt = binary_del(start, addr, ratio)
    n = len(addr)
    n_jtl = (2**n - 1) * ratio
    est_jj = n * (13 + 5) + 2 * n_jtl
    jjs = get_jj()
    assert jjs == est_jj
    print(f"{jjs=}")
    return dt, nxt


def cond_delay(inp: Wire, ctrl: Wire, cnt: int) -> Wire:
    "Add cnt of jtl delay to inp signal iff ctrl previously fired"
    slow, fast = dro_c(ctrl, inp)
    later = jtl_chain(slow, cnt)
    nxt = m(fast, later)
    return nxt


def binary_del(inp: Wire, addr: list[Wire], ratio: int) -> Wire:
    "delay inp wire by binary address, jtl/delay must be found by caller"
    n = len(addr)
    scores = [ratio * (2**i) for i in range(n)]
    nxt = inp
    for en, delay in zip(addr, scores):
        nxt = cond_delay(nxt, en, delay)
    return nxt


def monotone_del(inp: Wire, ctrls: list[Wire], scores: list[int]) -> Wire:
    "Add increasing delay for more signals active in order, to be used with sorter"
    delta = list(map(sub, scores[1:], scores[:-1]))
    # assert monotonic priorities
    assert min(delta) >= 0
    dt_scores = [scores[0]] + delta
    nxt = inp
    for en, delay in zip(ctrls, dt_scores):
        nxt = cond_delay(nxt, en, delay)
    return nxt

def mono_testcases(n_ctrl: int):
    "All in-order increasing bit sequences"
    return [[True]*i + [False]*(n_ctrl-i) for i in range(n_ctrl+1)]

def monotone_testdata(scores: list[int]) -> list[float]:
    "get resulting delays for set scores to debug"
    return [demo_monotone(tstcase, scores) for tstcase in mono_testcases(len(scores))]
        

def demo_monotone(enables: list[bool], scores: list[int]) -> float:
    "get measured delay for enable signals and priority levels"
    working_circuit().reset()
    ctrlists = [[10]*x for x in enables]
    ctrls = [inp_at(*x, name=f"ctrl{i}") for i, x in enumerate(ctrlists)]
    start = inp_at(20, name="start")
    signal = monotone_del(start, ctrls, scores)
    inspect(signal, "signal")
    sim = Simulation()
    events = sim.simulate()
    timer = events["signal"][0] - events["start"][0]
    return timer

def confirm_monotone(scores: list[int], min_dt: float):
    dt, ratio = coarse_delay(min_dt)
    cscores = [x*ratio for x in scores]
    data = monotone_testdata(cscores)
    norm = [x / dt for x in data]
    assert [x - i < 1e-3 for i, x in enumerate(norm)]
    print(f"{dt=}")
    n = len(scores)
    n_jtl = max(cscores)
    est_jj = n * (13 + 5) + 2 * n_jtl
    jjs = get_jj()
    assert jjs == est_jj
    print(f"{jjs=}")


def sim_temp(n: int, min_dt: float, sep: float):
    working_circuit().reset()
    fire_addrs = [
        [sep * x - 2 for x in range(2**n) if (x // 2**i % 2)] for i in range(n)
    ]
    strt = [sep * x for x in range(2**n)]
    start = inp_at(*strt, name="start")
    addrs = [inp_at(*x, name=f"a{i}") for i, x in enumerate(fire_addrs)]
    dt, out = temp_prior(addrs, start, min_dt)
    inspect(out, "out")
    sim = Simulation()
    events = sim.simulate()
    delays = list(map(sub, events["out"], events["start"]))
    d0 = min(delays)
    ndelays = [x - d0 for x in delays]
    norm = [x / dt for x in ndelays]
    assert [x - i < 1e-3 for i, x in enumerate(norm)]
    sim.plot()
    return d0, ndelays
