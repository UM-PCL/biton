from sfq_cells2 import jtl_chain, m, dro_c, JTL
from pylse import working_circuit, Wire, inp_at, Simulation, inspect
from math import ceil
from operator import sub


def temp_prior(addr: list[Wire], start: Wire, min_dt: float) -> tuple[float, Wire]:
    djtl: float = JTL.firing_delay
    ratio = ceil(min_dt / djtl)
    dt = djtl * ratio
    print(f"New {dt=}")
    n = len(addr)
    cnts = [ratio * (2**i) for i in range(n)]
    n_jtl = sum(cnts)
    nxt = start
    for adbit, cnt in zip(addr, cnts):
        slow, fast = dro_c(adbit, nxt)
        later = jtl_chain(slow, cnt)
        nxt = m(fast, later)
    est_jj = n * (13 + 5) + 2 * n_jtl
    jjs = sum(
        x.element.jjs
        for x in working_circuit()
        if x.element.name not in ["_Source", "InGen"]
    )
    assert jjs == est_jj
    print(f"{jjs=}")
    return dt, nxt


def sim_temp(n: int, min_dt: float, sep: float):
    working_circuit().reset()
    fire_addrs = [[sep*x -2 for x in range(2**n) if (x // 2**i % 2)] for i in range(n)]
    strt = [sep*x for x in range(2**n)]
    start = inp_at(*strt,name= "start")
    addrs = [inp_at(*x, name = f"a{i}") for i, x in enumerate(fire_addrs)]
    dt, out = temp_prior(addrs, start, min_dt)
    inspect(out, "out")
    sim = Simulation()
    events = sim.simulate()
    delays = list(map(sub, events["out"], events["start"]))
    d0 = min(delays)
    ndelays = [x-d0 for x in delays]
    norm = [x/dt for x in ndelays]
    assert [x-i < 1e-3 for i,x in enumerate(norm)]
    sim.plot()
    return d0, ndelays
