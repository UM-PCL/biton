from collections.abc import Callable
from sfq_cells2 import dro, jtl_chain, m, dro_c, JTL, split
from pylse import working_circuit, Wire, inp_at, Simulation, inspect
from math import ceil
from operator import sub
from helpers import get_jj, get_latency


def inp_list(l, *args, **kwargs):
    return inp_at(*l, *args, **kwargs)


def temp_posterior(q1: Wire, q2: Wire, t: Wire, start: Wire, clk: Wire) -> Wire:
    clkz = split(clk, n=5, firing_delay=0)
    dq, d0 = dro_c(q1, start, name_q="dq", name_q_not="d0")
    cq = dro(dq, clkz[0])
    dt_p, dt_n = dro_c(t, cq, name_q="dtp", name_q_not="dtn")
    ct_1 = dro(dt_p, clkz[1])
    ct_2 = dro(ct_1, clkz[4])
    dt = m(ct_2, dt_n, name="dt_m")
    dq2_p, dq2_n = dro_c(q2, dt, name_q="d2p", name_q_not="d2n")
    cq2 = dro(dq2_p, clkz[3])
    dq2 = m(cq2, d0)
    d_end = m(dq2, dq2_n)
    c_end = dro(d_end, clkz[2], name="signal")
    return c_end


def demo_delenc(q1: int, q2: int, t: int, clk: float = 20):
    "get measured delay for enable signals and priority levels"
    assert max(q1, q2, t) <= 1
    assert min(q1, q2, t) >= 0
    working_circuit().reset()
    wq1 = inp_list(q1 * [0], name="q1")
    wq2 = inp_list(q2 * [0], name="q2")
    wt = inp_list(t * [0], name="t")
    start = inp_at(10, name="start")
    wclk = inp_list([clk * i for i in range(1, 6)], name="clk")
    d_out = temp_posterior(wq1, wq2, wt, start, wclk)
    sim = Simulation()
    events = sim.simulate()
    sig = events["signal"]
    assert len(sig) == 1
    assert len(events["clk"]) == 5
    timer = sig[0]
    return timer


def test_posterior(clk: float = 40):
    hm = [
        demo_delenc(q1, q2, t, clk=clk)
        for t in [0, 1]
        for q1, q2 in [(0, 0), (1, 0), (1, 1)]
    ]
    d0 = hm[0]
    encs = [(d - d0) / clk for d in hm]
    expected = [0, 1, 2, 0, 3, 4]
    derr = [abs(dut - gold) for dut, gold in zip(encs, expected)]
    assert max(derr) < 1e-3
    return hm, encs
