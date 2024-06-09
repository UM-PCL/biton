from collections.abc import Callable
from itertools import product

from numpy.random import choice
from priority_encoder import sync_mtree
from sfq_cells2 import c, c_inv, dro, jtl_chain, m, dro_c, JTL, s, split
from pylse import working_circuit, Wire, inp_at, Simulation, inspect
from math import ceil, log2
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


def q12(q: list[Wire]):
    q0, w0 = s(q[0])
    q1, w1 = s(q[1])
    q2, w2 = s(q[2])
    q3, w3 = s(q[3])
    x1 = c(q0, q1)
    x2 = c_inv(w0, w1)
    x3 = c(q2, q3)
    x4 = c_inv(w2, w3)
    x5 = c_inv(x1, x3)
    x2a, x2b = s(x2)
    x4a, x4b = s(x4)
    x6 = c(x2a, x4a)
    x7 = c_inv(x2b, x4b, name="q1")
    x8 = c_inv(x5, x6, name="q2")
    return x7, x8


def testq12():
    x = [[0, 1]] * 4
    g = product(*x)
    outs = {}
    for q in g:
        working_circuit().reset()
        qs = [inp_list(x * [0], name=f"qin{i}") for i, x in enumerate(q)]
        q1, q2 = q12(qs)
        sim = Simulation()
        events = sim.simulate()
        e1 = events["q1"]
        e2 = events["q2"]
        assert len(e2) <= len(e1) <= 1
        t1 = e1[0] if len(e1) > 0 else -1
        t2 = e2[0] if len(e2) > 0 else -1
        outs[q] = (t1, t2)
    check = [
        ((sum(k) > 0, sum(k) > 1), (v1 > 0, v2 > 0)) for k, (v1, v2) in outs.items()
    ]
    assert all([x == y for x, y in check])
    d1, d2 = map(max, zip(*(outs.values())))
    return d1, d2


def guess_dels(d=9, clk=40):
    n_synd = (d + 1) * (d - 1) // 2
    n_quad = n_synd // 4
    d_mtree = 6.3 * ceil(log2(n_quad))
    t_eval = d_mtree + 1
    d_s2 = 3.6 + 35.9 + 1.2
    t_start = t_eval + d_s2
    t_clk0 = t_start + 9.5 + 2 * 6.3
    # final time to report = t_clk0 + 3.6 + 4T(clk)
    t_final = 4 * clk + t_clk0 + 3.6
    return t_eval, t_start, t_clk0, t_final


def test_score(d=9, clk=40):
    working_circuit().reset()
    t_eval, t_start, t_clk0, t_final = guess_dels(d, clk)
    n_synd = (d + 1) * (d - 1) // 2
    n_quad = n_synd // 4
    pq = 0.5
    pc = 0.05
    shall_qs = choice([True, False], size=4, p=[pq, 1 - pq])

    def new_qc():
        h = [choice([1, 0], p=[pc, 1 - pc]) for _ in range(n_quad)]
        return h if sum(h) > 0 else new_qc()

    cpxs = [
        [inp_list(x * [0]) for x in (new_qc() if shall_qs[i] else [0] * n_quad)]
        for i in range(4)
    ]
    eval = inp_at(t_eval, name="eval")
    start = inp_at(t_start, name="start")
    clkg = inp_list([clk * i + t_clk0 for i in range(5)], name="clk")
    evalz = split(eval, n=4, firing_delay=0)
    mnames = [f"mtree_{i}" for i in range(4)]
    qs = list(map(sync_mtree, cpxs, evalz, mnames))
    for i, q in enumerate(qs):
        inspect(q, f"quadrant{i}")
    q1, q2 = q12(qs)
    t = choice([True, False])
    wt = inp_list(t * [10], name="t")
    temp_posterior(q1, q2, wt, start, clkg)
    sim = Simulation()
    events = sim.simulate()
    k = events["signal"][0] - t_clk0
    ex_q1 = sum(shall_qs) > 0
    ex_q2 = sum(shall_qs) > 1
    exp_score = ex_q1 * (1 + t * 2 + ex_q2)
    assert abs(k - (exp_score * clk + 3.6)) < 1e-3
    # from IPython import embed
    #
    # embed()
    return k
