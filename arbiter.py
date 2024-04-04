from sortk import sortk, mergemax, events_io
from pylse import Wire, working_circuit
from sfq_cells2 import JTL, C, C_INV
import pylse
from random import choice
from math import inf, log2
from numpy.random import choice as npchoice


def reductor(
    inps: list[Wire],
    retins: list[Wire],
    retouts: list[Wire],
) -> list[Wire]:
    n = len(inps)
    k = len(retins)
    assert n // k >= 2
    hn = n // 2
    assert len(retouts) == n
    if n == 2 * k:
        inters1, inters2 = inps[:k], inps[k:]
        rets1, rets2 = retouts[:k], retouts[k:]
    else:
        inps1, inps2 = inps[:hn], inps[hn:]
        rets1 = [Wire() for _ in range(k)]
        rets2 = [Wire() for _ in range(k)]
        inters1 = reductor(inps1, rets1, retouts[:hn])
        inters2 = reductor(inps2, rets2, retouts[hn:])
    retm1 = [Wire() for _ in range(k)]
    retm2 = [Wire() for _ in range(k)]
    sorts1 = sortk(inters1, retm1, rets1)
    sorts2 = sortk(inters2, retm2, rets2)
    maxers = mergemax(sorts1, sorts2, retins, retm1, retm2)
    assert len(maxers) == k
    return maxers


def arbiter(k: int, inps: list[Wire]) -> list[Wire]:
    n = len(inps)
    # rets = [Wire() for _ in range(k)]
    fdel = get_del(k, n)
    max_in = 6 * minimum_sampling_del(k, n)
    rets = [pylse.inp_at(fdel + max_in, name=f"r{i}") for i in range(k)]
    retouts = [Wire() for _ in range(n)]
    maxers = reductor(inps, rets, retouts)
    for i, x in enumerate(maxers):
        pylse.inspect(x, f"max{i}")
    for i, x in enumerate(rets):
        pylse.inspect(x, f"r{i}")
    # for x, y in zip(maxers, rets):
    #     working_circuit().add_node(JTL(), [x], [y])
    return retouts


def get_del(k: int, n: int) -> float:
    lk = log2(k)
    depthn = log2(n) - lk
    depthk = lk * (lk + 1) // 2
    dla = 8.1
    dfa = 8.8
    dspl = 5.1
    dcell = max(dla, dfa)
    dcomp = dcell + (2 * dspl)
    dcmax = dla + dspl
    dlayer = (depthk * dcomp) + dcmax
    forward_delay = depthn * dlayer
    return forward_delay


def minimum_sampling_del(k: int, n: int) -> float:
    lk = log2(k)
    depthn = log2(n) - lk
    depthk = lk * (lk + 1) // 2
    dla = 8.1
    dfa = 8.8
    deltacell = abs(dla - dfa)
    delta = depthn * depthk * deltacell
    clk_del = max(delta, 10)
    return clk_del


clique_prob = [
    0.8057939982898317,
    0.11649655545541934,
    0.05665757307749672,
    0.009155941808821502,
    0.01115608050420479,
    0.000583709562690942,
    0.00010618326131122197,
    4.814339373482264e-05,
    1.5887836016949806e-06,
    2.093822647431851e-07,
    1.645295303988626e-08,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]


def clique_sample(n: int, p=clique_prob) -> list[int]:
    """Returns number of complex cliques from distribution
    for n logical qubits"""
    samp = npchoice(range(len(p)), size=n, p=p)
    return [int(x) for x in samp]


def demo_arbiter(k: int, inps: list[float], plot: bool = True):
    working_circuit().reset()
    n = len(inps)
    inplist = [pylse.inp_at(x, name=f"x{i}") for i, x in enumerate(inps)]
    topk = arbiter(k, inplist)
    for i, x in enumerate(topk):
        pylse.inspect(x, f"top{i}")
    sim = pylse.Simulation()
    events = sim.simulate()
    towatch = ["x", "top"]
    watchers = [[f"{x}{i}" for i in range(n)] for x in towatch]
    towatch2 = ["max", "r"]
    watchers2 = [[f"{x}{i}" for i in range(k)] for x in towatch2]
    watch_wires = sum(watchers + watchers2, [])
    if plot:
        sim.plot(wires_to_display=watch_wires)
    evio = events_io(events, towatch)
    check_arbitrage(k, *evio)
    return evio


def quick_arbiter(k: int, n: int, plot: bool = True):
    priority_limit = 6
    samps = clique_sample(n)
    clk_del = minimum_sampling_del(k, n)
    inps: list[float] = [clk_del * (min(x + 1, priority_limit)) for x in samps]
    demo_arbiter(k, inps, plot)


def check_arbitrage(k: int, x, o):
    ordx = sorted(x)
    winners = ordx[-k:]
    obool = [x < inf for x in o]
    chosen = sorted([qubit for qubit, sel in zip(x, obool) if sel])
    print(f"{(winners, chosen)=}")
    assert winners == chosen
