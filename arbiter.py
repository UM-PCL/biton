from sortk import sortk, mergemax, events_io
from pylse import Wire, working_circuit
from sfq_cells2 import JTL
import pylse
from random import choice
from math import inf, log2


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
    max_in = 110
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


def get_del(k: int, n:int) -> float:
    lk = log2(k)
    depthn = log2(n) - lk
    depthk = k*(k+1)//2
    dcell = 8.8
    dla = 8.1
    dlayer = (depthk * dcell) + dla
    forward_delay = depthn * dlayer
    return forward_delay


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
    inps: list[float] = [110 - int(log2(choice(range(1, 2**10 + 1)))) * 10 for _ in range(n)]
    demo_arbiter(k, inps, plot)


def check_arbitrage(k: int, x, o):
    ordx = sorted(x)
    winners = ordx[-k:]
    obool = [x < inf for x in o]
    chosen = sorted([qubit for qubit, sel in zip(x, obool) if sel])
    print(f"{(winners, chosen)=}")
    assert winners == chosen
