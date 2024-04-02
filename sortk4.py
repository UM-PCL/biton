from numpy import argsort
from operator import sub
from random import shuffle, choice
from itertools import groupby
from math import inf
from typing import Dict
from inhibitor import comp
from pylse import Wire
import pylse


def sortk4(inplist: list[Wire], retlist: list[Wire]):
    rt = [Wire() for _ in range(4)]
    x = [Wire() for _ in range(4)]
    y = [Wire() for _ in range(4)]
    rr = [Wire() for _ in range(4)]
    ro = [Wire() for _ in range(4)]
    o = [Wire() for _ in range(4)]
    x[0], x[1] = comp(inplist[0], inplist[1], rt[0], rt[1], ro[0], ro[1])
    x[3], x[2] = comp(inplist[3], inplist[2], rt[3], rt[2], ro[3], ro[2])
    y[0], y[2] = comp(x[0], x[2], rr[0], rr[2], rt[0], rt[2])
    y[1], y[3] = comp(x[1], x[3], rr[1], rr[3], rt[1], rt[3])
    o[0], o[1] = comp(y[0], y[1], retlist[0], retlist[1], rr[0], rr[1])
    o[2], o[3] = comp(y[2], y[3], retlist[2], retlist[3], rr[2], rr[3])
    # for i, z in enumerate(rr):
    #     pylse.inspect(z, f"rr{i}")
    # for i, z in enumerate(rt):
    #     pylse.inspect(z, f"rt{i}")
    # for i, z in enumerate(x):
    #     pylse.inspect(z, f"xx{i}")
    # for i, z in enumerate(y):
    #     pylse.inspect(z, f"y{i}")
    return o, ro


def demo_sortk4(ils: list[float], rls: list[bool], plot: bool = True):
    pylse.working_circuit().reset()
    inplist = [pylse.inp_at(x, name=f"x{i}") for i, x in enumerate(ils)]
    retlist = [pylse.inp_at(*([200] * x), name=f"r{i}") for i, x in enumerate(rls)]
    o, ro = sortk4(inplist, retlist)
    for i, x in enumerate(o):
        pylse.inspect(x, f"o{i}")
    for i, x in enumerate(ro):
        pylse.inspect(x, f"ro{i}")
    sim = pylse.Simulation()
    events = sim.simulate()
    towatch = ["x", "r", "o", "ro"]
    watchers = [[f"{x}{i}" for i in range(4)] for x in towatch]
    watch_wires = sum(watchers, [])
    if plot:
        sim.plot(wires_to_display=watch_wires)
    ex, er, eo, ero = events_io(events, towatch)
    check_out(ex, er, eo, ero)
    return events


def quick_sort4(plot: bool = True):
    rls = [choice([True, False]) for _ in range(4)]
    ils: list[float] = [10, 20, 30, 40]
    shuffle(ils)
    demo_sortk4(ils, rls, plot)


def events_io(events: Dict[str, list[float]], matchs: list[str]) -> list[list[float]]:
    def nonn(x):
        return "".join([i for i in x if not i.isdigit()])

    def evnorm(x: list[float]) -> list[float]:
        assert len(x) <= 1
        return [inf] if x == [] else x

    evks = sorted(events.keys())
    groupks = {
        k: sum([evnorm(events[x]) for x in v], []) for k, v in groupby(evks, nonn)
    }
    evio = [groupks[x] for x in matchs]
    return evio


def check_out(x, r, o, ro):
    order = list(argsort(x))
    rbool = [x < inf for x in r]
    robool = [x < inf for x in ro]
    assert max(map(sub, o, o[1:])) <= 0
    ordered_robool = [robool[i] for i in order]
    assert ordered_robool == rbool
