from numpy import argsort, log2
from operator import sub
from random import choice
from itertools import groupby
from math import inf
from typing import Dict
from inhibitor import cmax, comp
from pylse import Wire
import pylse


def alt(n: int, x: int) -> list[bool]:
    return [(i // x) % 2 == 1 for i in range(n)]


def sarrows(n: int, stride: int, xstride: int) -> list[tuple[int, int]]:
    s = stride * 2
    steps = n // s
    arr = [(i, i + stride) for i in range(stride)]
    arrn = [(j, i) for i, j in arr]
    alts = alt(steps, xstride)
    sarr = [
        (x * s + i, x * s + j)
        for x in range(steps)
        for i, j in (arrn if alts[x] else arr)
    ]
    return sarr


def lnums(x: int) -> list[tuple[int, int]]:
    xs = [2**i for i in range(x)]
    return list(zip(xs[::-1], xs))


def carrows(n: int) -> list[tuple[int, int, int]]:
    lgn = int(log2(n))
    layers = sum((lnums(i) for i in range(1, lgn + 1)), [])
    return [(n, x, y) for x, y in layers]


def layers(n: int) -> list[list[tuple[int, int]]]:
    return [sarrows(*x) for x in carrows(n)]


def mklayer(
    lcons: list[tuple[int, int]],
    inplist: list[Wire],
    retin: list[Wire],
    retback: list[Wire],
) -> list[Wire]:
    o = [Wire() for _ in inplist]
    for i, j in lcons:
        o[i], o[j] = comp(
            inplist[i], inplist[j], retin[i], retin[j], retback[i], retback[j]
        )
    return o


def sortk(
    inplist: list[Wire], retlist: list[Wire], rback: list[Wire]
) -> list[Wire]:
    n = len(inplist)
    assert len(retlist) == len(rback) == n
    las = layers(n)
    ln = len(las)
    r = [rback] + [[Wire() for _ in range(n)] for _ in range(ln - 1)] + [retlist]
    f = [inplist]
    for i, layer in enumerate(las):
        f.append(mklayer(layer, f[-1], r[i + 1], r[i]))
    bsorted = f[-1]
    return bsorted


def mergemax(
    inps1: list[Wire],
    inps2: list[Wire],
    rets: list[Wire],
    ret1: list[Wire],
    ret2: list[Wire],
) -> list[Wire]:
    res = map(cmax, inps1, reversed(inps2), rets, ret1, reversed(ret2))
    return list(res)


def demo_mmax(il1: list[float], il2: list[float], rls: list[bool], plot: bool = True):
    pylse.working_circuit().reset()
    n = len(il1)
    assert len(set(map(len, [il1, il2, rls]))) == 1
    inplist1 = [pylse.inp_at(x, name=f"x{i}") for i, x in enumerate(il1)]
    inplist2 = [pylse.inp_at(x, name=f"y{i}") for i, x in enumerate(il2)]
    retlist = [pylse.inp_at(*([150] * x), name=f"r{i}") for i, x in enumerate(rls)]
    ro1 = [Wire() for _ in range(n)]
    ro2 = [Wire() for _ in range(n)]
    o = mergemax(inplist1, inplist2, retlist, ro1, ro2)
    for i, x in enumerate(o):
        pylse.inspect(x, f"o{i}")
    for i, x in enumerate(ro1):
        pylse.inspect(x, f"rox{i}")
    for i, x in enumerate(ro2):
        pylse.inspect(x, f"roy{i}")
    sim = pylse.Simulation()
    events = sim.simulate()
    towatch = ["x", "y", "r", "o", "rox", "roy"]
    watchers = [[f"{x}{i}" for i in range(n)] for x in towatch]
    watch_wires = sum(watchers, [])
    if plot:
        sim.plot(wires_to_display=watch_wires)
    evio = events_io(events, towatch)
    check_merge(*evio)
    return events


def demo_sortk(ils: list[float], rls: list[bool], plot: bool = True):
    pylse.working_circuit().reset()
    n = len(ils)
    retwait = laydepth(n) * 20 + max(ils)
    inplist = [pylse.inp_at(x, name=f"x{i}") for i, x in enumerate(ils)]
    retlist = [pylse.inp_at(*([retwait] * x), name=f"r{i}") for i, x in enumerate(rls)]
    ro = [Wire() for _ in range(n)]
    o = sortk(inplist, retlist, ro)
    for i, x in enumerate(o):
        pylse.inspect(x, f"o{i}")
    for i, x in enumerate(ro):
        pylse.inspect(x, f"ro{i}")
    sim = pylse.Simulation()
    events = sim.simulate()
    towatch = ["x", "r", "o", "ro"]
    watchers = [[f"{x}{i}" for i in range(n)] for x in towatch]
    watch_wires = sum(watchers, [])
    if plot:
        sim.plot(wires_to_display=watch_wires)
    evio = events_io(events, towatch)
    check_out(*evio)
    return events


def quick_mmax(n, plot: bool = True):
    rls = [choice([True, False]) for _ in range(n)]
    ils1: list[float] = sorted([choice(range(6)) * 10 + 10 for _ in range(n)])
    ils2: list[float] = sorted([choice(range(6)) * 10 + 10 for _ in range(n)])
    demo_mmax(ils1, ils2, rls, plot)


def quick_sort(n, plot: bool = True):
    rls = [choice([True, False]) for _ in range(n)]
    # ils: list[float] = list(range(10, (n + 1) * 10, 10))
    # shuffle(ils)
    ils: list[float] = [choice(range(6)) * 10 + 10 for _ in range(n)]
    demo_sortk(ils, rls, plot)


def events_io(events: Dict[str, list[float]], matchs: list[str]) -> list[list[float]]:
    def nonn(x: str):
        return "".join([i for i in x if not i.isdigit()])

    def onlyn(x: str):
        return int("".join([i for i in x if i.isdigit()]))

    def evnorm(x: list[float]) -> list[float]:
        assert len(x) <= 1
        return [inf] if x == [] else x

    evks = sorted(events.keys())
    groupks = {
        k: sum([evnorm(events[x]) for x in sorted(v, key=onlyn)], [])
        for k, v in groupby(evks, nonn)
    }
    evio = [groupks[x] for x in matchs]
    return evio


def laydepth(n: int) -> int:
    hn = int(log2(n))
    depth = (hn * (hn + 1)) // 2
    return depth


def check_out(x, r, o, ro):
    delta = 0.8
    n = len(x)
    depth = laydepth(n)
    sdelta = depth * delta
    # print(f"{sdelta=}")
    order = list(argsort(x))
    rbool = [x < inf for x in r]
    robool = [x < inf for x in ro]
    maxdelta = max(map(sub, o, o[1:]))
    # print(f"{maxdelta=}")
    assert maxdelta <= sdelta
    if sum(rbool) == 0:
        print("no returns")
        return
    ordx = sorted(x)
    winners = [out for out, check in zip(ordx, rbool) if check]
    chosen = sorted([out for out, check in zip(x, robool) if check])
    # print(f"{winners=}, {chosen=}")
    assert winners == chosen
    ordered_robool = [robool[i] for i in order]
    oughts = [out for out, check in zip(o, ordered_robool) if check]
    haves = [out for out, check in zip(o, rbool) if check]
    # print(f"{oughts=}, {haves=}")
    diffmax = max(map(sub, oughts, haves))
    # print(f"{diffmax=}")
    assert diffmax <= sdelta


def check_merge(x, y, r, o, rox, roy):
    n = len(x)
    both = x + y
    bothr = rox + roy
    bothrbool = [x < inf for x in bothr]
    maxes = sorted(both)[n:]
    diffz = list(map(sub, maxes, sorted(o)))
    diffmax = max(diffz) - min(diffz)
    # print(f"{diffmax=}")
    assert diffmax <= 1
    order = list(argsort(o))
    rbool = [x < inf for x in r]
    ordered_rbool = [rbool[i] for i in order]
    oughts = [out for out, check in zip(maxes, ordered_rbool) if check]
    haves = sorted([out for out, check in zip(both, bothrbool) if check])
    # print(f"{oughts=}, {haves=}")
    assert oughts == haves
