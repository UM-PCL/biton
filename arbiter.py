from sortk import sortk, mergemax, events_io
from pylse import Wire, working_circuit
from pylse.circuit import InGen
import pylse
from math import inf, log2
from numpy.random import choice as npchoice
from tqdm import tqdm 


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
    last_level = n == 2 * k
    if last_level:
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
    sorts1 = sortk(inters1, retm1, rets1, prune=not last_level)
    sorts2 = sortk(inters2, retm2, rets2, prune=not last_level)
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
    dlayer = (lk * dcomp) + dcmax
    dlayer1 = (depthk * dcomp) + dcmax
    forward_delay = dlayer1 + (depthn-1) * dlayer
    return forward_delay


def get_back_del(k: int, n: int) -> float:
    lk = log2(k)
    depthn = log2(n) - lk
    depthk = lk * (lk + 1) // 2
    ddroc = 9.5
    dmg = 6.3
    dcomp = ddroc + dmg
    dcmax = ddroc
    dlayer1 = (depthk * dcomp) + dcmax
    dlayern = (lk * dcomp) + dcmax
    backdelay = dlayer1 + (depthn-1) * dlayern
    return backdelay


def minimum_sampling_del(k: int, n: int) -> float:
    lk = log2(k)
    depthn = log2(n) - lk
    depthk = lk * (lk + 1) // 2
    dla = 8.1
    dfa = 8.8
    deltacell = abs(dla - dfa)
    delta = (depthk + (depthn - 1) * lk) * deltacell
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


def info(k, n):
    info_dict = {}
    info_dict['n'] = n
    info_dict['k'] = k
    info_dict["temporal_distance"] = minimum_sampling_del(k, n)
    info_dict["forward_delay"] = get_del(k, n)
    info_dict["latest_input"] = 6 * info_dict["temporal_distance"]
    info_dict["return_start"] = info_dict["forward_delay"] + info_dict["latest_input"]
    info_dict["backwards_delay"] = get_back_del(k, n)
    info_dict["total_delay"] = info_dict["return_start"] + info_dict["backwards_delay"]
    info_dict["JJs"] = jj_estimation(k, n)
    return info_dict


def sim_arbiter(k: int, n: int, n_runs: int = 1, plot: bool = True):
    working_circuit().reset()
    priority_limit = 6
    clk_del = minimum_sampling_del(k, n)
    inplist = [Wire(name=f"x{i}") for i in range(n)]
    ingens = [InGen([]) for _ in range(n)]
    for i in range(n):
        working_circuit().add_node(
            ingens[i], [working_circuit().source_wire()], [inplist[i]]
        )
    topk = arbiter(k, inplist)
    towatch = ["x", "top"]
    watchers = [[f"{x}{i}" for i in range(n)] for x in towatch]
    towatch2 = ["max", "r"]
    watchers2 = [[f"{x}{i}" for i in range(k)] for x in towatch2]
    watch_wires = sum(watchers + watchers2, [])
    data = info(k, n)
    jjs = sum(
        x.element.jjs
        for x in working_circuit()
        if x.element.name not in ["_Source", "InGen"]
    )
    est_jj = jj_estimation(k, n)
    assert jjs == est_jj
    # print(f"{(n,k,jjs)=}")
    # print(data)
    for i, x in enumerate(topk):
        pylse.inspect(x, f"top{i}")
    runs = [0] if n_runs == 1 else tqdm(range(n_runs), desc=f"{(k,n)=}")
    for _ in runs:
        samps = clique_sample(n)
        inps: list[float] = [clk_del * (min(x + 1, priority_limit)) for x in samps]
        for ig, fire in zip(ingens, inps):
            ig.times = [fire]
        sim = pylse.Simulation()
        events = sim.simulate()
        if plot:
            sim.plot(wires_to_display=watch_wires)
        evio = events_io(events, towatch)
        check_arbitrage(k, *evio)
    # return data


def jj_estimation(k, n):
    la_jj = 4
    fa_jj = 4
    inh_jj = 13
    droc_jj = 13
    mg_jj = 5
    s_jj = 3
    comp_jj = la_jj + fa_jj + (5 * s_jj) + (2 * droc_jj) + inh_jj + (2 * mg_jj)
    cmax_jj = la_jj + (2 * s_jj) + inh_jj + droc_jj
    arrow_jj = comp_jj * k // 2
    lk = log2(k)
    # depthn = log2(n) - lk
    depthk = lk * (lk + 1) // 2
    sort_jj = arrow_jj * depthk
    bsort_jj = arrow_jj * lk
    mmax_jj = cmax_jj * k
    block1_jj = (2 * sort_jj) + mmax_jj
    blockn_jj = (2 * bsort_jj) + mmax_jj
    start_blocks = (n // k) // 2
    n_block = start_blocks - 1
    estimate_jj = (start_blocks * block1_jj) + (blockn_jj * n_block)
    # assert False
    return int(estimate_jj)


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
    # print(f"{(winners, chosen)=}")
    assert winners == chosen
    total_delay = max(ret for ret in o if ret < inf)
    predicted_delay = info(k, len(x))["total_delay"]
    # print(f"{(k,total_delay)=}")
    assert total_delay <= predicted_delay + 1
