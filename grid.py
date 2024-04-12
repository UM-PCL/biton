import numpy as np
from pylse import Wire, working_circuit, inp_at, Simulation
from sfq_cells2 import and_s, inv, jtl_chain, s, xor_s, xnor_s, split

grab = working_circuit().get_wire_by_name


def check(d: int, anc: np.ndarray):
    xnrs = np.zeros((d + 3, d + 1))
    # bs0 = np.array(bsynd[::2])
    # bs0.resize((3, 2))
    # bs1 = np.array(bsynd[1::2])
    # bs1.resize((3, 2))
    # anc[::2, 1::2], anc[1::2, ::2] = bs0, bs1
    down = np.pad(anc, ((2, 0), (0, 0)))
    up = np.pad(anc, ((0, 2), (0, 0)))
    xrs = np.logical_xor(down, up)[1:-1]
    ancp = np.pad(anc, ((1,1),(1,1)))
    for i in range(1, d+2):
        for j in range(1, d):
            xnrs[i,j] = ((i+j+1)%2)+(ancp[i+1,j+1]+ancp[i-1,j+1]+ancp[i+1,j-1]+ancp[i-1,j-1])
    xnrs = 1-(xnrs[1:-1,1:-1]%2)
    cmpx = xnrs * anc
    return anc, xrs, xnrs, cmpx


def grid(d: int, plot= False):
    working_circuit().reset()
    anc = np.zeros((d + 1, d - 1))
    anc[::2, 1::2] = 1
    anc[1::2, ::2] = 1
    xa = [(i, j) for i, j in np.argwhere(anc > 0)]
    nx = [(i, j) for i, j in np.argwhere(anc == 0)]
    bsynd = np.random.choice([True, False], len(xa), p=[0.3, 0.7])
    symptoms = syndromes(xa, bsynd)
    sympt = {k: s(v) for k, v in symptoms.items()}
    symptom_par = {k: v[0] for k, v in sympt.items()}
    symptom_sel = {k: v[1] for k, v in sympt.items()}
    topdown = {k: split(a) for k, a in symptom_par.items()}
    vert = [((i - 1, j), (i + 1, j)) for i, j in nx]
    fvert = {}
    for k, (a, b) in zip(nx, vert):
        fvert[k] = []
        if a in xa:
            fvert[k].append(topdown[a][0])
        if b in xa:
            fvert[k].append(topdown[b][1])
    validvert = {k: v for k, v in fvert.items() if len(v) > 1}
    # fhor = [[x for x in li if x in nx] for li in hor]
    nclk: int = int(np.floor((d - 1) ** 2 / 2))
    assert len(validvert) == nclk
    assert len(symptom_par) == np.ceil((d - 1) ** 2 / 2 + d - 1)
    clk = inp_at(20, name="clk")
    clk1, clks1 = s(clk)
    clkj1 = jtl_chain(clks1, 6)
    clk2, clks2 = s(clkj1)
    clkj2 = jtl_chain(clks2, 6)
    clk3, clks3 = s(clkj2)
    l1cls = dict(zip(validvert.keys(), split(clk1, n=nclk)))
    l1 = {
        k: opers[0]
        if len(opers) == 1
        else xor_s(opers[0], opers[1], l1cls[k], name=f"xr{k}")
        for k, opers in fvert.items()
    }
    leftright = {
        k: (a, a) if not (k[1] > 0 and k[1] < d - 2) else split(a)
        for k, a in l1.items()
    }
    hor = [[(i, j - 1), (i, j + 1)] for i, j in xa]
    fhor = {}
    for k, (a, b) in zip(xa, hor):
        fhor[k] = []
        if a in nx:
            fhor[k].append(leftright[a][0])
        if b in nx:
            fhor[k].append(leftright[b][1])
    # validhor = {k: v for k, v in fhor.items() if len(v) > 1}
    # nclkx: int = int(np.ceil(d / 2) * (d-3))
    # assert len(validhor) == nclkx
    l2cls = dict(zip(xa, split(clk2, n=len(xa))))
    l3cls = dict(zip(xa, split(clk3, n=len(xa))))
    l2 = {}
    for k, opers in fhor.items():
        loclk1 = l2cls[k]
        loclk2 = l3cls[k]
        if len(opers) == 1:
            reven = inv(opers[0], loclk1, name=f"xnr{k}")
        else:
            reven = xnor_s(opers[0], opers[1], loclk1, name=f"xnr{k}")
        l2[k] = and_s(reven, symptom_sel[k], loclk2, name=f"cmx{k}")
    sim = Simulation()
    events = sim.simulate()
    if plot:
        sim.plot()
    sind = np.zeros(anc.shape)
    for i, v in zip(xa, bsynd):
        sind[i] = v
    getxn = {k: len(events[f"xnr{k}"]) for k in xa}
    hmx = np.zeros(anc.shape)
    for i, v in getxn.items():
        hmx[i] = v
    getxr = {k: len(events[f"xr{k}"]) for k,v in fvert.items() if len(v) != 1}
    hmr = np.zeros(anc.shape)
    for i, v in getxr.items():
        hmr[i] = v
    getcm = {k: len(events[f"cmx{k}"]) for k in xa}
    hcm = np.zeros(anc.shape)
    for i, v in getcm.items():
        hcm[i] = v
    _, xrs, xnrs, cmpx = check(d, sind)
    assert np.all(xrs[1:-1] == hmr[1:-1])
    assert np.all(xnrs == hmx)
    assert np.all(cmpx == hcm)
    # import IPython
    # IPython.embed()
    return events, sind, hmr, hmx, hcm, check(d, sind)


def syndromes(
    pos: list[tuple[int, int]], synd_array: np.ndarray
) -> dict[tuple[int, int], Wire]:
    synd_times = [[10] * bool(x) for x in synd_array]
    synd_wires = {xy: inp_at(*syn, name=f"syn{xy}") for syn, xy in zip(synd_times, pos)}
    return synd_wires


# if __name__ == "__main__":
#     main()
