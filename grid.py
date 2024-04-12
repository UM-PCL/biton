import numpy as np
from pylse import Wire, working_circuit, inp_at, Simulation
from sfq_cells2 import and_s, inv, jtl_chain, s, xor_s, xnor_s, split

grab = working_circuit().get_wire_by_name


def main():
    d = 5
    anc = np.zeros((d + 1, d - 1))
    anc[::2, 1::2] = 1
    anc[1::2, ::2] = 1
    xa = [(i, j) for i, j in np.argwhere(anc > 0)]
    nx = [(i, j) for i, j in np.argwhere(anc == 0)]
    bsynd = np.random.choice([True, False], len(xa), p=[0.3, 0.7])
    symptoms = syndromes(xa, bsynd)
    sympt = {k: s(v) for k,v in symptoms.items()}
    symptom_par = {k: v[0] for k,v in sympt.items()}
    symptom_sel = {k: v[1] for k,v in sympt.items()}
    topdown = {
        k: split(a)
        for k, a in symptom_par.items()
    }
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
    clkj1 = jtl_chain(clks1, 3)
    clk2, slks2 = s(clkj1)
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
    l2 = {}
    for k, opers in fhor.items():
        loclk1, loclks1 = s(l2cls[k])
        loclk2 = jtl_chain(loclks1, 2)
        if len(opers) == 1:
            reven = inv(opers[0], loclk1, name=f"xnr{k}")
        else:
            reven = xnor_s(opers[0], opers[1], loclk1, name=f"xnr{k}")
        l2[k] = and_s(reven, symptom_sel[k], loclk2, name=f"cmx{k}")
    sim = Simulation()
    events = sim.simulate()
    sim.plot()


def syndromes(
    pos: list[tuple[int, int]], synd_array: np.ndarray
) -> dict[tuple[int, int], Wire]:
    synd_times = [[10] * x for x in synd_array]
    synd_wires = {xy: inp_at(*syn, name=f"syn{xy}") for syn, xy in zip(synd_times, pos)}
    return synd_wires


if __name__ == "__main__":
    main()
