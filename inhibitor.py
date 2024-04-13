import pylse
from pylse import Wire, working_circuit
from sfq_cells2 import C, C_INV, DRO_C, INH, M, s

nu = working_circuit().add_node


def comp(a, b, ret_min, ret_max, ret_a, ret_b):
    ax, a1 = s(a)
    a3, a4 = s(ax)
    bx, b1 = s(b)
    b3, b4 = s(bx)

    # Mixmax part
    xmax, xmin = Wire(), Wire()
    nu(C(), [a3, b3], [xmax])
    nu(C_INV(), [a4, b4], [xmin])

    out1 = Wire()
    inh1 = INH()
    routemax = DRO_C()
    routemin = DRO_C()
    outx1, outx2 = s(out1)

    back_a1, back_b1, back_a2, back_b2 = Wire(), Wire(), Wire(), Wire()
    nu(inh1, [a1, b1], [out1])
    nu(routemax, [outx1, ret_max], [back_b2, back_a2])
    nu(routemin, [outx2, ret_min], [back_a1, back_b1])
    nu(M(), [back_b1, back_b2], [ret_b])
    nu(M(), [back_a1, back_a2], [ret_a])
    return xmin, xmax


def simple_comp(a, b) -> tuple[Wire, Wire]:
    "2-comparator without return"
    a2, a1 = s(a)
    b2, b1 = s(b)
    # Mixmax part
    xmax, xmin = Wire(), Wire()
    nu(C(), [a1, b1], [xmax])
    nu(C_INV(), [a2, b2], [xmin])
    return xmin, xmax


def cmax(a: Wire, b: Wire, ret: Wire, ret_a: Wire, ret_b: Wire) -> Wire:
    ax, a1 = s(a)
    bx, b1 = s(b)

    # Mixmax part
    xmax = Wire()
    nu(C(), [ax, bx], [xmax])

    out1 = Wire()
    inh1 = INH()
    routemax = DRO_C()

    nu(inh1, [a1, b1], [out1])
    nu(routemax, [out1, ret], [ret_b, ret_a])
    return xmax


def demo_comp(t0, t1, tx, tn):
    pylse.working_circuit().reset()
    a = pylse.inp_at(t0, name="a")
    b = pylse.inp_at(t1, name="b")
    ret_max = pylse.inp_at(tx, name="rmax")
    ret_min = pylse.inp_at(tn, name="rmin")
    ret_a, ret_b = Wire(), Wire()
    xmin, xmax = comp(a, b, ret_max, ret_min, ret_a, ret_b)

    pylse.inspect(xmin, "xmin")
    pylse.inspect(xmax, "xmax")
    pylse.inspect(ret_b, "ret_b")
    pylse.inspect(ret_a, "ret_a")

    sim = pylse.Simulation()
    events = sim.simulate()
    sim.plot(
        wires_to_display=["a", "b", "rmax", "rmin", "xmax", "xmin", "ret_a", "ret_b"]
    )
    return events


def demo_cmax(t0, t1, tx):
    pylse.working_circuit().reset()
    a = pylse.inp_at(t0, name="a")
    b = pylse.inp_at(t1, name="b")
    ret = pylse.inp_at(tx, name="rmax")
    ret_a, ret_b = Wire(), Wire()
    xmax = cmax(a, b, ret, ret_a, ret_b)

    pylse.inspect(xmax, "xmax")
    pylse.inspect(ret_b, "ret_b")
    pylse.inspect(ret_a, "ret_a")

    sim = pylse.Simulation()
    events = sim.simulate()
    sim.plot(wires_to_display=["a", "b", "rmax", "xmax", "ret_a", "ret_b"])
    return events


if __name__ == "__main__":
    demo_cmax(10, 20, 50)
