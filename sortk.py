from inhibitor import comp
from pylse import Wire
import pylse

def sortk4(inplist: list[Wire], retlist: list[Wire]):
    rt = [Wire() for _ in range(4)]
    x = [Wire() for _ in range(4)]
    y = [Wire() for _ in range(4)]
    rr = [Wire() for _ in range(4)]
    # r = [Wire() for _ in range(4)]
    ro = [Wire() for _ in range(4)]
    o = [Wire() for _ in range(4)]
    x[0], x[1] = comp(inplist[0], inplist[1], rt[0], rt[1], ro[0], ro[1])
    x[3], x[2]= comp(inplist[3], inplist[2], rt[3], rt[2], ro[3], ro[2] )
    y[0], y[2] = comp(x[0], x[2], rr[0], rr[2]          , rt[0], rt[2])
    y[1], y[3] = comp(x[1], x[3], rr[1], rr[3]          , rt[1], rt[3])
    o[0], o[1] = comp(y[0], y[1], retlist[0], retlist[1], rr[0], rr[1])
    o[2], o[3] = comp(y[2], y[3], retlist[2], retlist[3], rr[2], rr[3])
    return x, y, o, rr, rt, ro

def demo_sortk4(ils: list[float], rls: list[bool]):
    pylse.working_circuit().reset()
    inplist = [pylse.inp_at(x, name=f'x{i}') for i, x in enumerate(ils)]
    retlist = [pylse.inp_at(*([200]*x), name=f'r{i}') for i, x in enumerate(rls)]
    xx, y, o, rr, rt, ro = sortk4(inplist, retlist)
    for i, x in enumerate(xx):
        pylse.inspect(x, f'xx{i}')
    for i, x in enumerate(y):
        pylse.inspect(x, f'y{i}')
    for i, x in enumerate(o):
        pylse.inspect(x, f'o{i}')
    for i, x in enumerate(ro):
        pylse.inspect(x, f'ro{i}')
    for i, x in enumerate(rr):
        pylse.inspect(x, f'rr{i}')
    for i, x in enumerate(rt):
        pylse.inspect(x, f'rt{i}')
    sim = pylse.Simulation()
    events = sim.simulate()
    sim.plot(wires_to_display=["x0","x1","x2","x3","r0","r1","r2","r3","o0","o1","o2","o3","ro0","ro1","ro2","ro3"])
    return events
