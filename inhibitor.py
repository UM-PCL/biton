import pylse
from sfq_cells2 import DRO_C, INH, JTL, M, s

# w1, w2, w3, w4, w5 = pylse.Wire(), pylse.Wire(), pylse.Wire(), pylse.Wire(), pylse.Wire()
pylse.working_circuit().reset()
out1, out2, a_p, b_p = pylse.Wire(), pylse.Wire(), pylse.Wire(), pylse.Wire()

inh1 = INH()
inh2 = INH()
routemax = DRO_C()
routemin = DRO_C()
tiebreak = JTL()

# def mux_s(a: pylse.Wire, b: pylse.Wire, sel: pylse.Wire, clk: pylse.Wire):
#     out = pylse.Wire()
#     pylse.working_circuit().add_node(Mux(), [a, b, sel, clk], [out])
#     return out


# clk = pylse.inp_at(*(i*50 for i in range(1, 12)), name='clk')

# On tie, both inhibits fire
# In this case, route rmax to a and rmin to b
a = pylse.inp_at(10, name='a')

b = pylse.inp_at(11, name='b')
a1, a2 = s(a)
b1, b2 = s(b)
outx1, outx2 = s(out1)

ret_max = pylse.inp_at(100, name='rmax')
ret_min = pylse.inp_at(50, name='rmin')
back_a1, back_b1, back_a2, back_b2 =  pylse.Wire(),  pylse.Wire(),  pylse.Wire(),  pylse.Wire()
ret_a, ret_b = pylse.Wire(),  pylse.Wire()
pylse.working_circuit().add_node(inh1, [a1, b1], [out1])
# pylse.working_circuit().add_node(inh2, [b2, a2], [out2])
pylse.working_circuit().add_node(routemax, [outx1, ret_max], [back_b2, back_a2])
pylse.working_circuit().add_node(routemin, [outx2, ret_min], [back_a1, back_b1])
pylse.working_circuit().add_node(M(), [back_b1, back_b2], [ret_b])
pylse.working_circuit().add_node(M(), [back_a1, back_a2], [ret_a])

# ...which we'll give a name by `inspect`ing it.
pylse.inspect(out1, 'afirst')
# pylse.inspect(out2, 'bfirst')
# pylse.inspect(a_p, 'a_p')
# pylse.inspect(b_p, 'b_p')
# pylse.inspect(back_a1, 'back_a1')
# pylse.inspect(back_b1, 'back_b1')
# pylse.inspect(back_a2, 'back_a2')
# pylse.inspect(back_b2, 'back_b2')
pylse.inspect(ret_b, 'ret_b')
pylse.inspect(ret_a, 'ret_a')

sim = pylse.Simulation()
events = sim.simulate()
sim.plot(wires_to_display=['a', 'b', 'afirst', 'rmax', 'rmin', 'ret_a', 'ret_b'])
# sim.plot(wires_to_display=['a', 'b', 'afirst', 'rmax', 'rmin', 'back_a1', 'back_b1', 'back_a2', 'back_b2'])
# sim.plot(wires_to_display=['a', 'b', 'a_p', 'b_p', 'bfirst', 'afirst'])
