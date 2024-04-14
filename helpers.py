from random import shuffle
from pylse import working_circuit
from numpy.random import choice


def get_latency(events: dict[str, list[float]]) -> float:
    return max(max(v, default=0) for v in events.values())


def get_jj():
    return sum(
        x.element.jjs
        for x in working_circuit()
        if x.element.name not in ["_Source", "InGen"]
    )


def sample_synd9():
    d = 9
    n = (d - 1) ** 2 // 2 + d - 1
    n_cpx = choice(range(len(pd9p5e2)), p=pd9p5e2)
    arr = [True] * n_cpx + [False] * (n - n_cpx)
    shuffle(arr)
    return arr


pd9p5e2 = [
    0.3912391131761947,
    0.12412938665012276,
    0.2437016865871852,
    0.07342152001014647,
    0.10278553999469055,
    0.027565786666506182,
    0.024792439999855986,
    0.006229993333299558,
    0.004420939999976968,
    0.0010052333333305863,
    0.000532080000000063,
    0.00011410000000001515,
    4.912000000000476e-05,
    9.226666666666585e-06,
    3.0333333333333006e-06,
    5.733333333333338e-07,
    2.0000000000000015e-07,
    2.6666666666666667e-08,
]
