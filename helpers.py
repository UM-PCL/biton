from pylse import working_circuit

def get_latency(events: dict[str, list[float]]) -> float:
    return max(max(v, default=0) for v in events.values())


def get_jj():
    return sum(
        x.element.jjs
        for x in working_circuit()
        if x.element.name not in ["_Source", "InGen"]
    )

