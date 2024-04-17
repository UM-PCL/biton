import matplotlib.pyplot as plt
import collections

def plotarbi(events_to_plot, wires_to_display):
    until = max(max(times, default=0) for times in events_to_plot.values()) + 5
    until = int(until)
    events = {w: events_to_plot[w] for w in wires_to_display}
    events = events.items()
    od = collections.OrderedDict(events)
    variables = list(od.keys())
    data = list(od.values())
    plt.rcParams["font.size"] = "30"
    plt.show()
    _, ax = plt.subplots()
    plt.eventplot(data, orientation='horizontal', color='black', linelengths=0.9, linewidths=4)
    ax.set_xlabel('Time (ps)')
    ax.set_xlim(-1, until)
    ax.set_xticks([x for x in range(until+1) if (x % 10 == 0)])
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    # ax.set_ylabel('Tracked Wires')
    # ax.set_ylim(-1, len(variables))
    ax.set_yticks([(i) for i in range(len(variables))])
    ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.grid(True)
    # plt.subplots_adjust(left=0.15, right=0.985, top=0.985, bottom=0.12)
    plt.tight_layout()
    plt.show()
