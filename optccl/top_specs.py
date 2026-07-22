import itertools
from .topologies import BandwidthConstraint, Topology, generate_ecs


def gen_doe():
    eps = 0.0001
    gpus = [1, 2, 3, 4]
    mems = [101, 102, 103, 104]
    switches = [201, 202, 203, 204, 301, 302, 303, 304]
    nics = [301, 302, 303, 304]
    edge_data = {}

    for i, j in itertools.permutations([1, 2, 3, 4], 2):
        edge_data[(i, j)] = 1

    for i in gpus:
        edge_data[(i, 200 + i)] = 1
        edge_data[(200 + i, i)] = 1
        edge_data[(200 + i, 300 + i)] = eps
        edge_data[(300 + i, 200 + i)] = eps

    for i in gpus:
        for j in gpus:
            edge_data[(200 + i, 100 + j)] = eps
            edge_data[(100 + j, 200 + i)] = eps

    for i in gpus + mems:
        pass
        edge_data[(i, i)] = 0

    bcs = []
    for i, j in itertools.permutations([1, 2, 3, 4], 2):
        bcs.append(BandwidthConstraint(f"nv {i, j}", [(i, j)], 100))
    for i in range(1, 5):
        bcs.append(BandwidthConstraint(f"up {i}", [(i, 200 + i)], 32))
        bcs.append(BandwidthConstraint(f"down {i}", [(200 + i, i)], 32))
        bcs.append(BandwidthConstraint(f"up {i}", [(300 + i, 200 + i)], 32))
        bcs.append(BandwidthConstraint(f"down {i}", [(200 + i, 300 + i)], 32))

    for i in range(1, 5):
        bcs.append(
            BandwidthConstraint(
                f"memory {i}",
                [(200 + j, 100 + i) for j in range(1, 5)]
                + [(100 + i, 200 + j) for j in range(1, 5)],
                50,
            )
        )

    doe = Topology.from_node_types(gpus, mems, switches, nics, edge_data, bcs)

    ecs = generate_ecs(doe)

    doe.add_ECs(ecs)

    return doe


def gen_doe_only_nv():
    eps = 0.001
    gpus = [1, 2, 3, 4]
    mems = []
    switches = [201, 202, 203, 204, 301, 302, 303, 304]
    nics = [301, 302, 303, 304]
    edge_data = {}

    for i, j in itertools.permutations([1, 2, 3, 4], 2):
        edge_data[(i, j)] = 1

    for i in gpus:
        edge_data[(i, 200 + i)] = 1
        edge_data[(200 + i, i)] = 1
        edge_data[(200 + i, 300 + i)] = eps
        edge_data[(300 + i, 200 + i)] = eps

    for i in gpus + mems:
        edge_data[(i, i)] = 0

    bcs = []
    for i, j in itertools.permutations([1, 2, 3, 4], 2):
        bcs.append(BandwidthConstraint(f"nv {i, j}", [(i, j)], 100))
    for i in range(1, 5):
        bcs.append(BandwidthConstraint(f"up {i}", [(i, 200 + i)], 32))
        bcs.append(BandwidthConstraint(f"down {i}", [(200 + i, i)], 32))
        bcs.append(BandwidthConstraint(f"up {i}", [(300 + i, 200 + i)], 32))
        bcs.append(BandwidthConstraint(f"down {i}", [(200 + i, 300 + i)], 32))

    doe = Topology.from_node_types(gpus, mems, switches, nics, edge_data, bcs)

    ecs = generate_ecs(doe)

    doe.add_ECs(ecs)

    return doe


def gen_doe_no_nv():
    eps = 0.001
    gpus = [1, 2, 3, 4]
    mems = [101, 102, 103, 104]
    switches = [201, 202, 203, 204, 301, 302, 303, 304]
    nics = [301, 302, 303, 304]
    edge_data = {}

    for i in gpus:
        edge_data[(i, 200 + i)] = 1
        edge_data[(200 + i, i)] = 1
        edge_data[(200 + i, 300 + i)] = eps
        edge_data[(300 + i, 200 + i)] = eps

    for i in gpus:
        for j in gpus:
            edge_data[(200 + i, 100 + j)] = eps
            edge_data[(100 + j, 200 + i)] = eps

    for i in gpus + mems:
        edge_data[(i, i)] = 0

    bcs = []
    for i in range(1, 5):
        bcs.append(BandwidthConstraint(f"up {i}", [(i, 200 + i)], 32))
        bcs.append(BandwidthConstraint(f"down {i}", [(200 + i, i)], 32))
        bcs.append(BandwidthConstraint(f"up {i}", [(300 + i, 200 + i)], 32))
        bcs.append(BandwidthConstraint(f"down {i}", [(200 + i, 300 + i)], 32))

    for i in range(1, 5):
        bcs.append(
            BandwidthConstraint(
                f"memory {i}",
                [(200 + j, 100 + i) for j in range(1, 5)]
                + [(100 + i, 200 + j) for j in range(1, 5)],
                50,
            )
        )

    doe = Topology.from_node_types(gpus, mems, switches, nics, edge_data, bcs)

    ecs = generate_ecs(doe)

    doe.add_ECs(ecs)

    return doe


def gen_a100():
    eps = 0.001
    nv_bw = 50
    pcie_bw = 32
    cross_bw = 96
    mem_bw = 200

    mems = [("m", 1), ("m", 2)]
    cpuA = [("s_1", 1), ("s_1", 2)]
    cpuB = [("s_2", 1), ("s_2", 2)]

    gpus = [
        ("g", 1),
        ("g", 2),
        ("g", 3),
        ("g", 4),
        ("g", 5),
        ("g", 6),
        ("g", 7),
        ("g", 8),
    ]
    nics = [
        ("n", 1),
        ("n", 2),
        ("n", 3),
        ("n", 4),
        ("n", 5),
        ("n", 6),
        ("n", 7),
        ("n", 8),
    ]
    pcie = [("p", 1), ("p", 2), ("p", 3), ("p", 4)]

    nvswitchs = [("nv", 1), ("nv", 2), ("nv", 3), ("nv", 4), ("nv", 5), ("nv", 6)]

    edge_data = {}
    bcs = []

    cross_lefts = []
    cross_rights = []
    mems_left = []
    mems_right = []

    for i in range(1, 5):
        edge_data[("p", i), ("g", 2 * i - 1)] = 1
        edge_data[("p", i), ("g", 2 * i)] = 1
        edge_data[("g", 2 * i - 1), ("p", i)] = 1
        edge_data[("g", 2 * i), ("p", i)] = 1
        bcs.append(
            BandwidthConstraint(f"pcie", [(("p", i), ("g", 2 * i - 1))], pcie_bw)
        )
        bcs.append(BandwidthConstraint(f"pcie", [(("p", i), ("g", 2 * i))], pcie_bw))
        bcs.append(
            BandwidthConstraint(f"pcie", [(("g", 2 * i - 1), ("p", i))], pcie_bw)
        )
        bcs.append(BandwidthConstraint(f"pcie", [(("g", 2 * i), ("p", i))], pcie_bw))

        edge_data[("p", i), ("n", 2 * i - 1)] = 1
        edge_data[("p", i), ("n", 2 * i)] = 1
        edge_data[("n", 2 * i - 1), ("p", i)] = 1
        edge_data[("n", 2 * i), ("p", i)] = 1
        bcs.append(
            BandwidthConstraint(f"pcie", [(("p", i), ("n", 2 * i - 1))], pcie_bw)
        )
        bcs.append(BandwidthConstraint(f"pcie", [(("p", i), ("n", 2 * i))], pcie_bw))
        bcs.append(
            BandwidthConstraint(f"pcie", [(("n", 2 * i - 1), ("p", i))], pcie_bw)
        )
        bcs.append(BandwidthConstraint(f"pcie", [(("n", 2 * i), ("p", i))], pcie_bw))

    edge_data[("p", 1), ("s_1", 1)] = eps
    edge_data[("p", 2), ("s_1", 1)] = eps
    edge_data[("p", 3), ("s_1", 2)] = eps
    edge_data[("p", 4), ("s_1", 2)] = eps

    edge_data[("p", 1), ("s_2", 1)] = eps
    edge_data[("p", 2), ("s_2", 1)] = eps
    edge_data[("p", 3), ("s_2", 2)] = eps
    edge_data[("p", 4), ("s_2", 2)] = eps

    bcs.append(
        BandwidthConstraint(
            f"pcie", [(("p", 1), ("s_1", 1)), (("p", 1), ("s_2", 1))], pcie_bw
        )
    )
    bcs.append(
        BandwidthConstraint(
            f"pcie", [(("p", 2), ("s_1", 1)), (("p", 2), ("s_2", 1))], pcie_bw
        )
    )
    bcs.append(
        BandwidthConstraint(
            f"pcie", [(("p", 3), ("s_1", 2)), (("p", 3), ("s_2", 2))], pcie_bw
        )
    )
    bcs.append(
        BandwidthConstraint(
            f"pcie", [(("p", 4), ("s_1", 2)), (("p", 4), ("s_2", 2))], pcie_bw
        )
    )

    edge_data[("s_2", 1), ("p", 1)] = eps
    edge_data[("s_2", 1), ("p", 2)] = eps
    edge_data[("s_2", 2), ("p", 3)] = eps
    edge_data[("s_2", 2), ("p", 4)] = eps
    bcs.append(BandwidthConstraint(f"pcie", [(("s_2", 1), ("p", 1))], pcie_bw))
    bcs.append(BandwidthConstraint(f"pcie", [(("s_2", 1), ("p", 2))], pcie_bw))
    bcs.append(BandwidthConstraint(f"pcie", [(("s_2", 2), ("p", 3))], pcie_bw))
    bcs.append(BandwidthConstraint(f"pcie", [(("s_2", 2), ("p", 4))], pcie_bw))

    edge_data[("s_1", 1), ("m", 1)] = eps
    edge_data[("s_1", 2), ("m", 2)] = eps
    mems_left.append((("s_1", 1), ("m", 1)))
    mems_right.append((("s_1", 2), ("m", 2)))

    edge_data[("m", 1), ("s_1", 1)] = eps
    edge_data[("m", 2), ("s_1", 2)] = eps
    mems_left.append((("m", 1), ("s_1", 1)))
    mems_right.append((("m", 2), ("s_1", 2)))

    edge_data[("m", 1), ("s_2", 1)] = eps
    edge_data[("m", 2), ("s_2", 2)] = eps
    mems_left.append((("m", 1), ("s_2", 1)))
    mems_right.append((("m", 2), ("s_2", 2)))

    edge_data[("s_1", 1), ("m", 2)] = eps
    edge_data[("s_1", 2), ("m", 1)] = eps
    mems_right.append((("s_1", 1), ("m", 2)))
    mems_left.append((("s_1", 2), ("m", 1)))
    cross_rights.append((("s_1", 1), ("m", 2)))
    cross_lefts.append((("s_1", 2), ("m", 1)))

    edge_data[("m", 1), ("s_2", 2)] = eps
    edge_data[("m", 2), ("s_2", 1)] = eps
    mems_left.append((("m", 1), ("s_2", 2)))
    mems_right.append((("m", 2), ("s_2", 1)))
    cross_rights.append((("m", 1), ("s_2", 2)))
    cross_lefts.append((("m", 2), ("s_2", 1)))

    bcs.append(BandwidthConstraint(f"mem", mems_left, mem_bw))
    bcs.append(BandwidthConstraint(f"mem", mems_right, mem_bw))

    bcs.append(BandwidthConstraint(f"cross", cross_rights, cross_bw))
    bcs.append(BandwidthConstraint(f"cross", cross_lefts, cross_bw))

    for i in range(1, 9):
        for j in range(1, 7):
            edge_data[("g", i), ("nv", j)] = 1
            edge_data[("nv", j), ("g", i)] = 1
            bcs.append(BandwidthConstraint(f"nv", [(("g", i), ("nv", j))], nv_bw))
            bcs.append(BandwidthConstraint(f"nv", [(("nv", j), ("g", i))], nv_bw))

    for i in gpus + mems:
        edge_data[(i, i)] = 0

    dgx_a100 = Topology.from_node_types(
        gpus, mems, cpuA + cpuB + nics + pcie + nvswitchs, nics, edge_data, bcs
    )

    ecs = generate_ecs(dgx_a100)

    dgx_a100.add_ECs(ecs)

    return dgx_a100


def gen_a100_no_nv():
    eps = 0.001
    pcie_bw = 32
    cross_bw = 96
    mem_bw = 200

    mems = [("m", 1), ("m", 2)]
    cpuA = [("s_1", 1), ("s_1", 2)]
    cpuB = [("s_2", 1), ("s_2", 2)]

    gpus = [
        ("g", 1),
        ("g", 2),
        ("g", 3),
        ("g", 4),
        ("g", 5),
        ("g", 6),
        ("g", 7),
        ("g", 8),
    ]
    nics = [
        ("n", 1),
        ("n", 2),
        ("n", 3),
        ("n", 4),
        ("n", 5),
        ("n", 6),
        ("n", 7),
        ("n", 8),
    ]
    pcie = [("p", 1), ("p", 2), ("p", 3), ("p", 4)]

    nvswitchs = []

    edge_data = {}
    bcs = []

    cross_lefts = []
    cross_rights = []
    mems_left = []
    mems_right = []

    for i in range(1, 5):
        edge_data[("p", i), ("g", 2 * i - 1)] = 1
        edge_data[("p", i), ("g", 2 * i)] = 1
        edge_data[("g", 2 * i - 1), ("p", i)] = 1
        edge_data[("g", 2 * i), ("p", i)] = 1
        bcs.append(
            BandwidthConstraint(f"pcie", [(("p", i), ("g", 2 * i - 1))], pcie_bw)
        )
        bcs.append(BandwidthConstraint(f"pcie", [(("p", i), ("g", 2 * i))], pcie_bw))
        bcs.append(
            BandwidthConstraint(f"pcie", [(("g", 2 * i - 1), ("p", i))], pcie_bw)
        )
        bcs.append(BandwidthConstraint(f"pcie", [(("g", 2 * i), ("p", i))], pcie_bw))

        edge_data[("p", i), ("n", 2 * i - 1)] = 1
        edge_data[("p", i), ("n", 2 * i)] = 1
        edge_data[("n", 2 * i - 1), ("p", i)] = 1
        edge_data[("n", 2 * i), ("p", i)] = 1
        bcs.append(
            BandwidthConstraint(f"pcie", [(("p", i), ("n", 2 * i - 1))], pcie_bw)
        )
        bcs.append(BandwidthConstraint(f"pcie", [(("p", i), ("n", 2 * i))], pcie_bw))
        bcs.append(
            BandwidthConstraint(f"pcie", [(("n", 2 * i - 1), ("p", i))], pcie_bw)
        )
        bcs.append(BandwidthConstraint(f"pcie", [(("n", 2 * i), ("p", i))], pcie_bw))

    edge_data[("p", 1), ("s_1", 1)] = eps
    edge_data[("p", 2), ("s_1", 1)] = eps
    edge_data[("p", 3), ("s_1", 2)] = eps
    edge_data[("p", 4), ("s_1", 2)] = eps

    edge_data[("p", 1), ("s_2", 1)] = eps
    edge_data[("p", 2), ("s_2", 1)] = eps
    edge_data[("p", 3), ("s_2", 2)] = eps
    edge_data[("p", 4), ("s_2", 2)] = eps

    bcs.append(
        BandwidthConstraint(
            f"pcie", [(("p", 1), ("s_1", 1)), (("p", 1), ("s_2", 1))], pcie_bw
        )
    )
    bcs.append(
        BandwidthConstraint(
            f"pcie", [(("p", 2), ("s_1", 1)), (("p", 2), ("s_2", 1))], pcie_bw
        )
    )
    bcs.append(
        BandwidthConstraint(
            f"pcie", [(("p", 3), ("s_1", 2)), (("p", 3), ("s_2", 2))], pcie_bw
        )
    )
    bcs.append(
        BandwidthConstraint(
            f"pcie", [(("p", 4), ("s_1", 2)), (("p", 4), ("s_2", 2))], pcie_bw
        )
    )

    edge_data[("s_2", 1), ("p", 1)] = eps
    edge_data[("s_2", 1), ("p", 2)] = eps
    edge_data[("s_2", 2), ("p", 3)] = eps
    edge_data[("s_2", 2), ("p", 4)] = eps
    bcs.append(BandwidthConstraint(f"pcie", [(("s_2", 1), ("p", 1))], pcie_bw))
    bcs.append(BandwidthConstraint(f"pcie", [(("s_2", 1), ("p", 2))], pcie_bw))
    bcs.append(BandwidthConstraint(f"pcie", [(("s_2", 2), ("p", 3))], pcie_bw))
    bcs.append(BandwidthConstraint(f"pcie", [(("s_2", 2), ("p", 4))], pcie_bw))

    edge_data[("s_1", 1), ("m", 1)] = eps
    edge_data[("s_1", 2), ("m", 2)] = eps
    mems_left.append((("s_1", 1), ("m", 1)))
    mems_right.append((("s_1", 2), ("m", 2)))

    edge_data[("m", 1), ("s_1", 1)] = eps
    edge_data[("m", 2), ("s_1", 2)] = eps
    mems_left.append((("m", 1), ("s_1", 1)))
    mems_right.append((("m", 2), ("s_1", 2)))

    edge_data[("m", 1), ("s_2", 1)] = eps
    edge_data[("m", 2), ("s_2", 2)] = eps
    mems_left.append((("m", 1), ("s_2", 1)))
    mems_right.append((("m", 2), ("s_2", 2)))

    edge_data[("s_1", 1), ("m", 2)] = eps
    edge_data[("s_1", 2), ("m", 1)] = eps
    mems_right.append((("s_1", 1), ("m", 2)))
    mems_left.append((("s_1", 2), ("m", 1)))
    cross_rights.append((("s_1", 1), ("m", 2)))
    cross_lefts.append((("s_1", 2), ("m", 1)))

    edge_data[("m", 1), ("s_2", 2)] = eps
    edge_data[("m", 2), ("s_2", 1)] = eps
    mems_left.append((("m", 1), ("s_2", 2)))
    mems_right.append((("m", 2), ("s_2", 1)))
    cross_rights.append((("m", 1), ("s_2", 2)))
    cross_lefts.append((("m", 2), ("s_2", 1)))

    bcs.append(BandwidthConstraint(f"mem", mems_left, mem_bw))
    bcs.append(BandwidthConstraint(f"mem", mems_right, mem_bw))

    bcs.append(BandwidthConstraint(f"cross", cross_rights, cross_bw))
    bcs.append(BandwidthConstraint(f"cross", cross_lefts, cross_bw))

    for i in gpus + mems:
        edge_data[(i, i)] = 0

    dgx_a100 = Topology.from_node_types(
        gpus, mems, cpuA + cpuB + nics + pcie + nvswitchs, nics, edge_data, bcs
    )

    ecs = generate_ecs(dgx_a100)

    dgx_a100.add_ECs(ecs)

    return dgx_a100


def gen_a100_only_nv():
    nv_bw = 50
    pcie_bw = 32

    mems = []
    cpuA = []
    cpuB = []

    gpus = [
        ("g", 1),
        ("g", 2),
        ("g", 3),
        ("g", 4),
        ("g", 5),
        ("g", 6),
        ("g", 7),
        ("g", 8),
    ]
    nics = [
        ("n", 1),
        ("n", 2),
        ("n", 3),
        ("n", 4),
        ("n", 5),
        ("n", 6),
        ("n", 7),
        ("n", 8),
    ]

    nvswitchs = [("nv", 1), ("nv", 2), ("nv", 3), ("nv", 4), ("nv", 5), ("nv", 6)]

    edge_data = {}
    bcs = []

    for i in range(1, 9):
        edge_data[("n", i), ("g", i)] = 2
        edge_data[("g", i), ("n", i)] = 2
        bcs.append(BandwidthConstraint(f"pcie", [(("n", i), ("g", i))], pcie_bw))
        bcs.append(BandwidthConstraint(f"pcie", [(("g", i), ("n", i))], pcie_bw))

    for i in range(1, 9):
        for j in range(1, 7):
            edge_data[("g", i), ("nv", j)] = 1
            edge_data[("nv", j), ("g", i)] = 1
            bcs.append(BandwidthConstraint(f"nv", [(("g", i), ("nv", j))], nv_bw))
            bcs.append(BandwidthConstraint(f"nv", [(("nv", j), ("g", i))], nv_bw))

    for i in gpus + mems:
        edge_data[(i, i)] = 0

    dgx_a100 = Topology.from_node_types(
        gpus, mems, cpuA + cpuB + nics + nvswitchs, nics, edge_data, bcs
    )

    ecs = generate_ecs(dgx_a100)

    dgx_a100.add_ECs(ecs)

    return dgx_a100
