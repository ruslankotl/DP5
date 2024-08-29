import re
import networkx as nx
import logging

logger = logging.getLogger(__name__)


def process_description(nmr_source):
    with open(nmr_source) as f:
        Cexp = f.readline()
        f.readline()
        Hexp = f.readline()
        equivalents = []
        omits = []
        f.readline()

        for line in f:
            if not "OMIT" in line and len(line) > 1:
                equivalents.append(line[:-1].split(","))
            elif "OMIT" in line:
                omits.extend(line[5:-1].split(","))
    logger.info("Read carbon NMR shifts")
    C_labels, C_exp = _parse_description(Cexp)
    logger.info("Read proton NMR shifts")
    H_labels, H_exp = _parse_description(Hexp)

    return C_labels, C_exp, H_labels, H_exp, equivalents, omits


def _parse_description(exp):

    if len(exp) > 0:
        # Replace all 'or' and 'OR' with ',', remove all spaces and 'any'
        texp = re.sub(r"or|OR", ",", exp, flags=re.DOTALL)
        texp = re.sub(r" ", "", texp, flags=re.DOTALL)
        # Get all assignments, split mulitassignments
        expLabels = re.findall(r"(?<=\().*?(?=\)|;)", texp, flags=re.DOTALL)
        expLabels = [x.replace("any", "") for x in expLabels]
        expLabels = [x.split(",") for x in expLabels]
        # Remove assignments and get shifts
        ShiftData = (re.sub(r"\(.*?\)", "", exp.strip(), flags=re.DOTALL)).split(",")
        logger.info(", ".join(ShiftData))
        expShifts = [float(x) for x in ShiftData]
    else:
        expLabels = []
        expShifts = []

    return expLabels, expShifts


def pairwise_assignment(calculated, experimental: list):
    sorted_calc = sorted(calculated, reverse=True)
    sorted_exp = sorted(experimental, reverse=True)
    assigned = [None] * len(calculated)

    for calc, exp in zip(sorted_calc, sorted_exp):
        index = list(calculated).index(calc)
        assigned[index] = exp

    return assigned


def matching_assignment(calculated, experimental, threshold=40):

    scaled = calculated

    # Create a bipartite graph
    G = nx.Graph()

    # Add nodes for calc and exp with a bipartite attribute
    calc_nodes = [("calc", i) for i in range(len(scaled))]
    exp_nodes = [("exp", i) for i in range(len(experimental))]
    G.add_nodes_from(calc_nodes, bipartite=0)
    G.add_nodes_from(exp_nodes, bipartite=1)

    # Add edges for all pairs within the threshold
    for i, c in enumerate(scaled):
        for j, e in enumerate(experimental):
            deviation = abs(c - e)
            if deviation <= threshold:
                G.add_edge(("calc", i), ("exp", j), weight=-deviation)

    # Find the maximum matching
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)

    matched_pair_indices = set()

    for pair in matching:
        # pair could be (('calc', i), ('exp', j)) or the reverse
        calc, exp = sorted(pair)
        # after sorting, 'calc' goes before 'exp'
        calc_index = calc[1]
        exp_index = exp[1]

        matched_pair_indices.add((calc_index, exp_index))

    # Now, use these indices to access values in calc and exp
    matched_pairs = [(calculated[i], experimental[j]) for i, j in matched_pair_indices]
    assigned = [None] * len(calculated)
    for calc_shift, exp_shift in matched_pairs:
        index = list(calculated).index(calc_shift)
        assigned[index] = exp_shift

    return assigned
