import re
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
    C_labels, C_exp = _parse_description(Cexp)
    H_labels, H_exp = _parse_description(Hexp)

    return C_labels, C_exp, H_labels, H_exp, equivalents, omits


def _parse_description(exp):

    if len(exp) > 0:
        # Replace all 'or' and 'OR' with ',', remove all spaces and 'any'
        texp = re.sub(r"or|OR", ",", exp, flags=re.DOTALL)
        texp = re.sub(r" ", "", texp, flags=re.DOTALL)
        # Get all assignments, split mulitassignments
        expLabels = re.findall(r"(?<=\().*?(?=\)|;)",
                               texp, flags=re.DOTALL)
        expLabels = [x.replace("any", "") for x in expLabels]
        expLabels = [x.split(",") for x in expLabels]
        # Remove assignments and get shifts
        ShiftData = (re.sub(r"\(.*?\)", "", exp.strip(), flags=re.DOTALL)).split(
            ","
        )
        logger.info(ShiftData)
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
