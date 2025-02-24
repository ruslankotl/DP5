from dp5.neural_net.nfp.layers import *
from dp5.neural_net.nfp.models import *

custom_layers = {
    'MessageLayer': MessageLayer,
    'EdgeNetwork': EdgeNetwork,
    'ReduceAtomToMol': ReduceAtomToMol,
    'ReduceBondToAtom': ReduceBondToAtom,
    'GatherAtomToBond': GatherAtomToBond,
    'GRUStep': GRUStep,
    'Embedding2D': Embedding2D,
    'Squeeze': Squeeze,
    'GraphModel': GraphModel,
    'masked_mean_squared_error': masked_mean_squared_error
}
