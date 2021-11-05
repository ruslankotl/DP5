import numpy as np
from numpy.random import seed

#seed(1)

#from tensorflow.random import set_seed

#set_seed(2)

import pickle

# Define Keras model

from tensorflow.keras.layers import (Input, Embedding, Dense,
                                 Concatenate, Multiply, Add)

from tensorflow.keras.models import  load_model


from nfp.preprocessing import  GraphSequence
from nfp.layers import ( Squeeze,ReduceBondToAtom,GatherAtomToBond, ReduceAtomToPro)

from nfp.models import GraphModel

from pathlib import Path

'''
def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -(np.atleast_2d(distances).T - (-mu + delta * k)) ** 2 / delta
    return np.exp(logits)

def atomic_number_tokenizer(atom):
    return atom.GetNumRadicalElectrons()

def _compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


class RBFSequence(GraphSequence):
    def process_data(self, batch_data):
        batch_data['distance_rbf'] = rbf_expansion(batch_data['distance'])

        offset = _compute_stacked_offsets(
            batch_data['n_pro'], batch_data['n_atom'])

        offset = np.where(batch_data['atom_index'] >= 0, offset, 0)
        batch_data['atom_index'] += offset

        del batch_data['n_atom']
        del batch_data['n_bond']
        del batch_data['distance']

        return batch_data


def Mol_iter(dfr):

    for index,r in dfr.iterrows():

        yield( r['Shift'],index)

def Mol_iter2(df):
    for index,r in df.iterrows():
        yield(r['Mol'], r['atom_index'])

def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()
    
def message_block(atom_features,atom_state, bond_state, connectivity):
    atom_state = Dense(atom_features, use_bias=False)(atom_state)

    source_atom_gather = GatherAtomToBond(1)
    target_atom_gather = GatherAtomToBond(0)

    source_atom = source_atom_gather([atom_state, connectivity])
    target_atom = target_atom_gather([atom_state, connectivity])

    # Edge update network
    bond_state_message = Concatenate()([source_atom, target_atom, bond_state])
    bond_state_message = Dense(2 * atom_features, activation='softplus')(bond_state_message)
    bond_state_message = Dense(atom_features)(bond_state_message)

    bond_state_message = Dense(atom_features, activation='softplus')(bond_state_message)
    bond_state_message = Dense(atom_features, activation='softplus')(bond_state_message)
    bond_state = Add()([bond_state_message, bond_state])

    # message function
    messages = Multiply()([source_atom, bond_state])
    messages = ReduceBondToAtom(reducer='sum')([messages, connectivity])

    # state transition function
    messages = Dense(atom_features, activation='softplus')(messages)
    messages = Dense(atom_features)(messages)
    atom_state = Add()([atom_state, messages])

    return atom_state, bond_state


def build_model(Settings,model_type):

    # Construct input sequences

    preprocessor = pickle.load(open(Path(Settings.ScriptDir)/"mean_model_preprocessor.p","rb"))

    # Raw (integer) graph inputs
    atom_index = Input(shape=(1,), name='atom_index', dtype='int32')
    atom_types = Input(shape=(1,), name='atom', dtype='int32')
    distance_rbf = Input(shape=(256,), name='distance_rbf', dtype='float32')
    connectivity = Input(shape=(2,), name='connectivity', dtype='int32')
    n_pro = Input(shape=(1,), name='n_pro', dtype='int32')

    squeeze = Squeeze()

    satom_index = squeeze(atom_index)
    satom_types = squeeze(atom_types)
    sn_pro = squeeze(n_pro)
    # Initialize RNN and MessageLayer instances
    atom_features = 256

    # Initialize the atom states
    atom_state = Embedding(
        preprocessor.atom_classes,
        atom_features, name='atom_embedding')(satom_types)

    bond_state = distance_rbf

    for _ in range(3):
        atom_state, bond_state = message_block(atom_features,atom_state, bond_state, connectivity)

    atom_state = ReduceAtomToPro(reducer='unsorted_mean')([atom_state, satom_index, sn_pro])

    trans_model_name = "NMRdb-CASCADEset_"+ model_type +"_mean_model_atom_features256.hdf5"

    trans_model = load_model( str(Path(Settings.ScriptDir) /  trans_model_name), custom_objects={'GraphModel': GraphModel,
                                                 'Squeeze': Squeeze,
                                                 'GatherAtomToBond': GatherAtomToBond,
                                                 'ReduceBondToAtom': ReduceBondToAtom,
                                                 'ReduceAtomToPro': ReduceAtomToPro})

    model = GraphModel([
        atom_index, atom_types, distance_rbf, connectivity, n_pro], [atom_state])

    model.compile()

    ####### transfer weights from dft model by name:

    model_layers = [ n.name for n in model.layers]

    for i, l in enumerate(model.layers):

        l.trainable = True

    for i, trans_layer in enumerate(trans_model.layers):

        if trans_layer.name in model_layers:

            model_layer = model.get_layer(name=trans_layer.name)

            model_layer.set_weights(trans_layer.get_weights())

    return model

def extract_Error_reps(model,test,Settings):

    batch_size = 1

    preprocessor = pickle.load(open( Path(Settings.ScriptDir) /  "mean_model_preprocessor.p","rb"))

    inputs_test = preprocessor.predict(Mol_iter2(test))

    test_sequence = RBFSequence(inputs_test, test.atom_index, batch_size)

    pca = pickle.load(open(Path(Settings.ScriptDir) / "pca_10_ERRORrep_Error_decomp.p", "rb"))

    reps = []

    for i,row in test.iterrows():

        yhat = model(test_sequence[i][0])

        r = [m for m in yhat.numpy()]

        X = pca.transform(r)

        reps.append(X)

        i+=1

    return reps
'''
def extract_Exp_reps(model,test,Settings):

    batch_size = 1

    preprocessor = pickle.load(open( Path(Settings.ScriptDir) /  "mean_model_preprocessor.p","rb"))

    inputs_test = preprocessor.predict(Mol_iter2(test))

    test_sequence = RBFSequence(inputs_test, test.atom_index, batch_size)

    pca = pickle.load(open(Path(Settings.ScriptDir) / "pca_10_EXP_decomp.p", "rb"))

    reps = []

    for i,row in test.iterrows():

        yhat = model(test_sequence[i][0])

        r = np.array([m for m in yhat.numpy()])

        X = pca.transform(r)

        reps.append(X)

        i+=1

    #do pca on the reps

    return reps

def build_Exp_predicting_model(Settings):

    filepath = "NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5"

    model = load_model( str(Path(Settings.ScriptDir) / filepath) , custom_objects={'GraphModel': GraphModel,
                                                 'Squeeze': Squeeze,
                                                 'GatherAtomToBond': GatherAtomToBond,
                                                 'ReduceBondToAtom': ReduceBondToAtom,
                                                 'ReduceAtomToPro': ReduceAtomToPro}, compile=False)

    model.compile()

    return model

def predict_shifts(model,test,Settings):

    batch_size = 1

    preprocessor = pickle.load(open( Path(Settings.ScriptDir) /  "mean_model_preprocessor.p","rb"))

    inputs_test = preprocessor.predict(Mol_iter2(test))

    test_sequence = RBFSequence(inputs_test, test.atom_index, batch_size)

    iso_shifts = []

    for i in test_sequence:

        yhat = model(i[0])

        shifts = np.array([m[0] for m in yhat.numpy()])

        iso_shifts.append(shifts)

    return iso_shifts

