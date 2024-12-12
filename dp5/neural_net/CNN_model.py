"""Implements the graph convolutional neural network for shift simulation and DP5 probaility estimation"""

from dp5.neural_net.nfp.preprocessing import GraphSequence
from dp5.neural_net.nfp.models import GraphModel
from dp5.neural_net.nfp.layers import (
    Squeeze,
    ReduceBondToAtom,
    GatherAtomToBond,
    ReduceAtomToPro,
)
from pathlib import Path
import pickle
import logging
import warnings
import tempfile
import zipfile
import os

import numpy as np
from numpy.random import seed
import sys
import pandas as pd

# seed(1)
# from tensorflow.random import set_seed
# set_seed(2)

# Define Keras model
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, Dense, Add, Embedding, Concatenate, Multiply, Add
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

from dp5.neural_net import nfp

# essential for models to work!!!
sys.modules["nfp"] = nfp

logger = logging.getLogger(__name__)


def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -((np.atleast_2d(distances).T - (-mu + delta * k)) ** 2) / delta
    return np.exp(logits)


def _atomic_number_tokenizer(atom):
    """suspicious code, likely redundant"""
    return atom.GetNumRadicalElectrons()


def _compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


class RBFSequence(GraphSequence):
    def process_data(self, batch_data):
        batch_data["distance_rbf"] = rbf_expansion(batch_data["distance"])

        offset = _compute_stacked_offsets(batch_data["n_pro"], batch_data["n_atom"])

        offset = np.where(batch_data["atom_index"] >= 0, offset, 0)
        batch_data["atom_index"] += offset

        del batch_data["n_atom"]
        del batch_data["n_bond"]
        del batch_data["distance"]

        return batch_data


def Mol_iter(dfr):

    for index, r in dfr.iterrows():

        yield (r["Shift"], index)


def Mol_iter2(df):
    for index, r in df.iterrows():
        yield (r["Mol"], r["atom_index"])


def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()


def message_block(atom_features, atom_state, bond_state, connectivity):
    atom_state = Dense(atom_features, use_bias=False)(atom_state)

    source_atom_gather = GatherAtomToBond(1)
    target_atom_gather = GatherAtomToBond(0)

    source_atom = source_atom_gather([atom_state, connectivity])
    target_atom = target_atom_gather([atom_state, connectivity])

    # Edge update network
    bond_state_message = Concatenate()([source_atom, target_atom, bond_state])
    bond_state_message = Dense(2 * atom_features, activation="softplus")(
        bond_state_message
    )
    bond_state_message = Dense(atom_features)(bond_state_message)

    bond_state_message = Dense(atom_features, activation="softplus")(bond_state_message)
    bond_state_message = Dense(atom_features, activation="softplus")(bond_state_message)
    bond_state = Add()([bond_state_message, bond_state])

    # message function
    messages = Multiply()([source_atom, bond_state])
    messages = ReduceBondToAtom(reducer="sum")([messages, connectivity])

    # state transition function
    messages = Dense(atom_features, activation="softplus")(messages)
    messages = Dense(atom_features)(messages)
    atom_state = Add()([atom_state, messages])

    return atom_state, bond_state


def build_model(model_file):
    """This model returns internal atomic representations from NMR prediction model"""
    # Construct input sequences

    preprocessor = pickle.load(
        open(Path(__file__).parent / "mean_model_preprocessor.p", "rb")
    )

    # Raw (integer) graph inputs
    atom_index = Input(shape=(1,), name="atom_index", dtype="int32")
    atom_types = Input(shape=(1,), name="atom", dtype="int32")
    distance_rbf = Input(shape=(256,), name="distance_rbf", dtype="float32")
    connectivity = Input(shape=(2,), name="connectivity", dtype="int32")
    n_pro = Input(shape=(1,), name="n_pro", dtype="int32")

    squeeze = Squeeze()

    satom_index = squeeze(atom_index)
    satom_types = squeeze(atom_types)
    sn_pro = squeeze(n_pro)
    # Initialize RNN and MessageLayer instances
    atom_features = 256

    # Initialize the atom states
    atom_state = Embedding(
        preprocessor.atom_classes, atom_features, name="atom_embedding"
    )(satom_types)

    bond_state = distance_rbf

    for _ in range(3):
        atom_state, bond_state = message_block(
            atom_features, atom_state, bond_state, connectivity
        )

    atom_state = ReduceAtomToPro(reducer="unsorted_mean")(
        [atom_state, satom_index, sn_pro]
    )

    trans_model = load_model(
        str(Path(__file__).parent / model_file),
        custom_objects={
            "GraphModel": GraphModel,
            "Squeeze": Squeeze,
            "GatherAtomToBond": GatherAtomToBond,
            "ReduceBondToAtom": ReduceBondToAtom,
            "ReduceAtomToPro": ReduceAtomToPro,
        },
    )

    model = GraphModel(
        [atom_index, atom_types, distance_rbf, connectivity, n_pro], [atom_state]
    )

    model.compile()

    # transfer weights from dft model by name:

    model_layers = [n.name for n in model.layers]

    for i, l in enumerate(model.layers):

        l.trainable = True

    for i, trans_layer in enumerate(trans_model.layers):

        if trans_layer.name in model_layers:

            model_layer = model.get_layer(name=trans_layer.name)

            model_layer.set_weights(trans_layer.get_weights())

    return model


def extract_representations(model, test, batch_size):
    with open(Path(__file__).parent / "mean_model_preprocessor.p", "rb") as f:
        preprocessor = pickle.load(f)
    inputs_test = preprocessor.predict(Mol_iter2(test))
    test_sequence = RBFSequence(inputs_test, test.atom_index, batch_size)
    reps_list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in test_sequence:
            yhat = model(i[0])
            indices = i[0]["n_pro"].cumsum()[:-1]
            reps = yhat.numpy()
            reps = np.split(reps, indices)
            # now supports batches!
            reps_list.extend(reps)

    return reps_list
    # finish later


def extract_Error_reps(model, test, Settings):

    batch_size = 1

    preprocessor = pickle.load(
        open(Path(Settings.ScriptDir) / "mean_model_preprocessor.p", "rb")
    )

    inputs_test = preprocessor.predict(Mol_iter2(test))

    test_sequence = RBFSequence(inputs_test, test.atom_index, batch_size)

    pca = pickle.load(
        open(Path(Settings.ScriptDir) / "pca_10_ERRORrep_Error_decomp.p", "rb")
    )

    reps = []

    for i, row in test.iterrows():

        yhat = model(test_sequence[i][0])

        r = [m for m in yhat.numpy()]

        X = pca.transform(r)

        reps.append(X)

        i += 1

    return reps


def extract_Exp_reps(model, test, Settings):

    batch_size = 1

    preprocessor = pickle.load(
        open(Path(Settings.ScriptDir) / "mean_model_preprocessor.p", "rb")
    )

    inputs_test = preprocessor.predict(Mol_iter2(test))

    test_sequence = RBFSequence(inputs_test, test.atom_index, batch_size)

    pca = pickle.load(open(Path(Settings.ScriptDir) / "pca_10_EXP_decomp.p", "rb"))

    reps = []

    for i, row in test.iterrows():

        yhat = model(test_sequence[i][0])

        r = np.array([m for m in yhat.numpy()])

        X = pca.transform(r)

        reps.append(X)

        i += 1

    # do pca on the reps

    return reps


def load_NMR_prediction_model(
    filepath="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
):
    """
    Loads NMR predicting model. Make sure your model is in neural_net folder!
    """

    model = load_model(
        str(Path(__file__).parent / filepath),
        custom_objects={
            "GraphModel": GraphModel,
            "Squeeze": Squeeze,
            "GatherAtomToBond": GatherAtomToBond,
            "ReduceBondToAtom": ReduceBondToAtom,
            "ReduceAtomToPro": ReduceAtomToPro,
        },
    )
    return model


def load_quantile_model(
    filepath="CASCADE_quantile_extended.keras",
):
    """
    Loads NMR predicting mode with quantile regression. Make sure your model is in neural_net folder!
    """

    model = load_model(
        str(Path(__file__).parent / filepath),
        custom_objects={
            "GraphModel": GraphModel,
            "Squeeze": Squeeze,
            "GatherAtomToBond": GatherAtomToBond,
            "ReduceBondToAtom": ReduceBondToAtom,
            "ReduceAtomToPro": ReduceAtomToPro,
            "_qloss": QuantileLoss(np.linspace(0.01, 0.99, 99)),
        },
        compile=False,
    )
    # separate compilation required to suppress a warning
    # https://stackoverflow.com/questions/77150716/loading-keras-model-issues-warning-skipping-variable-loading-for-optimizer-ada
    model.compile()
    return model


def get_shifts_and_labels(mols, atomic_symbol, model_path, batch_size=16):
    model = load_NMR_prediction_model(model_path)
    logger.info("Loaded NMR prediction model")
    all_shifts = []

    all_df, all_labels = mols_to_df(mols, atomic_symbol)

    logger.info(f"Ready to predict shifts for {atomic_symbol}")
    all_shifts = predict_shifts(model, all_df, batch_size=batch_size)

    return all_shifts, all_labels


def mols_to_df(mols, atomic_symbol):
    all_labels = []
    all_df = []

    for mol_id, mol in enumerate(mols):
        inds = [
            at.GetIdx() for at in mol[0].GetAtoms() if at.GetSymbol() == atomic_symbol
        ]
        mol_labels = [f"{atomic_symbol}{i+1}" for i in inds]
        all_labels.append(mol_labels)

        for conf_id, conf in enumerate(mol):
            all_df.append((mol_id, conf_id, conf, np.array(inds)))

    all_df = pd.DataFrame(all_df, columns=["mol_id", "conf_id", "Mol", "atom_index"])

    return all_df, all_labels


def predict_shifts(model, test, batch_size=16):
    """Predicts the shifts for all conformers of all molecules.
    - model: prediction_model
    - test: dataframe containing mol_id, conf_id, Mol object and atom indices column
    - batch_size: batch_size
    Returns:
    - list(molecules) of lists(geometries) of lists(shifts)
    """
    preprocessor = pickle.load(
        open(Path(__file__).parent / "mean_model_preprocessor.p", "rb")
    )

    inputs_test = preprocessor.predict(Mol_iter2(test))

    test_sequence = RBFSequence(inputs_test, test.atom_index, batch_size)

    iso_shifts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in test_sequence:
            # returns (n, 1) shape array of n shifts
            yhat = model(i[0])
            indices = i[0]["n_pro"].cumsum()[:-1]
            # must flatten it for shifts
            shifts = yhat.numpy().flatten()
            shifts = np.split(shifts, indices)
            # now supports batches!
            iso_shifts.extend(shifts)

    test["shift_arrays"] = iso_shifts
    all_shifts = [
        np.stack(shifts) for i, shifts in test.groupby("mol_id")["shift_arrays"]
    ]

    return all_shifts


class PercentileRegressor:
    # create a @classmethod to initialise a model from CASCADE or load an existing one
    def __init__(self, model, quantiles):

        self.quantiles = quantiles
        self.dims = len(self.quantiles)
        self.model = model

    def save(self, archive_path):
        # Use tempfile to create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the NumPy array in the temporary directory
            array_path = os.path.join(temp_dir, "array.npy")
            np.save(array_path, self.quantiles)

            # Save the TensorFlow model in the temporary directory
            model_path = os.path.join(temp_dir, "model.keras")
            self.model.save(model_path)

            # Create a ZIP archive containing both the array and the model
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add the NumPy array to the ZIP archive
                zipf.write(array_path, "array.npy")
                zipf.write(model_path, "model.keras")

    @classmethod
    def load(cls, archive_path):
        # Use tempfile to extract the ZIP archive to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the archive contents into the temporary directory
            with zipfile.ZipFile(archive_path, "r") as zipf:
                zipf.extractall(temp_dir)

            # Load the NumPy array
            array_path = os.path.join(temp_dir, "array.npy")
            arr = np.load(array_path)

            # Load the TensorFlow model
            model_dir = os.path.join(temp_dir, "model.keras")
            model = tf.keras.models.load_model(
                model_dir, custom_objects={"_qloss": QuantileLoss(arr)}
            )

            return cls(model, arr)

    @classmethod
    def from_cascade(
        cls,
        quantiles,
        model_path="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
    ):
        dims = len(quantiles)
        full_model = load_NMR_prediction_model(model_path)
        initial_learning_rate = 5e-4
        lr_schedule = ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)
        mean_shift = full_model.get_layer(name="atomwise_shift").weights[0][2].numpy()

        input_rep = Input(shape=(256,), name="input_rep")
        x = full_model.get_layer("loc_1")(input_rep)
        x = full_model.get_layer("loc_2")(x)
        x = full_model.get_layer("loc_3")(x)
        x = Dense(dims, name="loc_reduce")(x)
        output = Dense(dims, name="workaround", trainable=False)(x)

        model = Model(input_rep, output)

        w, b = full_model.get_layer("loc_reduce").weights
        w_c = np.broadcast_to(w, shape=(w.shape[0], dims))
        b_c = np.broadcast_to(b, shape=(dims))
        model.get_layer("loc_reduce").set_weights([w_c, b_c])

        w_w, b_w = np.eye(dims), np.broadcast_to(mean_shift, shape=(dims))
        model.get_layer("workaround").set_weights([w_w, b_w])

        model.compile(optimizer=optimizer, loss=QuantileLoss(quantiles))
        class1 = cls(model, quantiles)
        return class1

    def fit(self, *args, **kwargs):
        early_stopping_monitor = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        return self.model.fit(*args, **kwargs, callbacks=[early_stopping_monitor])

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)

    @keras.saving.register_keras_serializable()
    def _qloss(y, pred):
        I = tf.cast(y <= pred, tf.float32)
        d = K.abs(y - pred)
        correction = I * (1 - perc) + (1 - I) * perc
        # huber loss
        huber_loss = K.sum(
            correction * tf.where(d <= delta, 0.5 * d**2 / delta, d - 0.5 * delta), -1
        )
        # order loss
        q_order_loss = K.sum(K.maximum(0.0, pred[:, :-1] - pred[:, 1:] + 1e-6), -1)
        return huber_loss + q_order_loss

    return _qloss


class CASCADE_Quantile:
    def __init__(
        self,
        model,
        quantiles,
    ):
        self.model = model
        self.quantiles = quantiles

    @classmethod
    def from_cascade(
        cls,
        quantiles,
        model_path,
    ):
        """
        Creates a quantile regressor out of CASCADE model. Requires training
        Arguments:
          quantiles: list or numpy array of quantiles to see.
          model_path: path to saved model. Saves all pretrained model weights
        """
        full_model = load_model(
            model_path,
            custom_objects={
                "GraphModel": GraphModel,
                "Squeeze": Squeeze,
                "GatherAtomToBond": GatherAtomToBond,
                "ReduceBondToAtom": ReduceBondToAtom,
                "ReduceAtomToPro": ReduceAtomToPro,
            },
        )

        quantiles = np.array(quantiles)
        quantiles = np.sort(quantiles)
        dims = len(quantiles)

        rep, atomwise_shift = (
            full_model.get_layer(name="loc_3").output,
            full_model.get_layer(name="reduce_atom_to_pro_1").output,
        )
        rep = Dense(dims, name="loc_reduce")(rep)
        output = Add(name="final_layer")([rep, atomwise_shift])

        model = GraphModel(inputs=full_model.input, outputs=output)

        w, b = full_model.get_layer("loc_reduce").weights
        w_c = np.broadcast_to(w, shape=(w.shape[0], dims))
        b_c = np.broadcast_to(b, shape=(dims))
        model.get_layer("loc_reduce").set_weights([w_c, b_c])

        initial_learning_rate = 5e-4
        lr_schedule = ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)

        model.compile(optimizer=optimizer, loss=QuantileLoss(quantiles))

        ready = cls(model, quantiles)

        return ready

    def save(self, archive_path):
        # Use tempfile to create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the NumPy array in the temporary directory
            array_path = os.path.join(temp_dir, "array.npy")
            np.save(array_path, self.quantiles)

            # Save the TensorFlow model in the temporary directory
            model_path = os.path.join(temp_dir, "model.keras")
            self.model.save(model_path)

            # Create a ZIP archive containing both the array and the model
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add the NumPy array to the ZIP archive
                zipf.write(array_path, "array.npy")
                zipf.write(model_path, "model.keras")

    @classmethod
    def load(cls, archive_path):
        # Use tempfile to extract the ZIP archive to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the archive contents into the temporary directory
            with zipfile.ZipFile(archive_path, "r") as zipf:
                zipf.extractall(temp_dir)

            # Load the NumPy array
            array_path = os.path.join(temp_dir, "array.npy")
            arr = np.load(array_path)

            # Load the TensorFlow model
            model_dir = os.path.join(temp_dir, "model.keras")
            model = tf.keras.models.load_model(
                model_dir, custom_objects={"_qloss": QuantileLoss(arr)}
            )

            return cls(model, arr)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model(*args, **kwargs)
