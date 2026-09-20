"""Implements the graph neural network backend for shift simulation and DP5."""

from __future__ import annotations

import io
import json
import logging
import pickle
import tempfile
import warnings
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


def rbf_expansion(distances, mu=0, delta=0.1, kmax=256):
    k = np.arange(0, kmax)
    logits = -((np.atleast_2d(distances).T - (-mu + delta * k)) ** 2) / delta
    return np.exp(logits)


def _atomic_number_tokenizer(atom):
    """Suspicious code, likely redundant."""
    return atom.GetNumRadicalElectrons()


def _compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


def _as_1d_numpy(array_like, dtype=None):
    arr = np.asarray(array_like, dtype=dtype)
    return arr.reshape(-1)


def _resolve_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return Path(__file__).parent / path


def _segment_sum(values, index, dim_size):
    output_shape = (dim_size,) + tuple(values.shape[1:])
    output = values.new_zeros(output_shape)
    if dim_size == 0 or values.numel() == 0:
        return output
    valid = index >= 0
    if torch.any(valid):
        output.index_add_(0, index[valid], values[valid])
    return output


def _segment_mean(values, index, dim_size):
    sums = _segment_sum(values, index, dim_size)
    counts = values.new_zeros(dim_size)
    valid = index >= 0
    if torch.any(valid):
        counts.index_add_(0, index[valid], torch.ones_like(index[valid], dtype=values.dtype))
    view_shape = (dim_size,) + (1,) * (values.dim() - 1)
    return sums / counts.clamp_min(1).view(view_shape)


def _keras_hdf5_weights(path):
    with h5py.File(path, "r") as handle:
        root = handle["model_weights"]
        weights = {}
        for layer_name in root.keys():
            layer_group = root[layer_name]
            if layer_name in layer_group:
                layer_group = layer_group[layer_name]
            layer_weights = []
            if "kernel:0" in layer_group:
                layer_weights.append(np.array(layer_group["kernel:0"]))
            if "bias:0" in layer_group:
                layer_weights.append(np.array(layer_group["bias:0"]))
            if "embeddings:0" in layer_group:
                layer_weights.append(np.array(layer_group["embeddings:0"]))
            if layer_weights:
                weights[layer_name] = layer_weights
        return weights


def _keras_v3_weights_from_bytes(model_keras_bytes):
    with zipfile.ZipFile(io.BytesIO(model_keras_bytes)) as model_zip:
        weights_bytes = model_zip.read("model.weights.h5")
        with h5py.File(io.BytesIO(weights_bytes), "r") as handle:
            root = handle["_layer_checkpoint_dependencies"]
            weights = {}
            for layer_name in root.keys():
                layer_group = root[layer_name]
                if "vars" not in layer_group:
                    continue
                var_group = layer_group["vars"]
                layer_weights = [
                    np.array(var_group[key]) for key in sorted(var_group.keys(), key=int)
                ]
                if layer_weights:
                    weights[layer_name] = layer_weights
            return weights


def _keras_v3_model_bytes(path):
    return Path(path).read_bytes()


def _model_config_from_keras_bytes(model_keras_bytes):
    with zipfile.ZipFile(io.BytesIO(model_keras_bytes)) as model_zip:
        return json.loads(model_zip.read("config.json"))


def _assign_linear(module, weights):
    with torch.no_grad():
        kernel = torch.as_tensor(weights[0], dtype=module.weight.dtype)
        module.weight.copy_(kernel.T)
        if module.bias is not None and len(weights) > 1:
            module.bias.copy_(torch.as_tensor(weights[1], dtype=module.bias.dtype))


def _assign_embedding(module, weights):
    with torch.no_grad():
        module.weight.copy_(torch.as_tensor(weights[0], dtype=module.weight.dtype))


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class GraphBatchSequence:
    def __init__(self, inputs, y=None, batch_size=1, shuffle=True, final_batch=True):
        self._inputs = list(inputs)
        self._y = np.asarray(y, dtype=object) if y is not None else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.final_batch = final_batch

    def __len__(self):
        if self.final_batch:
            return int(np.ceil(len(self._inputs) / float(self.batch_size)))
        return int(np.floor(len(self._inputs) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(0, len(self._inputs))
            np.random.shuffle(indices)
            self._inputs = [self._inputs[i] for i in indices]
            if self._y is not None:
                self._y = self._y[indices]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(idx)
        batch_indexes = idx * self.batch_size + np.arange(0, self.batch_size)
        batch_indexes = batch_indexes[batch_indexes < len(self._inputs)]
        graphs = [_graph_from_input(self._inputs[i]) for i in batch_indexes]
        batch = Batch.from_data_list(graphs)
        if self._y is None:
            return batch
        targets = np.concatenate(
            [np.asarray(self._y[i], dtype=np.float32).reshape(-1) for i in batch_indexes]
        ).reshape(-1, 1)
        return batch, torch.as_tensor(targets, dtype=torch.float32)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class MessageBlock(nn.Module):
    def __init__(self, layer_names, atom_features=256):
        super().__init__()
        self.layer_names = layer_names
        self.atom_projection = nn.Linear(atom_features, atom_features, bias=False)
        self.edge_dense1 = nn.Linear(atom_features * 3, atom_features * 2)
        self.edge_dense2 = nn.Linear(atom_features * 2, atom_features)
        self.edge_dense3 = nn.Linear(atom_features, atom_features)
        self.edge_dense4 = nn.Linear(atom_features, atom_features)
        self.atom_dense1 = nn.Linear(atom_features, atom_features)
        self.atom_dense2 = nn.Linear(atom_features, atom_features)

    def keras_layers(self):
        return {
            self.layer_names[0]: self.atom_projection,
            self.layer_names[1]: self.edge_dense1,
            self.layer_names[2]: self.edge_dense2,
            self.layer_names[3]: self.edge_dense3,
            self.layer_names[4]: self.edge_dense4,
            self.layer_names[5]: self.atom_dense1,
            self.layer_names[6]: self.atom_dense2,
        }

    def forward(self, atom_state, bond_state, edge_index):
        atom_state = self.atom_projection(atom_state)
        source_atom = atom_state[edge_index[1]]
        target_atom = atom_state[edge_index[0]]

        bond_state_message = torch.cat([source_atom, target_atom, bond_state], dim=1)
        bond_state_message = F.softplus(self.edge_dense1(bond_state_message))
        bond_state_message = self.edge_dense2(bond_state_message)
        bond_state_message = F.softplus(self.edge_dense3(bond_state_message))
        bond_state_message = F.softplus(self.edge_dense4(bond_state_message))
        bond_state = bond_state + bond_state_message

        messages = source_atom * bond_state
        messages = _segment_sum(messages, edge_index[0], atom_state.shape[0])
        messages = F.softplus(self.atom_dense1(messages))
        messages = self.atom_dense2(messages)
        atom_state = atom_state + messages

        return atom_state, bond_state


class CascadeGraphModel(nn.Module):
    def __init__(self, atom_classes=13, atom_features=256, loc_output_dim=1):
        super().__init__()
        self.atom_features = atom_features
        self.atom_embedding = nn.Embedding(atom_classes, atom_features)
        self.message_blocks = nn.ModuleList(
            [
                MessageBlock(
                    ["dense", "dense_1", "dense_2", "dense_3", "dense_4", "dense_5", "dense_6"],
                    atom_features=atom_features,
                ),
                MessageBlock(
                    [
                        "dense_7",
                        "dense_8",
                        "dense_9",
                        "dense_10",
                        "dense_11",
                        "dense_12",
                        "dense_13",
                    ],
                    atom_features=atom_features,
                ),
                MessageBlock(
                    [
                        "dense_14",
                        "dense_15",
                        "dense_16",
                        "dense_17",
                        "dense_18",
                        "dense_19",
                        "dense_20",
                    ],
                    atom_features=atom_features,
                ),
            ]
        )
        self.loc_1 = nn.Linear(atom_features, 256)
        self.loc_2 = nn.Linear(256, 256)
        self.loc_3 = nn.Linear(256, 128)
        self.loc_reduce = nn.Linear(128, loc_output_dim)
        self.atomwise_shift = nn.Embedding(atom_classes, 1)

    def keras_layers(self):
        layers = {
            "atom_embedding": self.atom_embedding,
            "loc_1": self.loc_1,
            "loc_2": self.loc_2,
            "loc_3": self.loc_3,
            "loc_reduce": self.loc_reduce,
            "atomwise_shift": self.atomwise_shift,
        }
        for block in self.message_blocks:
            layers.update(block.keras_layers())
        return layers

    def encode(self, batch):
        atom_state = self.atom_embedding(batch.atom_types.long())
        bond_state = batch.edge_attr.float()
        edge_index = batch.edge_index.long()

        for block in self.message_blocks:
            atom_state, bond_state = block(atom_state, bond_state, edge_index)

        target_index, num_targets = _global_target_index(batch)
        representations = _segment_mean(atom_state, target_index, num_targets)
        atomwise_shift = self.atomwise_shift(batch.atom_types.long())
        atomwise_shift = _segment_mean(atomwise_shift, target_index, num_targets)

        return representations, atomwise_shift

    def forward(self, batch):
        representations, atomwise_shift = self.encode(batch)
        x = F.softplus(self.loc_1(representations))
        x = F.softplus(self.loc_2(x))
        x = F.softplus(self.loc_3(x))
        x = self.loc_reduce(x)
        return x + atomwise_shift


class RepresentationModel(nn.Module):
    def __init__(self, graph_model):
        super().__init__()
        self.graph_model = graph_model

    def forward(self, batch):
        representations, _ = self.graph_model.encode(batch)
        return representations


class PercentileMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.loc_1 = nn.Linear(256, 256)
        self.loc_2 = nn.Linear(256, 256)
        self.loc_3 = nn.Linear(256, 128)
        self.loc_reduce = nn.Linear(128, dims)
        self.workaround = nn.Linear(dims, dims)

    def forward(self, inputs):
        x = F.softplus(self.loc_1(inputs))
        x = F.softplus(self.loc_2(x))
        x = F.softplus(self.loc_3(x))
        x = self.loc_reduce(x)
        return self.workaround(x)


def _global_target_index(batch):
    n_pro = batch.n_pro.view(-1).long()
    offsets = torch.cumsum(
        torch.cat([n_pro.new_zeros(1), n_pro[:-1]], dim=0),
        dim=0,
    )
    target_index = batch.target_map.long().clone()
    valid = target_index >= 0
    if torch.any(valid):
        target_index[valid] += offsets[batch.batch[valid]]
    return target_index, int(n_pro.sum().item())


def _graph_from_input(raw_input):
    graph_input = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in raw_input.items()
    }
    if "distance_rbf" not in graph_input:
        graph_input["distance_rbf"] = rbf_expansion(graph_input["distance"])
    atom_types = _as_1d_numpy(graph_input["atom"], dtype=np.int64)
    target_map = _as_1d_numpy(graph_input["atom_index"], dtype=np.int64)
    connectivity = np.asarray(graph_input["connectivity"], dtype=np.int64).reshape(-1, 2)
    edge_attr = np.asarray(graph_input["distance_rbf"], dtype=np.float32).reshape(
        connectivity.shape[0], -1
    )
    n_pro = int(np.asarray(graph_input["n_pro"]).sum())
    return Data(
        atom_types=torch.as_tensor(atom_types, dtype=torch.long),
        edge_index=torch.as_tensor(connectivity.T, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32),
        target_map=torch.as_tensor(target_map, dtype=torch.long),
        n_pro=torch.tensor([n_pro], dtype=torch.long),
        num_nodes=int(atom_types.shape[0]),
    )


def _load_graph_state_dict_into_model(model, layer_weights):
    keras_layers = model.keras_layers()
    for layer_name, layer in keras_layers.items():
        if layer_name not in layer_weights:
            continue
        weights = layer_weights[layer_name]
        if isinstance(layer, nn.Embedding):
            _assign_embedding(layer, weights)
        elif isinstance(layer, nn.Linear):
            _assign_linear(layer, weights)
    model.eval()
    return model


def _load_graph_model(path):
    path = _resolve_path(path)
    if path.suffix == ".hdf5":
        model = CascadeGraphModel(loc_output_dim=1)
        weights = _keras_hdf5_weights(path)
        return _load_graph_state_dict_into_model(model, weights)
    if path.suffix == ".keras":
        keras_bytes = _keras_v3_model_bytes(path)
        config = _model_config_from_keras_bytes(keras_bytes)
        layers = {layer["name"]: layer for layer in config["config"]["layers"]}
        output_dim = layers["loc_reduce"]["config"]["units"]
        model = CascadeGraphModel(loc_output_dim=output_dim)
        weights = _keras_v3_weights_from_bytes(keras_bytes)
        return _load_graph_state_dict_into_model(model, weights)
    if path.suffix == ".pt":
        checkpoint = torch.load(path, map_location="cpu")
        loc_output_dim = checkpoint["loc_output_dim"]
        model = CascadeGraphModel(loc_output_dim=loc_output_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zipf:
            if "model.pt" in zipf.namelist():
                payload = torch.load(io.BytesIO(zipf.read("model.pt")), map_location="cpu")
                model = CascadeGraphModel(loc_output_dim=payload["loc_output_dim"])
                model.load_state_dict(payload["state_dict"])
                model.eval()
                return model
            keras_bytes = zipf.read("model.keras")
            config = _model_config_from_keras_bytes(keras_bytes)
            layers = {layer["name"]: layer for layer in config["config"]["layers"]}
            output_dim = layers["loc_reduce"]["config"]["units"]
            model = CascadeGraphModel(loc_output_dim=output_dim)
            weights = _keras_v3_weights_from_bytes(keras_bytes)
            return _load_graph_state_dict_into_model(model, weights)
    raise ValueError(f"Unsupported model format: {path}")


def _load_mlp_weights(path):
    path = _resolve_path(path)
    if path.suffix == ".pt":
        checkpoint = torch.load(path, map_location="cpu")
        model = PercentileMLP(checkpoint["dims"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
    if path.suffix == ".keras":
        keras_bytes = _keras_v3_model_bytes(path)
        weights = _keras_v3_weights_from_bytes(keras_bytes)
        config = _model_config_from_keras_bytes(keras_bytes)
        layers = {layer["name"]: layer for layer in config["config"]["layers"]}
        dims = layers["loc_reduce"]["config"]["units"]
        model = PercentileMLP(dims)
        for layer_name in ["loc_1", "loc_2", "loc_3", "loc_reduce", "workaround"]:
            if layer_name in weights:
                _assign_linear(getattr(model, layer_name), weights[layer_name])
        model.eval()
        return model
    raise ValueError(f"Unsupported percentile model format: {path}")


def _torch_save_bytes(payload):
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def Mol_iter(dfr):
    for index, row in dfr.iterrows():
        yield (row["Shift"], index)


def Mol_iter2(df):
    for index, row in df.iterrows():
        yield (row["Mol"], row["atom_index"])


def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()


def build_model(model_file):
    """Return the atomic-representation model derived from a prediction checkpoint."""
    return RepresentationModel(load_NMR_prediction_model(model_file))


def extract_representations(model, test, batch_size):
    with open(Path(__file__).parent / "mean_model_preprocessor.p", "rb") as handle:
        preprocessor = pickle.load(handle)
    inputs_test = preprocessor.predict(Mol_iter2(test))
    test_sequence = GraphBatchSequence(inputs_test, test.atom_index, batch_size)
    reps_list = []
    device = _model_device(model)
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            for batch in test_sequence:
                yhat = model(batch.to(device))
                indices = batch.n_pro.cpu().numpy().cumsum()[:-1]
                reps = yhat.detach().cpu().numpy()
                reps_list.extend(np.split(reps, indices))
    return reps_list


def extract_Error_reps(model, test, Settings):
    batch_size = 1
    preprocessor = pickle.load(
        open(Path(Settings.ScriptDir) / "mean_model_preprocessor.p", "rb")
    )
    inputs_test = preprocessor.predict(Mol_iter2(test))
    test_sequence = GraphBatchSequence(inputs_test, test.atom_index, batch_size)
    pca = pickle.load(
        open(Path(Settings.ScriptDir) / "pca_10_ERRORrep_Error_decomp.p", "rb")
    )

    reps = []
    device = _model_device(model)
    model.eval()
    with torch.no_grad():
        for i, row in test.iterrows():
            yhat = model(test_sequence[i].to(device))
            values = [m for m in yhat.detach().cpu().numpy()]
            reps.append(pca.transform(values))
    return reps


def extract_Exp_reps(model, test, Settings):
    batch_size = 1
    preprocessor = pickle.load(
        open(Path(Settings.ScriptDir) / "mean_model_preprocessor.p", "rb")
    )
    inputs_test = preprocessor.predict(Mol_iter2(test))
    test_sequence = GraphBatchSequence(inputs_test, test.atom_index, batch_size)
    pca = pickle.load(open(Path(Settings.ScriptDir) / "pca_10_EXP_decomp.p", "rb"))

    reps = []
    device = _model_device(model)
    model.eval()
    with torch.no_grad():
        for i, row in test.iterrows():
            yhat = model(test_sequence[i].to(device))
            values = np.array([m for m in yhat.detach().cpu().numpy()])
            reps.append(pca.transform(values))
    return reps


def load_NMR_prediction_model(
    filepath="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
):
    """Load the pretrained shift-prediction graph model."""
    return _load_graph_model(filepath)


def load_quantile_model(filepath="CASCADE_quantile_extended.keras"):
    """Load the pretrained graph quantile model."""
    return _load_graph_model(filepath)


def get_shifts_and_labels(mols, atomic_symbol, model_path, batch_size=16):
    model = load_NMR_prediction_model(model_path)
    logger.info("Loaded NMR prediction model")
    all_df, all_labels = mols_to_df(mols, atomic_symbol)
    logger.info(f"Ready to predict shifts for {atomic_symbol}")
    all_shifts = predict_shifts(model, all_df, batch_size=batch_size)
    return all_shifts, all_labels


def mols_to_df(mols, atomic_symbol):
    all_labels = []
    all_df = []

    for mol_id, mol in enumerate(mols):
        inds = [at.GetIdx() for at in mol[0].GetAtoms() if at.GetSymbol() == atomic_symbol]
        mol_labels = [f"{atomic_symbol}{i+1}" for i in inds]
        all_labels.append(mol_labels)

        for conf_id, conf in enumerate(mol):
            all_df.append((mol_id, conf_id, conf, np.array(inds)))

    all_df = pd.DataFrame(all_df, columns=["mol_id", "conf_id", "Mol", "atom_index"])
    return all_df, all_labels


def predict_shifts(model, test, batch_size=16):
    """Predict shifts for all conformers of all molecules."""
    preprocessor = pickle.load(open(Path(__file__).parent / "mean_model_preprocessor.p", "rb"))
    inputs_test = preprocessor.predict(Mol_iter2(test))
    test_sequence = GraphBatchSequence(inputs_test, test.atom_index, batch_size)

    iso_shifts = []
    device = _model_device(model)
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            for batch in test_sequence:
                yhat = model(batch.to(device))
                indices = batch.n_pro.cpu().numpy().cumsum()[:-1]
                shifts = yhat.detach().cpu().numpy().flatten()
                iso_shifts.extend(np.split(shifts, indices))

    test["shift_arrays"] = iso_shifts
    all_shifts = [
        np.stack(shifts) for i, shifts in test.groupby("mol_id", sort=False)["shift_arrays"]
    ]
    return all_shifts


class PercentileRegressor:
    def __init__(self, model, quantiles):
        self.quantiles = np.array(quantiles, dtype=float)
        self.dims = len(self.quantiles)
        self.model = model

    def save(self, archive_path):
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            array_buffer = io.BytesIO()
            np.save(array_buffer, self.quantiles)
            zipf.writestr("array.npy", array_buffer.getvalue())
            zipf.writestr(
                "model.pt",
                _torch_save_bytes(
                    {
                        "dims": self.dims,
                        "state_dict": self.model.state_dict(),
                    }
                ),
            )

    @classmethod
    def load(cls, archive_path):
        with zipfile.ZipFile(archive_path, "r") as zipf:
            arr = np.load(io.BytesIO(zipf.read("array.npy")))
            if "model.pt" in zipf.namelist():
                payload = torch.load(io.BytesIO(zipf.read("model.pt")), map_location="cpu")
                model = PercentileMLP(payload["dims"])
                model.load_state_dict(payload["state_dict"])
                model.eval()
                return cls(model, arr)
            model_bytes = zipf.read("model.keras")
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = Path(temp_dir) / "model.keras"
                model_path.write_bytes(model_bytes)
                return cls(_load_mlp_weights(model_path), arr)

    @classmethod
    def from_cascade(
        cls,
        quantiles,
        model_path="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
    ):
        quantiles = np.sort(np.array(quantiles).reshape(-1))
        dims = len(quantiles)
        full_model = load_NMR_prediction_model(model_path)
        model = PercentileMLP(dims)
        model.loc_1.load_state_dict(full_model.loc_1.state_dict())
        model.loc_2.load_state_dict(full_model.loc_2.state_dict())
        model.loc_3.load_state_dict(full_model.loc_3.state_dict())
        with torch.no_grad():
            model.loc_reduce.weight.copy_(full_model.loc_reduce.weight.repeat(dims, 1))
            model.loc_reduce.bias.copy_(full_model.loc_reduce.bias.repeat(dims))
            mean_shift = full_model.atomwise_shift.weight[2, 0].item()
            model.workaround.weight.copy_(torch.eye(dims))
            model.workaround.bias.copy_(torch.full((dims,), mean_shift))
        model.eval()
        return cls(model, quantiles)

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Training is not yet implemented for the PyTorch backend.")

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.as_tensor(args[0], dtype=torch.float32)
            outputs = self.model(inputs)
            return outputs.detach().cpu().numpy()


def QuantileLoss(perc, delta=1e-4):
    percentiles = torch.as_tensor(np.sort(np.array(perc).reshape(-1)), dtype=torch.float32)
    percentiles = percentiles.reshape(1, -1)

    def _qloss(y, pred):
        y_tensor = torch.as_tensor(y, dtype=pred.dtype, device=pred.device)
        indicator = (y_tensor <= pred).to(pred.dtype)
        delta_tensor = torch.as_tensor(delta, dtype=pred.dtype, device=pred.device)
        distance = torch.abs(y_tensor - pred)
        correction = indicator * (1 - percentiles.to(pred.device)) + (1 - indicator) * percentiles.to(
            pred.device
        )
        huber_loss = torch.sum(
            correction
            * torch.where(
                distance <= delta_tensor,
                0.5 * distance**2 / delta_tensor,
                distance - 0.5 * delta_tensor,
            ),
            dim=-1,
        )
        order_loss = torch.sum(torch.relu(pred[:, :-1] - pred[:, 1:] + 1e-6), dim=-1)
        return huber_loss + order_loss

    return _qloss


class CASCADE_Quantile:
    def __init__(self, model, quantiles):
        self.model = model
        self.quantiles = np.array(quantiles, dtype=float)

    @classmethod
    def from_cascade(
        cls,
        quantiles,
        model_path,
    ):
        full_model = load_NMR_prediction_model(model_path)
        quantiles = np.sort(np.array(quantiles))
        dims = len(quantiles)
        model = CascadeGraphModel(loc_output_dim=dims)
        shared_state = {
            key: value
            for key, value in full_model.state_dict().items()
            if not key.startswith("loc_reduce.")
        }
        model.load_state_dict(shared_state, strict=False)
        with torch.no_grad():
            model.loc_reduce.weight.copy_(full_model.loc_reduce.weight.repeat(dims, 1))
            model.loc_reduce.bias.copy_(full_model.loc_reduce.bias.repeat(dims))
        model.eval()
        return cls(model, quantiles)

    def save(self, archive_path):
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            array_buffer = io.BytesIO()
            np.save(array_buffer, self.quantiles)
            zipf.writestr("array.npy", array_buffer.getvalue())
            zipf.writestr(
                "model.pt",
                _torch_save_bytes(
                    {
                        "loc_output_dim": self.model.loc_reduce.out_features,
                        "state_dict": self.model.state_dict(),
                    }
                ),
            )

    @classmethod
    def load(cls, archive_path):
        archive_path = _resolve_path(archive_path)
        with zipfile.ZipFile(archive_path, "r") as zipf:
            quantiles = np.load(io.BytesIO(zipf.read("array.npy")))
            if "model.pt" in zipf.namelist():
                payload = torch.load(io.BytesIO(zipf.read("model.pt")), map_location="cpu")
                model = CascadeGraphModel(loc_output_dim=payload["loc_output_dim"])
                model.load_state_dict(payload["state_dict"])
                model.eval()
                return cls(model, quantiles)

            model_keras = io.BytesIO(zipf.read("model.keras"))
            weights = _keras_v3_weights_from_bytes(model_keras.getvalue())
            model = CascadeGraphModel(loc_output_dim=len(quantiles))
            _load_graph_state_dict_into_model(model, weights)
            return cls(model, quantiles)

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Training is not yet implemented for the PyTorch backend.")

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        self.model.eval()
        with torch.no_grad():
            return self.model(*args, **kwargs)


__all__ = [
    "CASCADE_Quantile",
    "GraphBatchSequence",
    "Mol_iter",
    "Mol_iter2",
    "PercentileRegressor",
    "QuantileLoss",
    "atomic_number_tokenizer",
    "build_model",
    "extract_Error_reps",
    "extract_Exp_reps",
    "extract_representations",
    "get_shifts_and_labels",
    "load_NMR_prediction_model",
    "load_quantile_model",
    "mols_to_df",
    "predict_shifts",
    "rbf_expansion",
]
