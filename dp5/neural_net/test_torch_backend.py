import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from dp5.neural_net.CNN_model import (
    CASCADE_Quantile,
    PercentileRegressor,
    build_model,
    load_NMR_prediction_model,
    load_quantile_model,
)


def _sample_batch():
    sample = Data(
        atom_types=torch.tensor([2, 3], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 256), dtype=torch.float32),
        target_map=torch.tensor([0, 1], dtype=torch.long),
        n_pro=torch.tensor([2], dtype=torch.long),
        num_nodes=2,
    )
    return Batch.from_data_list([sample])


class TorchBackendTests(unittest.TestCase):
    def test_load_shift_model(self):
        model = load_NMR_prediction_model()
        output = model(_sample_batch())
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_build_representation_model(self):
        model = build_model("NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5")
        output = model(_sample_batch())
        self.assertEqual(tuple(output.shape), (2, 256))

    def test_load_quantile_models(self):
        batch = _sample_batch()
        archive_path = (Path(__file__).parent / "NMRdb_CASCADE_99quantiles.zip").resolve()
        archive_model = CASCADE_Quantile.load(archive_path)
        archive_output = archive_model.model(batch)
        zip_model = load_quantile_model(archive_path)
        zip_output = zip_model(batch)

        with zipfile.ZipFile(archive_path, "r") as archive:
            with tempfile.TemporaryDirectory() as temp_dir:
                keras_path = Path(temp_dir) / "model.keras"
                keras_path.write_bytes(archive.read("model.keras"))
                direct_model = load_quantile_model(keras_path)

        direct_output = direct_model(batch)
        self.assertEqual(tuple(direct_output.shape), (2, 99))
        self.assertEqual(tuple(archive_output.shape), (2, 99))
        self.assertEqual(tuple(zip_output.shape), (2, 99))
        self.assertEqual(len(archive_model.quantiles), 99)
        self.assertTrue(torch.allclose(direct_output, archive_output))
        self.assertTrue(torch.allclose(zip_output, archive_output))

    def test_percentile_regressor_predict(self):
        regressor = PercentileRegressor.from_cascade([0.9, 0.1, 0.5])
        self.assertTrue(np.array_equal(regressor.quantiles, np.array([0.1, 0.5, 0.9])))
        outputs = regressor.predict(np.zeros((2, 256), dtype=np.float32))
        self.assertEqual(outputs.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
