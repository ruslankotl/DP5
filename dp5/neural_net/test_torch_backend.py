import unittest

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
        direct_model = load_quantile_model()
        direct_output = direct_model(_sample_batch())
        self.assertEqual(tuple(direct_output.shape), (2, 99))

        archive_model = CASCADE_Quantile.load("dp5/neural_net/NMRdb_CASCADE_99quantiles.zip")
        archive_output = archive_model.model(_sample_batch())
        self.assertEqual(tuple(archive_output.shape), (2, 99))
        self.assertEqual(len(archive_model.quantiles), 99)

    def test_percentile_regressor_predict(self):
        regressor = PercentileRegressor.from_cascade([0.1, 0.5, 0.9])
        outputs = regressor.predict(np.zeros((2, 256), dtype=np.float32))
        self.assertEqual(outputs.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
