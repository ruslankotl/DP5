import os
import tempfile
import unittest
from pathlib import Path

from dp5.load_config import (
    _resolve_cli_path_list,
    _resolve_dft_workdir,
    _resolve_output_folder,
    _resolve_path_list,
)


class _WorkingDirectory:
    def __init__(self, path: Path):
        self.path = path
        self.original = None

    def __enter__(self):
        self.original = Path.cwd()
        os.chdir(self.path)

    def __exit__(self, exc_type, exc, tb):
        os.chdir(self.original)


class LoadConfigPathResolutionTests(unittest.TestCase):
    def test_config_relative_inputs_resolve_from_config_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            resolved = _resolve_path_list(["inputs/mol.sdf"], config_dir)

            self.assertEqual(resolved, [str((config_dir / "inputs/mol.sdf").resolve())])

    def test_cli_inputs_resolve_from_invocation_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "run"
            workdir.mkdir()

            with _WorkingDirectory(workdir):
                resolved = _resolve_cli_path_list(["inputs/mol.sdf"])

            self.assertEqual(resolved, [str((workdir / "inputs/mol.sdf").resolve())])

    def test_default_output_folder_uses_first_structure_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            structure_dir = Path(tmpdir) / "structures"
            structure_dir.mkdir()
            structure = structure_dir / "mol.sdf"
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            output_dir = _resolve_output_folder("", "", [str(structure)], config_dir)

            self.assertEqual(output_dir, structure_dir.resolve())

    def test_default_output_folder_requires_structure_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            with self.assertRaisesRegex(
                ValueError, "Cannot infer output folder without any input paths"
            ):
                _resolve_output_folder("", "", [], config_dir)

    def test_default_output_folder_falls_back_to_other_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            nmr_dir = Path(tmpdir) / "nmr"
            nmr_dir.mkdir()

            output_dir = _resolve_output_folder(
                "", "", [], config_dir, [str(nmr_dir / "sample.dx")]
            )

            self.assertEqual(output_dir, nmr_dir.resolve())

    def test_dft_workdir_prefers_configured_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            output_dir = (Path(tmpdir) / "output").resolve()

            workdir = _resolve_dft_workdir("scratch/dft", output_dir, config_dir)

            self.assertEqual(workdir, str((config_dir / "scratch/dft").resolve()))


if __name__ == "__main__":
    unittest.main()
