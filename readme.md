An improved version of DP5 analysis developed by Howarth (DOI:[10.1039/D1SC04406K](https://doi.org/10.1039/D1SC04406K)). This codebase is refactored for legibility and maintainability.

We strongly recommend using a separate python environment created via `conda`, `uv`, or other solution of your choice to run this programme.
DP5 currently supports `python>=3.9,<=3.11` (due to the TensorFlow dependency range).

If you do not have `uv` installed yet:
- macOS (Homebrew): `brew install uv`
- Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `wget -qO- https://astral.sh/uv/install.sh | sh`)

To get started:
- clone this repository using `git clone https://github.com/ruslankotl/DP5.git`
- navigate to the folder on your machine
- create and activate a compatible environment, for example `uv venv --python 3.10 .venv && source .venv/bin/activate`
- install via `uv pip install -e .` 
- to also install documentation build dependencies, run `uv pip install -e ".[dev]"`
- run `pydp4 -s <SD_FILE> -n <NMR_FILE> -w w`

NMR files are provided as a list of shifts with assignments, e.g., `
153.0(any),125.6(any)`
Further documentation for workflow options is available [here](https://ruslankotl.github.io/DP5/)

Original DP5 code can be found at [https://github.com/Goodman-lab/DP5](https://github.com/Goodman-lab/DP5)
