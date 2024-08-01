from setuptools import setup, find_namespace_packages

setup(
    name="DP5",
    version="0.1",
    packages=find_namespace_packages(),
    install_requires=[
        # List your dependencies here
        "tomli",
        "numpy",
        "rdkit",
        "lmfit",
        "matplotlib",
        "openbabel-wheel",
        "scipy<1.10",
        "nmrglue",
    ],
    entry_points={
        "console_scripts": ["pydp4 = dp5.run.load_config:main"],
    },
)
