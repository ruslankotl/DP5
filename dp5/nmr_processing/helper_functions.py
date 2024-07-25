from pathlib import Path
import warnings

import numpy as np
import nmrglue as ng


__all__ = ['read_bruker', 'read_jcamp']


def _read_fid(path, reading_func, udic_function):
    """Abstract function for format-specific parsers.
    Arguments:
    - path(str): path to the file/folder
    - reading_func: function that reads file
    - udic_function: function that generates udic
    Returns:
    - nucleus(str): nucleus
    - total_spectral_ydata: complex array containing FID Data
    - uc: nmrglue unit conversion object
    Reads the FID file using the supplied function, returns nucleus for 1D experiments, 
    normalised intensity data, and nmrglue unit conversion object that defines the x-axis.
    """
    dic, total_spectral_ydata = reading_func(path)
    udic0 = udic_function(dic, total_spectral_ydata)
    nucleus = _get_nucleus_from_udic(udic0)
    if nucleus == '1H':
        total_spectral_ydata = preprocess_proton(total_spectral_ydata)
    elif nucleus == '13C':
        total_spectral_ydata = preprocess_carbon(total_spectral_ydata)
    # second generation of udic required after zero-filling
    udic = udic_function(dic, total_spectral_ydata)
    uc = ng.fileiobase.uc_from_udic(udic)
    return nucleus, total_spectral_ydata, uc


def _read_jcamp(file):
    with warnings.catch_warnings(record=True) as warns:
        warnings.filterwarnings(
            "error", category=UserWarning, message='^NTUPLES: multiple')
        dic, total_spectral_ydata = ng.jcampdx.read(file)  # read file

    total_spectral_ydata = total_spectral_ydata[0] + \
        1j * total_spectral_ydata[1]
    total_spectral_ydata = ng.proc_base.ifft_positive(total_spectral_ydata)
    return dic, total_spectral_ydata


def _read_bruker(folder):
    dic, total_spectral_ydata = ng.bruker.read(folder)
    total_spectral_ydata = ng.bruker.remove_digital_filter(
        dic, total_spectral_ydata)
    return dic, total_spectral_ydata


def read_jcamp(file):
    return _read_fid(file, reading_func=_read_jcamp, udic_function=jcamp_guess_udic)


def _get_nucleus_from_udic(udic: dict):
    ndims = udic.get('ndim', 0)
    if ndims == 1:
        nucleus = udic[0]['label']
    else:
        raise NotImplementedError(f"{ndims}-dimensional spectra not supported")
    return nucleus


def read_bruker(folder):
    """Wrapper function for reading bruker dataset folder. Note that the ydata comes normalised"""
    return _read_fid(folder, reading_func=_read_bruker, udic_function=ng.bruker.guess_udic)


def preprocess(total_spectral_ydata: np.array, zero_filling: int) -> np.array:
    """Method-agnostic zero-filling, Fourier transform, and normalisation
    Arguments:
    - total_spectral_ydata: numpy array containing time domain FID
    - zero_filling(int): double length of the time domain this number of times
    Returns:
    - Normalised complex frequency-domain data, highest modulus set to 1
    """
    # Zero-filling – FID time domain doubled 4 times (16 times original length)
    total_spectral_ydata = ng.proc_base.zf_double(
        total_spectral_ydata, zero_filling)
    # Fourier transform into frequency domain
    total_spectral_ydata = ng.proc_base.fft_positive(total_spectral_ydata)

    # Normalise the modulus for the most intense peak to 1
    total_spectral_ydata = normalise_intensities(total_spectral_ydata)

    return total_spectral_ydata


def normalise_intensities(ydata: np.ndarray):
    """Scales array by the largest modulus of the values"""
    return ydata / np.abs(ydata).max()


def preprocess_proton(total_spectral_ydata):
    return preprocess(total_spectral_ydata, zero_filling=4)


def preprocess_carbon(total_spectral_ydata):
    return preprocess(total_spectral_ydata, zero_filling=2)


def jcamp_guess_udic(dic, data):
    """
    Guess parameters of universal dictionary from dic, data pair.
    Parameters
    ----------
    dic : dict
        Dictionary of JCAMP-DX parameters.
    data : ndarray
        Array of NMR data.
    Returns
    -------
    udic : dict
        Universal dictionary of spectral parameters.
    """

    # create an empty universal dictionary
    udic = ng.fileiobase.create_blank_udic(1)

    # update default values (currently only 1D possible)
    # "label"
    try:
        label_value = dic[".OBSERVENUCLEUS"][0].replace("^", "")
        udic[0]["label"] = label_value
    except KeyError:
        # sometimes INSTRUMENTAL PARAMETERS is used:
        try:
            label_value = dic["INSTRUMENTALPARAMETERS"][0].replace("^", "")
            udic[0]["label"] = label_value
        except KeyError:
            pass

    # "obs"
    obs_freq = None
    try:
        obs_freq = float(dic[".OBSERVEFREQUENCY"][0])
        udic[0]["obs"] = obs_freq
    except KeyError:
        pass

    # "size"
    if isinstance(data, list):
        data = data[0]  # if list [R,I]
    if data is not None:
        udic[0]["size"] = len(data)

    # "sw"
    # get firstx, lastx and unit
    firstx, lastx, isppm = ng.jcampdx._find_firstx_lastx(dic)

    # ppm data: convert to Hz
    if isppm:
        if obs_freq:
            firstx = firstx * obs_freq
            lastx = lastx * obs_freq
        else:
            firstx, lastx = (None, None)

    if firstx is not None and lastx is not None:
        udic[0]["sw"] = abs(lastx - firstx)

        # keys not found in standard&required JCAMP-DX keys and thus left default:
        # car, complex, encoding

        udic[0]["car"] = firstx - abs(lastx - firstx) / 2

    return udic


def generalised_lorentzian(
    x: float, mu: float, std: float, v: float, A: float
) -> float:
    """
    Calculate the value of a generalized Lorentzian function at given x values.

    The generalized Lorentzian function is defined as:
    y = A * ((1 - v) / (1 + ((x - mu) / (std / 2)) ** 2) + v * (1 + (((x - mu) / (std / 2)) ** 2) /
    (1 + ((x - mu) / (std / 2)) ** 2 + ((x - mu) / (std / 2)) ** 4))

    Parameters:
    - x (float or array-like): The x values at which to evaluate the function.
    - mu (float): The mean or center of the distribution.
    - std (float): The standard deviation of the distribution.
    - v (float): A parameter controlling the shape of the function.
    - A (float): The amplitude of the function.

    Returns:
    - float or array-like: The value(s) of the generalized Lorentzian function at the given x value(s).
    """
    x1 = (mu - x) / (std / 2)
    y = (1 - v) * (1 / (1 + (x1) ** 2)) + (v) * (
        (1 + ((x1) ** 2) / 2) / (1 + (x1) ** 2 + (x1) ** 4)
    )
    y *= A

    return y


def lorentzian(p, w, p0, A):
    return generalised_lorentzian(p, p0, w, 1, A)


def methyl_protons(mol):

    protons = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            nbrprotons = []
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    nbrprotons.append(f"H{nbr.GetIdx()+1}")
            if len(nbrprotons) == 3:
                protons.append(nbrprotons)

    return protons


def labile_protons(mol):
    """
    Counts protons in OH functional groups.

    Assumes those to be exchangeable

    arguments:
        - mol: rdkit Mol object
    returns:
        - count number of labile protons
    """

    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    count += 1

    return count


def proton_count(mol) -> int:
    """
    Counts number of hydrogen atoms in a moecule

    arguments:
        - mol: RDKit Mol object
    returns:
        - number of protons (integer)
    """
    return sum([1 for at in mol.GetAtoms() if at.GetAtomicNum() == 1])
