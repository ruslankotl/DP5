import nmrglue as ng

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

        udic[0]['car'] = firstx  -  abs(lastx - firstx)/2

    return udic

def lorentzian(p, w, p0, A):
    """
    Simulate lorentzian curve
    """
    x = (p0 - p) / (w / 2)
    L = A / (1 + x ** 2)

    return L

def generalised_lorentzian(x, mu, std, v, A):
    x1 = (mu - x) / (std / 2)
    y = (1 - v) * (1 / (1 + (x1) ** 2)) + (v) * ((1 + ((x1) ** 2) / 2) / (1 + (x1) ** 2 + (x1) ** 4))
    y *= A

    return y

def lorentzian(p, w, p0, A):
    return generalised_lorentzian(p, p0, w, 1, A)