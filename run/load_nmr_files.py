"""expands NMRFile for input"""
from pathlib import Path

def find_nmr_files(nmr_path: str) -> list[Path]:
    """
    takes path for nmr file, returns the path of relevant files

    arguments:
    - nmr_path: relative path to NMRFile

    returns:
    list of paths to files 
    """

    _path = (Path(nmr_path)).resolve()
    nmr_paths = []

    if _path.is_dir():
        c_path = ''
        h_path = ''
        for file in _path.iterdir():
            if file.name.lower() == 'carbon' or file.name.lower()=='carbon.dx':
                c_path = file
            elif file.name.lower() == 'proton' or file.name.lower() == 'proton.dx':
                h_path = file
            if c_path and h_path:
                break
        nmr_paths = [c_path, h_path]

    else:
        nmr_paths.append(_path)

    return nmr_paths
