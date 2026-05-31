import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CConfig import config

cfg = config()


def Polyhedron_4_microphone_loc():
    """Return the fixed 4-channel tetrahedral microphone geometry."""
    return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])


def x_batch_switch(x_batch, switch_type):
    """Apply the axis permutation used for GSLR face-graph augmentation."""
    return x_batch_switch_face(x_batch, switch_type)


def ebd_switch(ebd_list, switch_type):
    """Permute source/listener xyz embedding consistently with the graph augmentation."""
    if switch_type == 0:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[0, 1, 2, 3, 4, 5]]
    elif switch_type == 1:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[0, 2, 1, 3, 5, 4]]
    elif switch_type == 2:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[1, 0, 2, 4, 3, 5]]
    elif switch_type == 3:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[1, 2, 0, 4, 5, 3]]
    elif switch_type == 4:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[2, 0, 1, 5, 3, 4]]
    elif switch_type == 5:
        ebd_list[[0, 1, 2, 3, 4, 5]] = ebd_list[[2, 1, 0, 5, 4, 3]]
    return ebd_list


def x_batch_switch_face(x_batch, switch_type):
    """Permute face-graph geometric feature columns for the six xyz orders."""
    cols = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    if switch_type == 0:
        perm = cols
    elif switch_type == 1:
        perm = [4, 6, 5, 7, 9, 8, 10, 12, 11, 13, 15, 14]
    elif switch_type == 2:
        perm = [5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15]
    elif switch_type == 3:
        perm = [5, 6, 4, 8, 9, 7, 11, 12, 10, 14, 15, 13]
    elif switch_type == 4:
        perm = [6, 4, 5, 9, 7, 8, 12, 10, 11, 15, 13, 14]
    elif switch_type == 5:
        perm = [6, 5, 4, 9, 8, 7, 12, 11, 10, 15, 14, 13]
    else:
        raise ValueError(f'Unsupported switch_type: {switch_type}')
    x_batch[:, cols] = x_batch[:, perm]
    return x_batch


def RIR_batch_switch(RIR_batch, switch_type):
    """Permute tetrahedral SRIR channels consistently with axis augmentation."""
    if switch_type == 0:
        perm = [0, 1, 2, 3]
    elif switch_type == 1:
        perm = [0, 1, 3, 2]
    elif switch_type == 2:
        perm = [0, 2, 1, 3]
    elif switch_type == 3:
        perm = [0, 2, 3, 1]
    elif switch_type == 4:
        perm = [0, 3, 1, 2]
    elif switch_type == 5:
        perm = [0, 3, 2, 1]
    else:
        raise ValueError(f'Unsupported switch_type: {switch_type}')
    RIR_batch[:, [0, 1, 2, 3], :] = RIR_batch[:, perm, :]
    return RIR_batch


def ev_batch_switch(ev_batch, switch_type):
    """Permute late-reverb envelope channels consistently with SRIR channels."""
    if switch_type == 0:
        perm = [0, 1, 2, 3]
    elif switch_type == 1:
        perm = [0, 1, 3, 2]
    elif switch_type == 2:
        perm = [0, 2, 1, 3]
    elif switch_type == 3:
        perm = [0, 2, 3, 1]
    elif switch_type == 4:
        perm = [0, 3, 1, 2]
    elif switch_type == 5:
        perm = [0, 3, 2, 1]
    else:
        raise ValueError(f'Unsupported switch_type: {switch_type}')
    ev_batch[:, [0, 1, 2, 3], :, :] = ev_batch[:, perm, :, :]
    return ev_batch
