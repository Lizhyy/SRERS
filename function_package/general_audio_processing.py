import numpy as np


def sig_energy(signal, rms=False):
    """Return signal energy or RMS-style energy used by the SRIR parameter encoder."""
    signal = np.asarray(signal)
    energy = np.sum(signal ** 2)
    if rms:
        return np.sqrt(energy / signal.size).astype('float32')
    return np.array(energy, dtype='float32')
