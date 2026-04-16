"""SRIR synthesis wrappers."""

from train.para_dataset.SRIR_decoder import (
    SRIR_Decoder as SRIRSynthesizer,
    full_RIR_dec as _legacy_full_rir_decoder,
)

def synthesize_full_srir(*args, **kwargs):
    """Compatibility wrapper around the original full-SRIR synthesis helper."""
    return _legacy_full_rir_decoder(*args, **kwargs)

__all__ = ["SRIRSynthesizer", "synthesize_full_srir"]
