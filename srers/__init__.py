"""SRERS public package interface.

This package provides a clearer, paper-aligned import surface on top of the
original research code, while keeping the legacy modules available.

Heavy training dependencies are imported lazily so that `import srers` works
without immediately importing the full training stack.
"""

from .config import create_config, save_config
from .utils.indexing import build_sample_index

__all__ = [
    "create_config",
    "save_config",
    "build_sample_index",
    "SRERSParameterDataset",
    "SRERSTrainer",
    "SRERSEvaluator",
    "SceneEncoderModel",
    "SRIRParameterDecoder",
    "compute_total_srir_loss",
    "compute_early_residual_loss",
    "compute_auxiliary_parameter_loss",
    "compute_late_reverb_loss",
    "synthesize_full_srir",
]

__version__ = "0.3.0"


def __getattr__(name):
    if name == "SRERSParameterDataset":
        from .data.dataset import SRERSParameterDataset
        return SRERSParameterDataset
    if name == "SRERSTrainer":
        from .engine.trainer import SRERSTrainer
        return SRERSTrainer
    if name == "SRERSEvaluator":
        from .engine.evaluator import SRERSEvaluator
        return SRERSEvaluator
    if name == "SceneEncoderModel":
        from .models.scene_encoder import SceneEncoderModel
        return SceneEncoderModel
    if name == "SRIRParameterDecoder":
        from .models.srir_decoder import SRIRParameterDecoder
        return SRIRParameterDecoder
    if name in {
        "compute_total_srir_loss",
        "compute_early_residual_loss",
        "compute_auxiliary_parameter_loss",
        "compute_late_reverb_loss",
    }:
        from .losses.srir_losses import (
            compute_total_srir_loss,
            compute_early_residual_loss,
            compute_auxiliary_parameter_loss,
            compute_late_reverb_loss,
        )
        return {
            "compute_total_srir_loss": compute_total_srir_loss,
            "compute_early_residual_loss": compute_early_residual_loss,
            "compute_auxiliary_parameter_loss": compute_auxiliary_parameter_loss,
            "compute_late_reverb_loss": compute_late_reverb_loss,
        }[name]
    if name == "synthesize_full_srir":
        from .synthesis.srir_synthesizer import synthesize_full_srir
        return synthesize_full_srir
    raise AttributeError(name)
