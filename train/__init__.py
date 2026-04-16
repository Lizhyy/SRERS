"""Training package for SRERS.

This package intentionally avoids importing the full training stack at module
import time, because some submodules require optional runtime dependencies.
"""

__all__ = ["SRERSTrainer", "GANTrainer"]


def __getattr__(name):
    if name in {"SRERSTrainer", "GANTrainer"}:
        from .trainer import SRERSTrainer, GANTrainer
        return {"SRERSTrainer": SRERSTrainer, "GANTrainer": GANTrainer}[name]
    raise AttributeError(name)
