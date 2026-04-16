"""Lightweight import smoke test for the cleaned repository layout."""

import importlib.util

import srers
from srers.config import create_config


def test_public_package_imports_exist():
    assert srers.__version__
    assert create_config
    assert srers.build_sample_index


def test_heavy_symbols_when_dependencies_available():
    if importlib.util.find_spec("torch_geometric") is None:
        return

    from srers.data import SRERSParameterDataset
    from srers.engine import SRERSEvaluator, SRERSTrainer
    from srers.models import SceneEncoderModel, SRIRParameterDecoder

    assert SRERSParameterDataset
    assert SRERSTrainer
    assert SRERSEvaluator
    assert SceneEncoderModel
    assert SRIRParameterDecoder
