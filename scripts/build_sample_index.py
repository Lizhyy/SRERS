#!/usr/bin/env python3
"""Wrapper script for build sample index."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from srers.cli.build_sample_index import main

if __name__ == '__main__':
    main()
