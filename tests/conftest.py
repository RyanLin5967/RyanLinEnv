"""Refuse test modules that would trip the live production freshpool study's owner scan.

See scripts/freshpool_launch/production/live.py: while the study's state is ``running``, a
session that collected any module naming an accelerator-owner pattern exits with code 2
before running anything, naming the modules. Other modules run normally.
"""

import importlib.util
from pathlib import Path

import pytest

LIVE = Path(__file__).resolve().parents[1] / "scripts" / "freshpool_launch" / "production" / (
    "live.py")


def _live():
    spec = importlib.util.spec_from_file_location("freshpool_production_live", LIVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_collection_modifyitems(session, config, items):
    if not LIVE.is_file():
        return
    live = _live()
    state = live.study_running()
    if state is None:
        return
    blocked = live.owner_naming_modules({item.path for item in items}, live.patterns())
    if blocked:
        pytest.exit(f"refused: the production freshpool study is running ({state}); these "
                    f"modules spawn or name accelerator-owner processes and would stop it: "
                    f"{blocked}", returncode=2)
