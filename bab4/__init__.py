"""Canonical BAB 4 validation pipeline."""

from bab4.config import Bab4Config

__all__ = ["Bab4Config", "run_all", "validate_manifest"]


def run_all(*args, **kwargs):
    from bab4.run_all import run_all as _run_all

    return _run_all(*args, **kwargs)


def validate_manifest(*args, **kwargs):
    from bab4.run_all import validate_manifest as _validate_manifest

    return _validate_manifest(*args, **kwargs)
