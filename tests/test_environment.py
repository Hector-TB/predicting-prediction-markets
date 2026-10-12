"""The installed library versions must be the pinned ones, and the ones the production release
was built with: its saved models only load (and predict identically) under those versions."""

import json
import re
import tomllib

import pytest
import release

import api  # noqa: F401  (sets up the import paths)
from api import ROOT

PINS = {m["pkg"]: m["ver"] for dep in tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
        if (m := re.fullmatch(r"(?P<pkg>[A-Za-z0-9_.-]+)==(?P<ver>\S+)", dep))}


def test_every_release_library_is_pinned_exactly():
    assert set(release.PINNED) <= set(PINS), f"not pinned with ==: {set(release.PINNED) - set(PINS)}"


def test_installed_versions_are_the_pinned_ones():
    have = release.environment()
    diff = {pkg: (PINS[pkg], have[pkg]) for pkg in release.PINNED if have[pkg] != PINS[pkg]}
    assert not diff, f"pinned vs installed: {diff} — run: uv pip install -e ."


def test_production_release_was_built_with_the_pinned_versions():
    prod = (release.RELEASES_DIR / "PRODUCTION").read_text().strip()
    manifest = json.loads((release.RELEASES_DIR / f"{prod}.json").read_text())
    want = release.release_environment(manifest)
    assert want, f"{prod} records no library versions"
    diff = {pkg: (v, PINS.get(pkg)) for pkg, v in want.items() if PINS.get(pkg) != v}
    assert not diff, f"{prod} built with vs pinned: {diff}"


def test_check_environment_names_each_mismatch(monkeypatch):
    monkeypatch.setattr(release, "environment", lambda: {"scikit-learn": "1.9.1", "xgboost": "3.2.0"})
    with pytest.raises(SystemExit, match=r"scikit-learn: built with 1\.7\.2, installed 1\.9\.1") as e:
        release.check_environment({"release": "rX", "environment": {"scikit-learn": "1.7.2", "xgboost": "3.2.0"}})
    assert "xgboost" not in str(e.value)
    release.check_environment({"release": "rX", "environment": {"xgboost": "3.2.0"}})   # match: no error


def test_releases_before_the_field_use_their_environment_file():
    assert release.release_environment({"release": "r2"})["scikit-learn"] == "1.7.2"
    with pytest.raises(SystemExit, match="records no library versions"):
        release.check_environment({"release": "r999"})
