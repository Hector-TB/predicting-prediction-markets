"""Load the production model release (ADR-019) for serving."""

import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib

import release as releases_cli   # scripts/release.py
from api import ROOT

log = logging.getLogger(__name__)

CACHE_DIR = ROOT / ".cache" / "releases"


@dataclass
class ServingRelease:
    name: str
    manifest: dict
    time_fill: dict                                   # ADR-025: time features with no end date
    models: dict = field(default_factory=dict)        # name -> (module, artifact)

    @property
    def dataset_version(self) -> str:
        return self.manifest["dataset_version"]

    @property
    def headline(self) -> str:
        """The model with the best test AUC in the release."""
        return max(self.manifest["models"],
                   key=lambda n: self.manifest["models"][n]["metrics"]["per_snapshot"]["auc"])


def time_fill(name: str, manifest: dict) -> dict:
    if "time_feature_fill" in manifest:
        return manifest["time_feature_fill"]
    # Releases before the manifest field (r1) keep it next to their record
    path = releases_cli.RELEASES_DIR / f"{name}_time_fill.json"
    return json.loads(path.read_text())["time_feature_fill"]


def cached(name: str) -> Path | None:
    """The cached release folder, if every artifact matches its manifest."""
    folder = CACHE_DIR / name
    path = folder / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    for entry in manifest["models"].values():
        art = folder / entry["artifact"]
        if not art.exists() or releases_cli.sha256_file(art) != entry["sha256"]:
            return None
    return folder


def load(name: str | None = None) -> ServingRelease:
    """`name`, else $RELEASE, else the S3 PRODUCTION pointer; downloaded once into .cache/."""
    s3 = None
    name = name or os.getenv("RELEASE")
    if not name:
        s3 = releases_cli.make_client()
        name = releases_cli.production(s3)
        if name is None:
            raise RuntimeError("No production release on S3.")

    folder = cached(name)
    if folder is None:
        log.info("Downloading release %s ...", name)
        manifest, folder = releases_cli.load_release(s3 or releases_cli.make_client(), name, CACHE_DIR)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest = json.loads((folder / "manifest.json").read_text())

    rel = ServingRelease(name, manifest, time_fill(name, manifest))
    for model, entry in manifest["models"].items():
        module = importlib.import_module(releases_cli.MODELS[model]["module"])
        rel.models[model] = (module, joblib.load(folder / entry["artifact"]))
    log.info("Serving release %s (dataset %s): %s", name, rel.dataset_version, ", ".join(rel.models))
    return rel
