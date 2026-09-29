"""
scripts/release.py
==================
Immutable model releases (ADR-019).

    s3://<bucket>/models/<release>/
        logistic_regression.joblib, gradient_boosting.joblib, random_forest.joblib
        predictions/<model>.csv        test-set predictions of that release's dataset version
        manifest.json                  dataset, commit, settings, features, metrics, calibration, status
    s3://<bucket>/models/PRODUCTION    name of the release the API serves

A copy of every manifest, the PRODUCTION pointer and a log of promotions live in
git under models/releases/. Releases are never overwritten or deleted.

Usage:
    python scripts/release.py init      # package the current local models as r1 and make it production
    python scripts/release.py status    # releases on S3 and which one is production

Strict mode: a release is refused unless the git tree is clean, the local
dataset matches its manifest, and every model reproduces its published test
predictions from the saved artifact.
"""

import argparse
import hashlib
import importlib
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RELEASES_DIR = ROOT / "models" / "releases"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATA_DIR))

from models.common.evaluation import (  # noqa: E402
    bootstrap_auc_diff, bootstrap_loss_diff, compute_metrics,
)
from models.common.training import load_dataset, market_weights  # noqa: E402
from sync import BUCKET, exists, git_info, make_client, sha256_file  # noqa: E402

log = logging.getLogger(__name__)

PREFIX = "models"
POINTER = f"{PREFIX}/PRODUCTION"
N_BOOT = 500
KEY = ["market_id", "snapshot_timestamp"]

# The models in a release (ADR-024): module with predict(artifact, df), local artifact,
# and the published prediction file + column used to verify the artifact.
MODELS = {
    "logistic_regression": {
        "module": "models.logistic_regression.train",
        "artifact": "models/logistic_regression/artifacts/model.joblib",
        "predictions": ("models/logistic_regression/predictions/predictions.csv", "pred_prob"),
    },
    "gradient_boosting": {
        "module": "models.gradient_boosting.train",
        "artifact": "models/gradient_boosting/artifacts/model.joblib",
        "predictions": ("models/gradient_boosting/predictions/predictions.csv", "pred_prob"),
    },
    "random_forest": {
        "module": "models.random_forest.train",
        "artifact": "models/random_forest/artifacts/rf_full_calibrated.pkl",
        "predictions": ("models/random_forest/predictions/test_predictions.csv", "proba_full_calibrated"),
    },
}


# ─────────────────────────────────────────────
# S3 / GIT HELPERS
# ─────────────────────────────────────────────

def release_keys(s3) -> list[str]:
    """Release names on S3 (r1, r2, …), in numeric order."""
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/", Delimiter="/")
    names = [p["Prefix"].split("/")[1] for p in r.get("CommonPrefixes", [])]
    return sorted((n for n in names if n.startswith("r") and n[1:].isdigit()), key=lambda n: int(n[1:]))


def read_json(s3, key: str) -> dict:
    return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())


def production(s3) -> str | None:
    return s3.get_object(Bucket=BUCKET, Key=POINTER)["Body"].read().decode().strip() if exists(s3, POINTER) else None


def upload(s3, path: Path, key: str) -> None:
    s3.upload_file(str(path), BUCKET, key)
    if s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"] != path.stat().st_size:
        raise SystemExit(f"ERROR: size mismatch after uploading {key}")


def append_log(line: str) -> None:
    path = RELEASES_DIR / "log.md"
    if not path.exists():
        path.write_text("# Release log\n\nPromotions and rollbacks, newest last (ADR-019).\n\n")
    with path.open("a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────
# STRICT-MODE CHECKS
# ─────────────────────────────────────────────

def check_git_clean() -> dict:
    info = git_info()
    if info["uncommitted_changes"]:
        raise SystemExit("ERROR: uncommitted changes — commit first, so the release records exact code (ADR-019).")
    return info


def check_dataset() -> dict:
    """The local data must be a published version, byte-identical to its manifest."""
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise SystemExit("ERROR: local data has no manifest — pull or publish a dataset version first (ADR-016).")
    manifest = json.loads(path.read_text())
    for fname in ["polymarket_markets_meta.csv", "polymarket_ml_dataset_clean.parquet"]:
        info = manifest["files"][fname]
        if sha256_file(DATA_DIR / fname) != info["sha256"]:
            raise SystemExit(f"ERROR: {fname} differs from dataset {manifest['version']}'s manifest.")
    log.info("  Dataset %s: local files match the manifest", manifest["version"])
    return manifest


# ─────────────────────────────────────────────
# MODEL DETAILS FOR THE MANIFEST
# ─────────────────────────────────────────────

def settings(name: str, art: dict) -> dict:
    """The tuned settings (reused by later refreshes, ADR-019 decision 7)."""
    if "params" in art:
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in art["params"].items()}
    if name == "logistic_regression":
        clf = art["pipeline"].named_steps["clf"]
        return {"clf__C": clf.C, "clf__penalty": clf.penalty, "clf__solver": clf.solver}
    if name == "gradient_boosting":
        p = art["clf"].get_params()
        return {"learning_rate": p["learning_rate"], "max_depth": p["max_depth"],
                "min_child_weight": p["min_child_weight"], "n_estimators": art["clf"].best_iteration + 1}
    rf = art["rf"]
    return {"max_depth": rf.max_depth, "min_samples_leaf": rf.min_samples_leaf}


def feature_importance(name: str, art: dict, features: list[str]) -> dict:
    if name == "logistic_regression":
        pre = art["pipeline"].named_steps["preprocessor"]
        names = list(pre.get_feature_names_out())
        values = art["pipeline"].named_steps["clf"].coef_[0]
    elif name == "gradient_boosting":
        names = list(art["preprocessor"].get_feature_names_out())
        values = art["clf"].feature_importances_
    else:
        names, values = features, art["rf"].feature_importances_
    return {n.split("__")[-1]: round(float(v), 6) for n, v in zip(names, values)}


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    return [{"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
             "mean_pred": round(float(p[idx == b].mean()), 4), "actual": round(float(y[idx == b].mean()), 4)}
            for b in range(n_bins) if (idx == b).any()]


def ci(d: dict) -> dict:
    return {k: round(float(d[k]), 5) for k in ("point", "ci_lo", "ci_hi")}


def score(test: pd.DataFrame, p: np.ndarray) -> dict:
    """Metrics vs the market price on the release's test rows (ADR-015)."""
    y, g, m = test["outcome"].values, test["market_id"].values, test["price_at_snapshot"].values
    out = {}
    for label, w in [("per_snapshot", None), ("per_market", market_weights(g))]:
        loss = bootstrap_loss_diff(y, m, p, groups=g, n_boot=N_BOOT, sample_weight=w)
        out[label] = {
            **{k: round(float(v), 5) for k, v in compute_metrics(y, p, sample_weight=w).items() if k != "n"},
            "vs_market": {
                "auc": ci(bootstrap_auc_diff(y, m, p, n_boot=N_BOOT, groups=g, verbose=False, sample_weight=w)),
                "log_loss": ci(loss["log_loss"]), "brier": ci(loss["brier"]),
            },
        }
    return out


# ─────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────

def cmd_init(s3, args) -> None:
    """Package the current local models (trained on the local dataset) as r1 and promote it."""
    if release_keys(s3) or exists(s3, POINTER):
        raise SystemExit("ERROR: releases already exist — init only creates r1.")
    code = check_git_clean()
    dataset = check_dataset()

    df = load_dataset()
    test = df[df["split"] == "test"].reset_index(drop=True)
    log.info("  Test set: %s rows, %s markets", f"{len(test):,}", f"{test['market_id'].nunique():,}")

    release, now = "r1", datetime.now(timezone.utc).isoformat(timespec="seconds")
    y, market = test["outcome"].values, test["price_at_snapshot"].values
    manifest = {
        "release": release, "status": "promoted", "created_at": now, "promoted_at": now, "parent": None,
        "notes": args.notes, "dataset_version": dataset["version"],
        "dataset_manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "code": code, "adrs": ["019", "024"],
        "test_set": {"rows": len(test), "markets": int(test["market_id"].nunique()),
                     "yes_rate": round(float(y.mean()), 4)},
        "market_baseline": {k: round(float(v), 5) for k, v in compute_metrics(y, market).items() if k != "n"},
        "models": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, spec in MODELS.items():
            log.info("\n%s", name)
            mod = importlib.import_module(spec["module"])
            art = joblib.load(ROOT / spec["artifact"])
            if name == "random_forest" and "encoder" not in art:
                # Artifacts before 2026-09-29 didn't save the encoder. It is fitted on the
                # sorted category names of the dataset it was trained on, so it can be rebuilt
                # exactly; the prediction check below proves it.
                art["encoder"] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(df[["category"]])
                log.info("  category encoder rebuilt from dataset %s: %s",
                         dataset["version"], art["encoder"].categories_[0].tolist())
            art.setdefault("features", getattr(mod, "FEATURES_FULL", None)
                           or mod.NUMERIC_FEATURES + mod.CATEGORICAL_FEATURES)
            art["params"] = settings(name, art)

            # Strict mode: the artifact must reproduce the published predictions
            p = mod.predict(art, test)
            csv, col = spec["predictions"]
            ref = pd.read_csv(ROOT / csv, usecols=KEY + [col, "dataset_version"])
            if set(ref["dataset_version"].astype(str)) != {dataset["version"]}:
                raise SystemExit(f"ERROR: {csv} is not from dataset {dataset['version']}.")
            ref["snapshot_timestamp"] = pd.to_datetime(ref["snapshot_timestamp"], utc=True, format="mixed")
            m = test[KEY].assign(p=p).merge(ref, on=KEY, how="left")
            diff = float(np.abs(m["p"] - m[col]).max())
            if m[col].isna().any() or diff > 1e-6:
                raise SystemExit(f"ERROR: {name} artifact does not reproduce {csv} (max diff {diff:.2e}).")
            log.info("  reproduces %s (max diff %.1e)", Path(csv).name, diff)

            art_path = tmp / f"{name}.joblib"
            joblib.dump(art, art_path)
            pred_path = tmp / f"{name}.csv"
            test[KEY + ["outcome"]].assign(pred_prob=p).to_csv(pred_path, index=False)

            log.info("  scoring vs the market (%d bootstrap resamples) ...", N_BOOT)
            manifest["models"][name] = {
                "artifact": f"{name}.joblib", "sha256": sha256_file(art_path),
                "predictions": f"predictions/{name}.csv",
                "features": list(art["features"]), "settings": art["params"],
                "threshold": round(float(art["threshold"]), 6),
                "metrics": score(test, p),
                "calibration": calibration_bins(y, p),
                "feature_importance": feature_importance(name, art, list(art["features"])),
            }
            s = manifest["models"][name]["metrics"]["per_snapshot"]
            log.info("  AUC %.4f (vs market %+.4f)  log-loss %.4f", s["auc"], s["vs_market"]["auc"]["point"], s["log_loss"])

        log.info("\nUploading %s to s3://%s/%s/%s/ ...", release, BUCKET, PREFIX, release)
        for name, entry in manifest["models"].items():
            upload(s3, tmp / entry["artifact"], f"{PREFIX}/{release}/{entry['artifact']}")
            upload(s3, tmp / f"{name}.csv", f"{PREFIX}/{release}/{entry['predictions']}")
        # Manifest last: a release without one is incomplete and ignored
        body = json.dumps(manifest, indent=2) + "\n"
        s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{release}/manifest.json", Body=body.encode())
        s3.put_object(Bucket=BUCKET, Key=POINTER, Body=f"{release}\n".encode())

    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    (RELEASES_DIR / f"{release}.json").write_text(body)
    (RELEASES_DIR / "PRODUCTION").write_text(f"{release}\n")
    append_log(f"- {now}: **{release}** created and promoted (init; dataset {dataset['version']}). {args.notes}")
    log.info("\n%s is production. Commit models/releases/.", release)


def cmd_status(s3, args) -> None:
    prod = production(s3)
    names = release_keys(s3)
    if not names:
        log.info("No releases yet — run: python scripts/release.py init")
        return
    log.info("%-5s %-10s %-8s %-20s %s", "", "status", "dataset", "created", "AUC vs market (per snapshot)")
    for n in names:
        key = f"{PREFIX}/{n}/manifest.json"
        if not exists(s3, key):
            log.info("%-5s (incomplete: no manifest)", n)
            continue
        m = read_json(s3, key)
        aucs = ", ".join(f"{k.split('_')[0]} {v['metrics']['per_snapshot']['vs_market']['auc']['point']:+.4f}"
                         for k, v in m["models"].items())
        log.info("%-5s %-10s %-8s %-20s %s%s", n, m["status"], m["dataset_version"], m["created_at"][:19],
                 aucs, "   ← PRODUCTION" if n == prod else "")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--notes", default="First release: the v3 models (ADR-024 protocol).")
    sub.add_parser("status")
    args = parser.parse_args()
    s3 = make_client()
    {"init": cmd_init, "status": cmd_status}[args.cmd](s3, args)


if __name__ == "__main__":
    main()
