"""
scripts/release.py
==================
Immutable model releases (ADR-019).

    s3://<bucket>/models/<release>/
        logistic_regression.joblib, gradient_boosting.joblib, random_forest.joblib
        predictions/<model>.csv        test-set predictions of that release's dataset version
        manifest.json                  dataset, commit, settings, features, metrics, calibration, status
    s3://<bucket>/models/PRODUCTION    name of the release the API serves

A copy of every manifest and change report, the PRODUCTION pointer and a log of
promotions live in git under models/releases/. Artifacts are never overwritten or
deleted; only a manifest's status changes: candidate → promoted | rejected, and
promoted → retired when rolled back.

Usage:
    python scripts/release.py init                       # package the current local models as r1 (production)
    python scripts/release.py candidate --from DIR       # package models trained into DIR/<model>/ as the next release
    python scripts/release.py report rN                  # change report vs production + promotion checks
    python scripts/release.py promote rN                 # make rN production (asks for confirmation)
    python scripts/release.py reject rN --reason "..."
    python scripts/release.py rollback [--to rN] --reason "..."   # asks for confirmation
    python scripts/release.py status

Testing without S3 writes: `candidate --dry-run` packages into DIR/<release>/, and
`report --local DIR/<release>` reports on it (written next to it, statuses unchanged).

Strict mode: a release is refused unless the git tree is clean, the local
dataset matches its manifest, and every model reproduces its test predictions
from the saved artifact.
"""

import argparse
import importlib
import json
import logging
import os
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
SHORT = {"logistic_regression": "LR", "gradient_boosting": "XGBoost", "random_forest": "RF"}
N_BOOT = 500
KEY = ["market_id", "snapshot_timestamp"]

# The models in a release (ADR-024): module with predict(artifact, df), and — relative to
# the model's folder (models/<name>/, or <train.py --out-dir>) — the artifact and the
# prediction file + column its train.py writes, used to verify the artifact.
MODELS = {
    "logistic_regression": {
        "module": "models.logistic_regression.train",
        "artifact": "artifacts/model.joblib",
        "predictions": ("predictions/predictions.csv", "pred_prob"),
    },
    "gradient_boosting": {
        "module": "models.gradient_boosting.train",
        "artifact": "artifacts/model.joblib",
        "predictions": ("predictions/predictions.csv", "pred_prob"),
    },
    "random_forest": {
        "module": "models.random_forest.train",
        "artifact": "artifacts/rf_full_calibrated.pkl",
        "predictions": ("predictions/test_predictions.csv", "proba_full_calibrated"),
    },
}
# model_runs.model_name of each release model (the names db/load_parquet.py uses)
DB_NAMES = {"logistic_regression": "logistic_regression_base", "gradient_boosting": "xgboost_base",
            "random_forest": "random_forest_full_calibrated"}


# ─────────────────────────────────────────────
# S3 / GIT HELPERS
# ─────────────────────────────────────────────

def release_keys(s3) -> list[str]:
    """Release names on S3 (r1, r2, …), in numeric order."""
    r = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/", Delimiter="/")
    names = [p["Prefix"].split("/")[1] for p in r.get("CommonPrefixes", [])]
    return sorted((n for n in names if n.startswith("r") and n[1:].isdigit()), key=lambda n: int(n[1:]))


def next_release(s3) -> str:
    names = release_keys(s3)
    return f"r{int(names[-1][1:]) + 1}" if names else "r1"


def read_json(s3, key: str) -> dict:
    return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())


def manifest_key(release: str) -> str:
    return f"{PREFIX}/{release}/manifest.json"


def read_manifest(s3, release: str) -> dict:
    if not exists(s3, manifest_key(release)):
        raise SystemExit(f"ERROR: no release {release} on S3 (or it is incomplete: no manifest).")
    return read_json(s3, manifest_key(release))


def save_manifest(s3, manifest: dict) -> None:
    """Write a manifest to S3 and its git copy (status changes; bucket versioning keeps the old one)."""
    body = json.dumps(manifest, indent=2) + "\n"
    s3.put_object(Bucket=BUCKET, Key=manifest_key(manifest["release"]), Body=body.encode())
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    (RELEASES_DIR / f"{manifest['release']}.json").write_text(body)


def production(s3) -> str | None:
    return s3.get_object(Bucket=BUCKET, Key=POINTER)["Body"].read().decode().strip() if exists(s3, POINTER) else None


def set_production(s3, release: str) -> None:
    s3.put_object(Bucket=BUCKET, Key=POINTER, Body=f"{release}\n".encode())
    (RELEASES_DIR / "PRODUCTION").write_text(f"{release}\n")


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


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def confirm(release: str, action: str) -> None:
    """Explicit confirmation (ADR-019): the user types the release name."""
    answer = input(f"\nType {release} to {action}: ").strip()
    if answer != release:
        raise SystemExit("Not confirmed — nothing changed.")


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


def load_test() -> pd.DataFrame:
    df = load_dataset()
    test = df[df["split"] == "test"].reset_index(drop=True)
    log.info("  Test set: %s rows, %s markets", f"{len(test):,}", f"{test['market_id'].nunique():,}")
    return test


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
# PACKAGING
# ─────────────────────────────────────────────

def package(release: str, model_dirs: dict[str, Path], out: Path, **fields) -> dict:
    """
    Verify each model's artifact against the test predictions its train.py wrote,
    then write <model>.joblib and predictions/<model>.csv into `out` and return
    the manifest (not yet written anywhere). `fields`: status, parent, notes, ….
    """
    code = check_git_clean()
    dataset = check_dataset()
    test = load_test()
    y, market = test["outcome"].values, test["price_at_snapshot"].values
    manifest = {
        "release": release, "created_at": now(), "promoted_at": None, **fields,
        "dataset_version": dataset["version"],
        "dataset_manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "code": code, "adrs": ["019", "024"],
        "test_set": {"rows": len(test), "markets": int(test["market_id"].nunique()),
                     "yes_rate": round(float(y.mean()), 4)},
        "market_baseline": {k: round(float(v), 5) for k, v in compute_metrics(y, market).items() if k != "n"},
        "models": {},
    }
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    for name, spec in MODELS.items():
        log.info("\n%s", name)
        mod = importlib.import_module(spec["module"])
        art = joblib.load(model_dirs[name] / spec["artifact"])
        if name == "random_forest" and "encoder" not in art:
            # Artifacts before 2026-09-29 didn't save the encoder. It is fitted on the
            # sorted category names of the dataset it was trained on, so it can be rebuilt
            # exactly; the prediction check below proves it.
            cats = load_dataset()[["category"]]
            art["encoder"] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(cats)
            log.info("  category encoder rebuilt from dataset %s: %s",
                     dataset["version"], art["encoder"].categories_[0].tolist())
        art.setdefault("features", getattr(mod, "FEATURES_FULL", None)
                       or mod.NUMERIC_FEATURES + mod.CATEGORICAL_FEATURES)
        art["params"] = settings(name, art)

        # Strict mode: the artifact must reproduce the predictions train.py wrote
        p = mod.predict(art, test)
        csv, col = spec["predictions"]
        csv = model_dirs[name] / csv
        ref = pd.read_csv(csv, usecols=KEY + [col, "dataset_version"])
        if set(ref["dataset_version"].astype(str)) != {dataset["version"]}:
            raise SystemExit(f"ERROR: {csv} is not from dataset {dataset['version']}.")
        ref["snapshot_timestamp"] = pd.to_datetime(ref["snapshot_timestamp"], utc=True, format="mixed")
        m = test[KEY].assign(p=p).merge(ref, on=KEY, how="left")
        diff = float(np.abs(m["p"] - m[col]).max())
        if m[col].isna().any() or diff > 1e-6:
            raise SystemExit(f"ERROR: {name} artifact does not reproduce {csv} (max diff {diff:.2e}).")
        log.info("  reproduces %s (max diff %.1e)", csv.name, diff)

        art_path = out / f"{name}.joblib"
        joblib.dump(art, art_path)
        test[KEY + ["outcome"]].assign(pred_prob=p).to_csv(out / "predictions" / f"{name}.csv", index=False)

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
    return manifest


def upload_release(s3, manifest: dict, out: Path) -> None:
    release = manifest["release"]
    if exists(s3, manifest_key(release)):
        raise SystemExit(f"ERROR: {release} already exists on S3 — releases are never overwritten (ADR-019).")
    log.info("\nUploading %s to s3://%s/%s/%s/ ...", release, BUCKET, PREFIX, release)
    for entry in manifest["models"].values():
        upload(s3, out / entry["artifact"], f"{PREFIX}/{release}/{entry['artifact']}")
        upload(s3, out / entry["predictions"], f"{PREFIX}/{release}/{entry['predictions']}")
    # Manifest last: a release without one is incomplete and ignored
    save_manifest(s3, manifest)


def load_release(s3, source: str, dest: Path) -> tuple[dict, Path]:
    """A release's manifest and a folder holding its artifacts: a local folder, or a release downloaded from S3."""
    if Path(source).is_dir():
        return json.loads((Path(source) / "manifest.json").read_text()), Path(source)
    manifest = read_manifest(s3, source)
    folder = dest / source
    folder.mkdir(parents=True, exist_ok=True)
    for entry in manifest["models"].values():
        path = folder / entry["artifact"]
        s3.download_file(BUCKET, f"{PREFIX}/{source}/{entry['artifact']}", str(path))
        if sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"ERROR: {source}/{entry['artifact']} doesn't match its manifest's sha256.")
    return manifest, folder


def predict_release(manifest: dict, folder: Path, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """P(YES) of each model of a release on `test`, loading one artifact at a time (RF is large)."""
    out = {}
    for name, entry in manifest["models"].items():
        mod = importlib.import_module(MODELS[name]["module"])
        out[name] = mod.predict(joblib.load(folder / entry["artifact"]), test)
    return out


# ─────────────────────────────────────────────
# COMMANDS
# ─────────────────────────────────────────────

def cmd_init(s3, args) -> None:
    """Package the current local models (trained on the local dataset) as r1 and promote it."""
    if release_keys(s3) or exists(s3, POINTER):
        raise SystemExit("ERROR: releases already exist — init only creates r1.")
    with tempfile.TemporaryDirectory() as tmp:
        manifest = package("r1", {n: ROOT / "models" / n for n in MODELS}, Path(tmp),
                           status="promoted", parent=None, notes=args.notes)
        manifest["promoted_at"] = manifest["created_at"]
        upload_release(s3, manifest, Path(tmp))
    set_production(s3, "r1")
    append_log(f"- {manifest['promoted_at']}: **r1** created and promoted (init; dataset "
               f"{manifest['dataset_version']}). {args.notes}")
    log.info("\nr1 is production. Commit models/releases/.")


def cmd_candidate(s3, args) -> None:
    """Package models trained into DIR/<model>/ (train.py --out-dir) as a candidate release."""
    prod = production(s3)
    if prod is None:
        raise SystemExit("ERROR: no production release — run init first.")
    release = args.name or next_release(s3)
    if exists(s3, manifest_key(release)):
        raise SystemExit(f"ERROR: {release} already exists on S3.")
    out = args.src / release
    manifest = package(release, {n: args.src / n for n in MODELS}, out,
                       status="candidate", parent=prod, notes=args.notes)
    if args.dry_run:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        log.info("\nDry run: %s packaged in %s (nothing uploaded). Report: release.py report --local %s",
                 release, out, out)
        return
    upload_release(s3, manifest, out)
    log.info("\n%s is a candidate. Next: python scripts/release.py report %s", release, release)


def cmd_report(s3, args) -> None:
    """Change report vs production + promotion checks; a failed check rejects the release."""
    import release_report

    prod_name = production(s3)
    source = str(args.local) if args.local else args.release
    with tempfile.TemporaryDirectory() as tmp:
        cand, cand_dir = load_release(s3, source, Path(tmp))
        if cand["release"] == prod_name and not args.local:
            raise SystemExit(f"ERROR: {prod_name} is production — nothing to compare it with.")
        prod, prod_dir = load_release(s3, prod_name, Path(tmp))
        dataset = check_dataset()
        if dataset["version"] != cand["dataset_version"]:
            raise SystemExit(f"ERROR: local data is {dataset['version']}, {cand['release']} was trained on "
                             f"{cand['dataset_version']} — pull it first (sync.py pull {cand['dataset_version']}).")
        test = load_test()
        log.info("  Predicting with %s and %s ...", prod["release"], cand["release"])
        p_prod = predict_release(prod, prod_dir, test)
        p_cand = predict_release(cand, cand_dir, test)

    log.info("  Scoring (%d bootstrap resamples per comparison) ...", args.n_boot)
    preds = {n: (p_prod[n], p_cand[n]) for n in cand["models"] if n in prod["models"]}
    md, checks = release_report.build(test, preds, prod, cand, args.n_boot)

    if args.local:
        path = args.local / "report.md"
        path.write_text(md)
        (args.local / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    else:
        path = RELEASES_DIR / f"{cand['release']}_report.md"
        path.write_text(md)
        cand["checks"] = {"against": prod["release"], "at": now(), **checks}
        if cand["status"] == "candidate" and not checks["passed"]:
            cand["status"] = "rejected"
            append_log(f"- {now()}: **{cand['release']}** rejected by the promotion checks "
                       f"(vs {prod['release']}; see {path.name}).")
        save_manifest(s3, cand)
    for name, c in checks["models"].items():
        for line in c["fails"]:
            log.info("  FAIL %s: %s", SHORT[name], line)
        for line in c["warnings"]:
            log.info("  warn %s: %s", SHORT[name], line)
    log.info("\nVerdict: %s — report: %s", "PASS" if checks["passed"] else "FAIL", path.relative_to(ROOT)
             if path.is_relative_to(ROOT) else path)


def cmd_promote(s3, args) -> None:
    cand = read_manifest(s3, args.release)
    if cand["status"] != "candidate":
        raise SystemExit(f"ERROR: {args.release} is {cand['status']} — only a candidate can be promoted.")
    checks = cand.get("checks")
    prod = production(s3)
    if not checks or checks["against"] != prod:
        raise SystemExit(f"ERROR: {args.release} has no report against production ({prod}) — "
                         f"run: python scripts/release.py report {args.release}")
    log.info("%s vs %s: checks passed. Per model (ΔAUC, Δlog-loss, candidate − production):", args.release, prod)
    for name, c in checks["models"].items():
        log.info("  %-8s %+.4f [%+.4f, %+.4f]   %+.4f [%+.4f, %+.4f]%s", SHORT[name],
                 *c["auc_diff"].values(), *c["log_loss_diff"].values(),
                 "".join(f"\n           warning: {w}" for w in c["warnings"]))
    log.info("Report: models/releases/%s_report.md", args.release)
    check_git_clean()
    confirm(args.release, f"make it production (replacing {prod})")

    cand["status"], cand["promoted_at"] = "promoted", now()
    save_manifest(s3, cand)
    set_production(s3, args.release)
    append_log(f"- {cand['promoted_at']}: **{args.release}** promoted, replacing {prod}. {args.reason or ''}".rstrip())
    load_model_runs(cand)
    log.info("\n%s is production. Commit models/releases/.", args.release)


def cmd_reject(s3, args) -> None:
    m = read_manifest(s3, args.release)
    if m["status"] != "candidate":
        raise SystemExit(f"ERROR: {args.release} is {m['status']} — only a candidate can be rejected.")
    m["status"] = "rejected"
    m["rejected_reason"] = args.reason
    save_manifest(s3, m)
    append_log(f"- {now()}: **{args.release}** rejected by hand. {args.reason}")
    log.info("%s rejected. Commit models/releases/.", args.release)


def cmd_rollback(s3, args) -> None:
    current = production(s3)
    if args.to:
        target = args.to
        if read_manifest(s3, target)["status"] != "promoted":
            raise SystemExit(f"ERROR: {target} was never promoted (or was retired) — pick a promoted release.")
    else:
        # The most recently promoted release other than the current one
        promoted = [m for m in (read_json(s3, manifest_key(n)) for n in release_keys(s3)
                                if exists(s3, manifest_key(n)))
                    if m["status"] == "promoted" and m["release"] != current]
        if not promoted:
            raise SystemExit("ERROR: no earlier promoted release to roll back to.")
        target = max(promoted, key=lambda m: m["promoted_at"])["release"]
    if target == current:
        raise SystemExit(f"ERROR: {target} is already production.")
    check_git_clean()
    confirm(target, f"roll production back from {current} to {target}")

    old = read_manifest(s3, current)
    old["status"], old["retired_at"] = "retired", now()
    save_manifest(s3, old)
    set_production(s3, target)
    append_log(f"- {now()}: **rollback** {current} → **{target}** ({current} retired). {args.reason}")
    log.info("\n%s is production again; %s retired. Commit models/releases/.", target, current)


def load_model_runs(manifest: dict) -> None:
    """One model_runs row per model of a promoted release (migration 005 adds `release`)."""
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    url = os.getenv("DATABASE_URL")
    if not url:
        log.warning("  DATABASE_URL not set — model_runs not updated. Rerun after setting it: "
                    "python scripts/release.py sync-db %s", manifest["release"])
        return
    with psycopg2.connect(url) as conn, conn.cursor() as cur:
        for name, entry in manifest["models"].items():
            cur.execute(
                """INSERT INTO model_runs (model_name, dataset_tag, release, notes, metrics, hyperparams, git_sha)
                   VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                   ON CONFLICT (model_name, dataset_tag, release)
                   DO UPDATE SET metrics = EXCLUDED.metrics, hyperparams = EXCLUDED.hyperparams,
                                 notes = EXCLUDED.notes, git_sha = EXCLUDED.git_sha""",
                (DB_NAMES[name], f"clean_{manifest['dataset_version']}", manifest["release"],
                 f"Release {manifest['release']} (ADR-019)", json.dumps(entry["metrics"]),
                 json.dumps(entry["settings"]), manifest["code"]["git_commit"]))
    log.info("  model_runs: %d rows for %s", len(manifest["models"]), manifest["release"])


def cmd_sync_db(s3, args) -> None:
    load_model_runs(read_manifest(s3, args.release))


def cmd_status(s3, args) -> None:
    prod = production(s3)
    names = release_keys(s3)
    if not names:
        log.info("No releases yet — run: python scripts/release.py init")
        return
    log.info("%-5s %-10s %-8s %-20s %s", "", "status", "dataset", "created", "AUC vs market (per snapshot)")
    for n in names:
        if not exists(s3, manifest_key(n)):
            log.info("%-5s (incomplete: no manifest)", n)
            continue
        m = read_json(s3, manifest_key(n))
        aucs = ", ".join(f"{SHORT.get(k, k)} {v['metrics']['per_snapshot']['vs_market']['auc']['point']:+.4f}"
                         for k, v in m["models"].items())
        log.info("%-5s %-10s %-8s %-20s %s%s", n, m["status"], m["dataset_version"], m["created_at"][:19],
                 aucs, "   ← PRODUCTION" if n == prod else "")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--notes", default="First release: the v3 models (ADR-024 protocol).")
    p = sub.add_parser("candidate")
    p.add_argument("--from", dest="src", type=Path, required=True, help="folder with <model>/artifacts, <model>/predictions")
    p.add_argument("--name", help="release name (default: the next free rN)")
    p.add_argument("--notes", default="")
    p.add_argument("--dry-run", action="store_true", help="package into DIR/<release>/ without uploading")
    p = sub.add_parser("report")
    p.add_argument("release", nargs="?")
    p.add_argument("--local", type=Path, help="report on a dry-run package instead (nothing on S3 changes)")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p = sub.add_parser("promote")
    p.add_argument("release")
    p.add_argument("--reason", default="")
    p = sub.add_parser("reject")
    p.add_argument("release")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("rollback")
    p.add_argument("--to")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("sync-db", help="(re)write a release's model_runs rows")
    p.add_argument("release")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.cmd == "report" and not (args.release or args.local):
        parser.error("report needs a release name or --local DIR")
    s3 = make_client()
    {"init": cmd_init, "candidate": cmd_candidate, "report": cmd_report, "promote": cmd_promote,
     "reject": cmd_reject, "rollback": cmd_rollback, "sync-db": cmd_sync_db, "status": cmd_status}[args.cmd](s3, args)


if __name__ == "__main__":
    main()
