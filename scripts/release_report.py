"""
scripts/release_report.py
=========================
Change report of a candidate release vs production, and the promotion checks
(ADR-019, amendment "one-command refresh"). Called by `release.py report`,
which loads both releases and predicts the candidate's test rows with each.

Both releases are scored on the SAME rows: the candidate's (newest) test set,
which production never trained on (T only moves forward). Sections:
  1. Data       — the two dataset manifests
  2. Code       — commits between the two releases; settings and feature diffs
  3. Metrics    — candidate vs production vs the market price, market-level CIs,
                  per snapshot and each market once; AUC by time left and category
  4. Attribution — production on its own test rows → on the new rows (the world
                  changed) → candidate on the new rows (the retrain changed things)
  5. Behaviour  — how far predictions move; threshold and feature-importance shifts
  Verdict       — the promotion checks
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.evaluation import (
    bootstrap_auc_diff, bootstrap_loss_diff, compute_metrics, expected_calibration_error,
)
from models.common.training import market_weights

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = ROOT / "data" / "manifests"

SHORT = {"logistic_regression": "LR", "gradient_boosting": "XGBoost", "random_forest": "RF"}
MAX_ECE_INCREASE = 0.01   # promotion check: calibration no worse than production by more than this
MOVE = 0.05               # behaviour: a prediction "moves" if it changes by more than this
TIME_LEFT = [("< 1 day", 0, 1), ("1–7 days", 1, 7), ("7–14 days", 7, 14), ("14–30 days", 14, 30),
             ("≥ 30 days", 30, np.inf)]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def ci(d: dict, fmt: str = "+.4f") -> str:
    return f"{d['point']:{fmt}} [{d['ci_lo']:{fmt}}, {d['ci_hi']:{fmt}}]"


def auc_or_dash(y, p) -> str:
    return f"{compute_metrics(y, p)['auc']:.4f}" if len(np.unique(y)) == 2 else "—"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def dataset_manifest(version: str) -> dict:
    path = MANIFESTS_DIR / f"{version}.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ─────────────────────────────────────────────
# SECTIONS
# ─────────────────────────────────────────────

def data_section(prod: dict, cand: dict) -> str:
    old, new = dataset_manifest(prod["dataset_version"]), dataset_manifest(cand["dataset_version"])
    if prod["dataset_version"] == cand["dataset_version"]:
        return f"## 1. Data\n\nSame dataset version ({cand['dataset_version']})."
    co, cn = old.get("contents", {}), new.get("contents", {})

    def get(c, *path):
        for p in path:
            c = c.get(p, {}) if isinstance(c, dict) else {}
        return c if c != {} else "—"

    def fmt(v):
        return f"{v:,}" if isinstance(v, int) else str(v)

    rows = [
        ["Fetched on", get(old, "fetched_on"), get(new, "fetched_on")],
        ["Markets in meta", get(co, "markets_fetched"), get(cn, "markets_fetched")],
        ["Train markets / snapshots", f"{fmt(get(co, 'clean_markets_by_split', 'train'))} / "
                                      f"{fmt(get(co, 'clean_snapshots_by_split', 'train'))}",
         f"{fmt(get(cn, 'clean_markets_by_split', 'train'))} / {fmt(get(cn, 'clean_snapshots_by_split', 'train'))}"],
        ["Test markets / snapshots", f"{fmt(get(co, 'clean_markets_by_split', 'test'))} / "
                                     f"{fmt(get(co, 'clean_snapshots_by_split', 'test'))}",
         f"{fmt(get(cn, 'clean_markets_by_split', 'test'))} / {fmt(get(cn, 'clean_snapshots_by_split', 'test'))}"],
        ["First test snapshot (≈ T)", str(get(co, "snapshot_range_by_split", "test"))[2:18],
         str(get(cn, "snapshot_range_by_split", "test"))[2:18]],
        ["Snapshot YES rate", get(co, "clean_snapshot_yes_rate"), get(cn, "clean_snapshot_yes_rate")],
    ]
    rows = [[r[0], fmt(r[1]), fmt(r[2])] for r in rows]
    return (f"## 1. Data\n\n{prod['release']} was trained on {prod['dataset_version']}, "
            f"{cand['release']} on {cand['dataset_version']}.\n\n"
            + table(["", prod["dataset_version"], cand["dataset_version"]], rows)
            + (f"\n\n{cand['dataset_version']} notes: {new['notes']}" if new.get("notes") else ""))


def code_section(prod: dict, cand: dict) -> str:
    a, b = prod["code"]["git_commit"], cand["code"]["git_commit"]
    parts = ["## 2. Code and settings"]
    if a == b:
        parts.append(f"Same commit ({b[:7]}).")
    else:
        commits = git("log", "--oneline", "--no-merges", f"{a}..{b}").splitlines()
        parts.append(f"{len(commits)} commits from {a[:7]} to {b[:7]} ({git('diff', '--shortstat', a, b)}):\n\n"
                     + "\n".join(f"- {c}" for c in commits[:40])
                     + (f"\n- … {len(commits) - 40} more" if len(commits) > 40 else ""))
    rows = []
    for name, mc in cand["models"].items():
        mp = prod["models"].get(name, {})
        changed = [f"{k}: {mp.get('settings', {}).get(k)} → {v}"
                   for k, v in mc["settings"].items() if mp.get("settings", {}).get(k) != v]
        added = sorted(set(mc["features"]) - set(mp.get("features", [])))
        removed = sorted(set(mp.get("features", [])) - set(mc["features"]))
        feats = "; ".join(filter(None, [f"+ {', '.join(added)}" if added else "",
                                        f"− {', '.join(removed)}" if removed else ""])) or "same"
        rows.append([SHORT.get(name, name), "; ".join(changed) or "same", feats,
                     f"{mp.get('threshold', float('nan')):.3f} → {mc['threshold']:.3f}"])
    parts.append(table(["Model", "Settings", "Features", "Threshold"], rows))
    return "\n\n".join(parts)


def metrics_section(test: pd.DataFrame, preds: dict, prod: dict, cand: dict, n_boot: int) -> tuple[str, dict]:
    """Returns the section and, per model, the candidate − production differences used by the checks."""
    y, g, market = test["outcome"].values, test["market_id"].values, test["price_at_snapshot"].values
    rp, rc = prod["release"], cand["release"]
    parts = [f"## 3. Metrics on {cand['dataset_version']}'s test rows\n\n"
             f"{len(test):,} snapshots, {test['market_id'].nunique():,} markets — rows neither release trained on. "
             f"95% CIs from {n_boot} market-level bootstrap resamples (ADR-015). Higher AUC is better; lower "
             "log-loss, Brier and calibration error (ECE) are better."]
    diffs = {}
    for label, w in [("Every snapshot counted", None), ("Each market counted once", market_weights(g))]:
        rows, drows = [], []
        base = compute_metrics(y, market, sample_weight=w)
        rows.append(["Market price", f"{base['auc']:.4f}", f"{base['log_loss']:.4f}", f"{base['brier']:.4f}",
                     f"{expected_calibration_error(y, market):.4f}" if w is None else "—"])
        for name, (pp, pc) in preds.items():
            for rel, p in [(rp, pp), (rc, pc)]:
                s = compute_metrics(y, p, sample_weight=w)
                rows.append([f"{SHORT[name]} {rel}", f"{s['auc']:.4f}", f"{s['log_loss']:.4f}", f"{s['brier']:.4f}",
                             f"{expected_calibration_error(y, p):.4f}" if w is None else "—"])
            a = bootstrap_auc_diff(y, pp, pc, n_boot=n_boot, groups=g, verbose=False, sample_weight=w)
            loss = bootstrap_loss_diff(y, pp, pc, groups=g, n_boot=n_boot, sample_weight=w)
            vs_market = bootstrap_auc_diff(y, market, pc, n_boot=n_boot, groups=g, verbose=False, sample_weight=w)
            drows.append([SHORT[name], ci(a), ci(loss["log_loss"]), ci(loss["brier"]), ci(vs_market)])
            if w is None:
                diffs[name] = {"auc": a, "log_loss": loss["log_loss"], "vs_market": vs_market,
                               "ece": (expected_calibration_error(y, pp), expected_calibration_error(y, pc))}
        parts.append(f"### {label}\n\n" + table(["", "AUC-ROC", "Log-loss", "Brier", "ECE"], rows)
                     + f"\n\n{rc} − {rp}, and {rc} − market:\n\n"
                     + table(["Model", "ΔAUC", "ΔLog-loss", "ΔBrier", f"ΔAUC {rc} vs market"], drows))

    cols = ["Market"] + [f"{SHORT[n]} {r}" for n in preds for r in (rp, rc)]
    rows = []
    for tname, lo, hi in TIME_LEFT:
        m = (test["days_before_close"] >= lo) & (test["days_before_close"] < hi)
        rows.append([tname, f"{m.sum():,}", f"{test.loc[m, 'market_id'].nunique():,}", auc_or_dash(y[m], market[m])]
                    + [auc_or_dash(y[m], p[m]) for pair in preds.values() for p in pair])
    parts.append("### AUC by time left before close\n\n" + table(["Time left", "Snapshots", "Markets"] + cols, rows))

    rows = []
    cats = test["category"].fillna("other")
    for cat in cats.value_counts().index:
        m = (cats == cat).values
        rows.append([cat, f"{m.sum():,}", f"{test.loc[m, 'market_id'].nunique():,}", auc_or_dash(y[m], market[m])]
                    + [auc_or_dash(y[m], p[m]) for pair in preds.values() for p in pair])
    parts.append("### AUC by category\n\n" + table(["Category", "Snapshots", "Markets"] + cols, rows))
    return "\n\n".join(parts), diffs


def attribution_section(test: pd.DataFrame, preds: dict, prod: dict, cand: dict) -> str:
    y, market = test["outcome"].values, test["price_at_snapshot"].values
    rp, rc, vp, vc = prod["release"], cand["release"], prod["dataset_version"], cand["dataset_version"]
    m_new = compute_metrics(y, market)
    rows = []
    for name, (pp, pc) in preds.items():
        own = prod["models"][name]["metrics"]["per_snapshot"]
        m_own = prod["market_baseline"]
        new_p, new_c = compute_metrics(y, pp), compute_metrics(y, pc)
        rows += [
            [SHORT[name], f"{rp} on {vp} test", f"{own['auc']:.4f}", f"{own['auc'] - m_own['auc']:+.4f}",
             f"{own['log_loss']:.4f}"],
            ["", f"{rp} on {vc} test", f"{new_p['auc']:.4f}", f"{new_p['auc'] - m_new['auc']:+.4f}",
             f"{new_p['log_loss']:.4f}"],
            ["", f"{rc} on {vc} test", f"{new_c['auc']:.4f}", f"{new_c['auc'] - m_new['auc']:+.4f}",
             f"{new_c['log_loss']:.4f}"],
        ]
    return (f"## 4. Attribution\n\nRow 1 → row 2: same model, newer markets — the change due to the data or the world "
            f"(market price AUC {m_own['auc']:.4f} on {vp} test, {m_new['auc']:.4f} on {vc} test). "
            "Row 2 → row 3: same markets, retrained model — the change due to the retrain.\n\n"
            + table(["Model", "Scored", "AUC-ROC", "ΔAUC vs market", "Log-loss"], rows))


def behaviour_section(test: pd.DataFrame, preds: dict, prod: dict, cand: dict) -> str:
    rp, rc = prod["release"], cand["release"]
    rows, shifts = [], []
    for name, (pp, pc) in preds.items():
        d = np.abs(pc - pp)
        tp, tc = prod["models"][name]["threshold"], cand["models"][name]["threshold"]
        flips = ((pp >= tp) != (pc >= tc)).mean()
        rows.append([SHORT[name], f"{d.mean():.4f}", f"{np.percentile(d, 95):.4f}", f"{(d > MOVE).mean():.1%}",
                     f"{flips:.1%}"])
        fp, fc = prod["models"][name].get("feature_importance", {}), cand["models"][name].get("feature_importance", {})
        change = sorted(((abs(fc.get(k, 0) - fp.get(k, 0)), k) for k in set(fp) | set(fc)), reverse=True)[:5]
        shifts.append([SHORT[name], "; ".join(f"{k} {fp.get(k, 0):.3f} → {fc.get(k, 0):.3f}" for _, k in change)])
    return (f"## 5. Behaviour\n\nHow far {rc}'s probabilities move from {rp}'s on the same rows, and how often the "
            "YES/NO label flips (each at its own threshold).\n\n"
            + table(["Model", "Mean |Δp|", "95th pct |Δp|", f"|Δp| > {MOVE}", "Label flips"], rows)
            + "\n\nLargest feature-importance shifts (LR: coefficients; XGBoost: gain; RF: impurity):\n\n"
            + table(["Model", "Feature: production → candidate"], shifts))


def run_checks(diffs: dict) -> dict:
    """ADR-019 promotion checks, per model, per snapshot. Any failure fails the release."""
    out = {"passed": True, "models": {}}
    for name, d in diffs.items():
        ece_p, ece_c = d["ece"]
        fails, warns = [], []
        if d["auc"]["ci_hi"] < 0:
            fails.append(f"AUC regression: ΔAUC {ci(d['auc'])} lies entirely below 0")
        if d["log_loss"]["ci_lo"] > 0:
            fails.append(f"log-loss regression: Δlog-loss {ci(d['log_loss'])} lies entirely above 0")
        if ece_c - ece_p > MAX_ECE_INCREASE:
            fails.append(f"calibration: ECE {ece_p:.4f} → {ece_c:.4f} (more than +{MAX_ECE_INCREASE})")
        if d["vs_market"]["ci_lo"] <= 0:
            warns.append(f"no significant edge over the market: ΔAUC {ci(d['vs_market'])}")
        out["models"][name] = {
            "fails": fails, "warnings": warns,
            "auc_diff": {k: round(float(d["auc"][k]), 5) for k in ("point", "ci_lo", "ci_hi")},
            "log_loss_diff": {k: round(float(d["log_loss"][k]), 5) for k in ("point", "ci_lo", "ci_hi")},
            "ece": [round(ece_p, 5), round(ece_c, 5)],
        }
        out["passed"] &= not fails
    return out


def verdict_section(checks: dict, prod: dict, cand: dict) -> str:
    rows = [[SHORT[n], "FAIL" if c["fails"] else "pass", "<br>".join(c["fails"] + [f"warning: {w}" for w in c["warnings"]])
             or "—"] for n, c in checks["models"].items()]
    head = (f"**PASS** — {cand['release']} can be promoted: `python scripts/release.py promote {cand['release']}`"
            if checks["passed"] else
            f"**FAIL** — {cand['release']} is rejected; {prod['release']} stays in production.")
    return ("## Verdict\n\n" + head + "\n\nChecks per model on every test snapshot: no significant AUC regression "
            "(95% CI of ΔAUC entirely below 0), no significant log-loss regression (CI entirely above 0), expected "
            f"calibration error at most {MAX_ECE_INCREASE} above production. A missing edge over the market only "
            "warns.\n\n" + table(["Model", "Result", "Details"], rows))


# ─────────────────────────────────────────────

def build(test: pd.DataFrame, preds: dict, prod: dict, cand: dict, n_boot: int) -> tuple[str, dict]:
    """preds: model name → (production's P(YES), candidate's P(YES)) on `test`. Returns (markdown, checks)."""
    metrics, diffs = metrics_section(test, preds, prod, cand, n_boot)
    checks = run_checks(diffs)
    md = "\n\n".join([
        f"# Release {cand['release']} vs production {prod['release']}",
        f"Candidate {cand['release']}: dataset {cand['dataset_version']}, commit {cand['code']['git_commit'][:7]}, "
        f"created {cand['created_at']}. Production {prod['release']}: dataset {prod['dataset_version']}, "
        f"commit {prod['code']['git_commit'][:7]}. Generated by `scripts/release.py report` (ADR-019).",
        verdict_section(checks, prod, cand),
        data_section(prod, cand),
        code_section(prod, cand),
        metrics,
        attribution_section(test, preds, prod, cand),
        behaviour_section(test, preds, prod, cand),
    ])
    return md + "\n", checks
