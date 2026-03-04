"""Data audit report generation for FTIR manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import REPORTS_DIR


def _explode_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, rec in df.iterrows():
        flags_raw = str(rec.get("quality_flags", "") or "").strip()
        if not flags_raw:
            continue
        for flag in [f.strip() for f in flags_raw.split(";") if f.strip()]:
            rows.append(
                {
                    "sample_id": rec["sample_id"],
                    "source_path": rec["source_path"],
                    "species": rec["species"],
                    "split": rec["split"],
                    "issue": flag,
                }
            )
    return pd.DataFrame(rows)


def _find_unindexed_lab_files(reference_root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    indexed_paths = {Path(p).resolve() for p in manifest["source_path"].tolist()}

    rows: list[dict[str, object]] = []
    for lab in reference_root.rglob("*.lab"):
        rp = lab.resolve()
        if rp not in indexed_paths:
            rows.append(
                {
                    "sample_id": "",
                    "source_path": str(rp),
                    "species": "",
                    "split": "",
                    "issue": "filesystem_not_in_manifest",
                }
            )

    for lab in reference_root.rglob("*.LAB"):
        rp = lab.resolve()
        if rp not in indexed_paths:
            rows.append(
                {
                    "sample_id": "",
                    "source_path": str(rp),
                    "species": "",
                    "split": "",
                    "issue": "filesystem_not_in_manifest",
                }
            )

    return pd.DataFrame(rows)


def audit_manifest(
    manifest_path: Path,
    *,
    reference_root: Path,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Path]:
    """Generate class distribution and anomaly reports for a manifest."""
    manifest = pd.read_csv(manifest_path)

    reports_dir.mkdir(parents=True, exist_ok=True)
    class_csv = reports_dir / "class_distribution.csv"
    anomaly_csv = reports_dir / "label_anomalies.csv"
    audit_md = reports_dir / "data_audit.md"

    # Class distribution
    grp = (
        manifest.groupby(["species", "is_sparse_class", "split"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )

    pivot = grp.pivot_table(
        index=["species", "is_sparse_class"],
        columns="split",
        values="n",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    stats = (
        manifest.groupby("species")["concentration_ppmv"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
    )

    class_df = stats.merge(pivot, on="species", how="left")
    class_df = class_df.sort_values(["is_sparse_class", "species"]).reset_index(drop=True)
    class_df.to_csv(class_csv, index=False)

    # Anomalies
    quality_issues = _explode_quality_flags(manifest)
    unindexed = _find_unindexed_lab_files(reference_root, manifest)
    anomalies = pd.concat([quality_issues, unindexed], ignore_index=True)
    anomalies = anomalies.sort_values(["issue", "species", "source_path"]).reset_index(drop=True)
    anomalies.to_csv(anomaly_csv, index=False)

    # Markdown summary
    total = len(manifest)
    n_sparse = int(manifest["is_sparse_class"].sum())
    by_split = manifest["split"].value_counts().to_dict()
    issue_counts = anomalies["issue"].value_counts().to_dict()

    lines = [
        "# FTIR Data Audit",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Total rows: `{total}`",
        f"- Sparse-class rows: `{n_sparse}`",
        f"- Splits: `{by_split}`",
        "",
        "## Outputs",
        f"- Class distribution: `{class_csv}`",
        f"- Label anomalies: `{anomaly_csv}`",
        "",
        "## Issue Counts",
    ]

    if issue_counts:
        for issue, count in sorted(issue_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{issue}`: `{count}`")
    else:
        lines.append("- None")

    with audit_md.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    return {
        "class_distribution": class_csv,
        "label_anomalies": anomaly_csv,
        "data_audit": audit_md,
    }
