from __future__ import annotations

import numpy as np
import pytest

from ftir_analysis.training import _build_epoch_summary, _epoch_summary_payload


def test_epoch_summary_payload_includes_selection_and_group_metrics() -> None:
    all_log_mae = np.array([3.0, 2.0, 1.0, 1.5, 1.0, 0.5, 0.8, 0.7, 0.6, 0.4, 0.3], dtype=np.float32)
    synth_log_mae = all_log_mae + 0.1
    ref_log_mae = all_log_mae - 0.2

    summary = _build_epoch_summary(
        epoch=4,
        val_loss=1.23,
        val_log_mae=all_log_mae,
        synth_log_mae=synth_log_mae,
        ref_log_mae=ref_log_mae,
        zero_baseline_mean_all=1.9,
        zero_baseline_mean_ref=0.8,
        species_beating_zero=6,
    )
    payload = _epoch_summary_payload(summary)

    assert payload["selection_metric"] == "ref_val_log_mae_mean"
    assert payload["selection_metric_value"] == pytest.approx(float(ref_log_mae.mean()))
    assert payload["species_beating_zero_baseline"] == 6

    grouped = payload["group_log_mae_mean"]
    assert grouped["all"]["major"] == pytest.approx(np.mean(all_log_mae[:6]))
    assert grouped["all"]["trace"] == pytest.approx(np.mean(all_log_mae[6:]))
    assert grouped["synth"]["major"] == pytest.approx(np.mean(synth_log_mae[:6]))
    assert grouped["ref"]["trace"] == pytest.approx(np.mean(ref_log_mae[6:]))
