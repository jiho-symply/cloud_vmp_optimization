from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/audit_google2019_residual_fix.py"
SPEC = importlib.util.spec_from_file_location("audit_google2019_residual_fix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _selection(*keys: str) -> dict[str, object]:
    return {
        "selected": {
            "on_demand": [
                {"stable_key": key, "vm_id": f"vm-{key}"} for key in keys
            ],
            "spot": [],
            "batch_jobs": [],
        }
    }


def test_selection_comparison_uses_stable_raw_keys_not_vm_ids() -> None:
    before = _selection("1:2", "3:4")
    after = {
        "selected": {
            "on_demand": [
                {"stable_key": "1:2", "vm_id": "renumbered-a"},
                {"stable_key": "5:6", "vm_id": "renumbered-b"},
            ],
            "spot": [],
            "batch_jobs": [],
        }
    }

    comparison = audit._selection_comparison(before, after)["on_demand"]

    assert comparison["stable_key_set_unchanged"] is False
    assert [row["stable_key"] for row in comparison["removed"]] == ["3:4"]
    assert [row["stable_key"] for row in comparison["added"]] == ["5:6"]


def test_histogram_quantile_is_frequency_weighted() -> None:
    histogram = audit.Counter({1_000_000: 1, audit.T5_US: 5})

    assert audit._histogram_quantile(histogram, 0.50) == audit.T5_US
    assert audit._histogram_quantile(histogram, 0.95) == audit.T5_US
