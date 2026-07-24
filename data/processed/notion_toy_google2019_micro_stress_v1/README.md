# Google 2019 micro capacity-pressure benchmark

This directory is derived from the canonical v5 pool without modifying it. It contains the top ten eligible on-demand, spot, and batch units by coverage-weighted average CPU+memory usage.  The monotone OD-memory stress transform is intentionally synthetic and proves a five-server scenario-0 lower bound; it must not be described as a representative Google-trace sample.  See `stress_transform.json`, `selection_manifest.csv`, and `validation_report.json` for the audit trail.
