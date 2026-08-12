# Google 2019 unscaled top-average-usage micro fixture

This directory is derived from the canonical v5 pool without modifying it. The ten eligible rows in each class with the largest coverage-weighted mean CPU plus memory usage are selected. Observed scenario-0 usage, q, requests, lifecycle, and provenance values are copied exactly. There is no target-q scaling, usage rescaling, memory-pressure transform, or minimum-server guarantee.

The `on_demand`, `spot`, and `batch_candidate` labels are Google trace proxy classes, not observed public-cloud purchase types. The optimization loader later converts selected batch candidates into at most three flexible workload families. See `selection_policy.json`, `selection_manifest.csv`, and `validation_report.json`.
