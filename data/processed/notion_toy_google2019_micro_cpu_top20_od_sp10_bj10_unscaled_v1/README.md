# Google 2019 CPU-heavy OD20 + frozen SP10/BJ10 fixture

This directory is derived from the canonical v5 pool without modifying it. The 20 eligible on-demand VMs with the largest coverage-weighted scenario-0 mean CPU usage are selected. The 10 spot and 10 batch-candidate IDs, including their within-class order, are copied exactly from the existing unscaled top-10 reference fixture.

All output request rows and scenario-0 usage rows are copied from the canonical source. Configured q, resource requests, CPU and memory usage, lifecycle, and provenance values are not changed. There is no target-q scaling, usage rescaling, or memory-pressure transform.

The frozen spot guarantee applies to the stored request and canonical scenario-0 usage rows. The current optimization loader regenerates later OD/SP scenarios from one RNG stream, so changing the OD set can change synthetic spot draws even though the stored spot source data is identical.

The workload-class labels are Google trace proxy classes, not observed public-cloud purchase types. The optimization loader later generates the remaining workload scenarios and converts the selected batch candidates into flexible workload families. See `selection_policy.json`, `selection_manifest.csv`, and `validation_report.json` for exact provenance and validation.
