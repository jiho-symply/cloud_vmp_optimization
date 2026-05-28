# References for the Stochastic Temporal VMP Model

## Summary

This directory justifies the current model as a stochastic temporal virtual-machine-placement abstraction. The argument is model-centric: Azure, Google, Alibaba, Bitbrains, or other traces may motivate parameters, but the formulation is not an Azure CPU-trace-only study.

Start with [09_notation_and_terms.md](09_notation_and_terms.md) if the model notation or terminology is unfamiliar. The topic files use occasional monospace variable names for implementation readability, but the intended paper-style notation is summarized there.

The main literature finding is important:

- No verified Q1/top-venue paper found here contains the exact full combination of temporal VM placement, stochastic realized VM resource usage, two-stage recourse, on-demand migration, spot/preemptible suspension, Managed Batch reservation, server activation, and CPU-only capacity.
- The model is nevertheless defensible as a synthesis of established modeling families: temporal VMP/TBPP, stochastic/chance-constrained bin packing, stochastic VM placement/load balancing, priority-aware cloud scheduling, and energy-aware consolidation.
- Strong claims should be made about the abstraction's usefulness for studying tradeoffs. Production-completeness claims should be avoided.

## Argument Map

| Modeling choice | Verdict | Strength | Primary support | Main caveat |
|---|---|---|---|---|
| Temporal VMP as bin packing over time | Natural and well supported. | Strong | Aydin et al. COR 2020; Dell'Amico et al. COR 2020; Martinovic et al. EJOR 2023; Muir et al. INFORMS Journal on Optimization 2024; Buchbinder et al. SIGMETRICS 2021; Barbalho et al. MLSys 2023. | Deterministic TBPP and prediction-based dynamic bin-packing papers do not by themselves justify stochastic recourse or workload classes. |
| Stochastic realized VM usage | Well supported as a bridge. | Strong | Cohen et al. Management Science 2019; Zhou et al. IEEE Access 2019; Chen et al. FGCS 2020; Yu et al. IEEE TCC 2020; Yan et al. KDD 2022; Hong et al. POMACS/SIGMETRICS 2023/2024; Martinovic and Selch COR 2021. | Exact stochastic temporal VMP with the current recourse structure was not found; scenario probabilities and correlations need calibration. |
| SLA as resource contention / overload risk | Defensible as a placement-level proxy. | Moderate | Cohen et al. Management Science 2019; Yu et al. IEEE TCC 2020; Zhou et al. IEEE Access 2019; Yan et al. KDD 2022. | Resource overflow is not the same as latency, availability, queueing delay, deadline miss, or tenant-level SLA unless those metrics are explicitly mapped. |
| On-demand, spot/preemptible, and Managed Batch roles | Realistic as scheduling roles. | Strong | Borg EuroSys 2015; Resource Central SOSP 2017; Curino et al. SoCC 2014; SpotOn SoCC 2015; SpotCheck EuroSys 2015; SciSpot IEEE TPDS 2022. | Spot/preemptible is a capacity contract or priority property; batch/deferable is a workload/service property. They can overlap in real systems. |
| On-demand migration recourse | Valid as a high-level control lever. | Moderate | Ahmad et al. JNCA 2015; Le Computer Science Review 2020; Clark et al. NSDI 2005; pMapper Middleware 2008. | A binary migration event does not model bandwidth, downtime, dirty pages, memory footprint, or source/destination double reservation. |
| Spot/preemptible suspension | Better framed as curtailment/preemption unless state is modeled. | Moderate for curtailment, weak for true suspension | SpotOn SoCC 2015; SpotCheck EuroSys 2015; SciSpot IEEE TPDS 2022; Resource Deflation EuroSys 2019; VM Deflation HPDC 2020. | True suspension needs state preservation, checkpoint/restart or hibernation cost, warning/lifetime semantics, capacity reclamation, and service-loss accounting. |
| Managed Batch reservation and fluid processing | Defensible for aggregate divisible or many-task work. | Moderate | Curino et al. SoCC 2014; Apollo OSDI 2014; TR-Spark SoCC 2016; Kang et al. JPDC 2018; Yan et al. KDD 2022. | Weak for a single indivisible job, precedence-constrained workflow, stateful job, or exact deadline schedule without integer task variables. |
| Two-stage stochastic planning | Appropriate for planned capacity plus scenario recourse. | Moderate to strong | Crainic et al. EJOR 2016; Shen and Wang Service Science 2014; Zhang et al. INFORMS JOC 2020. | Multi-stage, online, or rolling-horizon/MPC models may be better when decisions are revised continuously. |
| 24-hour horizon | Defensible only as a daily/look-ahead abstraction. | Moderate | Shen and Wang Service Science 2014; Youssef and Krishnamurthy 2017; SciSpot IEEE TPDS 2022; Gaggero and Caviglione IEEE TASE 2019. | Not universal; use carry-over state, terminal penalties, or rolling horizon for jobs crossing the boundary. |
| CPU-only capacity | Acceptable as a single-bottleneck prototype. | Weak to moderate | CPU-utilization energy models; scalar TBPP papers; CPU-focused overcommitment studies. | For credible general VMP, CPU+memory is the minimum. Network/storage/topology are needed for VDCs, data locality, bandwidth guarantees, or I/O-sensitive workloads. |
| CPU+memory and richer resources | Recommended extension path. | Strong | Borg EuroSys 2015; DRF NSDI 2011; Tetris SIGCOMM 2014; Chen et al. FGCS 2020; Netsolver Artificial Intelligence 2020; Meng et al. INFOCOM 2010. | More dimensions require trace support and increase solve complexity; adding unverified dimensions can create false precision. |
| Powered-server time and load-dependent energy | Useful placement-level server-energy proxy. | Moderate to strong | Dayarathna et al. IEEE COMST 2016; Jin et al. Applied Energy 2020; Zhang et al. SIMPAT 2013; Beloglazov et al. FGCS 2012; Meroni and Guitart FGCS 2026. | Not facility energy. Cooling, PUE, network power, storage power, thermal effects, heterogeneous nonlinear profiles, and migration dynamics are omitted. |
| Fire-up cost | Should be modeled separately from powered time. | Strong as caveat | Aydin et al. COR 2020; Martinovic et al. COR 2021. | $\sum_{s,t} u_{st}$ charges active periods. Fire-up needs transition variables such as $q_{st} \ge u_{st}-u_{s,t-1}$. |

## File Guide

- [00_current_model_summary.md](00_current_model_summary.md): summarizes the repository formulation and states the supported claim.
- [01_temporal_bin_packing_and_vmp.md](01_temporal_bin_packing_and_vmp.md): justifies the temporal VMP/TBPP skeleton and explains the bridge to stochastic usage.
- [02_workload_classes_od_sp_batch.md](02_workload_classes_od_sp_batch.md): analyzes on-demand, spot/preemptible, and Managed Batch as scheduling roles.
- [03_recourse_migration_suspension_batch.md](03_recourse_migration_suspension_batch.md): reviews migration, spot curtailment/suspension, and Managed Batch processing recourse.
- [04_stochastic_demand_and_risk.md](04_stochastic_demand_and_risk.md): covers stochastic VM usage, chance constraints, CVaR, robust alternatives, and SLA-risk proxies.
- [05_two_stage_24h_planning.md](05_two_stage_24h_planning.md): compares two-stage, multi-stage, online, and rolling-horizon models and qualifies the 24-hour horizon.
- [06_resource_dimensions.md](06_resource_dimensions.md): explains CPU-only as a prototype and CPU+memory/network/storage as stronger placement models.
- [07_energy_and_power_model.md](07_energy_and_power_model.md): reviews powered-server time, load-dependent power, fire-up distinction, and migration energy/cost.
- [08_post2017_q1_stochastic_temporal_vmp_screen.md](08_post2017_q1_stochastic_temporal_vmp_screen.md): screens post-2017 Q1/top-venue papers for the stochastic temporal VMP/TBPP intersection.
- [09_notation_and_terms.md](09_notation_and_terms.md): defines terms, sets, parameters, variables, representative constraints, and paper-style equations aligned with the Notion model page.

## Recommended Framing

Use this wording:

> We model provider-side cloud operation as a stochastic temporal VM placement problem. The formulation combines temporal bin-packing for time-windowed placement, stochastic/chance-constrained bin packing for uncertain realized resource usage, priority-aware recourse for on-demand and revocable workloads, Managed Batch reservation for delay-tolerant aggregate work, and a placement-level server-energy objective. The model is a compact research abstraction, not a complete production scheduler.

Use this wording for the 24-hour horizon:

> The 24-hour horizon is a daily planning or look-ahead window. It is appropriate when workload forecasts, batch-service windows, energy accounting, or transient-VM semantics have daily structure. Online, multi-stage, and rolling-horizon formulations are natural alternatives when decisions are revised continuously.

Use this wording for SLA:

> Resource contention and capacity overflow are modeled as placement-level SLA-risk proxies. They capture capacity-driven violations but do not fully model latency, availability, deadline miss, queueing delay, or tenant-level service guarantees.

Avoid these stronger claims:

- "The literature already contains this exact model."
- "CPU-only is sufficient for production VMP."
- "All cloud workloads naturally fall into on-demand, spot, and batch classes."
- "Spot suspension is realistic without checkpoint, restart, hibernation, or state-cost modeling."
- "A 24-hour horizon is the standard for cloud data-center operation."
- "The energy objective predicts full data-center energy consumption."

## Bottom Line

The current model is defensible as a stochastic temporal VMP/TBPP prototype for studying consolidation, server activation, stochastic usage, migration, revocable-capacity curtailment, Managed Batch reservation, and SLA-risk tradeoffs. The strongest paper narrative is that the formulation combines well-established components in a compact MILP. The limitations are also clear: CPU-only capacity, simplified migration/suspension, aggregate batch processing, resource-contention SLA, and a conditional 24-hour horizon must be presented as modeling choices rather than production-complete assumptions.
