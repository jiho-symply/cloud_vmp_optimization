# Current Model Summary

## Claim

The current model is a research prototype for stochastic temporal virtual machine placement. It is best described as a two-stage stochastic temporal bin-packing model with priority-aware recourse. Physical servers are bins, VM and batch workloads consume server-time capacity, the first stage selects a baseline plan, and each demand scenario triggers migration, suspension, overload accounting, and batch-processing decisions.

The model should be positioned as a synthesis of several established modeling lines rather than as a direct reproduction of one prior paper. Deterministic temporal VMP/TBPP papers justify the time-windowed placement skeleton, stochastic bin-packing and stochastic VMP papers justify random realized resource usage, and cloud systems papers justify workload-priority and recourse mechanisms.

## How the Current Model Matches

The first stage chooses time-indexed server activation and baseline workload placement before scenario-specific demand is known. In paper terminology, these are here-and-now decisions. The full notation is collected in `09_notation_and_terms.md`; the same variables are summarized here:

- $u_{st}$ and $u_s$ represent powered servers and horizon-level server usage.
- $x_{is}$ assigns each on-demand VM to an initial server.
- $y_{js}$ assigns each spot/preemptible VM to a baseline server.
- $b_{kst}$ reserves server-time processing slots for each managed batch job.

The second stage reacts to each demand scenario. These are recourse decisions:

- $x^R_{ist}(\xi)$ is the realized on-demand placement after optional migration.
- $m_{it}(\xi)$ detects whether an on-demand VM changes server between active periods, with at most one migration per VM per scenario.
- $y^R_{jst}(\xi)$ is the realized active state of a spot/preemptible VM after server-time curtailment.
- $z_{kst}(\xi)$ is continuous processed batch volume on reserved slots.
- $\phi_{st}(\xi)$, $\eta_s(\xi)$, $\gamma_{st}(\xi)$, and $\delta_j(\xi)$ track on-demand overload/SLA violation and low-priority suspension or curtailment events.

The model uses scenario probabilities in both the risk constraints and the energy objective. The baseline formulation enforces chance-style service limits for on-demand and spot classes; related variants replace these with CVaR-style risk constraints, robust all-scenario capacity satisfaction, or multi-stage/rolling reoptimization.

SLA is represented through resource contention and overload indicators. This is defensible for a placement-level capacity model because several cloud bin-packing and overcommitment papers use overflow probability or capacity-violation probability as the service-risk object. It remains a proxy: application-level SLA may instead be latency, availability, throughput, deadline completion, or tenant-level fairness.

The capacity abstraction is currently scalar CPU capacity. This is defensible as a narrow prototype or as a model for CPU-dominant workloads, but it is not a complete production-grade VMP resource model. For realistic general-purpose cloud placement, CPU and memory are usually minimum dimensions, and network/storage constraints become important for communication-heavy, data-intensive, or storage-sensitive workloads.

Batch work is best interpreted as a Managed Batch style planning abstraction: the platform accepts delay-tolerant work, reserves or allocates capacity ahead of time, and processes an aggregate backlog across the horizon. The continuous $z_{kst}(\xi)$ variable is appropriate for divisible work, many independent tasks, or an aggregate processing requirement. It should not be used to claim exact modeling of a single indivisible workflow unless task-level constraints are added.

## Evidence Table

This file defines the model to be justified by the other topic files. Evidence appears in `01_*.md` through `07_*.md`.

| Paper | Venue | Verified link | What it supports | Limits for this model |
|---|---|---|---|---|
| Aydin et al., "Multi-objective temporal bin packing problem: An application in cloud computing" | Computers & Operations Research 2020 | [DOI](https://doi.org/10.1016/j.cor.2020.104959) | Directly supports the time-windowed VMP-as-TBPP skeleton and server fire-up objectives. | Deterministic requests; no stochastic recourse or workload classes. |
| Cohen et al., "Overcommitment in Cloud Services: Bin Packing with Chance Constraints" | Management Science 2019 | [DOI](https://doi.org/10.1287/mnsc.2018.3091) | Supports using capacity-overflow probability as a cloud overcommitment/service-risk abstraction. | Online overcommitment model, not temporal two-stage VMP. |
| Yan et al., "Solving the Batch Stochastic Bin Packing Problem in Cloud" | ACM SIGKDD 2022 | [DOI](https://doi.org/10.1145/3534678.3539334) | Supports cloud batch packing with stochastic dynamic resource usage and chance constraints. | Container/service batch packing, not VM placement with migration and spot suspension. |
| Borg and Resource Central papers | EuroSys 2015; SOSP 2017 | [Borg DOI](https://doi.org/10.1145/2741948.2741964); [Resource Central DOI](https://doi.org/10.1145/3132747.3132772) | Support priority-aware large-cluster/cloud resource management, overcommitment, prediction, and class-aware scheduling. | Systems evidence, not a direct MILP/TBPP formulation. |

## Different Modeling Choices in the Literature

The model intentionally abstracts away several dimensions common in production systems and in richer placement papers. These are not defects if the paper labels them as scope boundaries:

- Heterogeneous physical machines and VM types.
- Multi-resource capacity vectors including memory, network, disk, storage I/O, and accelerator resources.
- Explicit migration arcs, migration duration, dirty-page behavior, and source/destination double reservation.
- Network topology, affinity/anti-affinity, latency, and fault-domain constraints.
- Online arrivals, prediction updates, and rolling reoptimization.
- Explicit job precedence, checkpointing, replication, and deadline constraints for batch workloads.
- Application-level SLA metrics such as latency, availability, tail response time, and per-tenant fairness.

## Caveats

The current formulation should not be presented as a full production scheduler. It is a compact research abstraction designed to isolate stochastic demand, temporal capacity, priority classes, migration/suspension recourse, Managed Batch style processing, and energy-aware consolidation in one MILP family.

The strongest result claim should be comparative rather than universal: the model can compare how risk controls, migration limits, low-priority curtailment, batch reservation, and server activation costs interact under the chosen scenario set. It cannot prove that CPU-only placement, simple suspension, or a 24-hour horizon is universally sufficient.

## Verdict

The model is well framed as a stochastic temporal VMP/TBPP prototype. The strongest claims should be about the suitability of the abstraction for studying risk-aware consolidation and recourse tradeoffs. Claims about production completeness should be limited and explicitly caveated.
