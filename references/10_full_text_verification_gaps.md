# Full-Text Verification Gaps

## Purpose

This file records claims that are useful for the current stochastic temporal VM placement argument but should be checked against the full paper before they are used as strong evidence in a manuscript.

The entries below are not invalid citations. Most of them have stable DOI, publisher, DBLP, institutional, or author-page verification elsewhere in this directory. The issue is narrower: some detailed modeling claims should be confirmed from the full text, formulation, experiments, or system design sections before the paper relies on them heavily.

Use this file when collecting PDFs. After reading a paper, update the relevant topic file and mark the entry here as one of:

- Confirmed: the full text supports the claim as written.
- Narrowed: the full text supports a weaker or more conditional claim.
- Rejected: the claim should be removed or replaced.
- Superseded: a better source now supports the same point.

## Highest-Priority Claim Gaps

| Priority | Claim to verify | Candidate papers | What to check in the full text | If unsupported, revise to |
|---|---|---|---|---|
| P0 | VM placement can be represented as temporal bin packing in a cloud setting, with VM time windows, capacity constraints, and server activation/fire-up objectives. | Aydin et al. COR 2020; de Cauwer et al. IEEE ICTAI 2016; Dell'Amico et al. COR 2020; Martinovic et al. COR 2021; Martinovic et al. EJOR 2023. | Confirm whether VMs or jobs have fixed start/end times, whether physical machines are bins, whether capacity is checked over time, whether fire-up is a transition count or active-server state, and whether the model is single-resource or multi-resource. | Keep temporal bin packing as the deterministic skeleton, but avoid saying that these papers justify stochastic recourse, workload classes, or CPU-only production realism. |
| P0 | No single Q1/top-venue paper found here exactly combines temporal VM placement, stochastic realized VM resource usage, two-stage recourse, migration, spot/preemptible suspension, Managed Batch reservation, and energy-aware activation. | Aydin et al. COR 2020; Yan et al. KDD 2022; Hong et al. POMACS/SIGMETRICS 2023/2024; Yu et al. IEEE TCC 2020; Zhou et al. IEEE Access 2019; Chen et al. FGCS 2020; Zhang et al. INFORMS JOC 2020; Bulbul et al. EJOR 2021; Meroni and Guitart FGCS 2026. | Check whether any paper actually includes both temporal windows and stochastic item/resource usage, and whether it includes operational recourse such as migration, suspension/preemption, batch processing, and energy. | Present the model as a synthesis of established components, not as a direct extension of one exact prior formulation. |
| P0 | Stochastic realized VM usage is defensible as random or time-varying item size in placement/bin-packing constraints. | Cohen et al. Management Science 2019; Zhou et al. IEEE Access 2019; Chen et al. FGCS 2020; Yu et al. IEEE TCC 2020; Hong et al. POMACS/SIGMETRICS 2023/2024; Martinovic and Selch COR 2021; Martinovic et al. TOP 2022. | Confirm whether uncertainty is in per-VM resource size, aggregate demand, arrivals, lifetime, bandwidth, or algorithmic randomization. Check independence/correlation assumptions, distributional assumptions, and whether overflow is per-server, global, or expected-cost based. | State exactly which uncertainty type is supported. Do not cite lifetime or arrival uncertainty as direct support for stochastic CPU usage unless the paper models resource-size uncertainty. |
| P0 | SLA as resource contention or overload risk is a valid placement-level proxy. | Cohen et al. Management Science 2019; Yu et al. IEEE TCC 2020; Zhou et al. IEEE Access 2019; Chen et al. FGCS 2020; Resource Central SOSP 2017; Borg EuroSys 2015. | Check how each paper defines violation, overload, hotspot, service quality, or SLA. Identify whether the risk event is physical capacity overflow, probabilistic overflow, latency degradation, availability, throughput, job delay, or cost. | Use "resource-contention risk proxy" rather than "SLA compliance" unless the cited paper directly maps contention to the target SLA metric. |
| P0 | Managed Batch can be modeled as provider-side capacity reservation plus aggregate processing of delay-tolerant work. | Curino et al. SoCC 2014; Apollo OSDI 2014; Yan et al. KDD 2022; TR-Spark SoCC 2016; Kang et al. JPDC 2018; SciSpot IEEE TPDS 2022. | Confirm whether the system accepts batch jobs, reserves future resources, allocates guaranteed resources, schedules many tasks, uses divisible work, or handles deadlines. Check whether the model is provider-side scheduling, customer-side spot management, or cluster workload management. | Describe $b_{kst}$ and $z_{kst}(\xi)$ as an aggregate/divisible planning relaxation for many independent tasks or server-time accounting, not exact task-level scheduling. |
| P0 | Spot "suspension" is realistic only if preemption state, reclaimed capacity, checkpoint/restart or migration cost, warning/lifetime semantics, and lost-service accounting are represented. | SpotOn SoCC 2015; SpotCheck EuroSys 2015; SciSpot IEEE TPDS 2022; Resource Deflation EuroSys 2019; Cloud-scale VM Deflation HPDC 2020; TR-Spark SoCC 2016. | Check whether the paper uses termination, preemption, suspension, checkpointing, hibernation, migration, replication, deflation, market switching, or fallback to on-demand. Check whether it models warning time, maximum lifetime, preemption probability, lost work, or resumed state. | Prefer "curtailment," "preemption," or "unserved low-priority VM-time" unless state preservation and restart/suspend costs are explicitly modeled. |
| P0 | Two-stage stochastic planning over a 24-hour horizon is defensible as a daily/look-ahead abstraction, not a universal production scheduler. | Crainic et al. EJOR 2016; Shen and Wang Service Science 2014; Zhang et al. INFORMS JOC 2020; Bulbul et al. EJOR 2021; Gaggero and Caviglione IEEE TASE 2019; Youssef and Krishnamurthy 2017; SciSpot IEEE TPDS 2022. | Confirm the decision stages, nonanticipativity structure, recourse variables, planning interval length, and whether 24 hours is a daily workload cycle, product semantic, or merely an experimental horizon. | Use "24-hour planning/look-ahead window" and discuss rolling-horizon or multi-stage alternatives. Avoid saying cloud VMP is naturally 24-hour. |
| P0 | CPU-only capacity is acceptable only as a prototype or CPU-dominant abstraction; CPU+memory is the minimum credible general VMP vector. | Borg EuroSys 2015; DRF NSDI 2011; Tetris SIGCOMM 2014; Chen et al. FGCS 2020; Netsolver AI 2020; Wei Chen et al. FGCS 2020; Lopez-Pires et al. FGCS 2018; Meroni and Guitart FGCS 2026. | Check which resources are modeled as hard constraints, soft usage, bottlenecks, or objectives. Confirm whether CPU and memory are both enforced in VMP formulations, and when network/storage/topology are first-order. | Keep CPU-only as a deliberate single-resource prototype. For production realism, add CPU+memory first, then network/storage/topology only when needed by workload/SLA claims. |

## Paper-Level Reading Queue

### Temporal VMP/TBPP Skeleton

| Priority | Paper | Current use | Full-text questions |
|---|---|---|---|
| P0 | Aydin, Muter, and Birbil, "Multi-objective temporal bin packing problem: An application in cloud computing" (COR 2020). | Main support for VMP-as-TBPP and server fire-ups. | Does the formulation explicitly use VM requests, fixed lifetimes, and physical servers? Is capacity scalar CPU? How exactly are fire-ups defined? Does the paper mention why one resource is used? |
| P1 | Dell'Amico, Furini, and Iori, "A branch-and-price algorithm for the temporal bin packing problem" (COR 2020). | Mathematical TBPP backbone. | Confirm item-window and capacity definitions, continuous-vs-discrete time handling, objective, and whether there are any remarks relevant to cloud or server activation. |
| P1 | Martinovic, Strasdat, and Selch, "Compact integer linear programming formulations for the temporal bin packing problem with fire-ups" (COR 2021). | Fire-up/activation modeling support. | Confirm whether fire-up is a transition variable, whether the objective is bin count plus fire-up count, and whether the model is homogeneous/deterministic. |
| P1 | Martinovic, Strasdat, Valerio de Carvalho, and Furini, "A combinatorial flow-based formulation for temporal bin packing problems" (EJOR 2023). | Recent high-quality TBPP/TBPP-FU support. | Confirm problem variants covered and whether anything extends toward stochasticity, active-bin time, or server activation beyond deterministic TBPP. |
| P1 | Muir, Marshall, and Toriello, "Temporal Bin Packing with Half-Capacity Jobs" (INFORMS Journal on Optimization 2024). | Active-bin-time objective and cloud motivation. | Check whether "time-averaged number of active bins" is the exact objective and how directly the paper motivates cloud computing. |
| P2 | Martinovic and Strasdat 2024/2025 TBPP-FU and busy-time papers. | Recent OR context for active-server time and heuristics. | Confirm whether these are mostly theoretical/heuristic support and not direct stochastic VMP evidence. |

### Stochastic Usage, Risk, and SLA Proxy

| Priority | Paper | Current use | Full-text questions |
|---|---|---|---|
| P0 | Cohen et al., "Overcommitment in Cloud Services: Bin Packing with Chance Constraints" (Management Science 2019). | Strong cloud overcommitment/chance constraint evidence. | What is the exact random variable? How is violation probability defined? Is the claim about requested capacity versus realized utilization? Does it discuss SLA or only capacity violation? |
| P0 | Yu et al., "Stochastic Load Balancing for Virtual Resource Management in Datacenters" (IEEE TCC 2020). | Stochastic VM workload, overload probability, migration, and SLA-contention bridge. | Confirm workload model, resource dimensions, migration-cost model, and wording linking hotspots/resource overload to SLA degradation. |
| P0 | Yan et al., "Solving the Batch Stochastic Bin Packing Problem in Cloud" (KDD 2022). | Recent cloud-scale stochastic packing and chance constraints. | Are objects containers, services, jobs, or VMs? Is the packing daily/batch? What is "Used Capacity at Confidence"? Does it support managed-batch reservation or only stochastic packing onto nonempty machines? |
| P0 | Hong, Xie, and Wang, "Near-Optimal Stochastic Bin-Packing in Large Service Systems with Time-Varying Item Sizes" (POMACS/SIGMETRICS 2023/2024). | Closest theoretical support for time-varying stochastic item sizes motivated by VM/container scheduling. | Confirm the service-system assumptions, time-varying item-size process, arrival/departure model, and whether the VM scheduling motivation is explicit or illustrative. |
| P1 | Zhou et al., "Stochastic Virtual Machine Placement for Cloud Data Centers Under Resource Requirement Variations" (IEEE Access 2019). | Direct stochastic VMP with CPU/memory random requirements. | Confirm exact probability constraints, whether both CPU and memory are random, and how the solution method differs from MILP/TBPP. |
| P1 | Chen et al., "Stochastic scheduling for variation-aware virtual machine placement in a cloud computing CPS" (FGCS 2020). | Variation-aware stochastic VMP support. | Confirm whether CPU/memory request variation is modeled probabilistically and whether this is placement, scheduling, or feasibility filtering. |
| P1 | Martinovic and Selch, "Mathematical models and approximate solution approaches for the stochastic bin packing problem" (COR 2021). | OR support for stochastic item-size bin packing. | Confirm distribution assumptions, exact formulation, approximation method, and whether server consolidation is an explicit motivation or only a possible application. |
| P1 | Martinovic, Haehnel, Scheithauer, and Dargie, "An introduction to stochastic bin packing-based server consolidation with conflicts" (TOP 2022). | Stochastic server consolidation with conflicts. | Confirm whether the paper explicitly mentions stochastic TBPP as a future/natural extension and what Google data-center instances are used. |
| P2 | Thabet et al., CVaR for co-location attack risk in VMP (FGCS 2023). | Cloud-specific evidence that CVaR can be embedded in VMP. | Confirm the CVaR formulation and make sure it is not cited for resource contention or stochastic CPU usage. |

### Two-Stage, Multi-Stage, and 24-Hour Planning

| Priority | Paper | Current use | Full-text questions |
|---|---|---|---|
| P0 | Shen and Wang, "Stochastic Modeling and Approaches for Managing Energy Footprints in Cloud Computing Service" (Service Science 2014). | Closest finite-period stochastic cloud energy/service planning support. | Confirm exact time horizon, scenario MIP structure, switching variables, delay/backlog penalties, chance constraints, and benchmark two-stage model. |
| P0 | Crainic et al., "Logistics capacity planning: A stochastic bin packing formulation..." (EJOR 2016). | Methodological two-stage SBPP recourse support. | Confirm first-stage capacity/bin decisions, second-stage recourse, and which part is truly analogous to server activation plus scenario recourse. |
| P0 | Zhang, Denton, and Xie, "Branch and Price for Chance-Constrained Bin Packing" (INFORMS JOC 2020). | Two-stage chance-constrained bin packing machinery. | Confirm the two-stage stochastic MIP formulation, distributionally robust variant, and whether recourse has item reassignment or only feasibility accounting. |
| P1 | Bulbul, Noyan, and Erol, "Multi-stage stochastic programming models for provisioning cloud computing resources" (EJOR 2021). | Counterpoint: multi-stage cloud planning. | Confirm procurement/reservation decisions, chance constraints, horizon length, and why it is customer-side provisioning rather than provider-side VMP. |
| P1 | Gaggero and Caviglione, "MPC for Energy-Efficient, Quality-Aware, and Secure VMP" (IEEE TASE 2019). | Rolling-horizon/MPC alternative. | Confirm how horizon and replanning work, and whether it supports treating 24 hours as a look-ahead window rather than a one-shot commitment. |
| P1 | Youssef and Krishnamurthy, burstiness-aware 24-hour service planning (2017). | Concrete 24-hour planning example. | Confirm the 24-hour/30-minute interval setup and whether it is peer-reviewed enough to remain secondary evidence. |
| P1 | SciSpot (IEEE TPDS 2022). | Product-specific 24-hour preemptible-VM semantics. | Confirm exactly how 24-hour lifetime or preemption timing is modeled and avoid generalizing it to all spot markets. |

### Workload Classes, Managed Batch, and Spot/Preemptible Semantics

| Priority | Paper | Current use | Full-text questions |
|---|---|---|---|
| P0 | Curino et al., "Reservation-based Scheduling: If You're Late Don't Blame Us!" (SoCC 2014). | Strongest support for reservation-style managed batch capacity. | Confirm MILP/reservation language, production/best-effort sharing, future resource reservation, and whether this is appropriate for $b_{kst}$ as reserved server-time. |
| P0 | SpotOn (SoCC 2015). | Spot versus on-demand, batch service on spot, checkpoint/migration/replication. | Confirm fault-tolerance mechanisms, revocation warning assumptions, batch-job model, and whether the service is customer-side. |
| P0 | SpotCheck (EuroSys 2015). | Spot recourse and derivative IaaS abstraction. | Confirm whether recourse is migration, checkpoint/restore, fallback capacity, or market switching; do not cite it as simple suspension. |
| P0 | SciSpot (IEEE TPDS 2022). | Preemptible VMs and scientific bag-of-jobs support. | Confirm temporally constrained preemption model, job independence assumptions, and whether it justifies aggregate batch processing or task-level scheduling. |
| P1 | Apollo (OSDI 2014). | Managed cloud-scale batch scheduling. | Confirm guaranteed resources, future availability estimates, and how many-task scheduling differs from continuous $z_{kst}(\xi)$. |
| P1 | TR-Spark (SoCC 2016). | Batch analytics on transient resources. | Confirm whether lineage-aware checkpointing and eviction handling justify only data-parallel batch recourse, not arbitrary VM suspension. |
| P1 | Resource Deflation (EuroSys 2019) and Cloud-scale VM Deflation (HPDC 2020). | Alternative to binary suspension/preemption. | Confirm partial-resource reclamation variables, operational assumptions, and whether deflation can be modeled by a continuous spot allocation variable rather than binary service. |
| P1 | Borg EuroSys 2015 and Borg Next Generation EuroSys 2020. | Priority-aware mixed workloads and batch/service separation. | Confirm priority classes, batch tiers, overcommitment, admission control, and resource dimensions. Avoid mapping internal Borg classes too literally onto public on-demand/spot/batch products. |
| P1 | Resource Central SOSP 2017. | Prediction-informed class-aware VM scheduling. | Confirm production/non-production labels, predicted utilization/lifetime features, and whether it supports stochastic usage or only prediction-informed oversubscription. |

### Resource Dimensions and Energy

| Priority | Paper | Current use | Full-text questions |
|---|---|---|---|
| P0 | Tetris (SIGCOMM 2014). | Multi-resource packing, fragmentation, disk/network importance. | Confirm exact resource dimensions and claims about why ignoring network/disk can hurt packing quality. |
| P0 | Netsolver (Artificial Intelligence 2020). | CPU/RAM/storage plus bandwidth-flow feasibility for VDC allocation. | Confirm local server constraints, global bandwidth constraints, and whether it should be cited only for VDC/network-topology claims. |
| P1 | Wei Chen et al., "Exact algorithms for energy-efficient VMP in data centers" (FGCS 2020). | CPU+memory as standard minimum in energy-aware VMP. | Confirm dimensions are execution time, CPU, and memory; confirm objective includes working and idle energy. |
| P1 | Lopez-Pires et al., VMP under uncertainty in overbooked clouds (FGCS 2018). | Multi-resource uncertainty and overbooking extension path. | Confirm whether CPU/RAM/networking are uncertain or constrained and whether model is elastic IaaS rather than temporal TBPP. |
| P1 | Meroni and Guitart, "Scalable energy-aware VM allocation..." (FGCS 2026). | Latest energy-aware mathematical programming VMP with richer dimensions. | Confirm heterogeneous PM capacities, CPU/memory load measures, performance profiles, power profiles, and migration costs. |
| P1 | Jin et al., server power models review (Applied Energy 2020). | Idle+load server power proxy and heterogeneity caveat. | Confirm model taxonomy and whether linear utilization-based power is an accepted approximation or only one among many. |
| P1 | Dayarathna et al., data center energy modeling survey (IEEE COMST 2016). | Boundary between server-side IT energy and facility energy. | Confirm categories: IT, cooling, power delivery, PUE/facility terms. |
| P1 | VMPlanner (Computer Networks 2013). | Network-power counterexample. | Confirm whether joint VM placement and traffic routing can reduce network power and when server-only energy is insufficient. |
| P1 | Migration surveys and pMapper. | Migration cost should not be free. | Confirm migration metrics: downtime, total migration time, memory transfer, bandwidth, performance impact, and whether pMapper models migration cost in placement. |

## Cross-Cutting Questions for Full-Text Review

Use these questions when reading any supplied paper:

1. What is the scheduled object: VM, container, service, task, job, application, tenant graph, or capacity unit?
2. What is random: resource usage, lifetime, arrival rate, preemption time, price, availability, or algorithmic choice?
3. What is temporal: fixed activity window, dynamic online arrival, daily planning interval, rolling horizon, or multi-stage scenario tree?
4. What is the capacity unit: physical server, cluster, market, VM type, link, rack, data center, or generic bin?
5. What is the service-risk event: capacity overflow, hotspot, latency, availability, unserved VM-time, lost progress, missed deadline, or cost?
6. Does recourse actually change placement, buy capacity, migrate, suspend, preempt, checkpoint, restart, replicate, defer, or only charge a penalty?
7. Are CPU and memory both hard constraints? If not, does the paper justify a single-resource abstraction?
8. Is the paper direct cloud/VMP evidence, or methodological bridge evidence from bin packing, stochastic programming, or another domain?

## How to Update the Reference Files After Reading PDFs

When a paper is provided and reviewed:

1. Add a short note here under the relevant row if the claim is confirmed, narrowed, rejected, or superseded.
2. Update the evidence-table row in the corresponding topic file only if the full text changes the current claim.
3. If the full text supports a stronger claim, strengthen wording only for the exact component it supports.
4. If the full text supports a weaker claim, narrow the statement and move the paper to bridge or supplemental evidence.
5. Keep "no exact-match paper found" unless a paper truly includes temporal VMP, stochastic realized resource usage, two-stage recourse, workload classes, and energy-aware activation together.

## Current Action Items

- Collect full texts first for the P0 entries above, especially Aydin et al. 2020, Cohen et al. 2019, Yan et al. 2022, Hong et al. 2023/2024, Yu et al. 2020, Curino et al. 2014, SpotOn 2015, SpotCheck 2015, SciSpot 2022, Shen and Wang 2014, Crainic et al. 2016, Zhang et al. 2020, and Meroni and Guitart 2026.
- After each PDF is reviewed, update this file before changing the synthesis in `README.md`.
- Do not upgrade any claim from moderate to strong solely from a DOI, abstract, citation count, or venue quality. Upgrade only after the formulation or system design section is checked.
