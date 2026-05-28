# Energy and Power Model

## Claim

The current objective, which combines powered-server time, load-dependent CPU energy, and migration cost, is a defensible research abstraction for energy-aware VM placement. It matches a common consolidation logic: keep fewer servers powered, account for utilization-dependent power while active, and penalize migration because it consumes resources and can affect service quality.

The strongest claim is conditional: this is a compact IT-energy model, not a full data-center energy model. It is appropriate for studying placement and recourse tradeoffs, but it does not model cooling, power delivery, network equipment, heterogeneous server power curves, memory/storage/network power, thermal effects, or detailed live-migration dynamics.

The model should also distinguish powered-server time from fire-up events. A term like $\sum_{s,t} u_{st}$ charges every active server-period. That is a powered-time or active-time cost. A fire-up cost charges transitions from off to on, for example $\max\{0,u_{st}-u_{s,t-1}\}$. These terms encourage different behavior: powered-time discourages keeping servers on, while fire-up cost discourages frequent on/off cycling. If the current objective only has active server-time, it should not be described as explicitly modeling startup wear, boot delay, or fire-up energy unless a transition variable is added.

## How the Current Model Matches

The current model uses three energy/cost terms:

- Idle/server activation energy: $E^{idle}\sum_{s,t}u_{st}$.
- CPU-dependent energy: $E^{cpu}\sum_\xi p_\xi\sum_{s,t}\bar L_{st}(\xi)/C$, where $\bar L_{st}(\xi)$ is capped by server capacity.
- Migration energy/cost: $E^{mig}\sum_\xi p_\xi\sum_{i,t}m_{it}(\xi)$.

This aligns with a standard VM consolidation view. Server activation captures the fixed cost of keeping a physical machine on, utilization-dependent load captures the dynamic part of server power, and migration cost discourages excessive reconfiguration. Capping `barL` at capacity is consistent with interpreting overload as unmet demand, throttling, or SLA loss rather than as physical CPU utilization above 100 percent.

The objective is closest to an affine server-power proxy:

$$
P_s(t,\xi)
= P^{idle}_s u_{st}
+ \alpha_s \bar L_{st}(\xi)
+ \text{migration costs}.
$$

This structure is common in placement-level optimization because it is simple, convex/linearizable, and compatible with bin-packing constraints. Its main limitation is that real servers can have heterogeneous, nonlinear, workload-dependent, and component-level power behavior.

## Evidence Table

| Paper | Venue | Verified link | What it supports | Limits for this model |
|---|---|---|---|---|
| Dayarathna, Wen, and Fan, "Data Center Energy Consumption Modeling: A Survey" | IEEE Communications Surveys & Tutorials 18(1), 2016 | [DOI: 10.1109/COMST.2015.2481183](https://doi.org/10.1109/COMST.2015.2481183); [DBLP record](https://dblp.org/rec/journals/comsur/DayarathnaWF16) | This survey separates data-center energy modeling into IT equipment, cooling, power delivery, and broader infrastructure. It supports a clear boundary statement: the current model is server-side IT energy, not facility energy. It also motivates why cooling and thermal coupling cannot be inferred from server activation alone. | Broader than the current model. It supports the limitation, not a requirement that every VMP prototype include PUE, cooling, or power-delivery constraints. |
| Jin, Bai, Yang, Mao, and Xu, "A review of power consumption models of servers in data centers" | Applied Energy 265, 2020 | [DOI: 10.1016/j.apenergy.2020.114806](https://doi.org/10.1016/j.apenergy.2020.114806); [CoLab metadata](https://colab.ws/articles/10.1016%2Fj.apenergy.2020.114806) | This review is the strongest support for using server power models as the placement-level energy core. It classifies baseline-plus-active models, including linear and nonlinear variants, and notes that the relation between idle power, peak power, and load differs across server generations. This supports the current idle+load proxy while requiring a caveat about heterogeneity and nonlinearity. | It is a server-power modeling review, not a VM placement formulation. It does not justify a single universal coefficient for all machines or workloads. |
| Zhang, Lu, Qin, and Zhao, "A high-level energy consumption model for heterogeneous data centers" | Simulation Modelling Practice and Theory 39, 2013 | [DOI: 10.1016/j.simpat.2013.05.006](https://doi.org/10.1016/j.simpat.2013.05.006); [ScienceDirect page](https://www.sciencedirect.com/science/article/pii/S1569190X13000853) | This paper supports high-level utilization-based energy modeling in heterogeneous data centers. Its value here is practical: optimization models often need tractable power proxies based on utilization and server profiles rather than detailed component simulation. | The paper also shows why heterogeneity matters. A single linear CPU coefficient is a simplification and may be inaccurate for some servers or workload mixes. |
| Beloglazov, Abawajy, and Buyya, "Energy-aware resource allocation heuristics for efficient management of data centers for Cloud computing" | Future Generation Computer Systems 28(5), 2012 | [DOI: 10.1016/j.future.2011.04.017](https://doi.org/10.1016/j.future.2011.04.017); [author PDF](https://beloglazov.info/papers/2012-fgcs-vm-consolidation-heuristics.pdf) | This is direct cloud-management evidence for the consolidation logic used by the current model: detect overloaded and underloaded hosts, consolidate VMs, migrate VMs, and switch idle nodes to low-power states while considering SLA violations. It supports active-server minimization plus migration penalties as a standard energy-aware VMP idea. | It uses heuristic simulation rather than a two-stage stochastic MILP. Its migration and SLA models are not the same as the current formulation. |
| Aydin, Muter, and Birbil, "Multi-objective temporal bin packing problem: An application in cloud computing" | Computers & Operations Research 121, 2020 | [DOI: 10.1016/j.cor.2020.104959](https://doi.org/10.1016/j.cor.2020.104959); [Warwick repository page](https://wrap.warwick.ac.uk/id/eprint/136154/) | This paper is important for the distinction between using fewer servers and reducing fire-ups in temporal placement. It models cloud VM placement as temporal bin packing and treats server activation/fire-up behavior as an operational objective. It supports adding a transition-based fire-up term if the current model wants to penalize on/off cycling. | It is deterministic and does not include stochastic recourse, chance constraints, detailed power curves, or live-migration dynamics. |
| Meroni and Guitart, "Scalable energy-aware VM allocation on cloud data centers through mathematical programming models" | Future Generation Computer Systems 174, 2026 | [DOI: 10.1016/j.future.2025.108011](https://doi.org/10.1016/j.future.2025.108011); [ScienceDirect page](https://www.sciencedirect.com/science/article/pii/S0167739X25003061); [DBLP record](https://dblp.org/rec/journals/fgcs/MeroniG26) | This recent paper supports mathematical-programming-based VM allocation with heterogeneous physical machines, distinct capacities, power-consumption profiles, performance profiles, and migration costs. It is strong evidence that energy-aware MP formulations can be credible while including richer power and performance details than the current prototype. | Richer than the current prototype. It does not imply that a single CPU-only, homogeneous, linear energy model is production-realistic. |
| Ahmad, Gani, Hamid, Shiraz, Yousafzai, and Xia, "A survey on virtual machine migration and server consolidation frameworks for cloud data centers" | Journal of Network and Computer Applications 52, 2015 | [DOI: 10.1016/j.jnca.2015.02.002](https://doi.org/10.1016/j.jnca.2015.02.002); [ScienceDirect page](https://www.sciencedirect.com/science/article/pii/S1084804515000561) | This survey supports migration and server consolidation as standard mechanisms for load balancing, power management, and resource management. It also emphasizes that migration has overheads and can affect performance, motivating a migration cost or penalty term. | Survey-level evidence. It supports penalizing migration, but not necessarily a single binary event cost independent of memory size, dirty-page rate, network bandwidth, or downtime. |
| Clark, Fraser, Hand, Hansen, Jul, Limpach, Pratt, and Warfield, "Live Migration of Virtual Machines" | USENIX NSDI 2005 | [USENIX official page](https://www.usenix.org/conference/nsdi-05/live-migration-virtual-machines); [DBLP record](https://dblp.org/rec/conf/nsdi/ClarkFHHJLPW05) | This systems paper establishes live migration as an operational tool for load balancing, maintenance, and data-center management, and shows that migration has measurable phases and downtime behavior. It supports the existence of migration as a recourse action and highlights why a detailed model would include duration, memory transfer, and service disruption. | It is not an energy-optimization paper and predates modern clouds. It supports migration mechanics and overhead, not the energy coefficient used in the current objective. |
| Verma, Ahuja, and Neogi, "pMapper: Power and Migration Cost Aware Application Placement in Virtualized Systems" | Middleware 2008 | [DOI: 10.1007/978-3-540-89856-6_13](https://doi.org/10.1007/978-3-540-89856-6_13); [DBLP record](https://dblp.org/rec/conf/middleware/VermaAN08) | pMapper is direct evidence that power-aware placement should account for migration cost in addition to power savings. It models placement in heterogeneous virtualized clusters and treats migration cost as part of the dynamic placement decision. | It is older and not a stochastic temporal model. It is useful for the migration-cost principle, not for the exact current cost structure. |
| Fang, Liang, Li, Chiaraviglio, and Xiong, "VMPlanner: Optimizing virtual machine placement and traffic flow routing to reduce network power costs in cloud data centers" | Computer Networks 57(1), 2013 | [DOI: 10.1016/j.comnet.2012.09.008](https://doi.org/10.1016/j.comnet.2012.09.008); [ScienceDirect page](https://www.sciencedirect.com/science/article/pii/S1389128612003301) | VMPlanner shows an alternative energy-aware VMP direction where network equipment power and traffic routing are modeled explicitly. It is useful as a counterexample to server-only energy: when traffic is important, optimizing only server activation and CPU load can miss network power savings and routing feasibility. | It targets network power and routing rather than stochastic server consolidation. It indicates a limitation of the current model, not a required extension for CPU-dominant workloads. |

## Different Modeling Choices in the Literature

Several papers model energy more richly than the current prototype.

Powered-time versus fire-up events. The current $\sum_{s,t}u_{st}$ term charges the number of active server-periods. Aydin et al. show that temporal bin-packing formulations can also care about fire-ups or server activation patterns. If the paper wants to claim fire-up minimization, add transition variables. If not, call the current term powered-server time or active-server energy.

Idle plus load-dependent server power. Jin et al. and Zhang et al. support tractable utilization-based server power models, but they also show that power curves vary by server generation and hardware profile. A homogeneous linear model is acceptable for a compact optimization prototype, while heterogeneous piecewise-linear or nonlinear profiles are stronger for production-oriented claims.

Server-only IT energy versus facility energy. Dayarathna et al. show that data-center energy includes IT equipment, cooling, and infrastructure. The current objective is server-side IT energy. It should not be interpreted as total facility energy, carbon emissions, or PUE-aware optimization unless the model adds those terms.

Server-only energy versus network-aware energy. VMPlanner models VM placement jointly with traffic-flow routing to reduce network power. This is a direct counterexample to the idea that server consolidation alone is always enough. For network-intensive services, the current model should be described as server-energy-aware, not full data-center-energy-aware.

Coarse migration penalty versus migration dynamics. Ahmad et al., Clark et al., and pMapper support migration as a standard recourse mechanism with real overhead. A binary event cost is a reasonable optimization penalty when detailed migration data are unavailable. A detailed model would include memory footprint, dirty-page rate, migration bandwidth, source-destination double resource reservation, migration duration, and downtime or performance degradation.

## Modeling Recommendations

Use the following language for the current model:

> The objective is a placement-level server-energy proxy. It charges powered-server time, utilization-dependent CPU energy, and a coarse migration penalty. It is intended to compare placement and recourse policies, not to predict full facility energy consumption.

If the formulation has only $\sum_{s,t}u_{st}$, use "powered-server time" or "active-server energy" instead of "fire-up cost." To model fire-up explicitly, add a binary transition variable:

$$
q_{st} \ge u_{st}-u_{s,t-1},
\qquad
q_{st} \le u_{st},
\qquad
q_{st} \le 1-u_{s,t-1}.
$$

Then add $E^{start}\sum_{s,t}q_{st}$ or a startup-delay constraint.

If the model keeps one migration coefficient, describe it as a reconfiguration penalty. For stronger realism, scale migration cost by VM memory size, expected dirty-page behavior, or bandwidth use, and limit the number of concurrent migrations per host or link.

If the model later adds memory/network/storage resource dimensions, the energy objective should be revisited. CPU-only dynamic energy is a reasonable first proxy, but memory pressure, storage I/O, and network traffic can affect both server and network energy.

## Caveats

The capped load term is a modeling choice, not a universal power law. It is reasonable when overload means unmet service, throttling, or SLA violation and physical CPU utilization cannot exceed capacity. If the research question is queueing delay, performance degradation, or thermal behavior under saturation, a more detailed performance-power model is needed.

The migration term is also coarse. It captures that migration is not free, but it does not model source/destination double reservation, migration duration, bandwidth contention, downtime, dirty-page rate, or application sensitivity. It is therefore appropriate for discouraging unnecessary recourse, but not for evaluating migration protocols.

The powered-server term does not distinguish an already-on server from a newly-started server. If startup delay, boot energy, hardware wear, or operator limits on power cycling matter, a fire-up transition term is needed.

The model is server-energy-aware, not facility-energy-aware. It excludes cooling, power distribution, UPS losses, network switches, storage systems, and thermal coupling. This is acceptable for placement experiments focused on server consolidation, but it should be explicit.

## Verdict

The current energy objective is well justified as a compact placement-level energy proxy: it captures powered-server time, utilization-dependent CPU work, and migration cost. It should be described as a server-side IT-energy abstraction for optimization experiments.

The wording should be careful. The model supports claims about energy-aware consolidation and recourse tradeoffs. It does not support claims about exact data-center energy, cooling, network power, startup wear, or migration protocol performance unless those mechanisms are explicitly added. Stronger production claims require heterogeneous power profiles, CPU+memory resources, network/storage resources where relevant, migration-duration constraints, and possibly cooling or facility-level energy terms.
