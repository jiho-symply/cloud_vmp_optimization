# Resource Dimensions for Credible VM Placement

## Claim

The current scalar CPU formulation is defensible only as a narrow prototype abstraction for stochastic temporal VM placement, or as a model of CPU-dominant workloads. It should not be described as a production-realistic VM placement model.

For credible general-purpose VM placement, the minimum resource vector is CPU plus memory. CPU determines compute feasibility and most consolidation and server-power objectives, but memory is a hard admission constraint for ordinary VMs and is not safely interchangeable with CPU. A server with spare CPU but insufficient memory cannot host an additional VM without swapping, ballooning, eviction, or SLA risk. Conversely, a server with spare memory but no CPU headroom can also be infeasible. This is why CPU-only should be framed as a single-bottleneck relaxation, not as a full feasibility model.

Network and storage dimensions become necessary when the model claims to represent communication-heavy services, virtual data centers, data locality, storage capacity, disk I/O, host-level bandwidth, or end-to-end bandwidth guarantees. Topology becomes necessary when the cost or feasibility of two VMs depends on whether they are placed on the same host, rack, aggregation block, cluster, or region. In those settings, CPU-only placement can accept allocations that are mathematically feasible in the prototype but infeasible, congested, or power-inefficient in a real system.

The strongest framing is therefore:

- CPU-only: acceptable for isolating stochastic temporal placement, priority-class recourse, server activation, and energy-aware consolidation mechanics.
- CPU + memory: minimum credible server-capacity model for ordinary VM-to-server placement.
- CPU + memory + host-level network/storage: required when workloads are communication-heavy, data-intensive, I/O-sensitive, or bandwidth-constrained.
- CPU + memory + topology-aware network/storage: required for virtual data centers, tenant graphs, end-to-end bandwidth guarantees, affinity/anti-affinity, and placement decisions whose quality depends on communication distance.

## How the Current Model Matches

The current model uses one scalar server-time capacity dimension. This maps naturally to CPU capacity, CPU-normalized demand, or a single dominant bottleneck abstraction. It lets the formulation focus on the stochastic and temporal decisions:

- first-stage server activation and baseline placement;
- scenario-specific realized placement;
- migration, suspension, overload, and batch-processing recourse;
- chance-style or CVaR-style service-risk constraints;
- energy-aware consolidation over time.

This is a legitimate prototype choice because the time and scenario structure is mostly orthogonal to the number of resource dimensions. A multi-resource extension would replace each scalar capacity constraint with a vector of capacity constraints, for example CPU and memory per server-time, and optionally add host bandwidth, storage, disk-I/O, or link-flow constraints. The current model already has the right placement indices to support such an extension, but it does not currently enforce those dimensions.

The model should therefore be presented as a stochastic temporal VMP prototype with a CPU-only capacity abstraction, not as a complete cloud scheduler. Its credibility depends on saying explicitly that it studies recourse and consolidation tradeoffs under a reduced resource model. If SLA is currently measured through resource contention, the claim should also be limited: resource contention is a reasonable proxy for capacity-driven SLA risk, but it does not cover latency, throughput, memory pressure, network congestion, disk I/O, availability, or tenant-level service objectives.

## Evidence Table

| Paper | Venue | Verified link | What it supports | Limits for this model |
|---|---|---|---|---|
| Verma, Pedrosa, Korupolu, Oppenheimer, Tune, and Wilkes, "Large-scale cluster management at Google with Borg" | EuroSys 2015 | [DOI: 10.1145/2741948.2741964](https://doi.org/10.1145/2741948.2741964); [DBLP record](https://dblp.org/rec/conf/eurosys/VermaPKOTW15) | Borg is useful operational evidence because it treats production cluster management as a multi-resource problem, with task requests and usage for resources such as CPU, memory, and disk. It also distinguishes requested resources from observed usage, which supports the broader modeling point that placement feasibility and realized contention are not captured by a single CPU scalar. | Borg schedules jobs/tasks in Google's cluster system rather than VMs in a stochastic temporal MILP. It supports multi-resource realism and request-vs-usage thinking, not the exact formulation. |
| Ghodsi, Zaharia, Hindman, Konwinski, Shenker, and Stoica, "Dominant Resource Fairness: Fair Allocation of Multiple Resource Types" | USENIX NSDI 2011 | [USENIX official page](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types); [DBLP record](https://dblp.org/rec/conf/nsdi/GhodsiZHKSS11) | DRF formalizes allocation when users consume different dominant resources. It is strong conceptual evidence that CPU-only normalization loses important information when users have heterogeneous CPU-memory-disk-network mixes. It supports the statement that CPU+memory is a minimum credible capacity vector and that fairness/contention analysis can change under multi-resource demands. | DRF is a fairness mechanism, not a VM-to-server placement or migration model. It does not decide topology, stochastic recourse, or temporal bin-packing feasibility. |
| Grandl, Ananthanarayanan, Kandula, Rao, and Akella, "Multi-resource packing for cluster schedulers" | ACM SIGCOMM 2014 | [DOI: 10.1145/2619239.2626334](https://doi.org/10.1145/2619239.2626334); [ACM page](https://dl.acm.org/doi/10.1145/2619239.2626334) | Tetris argues that modern cluster tasks have diverse CPU, memory, disk, and network requirements, and that ignoring disk/network can cause fragmentation, over-allocation, and interference. It supports treating CPU-only as a narrow abstraction rather than a general scheduling model. | Tetris is a cluster task scheduler, not a VM placement MILP. Its objective and online heuristic setting differ from this stochastic temporal VMP model. |
| Meng, Pappas, and Zhang, "Improving the Scalability of Data Center Networks with Traffic-aware Virtual Machine Placement" | IEEE INFOCOM 2010 | [DOI: 10.1109/INFCOM.2010.5461930](https://doi.org/10.1109/INFCOM.2010.5461930); [IBM Research page](https://research.ibm.com/publications/improving-the-scalability-of-data-center-networks-with-traffic-aware-virtual-machine-placement); [DBLP record](https://dblp.org/rec/conf/infocom/MengPZ10) | This paper directly models VM placement as a way to align inter-VM traffic with network distance, placing heavily communicating VMs close together. It supports adding topology-aware network terms when the workload is a multi-VM service with nontrivial traffic matrices. | It focuses on network scalability and traffic-aware placement, not CPU/memory bin packing or stochastic temporal recourse. It is an argument for an extension, not a requirement for every CPU-dominant prototype. |
| Bayless, Kodirov, Iqbal, Beschastnikh, Hoos, and Hu, "Scalable constraint-based virtual data center allocation" | Artificial Intelligence 278, 2020 | [DOI: 10.1016/j.artint.2019.103196](https://doi.org/10.1016/j.artint.2019.103196); [ScienceDirect page](https://www.sciencedirect.com/science/article/pii/S000437021930195X) | Netsolver models VDC allocation with local server constraints for CPU, RAM, and storage, plus global bandwidth constraints as network flows. It is direct evidence that credible virtual data center placement may need both server resource dimensions and network/link feasibility. | It targets virtual data center allocation with bandwidth guarantees, not stochastic temporal consolidation. Its stronger network model would substantially increase this model's data requirements and solve complexity. |
| Masdari, Nabavi, and Ahmadi, "An overview of virtual machine placement schemes in cloud computing" | Journal of Network and Computer Applications 66, 2016 | [DOI: 10.1016/j.jnca.2016.01.011](https://doi.org/10.1016/j.jnca.2016.01.011); [ScienceDirect page](https://www.sciencedirect.com/science/article/pii/S1084804516000291) | This survey frames VM placement as selecting a physical machine that can provide required VM resources, including CPU, memory, storage, and network bandwidth. It also reviews objectives such as utilization, power, performance, migration, and load balancing. It supports the claim that resource dimensionality is a standard VMP design choice rather than an optional detail. | As a survey, it does not validate one exact formulation. Some surveyed schemes use different assumptions and do not all require the same resource vector. |
| Silva Filho, Monteiro, Inacio, and Freire, "Approaches for optimizing virtual machine placement and migration in cloud environments: A survey" | Journal of Parallel and Distributed Computing 111, 2018 | [DOI: 10.1016/j.jpdc.2017.08.010](https://doi.org/10.1016/j.jpdc.2017.08.010); [DBLP record](https://dblp.org/rec/journals/jpdc/FilhoMIF18) | This survey identifies VM placement and migration as multi-objective and multi-constraint problems. It is useful here because it connects placement quality to CPU, RAM, storage, load balancing, network links, migration, and energy. It supports using more than CPU for broad cloud-placement claims. | It is a literature map, not a single operational model. It supports the caveat direction, not a unique minimum dimension set for every experiment. |
| Wei Chen, Zhi-Hua Hu, and You-Gan Wang, "Exact algorithms for energy-efficient virtual machine placement in data centers" | Future Generation Computer Systems 106, 2020 | [DOI: 10.1016/j.future.2019.12.043](https://doi.org/10.1016/j.future.2019.12.043); [ScienceDirect page](https://www.sciencedirect.com/science/article/pii/S0167739X19319594) | This FGCS VMP paper formulates energy-efficient placement as a three-dimensional bin-packing problem over execution time, CPU, and memory. The server capacity constraints include CPU and memory requirements, and the objective includes working and idle machine energy. It is direct support for CPU+memory as a standard minimum in energy-aware VMP. | It does not include network or storage feasibility, so it supports the minimum CPU+memory tier rather than full production realism. |
| Lopez-Pires, Baran, Benitez, Zalimben, and Amarilla, "Virtual machine placement for elastic infrastructures in overbooked cloud computing datacenters under uncertainty" | Future Generation Computer Systems 79, 2018 | [DOI: 10.1016/j.future.2017.09.021](https://doi.org/10.1016/j.future.2017.09.021); [ScienceDirect page](https://www.sciencedirect.com/science/article/pii/S0167739X17303126) | This paper models VMP under uncertainty with elastic services, overbooking, CPU/RAM server resources, and networking resources. It supports the point that uncertainty-aware VMP can be multi-resource, not only scalar CPU. | It uses a richer elastic/overbooked IaaS setting and multiple objectives. It is evidence for an extension path, not evidence that the current scalar model is complete. |
| Meroni and Guitart, "Scalable energy-aware VM allocation on cloud data centers through mathematical programming models" | Future Generation Computer Systems 174, 2026 | [DOI: 10.1016/j.future.2025.108011](https://doi.org/10.1016/j.future.2025.108011); [ScienceDirect page](https://www.sciencedirect.com/science/article/pii/S0167739X25003061); [DBLP record](https://dblp.org/rec/journals/fgcs/MeroniG26) | This recent FGCS paper is useful because its mathematical programming VM allocation model explicitly allows heterogeneous PM capacities and CPU/memory load measures, and connects those loads to performance and power profiles. It supports the idea that a realistic MP formulation can retain optimization structure while adding CPU and memory dimensions. | Its value here is methodological and dimensional. It does not make CPU-only production-realistic, and its heterogeneity/scalability focus is broader than this document's resource-dimension question. |

## Different Modeling Choices in the Literature

A CPU-only scalar model is the smallest abstraction. It is useful when the research question is about stochastic temporal recourse, consolidation, and priority handling, and when workload demand can reasonably be treated as CPU-dominant or normalized to one bottleneck dimension. This is the current model's position. The model should say that it studies a one-dimensional capacity abstraction, not that real VM placement is one-dimensional.

A CPU+memory vector model is the minimum credible general-purpose VM placement model. A VM with enough CPU but insufficient memory is not placeable in ordinary infrastructure. Chen et al. model energy-aware VMP as bin packing over time, CPU, and memory, while Meroni and Guitart use CPU and memory loads when connecting placement to performance and power profiles. Surveys in JNCA and JPDC also treat CPU and memory as recurring placement dimensions.

A multi-dimensional server model adds storage capacity, disk I/O, and host-level network bandwidth. This is closer to general production placement because VM shapes and workloads can be compute-intensive, memory-intensive, storage-intensive, or communication-intensive. Tetris provides strong systems evidence that ignoring disk and network can create resource fragmentation and interference even when CPU and memory fit. Borg provides production-system evidence that cluster managers track multiple resource dimensions and separate requests from observed usage.

A topology-aware network model goes beyond host-level bandwidth. Meng et al. show that VM placement can be optimized around inter-VM traffic and communication distance. Netsolver goes further by modeling local VM resources separately from global network-flow feasibility. This is the right modeling choice for virtual data centers, bandwidth reservations, multi-path routing, tenant topologies, or placement decisions where two VMs' communication cost depends on whether they share a server, rack, cluster, or availability zone.

An uncertainty-aware multi-resource model treats resource usage, elastic scaling, and overbooking as scenario-dependent. Lopez-Pires et al. show one route: uncertain CPU/RAM and networking utilization, service elasticity, and reconfiguration decisions. This is closest in spirit to the current stochastic model, but with a richer resource vector and a different online/offline reconfiguration scheme.

An energy-aware mathematical-programming model may keep the resource vector modest while adding machine heterogeneity, performance profiles, power curves, and scalable decomposition/scaffolding. Meroni and Guitart show that modern mathematical-programming approaches can still be credible for large VM allocation studies, but they do not remove the need to caveat omitted resources.

## Modeling Recommendations

For the current paper, the cleanest statement is:

> We model server capacity as a single CPU-normalized stochastic resource in order to isolate the temporal placement, uncertainty, and recourse structure. This is a prototype abstraction. A production VM placement model would normally include at least CPU and memory, and would add network/storage constraints when workloads require bandwidth, data locality, or topology-aware placement.

If the model wants to keep CPU-only, it should avoid claims such as "feasible VM placement in production clouds" or "complete SLA-aware placement." Better wording is "CPU-contention-aware placement," "capacity-risk proxy," or "server-consolidation prototype."

If one extension is added, add memory first. This is the smallest change that materially improves placement feasibility. The extension is straightforward:

- Replace scalar VM demand $d_{it}(\xi)$ with $d^{cpu}_{it}(\xi)$ and $d^{mem}_{it}(\xi)$.
- Replace scalar capacity $C_s$ with $C^{cpu}_s$ and $C^{mem}_s$.
- Enforce both capacity constraints for every server, time, and scenario.
- Keep the same temporal, stochastic, migration, suspension, and batch-processing structure.

If the paper discusses multi-VM applications, add network next. A host-level bandwidth constraint is a cheap approximation. A topology-aware model with link variables or flow constraints is more faithful but changes the problem class and data requirements.

## Caveats

CPU-only placement can be empirically useful, but it is a reduced model. It can overstate feasibility whenever memory, storage, or network is the true bottleneck. It can also understate fragmentation because a server with spare CPU may be unusable for a memory-heavy or I/O-heavy VM.

CPU+memory is a minimum, not a complete production model. It still misses network interference, storage capacity and I/O, topology, fault domains, affinity/anti-affinity, accelerator constraints, thermal constraints, and migration bandwidth.

Network and storage can be modeled at different levels. A simple host-level bandwidth or storage-capacity constraint is cheaper but weaker. A full topology-aware model with link capacities and flow variables is more faithful for VDCs, tenant graphs, and bandwidth guarantees, but it greatly increases data needs and solve complexity.

Adding dimensions without data can create false precision. If the available traces or experiments only support CPU demand, it is better to state a CPU-only prototype assumption than to invent unverified memory, network, or storage distributions.

Resource-contention SLA is a partial SLA proxy. It is aligned with capacity-risk and overcommitment literature, but it is not equivalent to application-level latency, throughput, availability, or tenant-level SLA penalties unless those metrics are explicitly modeled or empirically calibrated.

## Verdict

The current CPU-only capacity model is defensible as a narrow research prototype for studying stochastic temporal VM placement, recourse, priority classes, and energy-aware consolidation. It is not defensible as a full production-realistic VMP model.

For a credible general-purpose VM placement claim, CPU and memory should be treated as the minimum resource dimensions. Network and storage should be added whenever the modeled services include communication, data locality, bandwidth guarantees, storage capacity, disk I/O, or topology-sensitive behavior. The current model is therefore acceptable if the paper explicitly labels CPU-only as a prototype abstraction and avoids claims that require CPU+memory+network/storage realism.
