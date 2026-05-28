# Notation and Terms for the Stochastic Temporal VMP Model

## Purpose

This file defines the terminology and mathematical notation used across the reference notes. The goal is to make the modeling argument readable without requiring the reader to inspect implementation code.

The notation follows the current Notion model page, "VM type modeling (1)", with light cleanup for paper-style writing. Other topic files may still use monospace names when referring to implementation variables, but mathematical statements should use the notation here.

## Core Terms

| Term | Meaning in this model | Modeling role |
|---|---|---|
| Virtual machine placement (VMP) | Assigning VM workloads to physical servers subject to server capacity and service constraints. | The cloud operation being modeled. |
| Temporal bin packing problem (TBPP) | A bin-packing model where each item consumes bin capacity only during its active time window. | Physical servers are bins; VMs or reserved batch work are items or workload volumes; time periods define active windows. |
| Two-stage stochastic program (2SP) | A model with first-stage decisions made before the scenario is known and second-stage recourse decisions made after uncertainty is realized. | Baseline placement and reservation are first-stage; migration, suspension/curtailment, overload accounting, and batch processing are second-stage. |
| Scenario | One realization of uncertain VM resource usage or workload demand. | Indexed by $\xi \in \Xi$ with probability $p_\xi$. |
| Recourse | A corrective action after uncertainty is realized. | Includes on-demand migration, spot/preemptible curtailment, overload indicators, and realized batch processing. |
| Chance constraint | A constraint that limits the probability of a bad event. | Used for on-demand overload and spot/preemptible suspension probability. |
| CVaR | Conditional Value-at-Risk, the expected loss in the tail beyond a confidence level. | Alternative to chance constraints when the severity of bad scenarios matters. |
| Robust optimization | Enforcing feasibility for all scenarios or all realizations in an uncertainty set. | Alternative when scenario probabilities are unreliable. |
| On-demand VM | Protected, high-priority VM demand that should remain served with high probability. | Violation is modeled as expensive overload/SLA risk; migration is allowed as recourse. |
| Spot/preemptible VM | Revocable or lower-priority VM demand that may be interrupted, curtailed, or suspended under scarcity. | Suspension/curtailment is allowed, but true suspension requires state semantics such as checkpoint/restart or hibernation. |
| Managed Batch | A provider/platform-managed service that accepts delay-tolerant work, reserves or allocates capacity over time, and processes work before a completion target. | Modeled as reserved server-time and aggregate processed workload volume. |
| Resource-contention SLA proxy | Treating capacity overflow or overload as the bad service event. | Defensible for placement-level capacity risk, but not equivalent to latency, availability, deadline, or tenant-level SLA unless mapped explicitly. |
| Powered-server time | Charging every period in which a server is on. | Modeled by $\sum_{s,t} u_{st}$. |
| Fire-up/startup | Charging transitions from off to on. | Requires an explicit transition variable, not only $u_{st}$. |

## Sets and Indices

| Symbol | Definition |
|---|---|
| $I$ | Set of on-demand VMs, indexed by $i \in I$. |
| $J$ | Set of spot/preemptible VMs, indexed by $j \in J$. |
| $K$ | Set of managed batch jobs, indexed by $k \in K$. |
| $S$ | Set of homogeneous physical servers, indexed by $s \in S$. |
| $T$ | Set of time periods, indexed by $t \in T$. |
| $T_i \subseteq T$ | Active periods of on-demand VM $i$, e.g. $T_i=\{a_i,\ldots,d_i\}$. |
| $T_j \subseteq T$ | Active periods of spot/preemptible VM $j$, e.g. $T_j=\{a_j,\ldots,d_j\}$. |
| $\Xi$ | Set of scenarios, indexed by $\xi \in \Xi$. |

## Parameters

| Symbol | Definition |
|---|---|
| $C$ | CPU-normalized capacity of each homogeneous server. |
| $d^O_{it}(\xi)$ | Realized resource usage of on-demand VM $i$ at time $t$ in scenario $\xi$. |
| $d^S_{jt}(\xi)$ | Realized resource usage of spot/preemptible VM $j$ at time $t$ in scenario $\xi$. |
| $r^B_k$ | Reserved batch VM CPU capacity for managed batch job $k$. |
| $\bar d^B_k(\xi)$ | Scenario-dependent average CPU usage for batch job $k$. |
| $H^B_k$ | Total processing time requirement of batch job $k$. |
| $W_k$ | Total workload amount of batch job $k$, often $W_k = H^B_k \bar d^B_k$. |
| $\kappa$ | Batch reservation slack factor. |
| $N^B_k$ | Number of batch VM time slots reserved for job $k$, e.g. $N^B_k = \left\lceil \frac{(1+\kappa)W_k}{r^B_k} \right\rceil$. |
| $p_\xi$ | Probability of scenario $\xi$. |
| $\varepsilon^{OD}$ | Allowable probability of on-demand SLA/resource-contention violation. |
| $\varepsilon^{SP}$ | Allowable probability that a spot/preemptible VM is suspended or curtailed at least once. |
| $\rho$ | Minimum served-time ratio for each spot/preemptible VM. |
| $M$ | Sufficiently large constant for indicator/big-M constraints. |

## First-Stage Variables

These variables are selected before scenario $\xi$ is known.

| Symbol | Domain | Definition |
|---|---|---|
| $x_{is}$ | $\{0,1\}$ | 1 if on-demand VM $i$ is initially placed on server $s$. |
| $y_{js}$ | $\{0,1\}$ | 1 if spot/preemptible VM $j$ is initially placed on server $s$. |
| $b_{kst}$ | $\{0,1\}$ | 1 if managed batch capacity for job $k$ is reserved on server $s$ at time $t$. |
| $u_{st}$ | $\{0,1\}$ | 1 if server $s$ is powered on at time $t$. |
| $u_s$ | $\{0,1\}$ | 1 if server $s$ is used at least once in the horizon. |

## Second-Stage Variables

These variables are scenario-dependent recourse decisions.

| Symbol | Domain | Definition |
|---|---|---|
| $x^R_{ist}(\xi)$ | $\{0,1\}$ | 1 if on-demand VM $i$ is active on server $s$ at time $t$ in scenario $\xi$. |
| $m_{it}(\xi)$ | $\{0,1\}$ | 1 if on-demand VM $i$ migrates at time $t$ in scenario $\xi$. |
| $y^R_{jst}(\xi)$ | $\{0,1\}$ | 1 if spot/preemptible VM $j$ remains active on server $s$ at time $t$ in scenario $\xi$. |
| $z_{kst}(\xi)$ | $\mathbb{R}_+$ | Batch workload volume of job $k$ processed on server $s$ at time $t$ in scenario $\xi$. |
| $\gamma_{st}(\xi)$ | $\{0,1\}$ | 1 if lower-priority work on server $s$ at time $t$ is suspended or curtailed in scenario $\xi$. |
| $\phi_{st}(\xi)$ | $\{0,1\}$ | 1 if on-demand resource-contention/SLA violation occurs on server $s$ at time $t$ in scenario $\xi$. |
| $\eta_s(\xi)$ | $\{0,1\}$ | 1 if server $s$ has at least one on-demand violation over the horizon in scenario $\xi$. |
| $\delta_j(\xi)$ | $\{0,1\}$ | 1 if spot/preemptible VM $j$ is suspended or curtailed at least once in scenario $\xi$. |

## Representative Constraints

### Baseline Placement

Each on-demand VM and each spot/preemptible VM receives one initial server:

$$
\sum_{s \in S} x_{is} = 1 \qquad \forall i \in I,
$$

$$
\sum_{s \in S} y_{js} = 1 \qquad \forall j \in J.
$$

Spot/preemptible baseline placement requires the assigned server to be active during the VM's active window:

$$
y_{js} \le u_{st}
\qquad
\forall j \in J,\ s \in S,\ t \in T_j.
$$

### Managed Batch Reservation

Batch reservation consumes server-time capacity only when the server is on:

$$
b_{kst} \le u_{st}
\qquad
\forall k \in K,\ s \in S,\ t \in T.
$$

The total reserved batch slots are bounded by the reservation budget:

$$
\sum_{s \in S}\sum_{t \in T} b_{kst} \le N^B_k
\qquad
\forall k \in K.
$$

### On-Demand Recourse and Migration

Each active on-demand VM must be assigned to one active server in each scenario:

$$
\sum_{s \in S} x^R_{ist}(\xi) = 1
\qquad
\forall i \in I,\ t \in T_i,\ \xi \in \Xi.
$$

The realized placement can be linked to initial placement at the first active time:

$$
x^R_{is a_i}(\xi) = x_{is}
\qquad
\forall i \in I,\ s \in S,\ \xi \in \Xi.
$$

Migration can be detected by changes in realized placement:

$$
m_{it}(\xi) \ge x^R_{ist}(\xi)-x^R_{is,t-1}(\xi),
\qquad
\forall i,s,t \in T_i \setminus \{a_i\},\xi,
$$

$$
m_{it}(\xi) \ge x^R_{is,t-1}(\xi)-x^R_{ist}(\xi),
\qquad
\forall i,s,t \in T_i \setminus \{a_i\},\xi.
$$

If the model allows at most one migration per on-demand VM and scenario:

$$
\sum_{t \in T_i \setminus \{a_i\}} m_{it}(\xi) \le 1
\qquad
\forall i \in I,\ \xi \in \Xi.
$$

### Spot/Preemptible Curtailment

A spot/preemptible VM can only be served on its baseline server and is turned off when the corresponding server-time curtailment indicator is active:

$$
y^R_{jst}(\xi) \le y_{js}
\qquad
\forall j,s,t \in T_j,\xi,
$$

$$
y^R_{jst}(\xi) \le 1-\gamma_{st}(\xi)
\qquad
\forall j,s,t \in T_j,\xi,
$$

$$
y^R_{jst}(\xi) \ge y_{js}-\gamma_{st}(\xi)
\qquad
\forall j,s,t \in T_j,\xi.
$$

This is more accurately called curtailment or preemption unless the formulation also models state preservation, checkpoint/restart, hibernation, or migration for the spot VM.

### Managed Batch Processing

The processed volume should meet the accepted batch workload target:

$$
\sum_{s \in S}\sum_{t \in T} z_{kst}(\xi) \ge W_k
\qquad
\forall k \in K,\ \xi \in \Xi.
$$

Processing is limited by reserved server-time capacity:

$$
z_{kst}(\xi) \le r^B_k b_{kst}
\qquad
\forall k,s,t,\xi.
$$

If batch processing is also curtailed when low-priority work is suspended:

$$
z_{kst}(\xi) \le r^B_k \left(1-\gamma_{st}(\xi)\right)
\qquad
\forall k,s,t,\xi.
$$

### Resource Load

Scenario-dependent server load can be decomposed by workload class:

$$
\ell^O_{st}(\xi)=\sum_{i\in I} d^O_{it}(\xi)x^R_{ist}(\xi),
$$

$$
\ell^S_{st}(\xi)=\sum_{j\in J} d^S_{jt}(\xi)y^R_{jst}(\xi),
$$

$$
\ell^B_{st}(\xi)=\sum_{k\in K} z_{kst}(\xi),
$$

$$
L_{st}(\xi)=\ell^O_{st}(\xi)+\ell^S_{st}(\xi)+\ell^B_{st}(\xi).
$$

For energy accounting, a capped load variable may be used:

$$
\bar L_{st}(\xi) \le C,
\qquad
\bar L_{st}(\xi) + e_{st}(\xi) = L_{st}(\xi),
\qquad
e_{st}(\xi) \ge 0.
$$

Here, $e_{st}(\xi)$ is the excess demand. The overload indicator $\phi_{st}(\xi)$ should be linked to this excess through a clear big-M or indicator formulation.

### Chance Constraint for On-Demand Resource-Contention Risk

A common placement-level SLA proxy is to limit the probability that on-demand load exceeds active server capacity:

$$
\ell^O_{st}(\xi) \le C u_{st} + M\phi_{st}(\xi)
\qquad
\forall s,t,\xi.
$$

The horizon-level server violation indicator can be linked as:

$$
\eta_s(\xi) \ge \phi_{st}(\xi)
\qquad
\forall s,t,\xi.
$$

Then the per-server chance constraint is:

$$
\sum_{\xi \in \Xi} p_\xi \eta_s(\xi)
\le \varepsilon^{OD}
\qquad
\forall s \in S.
$$

This is a resource-contention SLA proxy. It is not automatically a tenant-level, fleet-level, latency, or availability SLA.

### Chance Constraint for Spot/Preemptible Curtailment

The event that spot/preemptible VM $j$ is unserved at time $t$ can be linked to $\delta_j(\xi)$:

$$
\delta_j(\xi) \ge 1-\sum_{s \in S} y^R_{jst}(\xi)
\qquad
\forall j \in J,\ t \in T_j,\ \xi \in \Xi.
$$

The chance constraint is:

$$
\sum_{\xi \in \Xi} p_\xi \delta_j(\xi)
\le \varepsilon^{SP}
\qquad
\forall j \in J.
$$

A minimum served-time ratio can also be enforced:

$$
\sum_{t \in T_j}\sum_{s \in S} y^R_{jst}(\xi)
\ge \rho |T_j|
\qquad
\forall j \in J,\ \xi \in \Xi.
$$

## Objective Functions

### Server Count and Migration Proxy

A server-count objective can use $u_s$:

$$
\min \sum_{s \in S} u_s
 + \lambda \sum_{\xi \in \Xi}p_\xi
 \frac{1}{|I|}
 \sum_{i \in I}\sum_{t \in T_i}m_{it}(\xi),
$$

with

$$
u_s \ge u_{st}
\qquad
\forall s \in S,\ t \in T.
$$

### Placement-Level Energy Proxy

A compact energy objective can be written as:

$$
\min
\sum_{s \in S}\sum_{t \in T} E^{idle} u_{st}
+
\sum_{\xi \in \Xi}p_\xi
\left(
\sum_{s \in S}\sum_{t \in T}
E^{cpu}\frac{\bar L_{st}(\xi)}{C}
+
E^{mig}\sum_{i \in I}\sum_{t \in T_i\setminus\{a_i\}}m_{it}(\xi)
\right).
$$

This is a server-side IT-energy proxy. It is not a full facility energy model.

### Fire-Up Extension

Powered-server time and fire-up events are different. If startup events matter, introduce:

$$
q_{st} \ge u_{st}-u_{s,t-1}
\qquad
\forall s,t,
$$

and add a term such as:

$$
E^{start}\sum_{s \in S}\sum_{t \in T}q_{st}.
$$

## Reading Guide for the Other Reference Files

When the other topic files use expressions such as `x[i,s]` or `u[s,t]`, read them as implementation-style names for the mathematical variables $x_{is}$ and $u_{st}$ defined here.

When the other topic files discuss "spot suspension," read it as a high-level low-priority curtailment event unless the paper explicitly adds state-preserving suspension, checkpoint/restart, or migration semantics.

When the other topic files discuss "SLA violation," read it as resource-contention or capacity-overflow risk unless the paper explicitly models latency, availability, deadline miss, or tenant-level service guarantees.

## Verdict

The reference notes are easier to read if this notation file is used as the entry point. It preserves the current temporal bin-packing and two-stage stochastic-programming framing while making clear which expressions are mathematical variables, which terms are operational interpretations, and which service-quality measures are only proxies.
