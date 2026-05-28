# Reference Writing Instructions

This directory justifies the current stochastic temporal VM placement model from a modeling and operations perspective. Do not frame the model as an Azure CPU trace-only model. Traces such as Azure, Google/Borg, Alibaba, Bitbrains, and Google cluster traces may be used as empirical evidence, but the argument must be about the modeling abstraction.

## Allowed Sources

Primary evidence should come from peer-reviewed venues of comparable quality to:

- IEEE Transactions, IEEE/ACM Transactions, IEEE Access, IEEE CLOUD, IEEE INFOCOM, CCGrid, HPDC, SC.
- Computers & Operations Research, European Journal of Operational Research, Operations Research, Management Science, INFORMS Journal on Computing, TOP, 4OR, Annals of Operations Research.
- Future Generation Computer Systems, Journal of Network and Computer Applications, Journal of Parallel and Distributed Computing, Artificial Intelligence, Computer Science Review, Applied Energy.
- Top systems/data venues such as SOSP, OSDI, NSDI, EuroSys, SoCC, SIGCOMM, SIGMETRICS/POMACS, KDD, MLSys, and Performance Evaluation Review companion publications for SIGMETRICS papers.

Do not use arXiv-only manuscripts, blogs, Wikipedia, or vendor marketing pages as primary evidence. Vendor documentation may be cited only for operational semantics such as on-demand, spot, or preemptible instance behavior.

When a source is outside this quality list but still useful, label it as supplemental and state why it is not primary evidence. Do not let a supplemental source carry a central modeling claim.

## Required Verification

For every cited paper, verify at least one stable source before citing it:

- Publisher page, DOI landing page, ACM/IEEE/Elsevier/Springer/INFORMS page, DBLP entry, official author PDF, or institutional repository.
- Record the DOI or stable link in the evidence table whenever available.
- Do not cite a paper if the title, authors, venue, year, or claim cannot be verified.
- Avoid broad claims such as "the literature proves" unless the cited paper directly supports the statement.
- Separate exact support from bridge support. Exact support means the paper models the same object or decision. Bridge support means it supports one component, such as temporal packing, stochastic item sizes, chance constraints, prediction-based VM allocation, or migration cost.
- For every bridge citation, state the missing dimensions explicitly: for example, "not VM-specific," "not temporal," "not stochastic," "not two-stage," or "not production VMP."

## Required Markdown Structure

Each topic file must use the following sections:

1. `# <topic title>`
2. `## Claim`
3. `## How the Current Model Matches`
4. `## Evidence Table`
5. `## Different Modeling Choices in the Literature`
6. `## Caveats`
7. `## Verdict`

The evidence table must include these columns:

`Paper | Venue | Verified link | What it supports | Limits for this model`

Each evidence-table row should be concrete enough to be auditable. Prefer statements such as "models VM CPU/memory requirements as random variables with probabilistic overflow constraints" over vague statements such as "supports stochastic modeling."

Use `09_notation_and_terms.md` as the canonical notation and terminology entry point. Topic files may mention implementation-style variable names in monospace when needed, but equations and formal model statements should use LaTeX-style notation such as $x_{is}$, $u_{st}$, and $x^R_{ist}(\xi)$.

## Review Process

Each worker owns exactly one topic file. Workers may read other files in this directory but must not edit files they do not own. The main agent reviews each topic file for:

- Citation existence and venue quality.
- Whether the cited evidence actually supports the modeling claim.
- Missing counterexamples or alternative modeling choices.
- Overstated claims, especially around CPU-only modeling, 24-hour horizons, and fluid batch processing.
- Alignment with `00_current_model_summary.md`.

Weak or conditional claims must be labeled as such. A useful limitation is better than a forced justification.

## Current Modeling Questions to Verify

Each topic file should help answer at least one of these questions:

- Is temporal bin packing a defensible abstraction for VM placement, or is the cited paper only a general bin-packing method?
- Is stochastic VM resource usage modeled directly, or is the paper only about stochastic arrivals, durations, prices, or algorithmic randomization?
- Does the paper support provider-side placement, customer-side provisioning, container scheduling, service placement, or only a methodological analog?
- Does the model's SLA interpretation as resource-contention or overload risk match the paper, or does the paper use latency, availability, job completion, or cost instead?
- Does spot/preemptible suspension correspond to realistic interruption/preemption semantics, or should checkpointing, restart, migration, replication, warning time, or remaining lifetime be modeled?
- Does batch processing correspond to a Managed Batch style service that reserves/allocates capacity for delay-tolerant work, or is it only generic low-priority cluster work?
- Is CPU-only capacity acceptable for the stated experiment, and what minimum dimensions would be required for a stronger production VMP claim?

## Claim-Strength Labels

Use these labels consistently:

- Strong: direct peer-reviewed support for the same modeling component.
- Moderate: direct support for a related cloud/cluster setting, or strong methodological support with one missing operational dimension.
- Weak/conditional: useful only under explicit assumptions such as divisible batch work, CPU-dominant workloads, or daily planning cycles.
- Supplemental: informative but outside the preferred venue set, outside the cloud/VMP setting, or useful only for operational semantics.
