# Indexed HSG: 2 × 2 ablation

Four variants use the same policy, spatial hash and adjacency indices. HSG additionally caches regional totals and best candidates.
Synthetic graph streams measure computation only. They do not represent robot reachability or RGB-D accuracy.
All reported B times include full-snapshot diffing, index updates, summaries and querying. Initial setup is reported separately.

| Nodes | Update | Query | Flat full ms | Flat inc ms | HSG full ms | HSG inc ms | Flat-inc / HSG-inc |
|---:|---|---|---:|---:|---:|---:|---:|
| 27 | local | hidden | 0.344 | 0.175 | 0.344 | 0.172 | 1.04x |
| 27 | local | visible | 0.340 | 0.169 | 0.344 | 0.176 | 0.98x |
| 27 | global | hidden | 0.340 | 0.358 | 0.344 | 0.356 | 1.01x |
| 27 | global | visible | 0.335 | 0.350 | 0.346 | 0.356 | 0.98x |
| 216 | local | hidden | 2.299 | 0.381 | 2.303 | 0.324 | 1.20x |
| 216 | local | visible | 2.334 | 0.417 | 2.350 | 0.330 | 1.28x |
| 216 | global | hidden | 2.635 | 2.900 | 2.584 | 2.754 | 1.05x |
| 216 | global | visible | 2.620 | 2.977 | 2.643 | 2.825 | 1.05x |
| 1728 | local | hidden | 26.125 | 2.116 | 26.069 | 1.816 | 1.19x |
| 1728 | local | visible | 24.459 | 1.696 | 24.877 | 1.656 | 1.02x |
| 1728 | global | hidden | 27.173 | 29.206 | 27.934 | 29.120 | 1.01x |
| 1728 | global | visible | 26.776 | 28.819 | 27.622 | 28.679 | 1.01x |

Times are medians after frame 0. Ratios use per-seed medians (10 seeds), not independent frame-level significance tests.
Seed-level bootstrap 95% intervals and ranges are in summary.json. These are conditional timing results on this machine, not universal speed guarantees.
Global updates and visible-target queries are included as controls; HSG need not improve these workloads.

## Physical validation

- flat_full: 4/8 physical retrievals.
- flat_incremental: 4/8 physical retrievals.
- hsg_full: 4/8 physical retrievals.
- hsg_incremental: 4/8 physical retrievals.

All four variants produced bitwise identical command sequences per physical scene. Failed scenes remain in the denominator.
This validates behavioral equivalence, not a higher physical success rate.

## Allocation measurements

Separate tracemalloc runs count additional planner-owned Python allocations, excluding prebuilt input graphs and native/GPU memory.

| Nodes | Variant | Retained KiB | Peak KiB |
|---:|---|---:|---:|
| 27 | flat_full | 45.7 | 56.7 |
| 27 | flat_incremental | 48.8 | 54.3 |
| 27 | hsg_full | 40.4 | 51.6 |
| 27 | hsg_incremental | 48.7 | 54.3 |
| 216 | flat_full | 258.4 | 344.6 |
| 216 | flat_incremental | 294.5 | 333.8 |
| 216 | hsg_full | 242.3 | 328.7 |
| 216 | hsg_incremental | 294.2 | 333.7 |
| 1728 | flat_full | 2582.7 | 3285.6 |
| 1728 | flat_incremental | 2347.1 | 3116.6 |
| 1728 | hsg_full | 2477.6 | 3181.1 |
| 1728 | hsg_incremental | 2359.7 | 3129.6 |

Full source hashes, per-frame work counts, cold-start times, input-build-plus-B times and seed ranges are in compute.json.gz / summary.json.
Input construction in this synthetic experiment is not RGB-D inference time. No end-to-end robot speedup follows from B-only measurements.
