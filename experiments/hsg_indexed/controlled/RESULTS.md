# Indexed HSG: 2 × 2 ablation

Four variants use the same policy, spatial hash and adjacency indices. HSG additionally caches regional totals and best candidates.
Synthetic graph streams measure computation only. They do not represent robot reachability or RGB-D accuracy.
All reported B times include full-snapshot diffing, index updates, summaries and querying. Initial setup is reported separately.
Protocol: serial; order offset=(frame+seed)%4; cyclic GC disabled during timed B calls; snapshot diff and summary maintenance included. Reference counting remains enabled. GC outside the B call and RGB-D acquisition are not timed as B.

| Nodes | Update | Query | Flat full ms | Flat inc ms | HSG full ms | HSG inc ms | Flat-inc / HSG-inc |
|---:|---|---|---:|---:|---:|---:|---:|
| 27 | local | hidden | 0.333 | 0.169 | 0.333 | 0.166 | 1.01x |
| 27 | local | visible | 0.328 | 0.162 | 0.334 | 0.163 | 0.99x |
| 27 | global | hidden | 0.344 | 0.358 | 0.345 | 0.358 | 0.99x |
| 27 | global | visible | 0.333 | 0.347 | 0.338 | 0.353 | 0.99x |
| 216 | local | hidden | 2.471 | 0.371 | 2.444 | 0.342 | 1.09x |
| 216 | local | visible | 2.616 | 0.359 | 2.678 | 0.353 | 1.00x |
| 216 | global | hidden | 2.709 | 2.907 | 2.715 | 2.915 | 0.99x |
| 216 | global | visible | 2.744 | 2.959 | 2.772 | 2.938 | 1.01x |
| 1728 | local | hidden | 23.173 | 2.070 | 23.288 | 1.780 | 1.15x |
| 1728 | local | visible | 23.712 | 1.804 | 24.047 | 1.826 | 0.98x |
| 1728 | global | hidden | 25.823 | 27.923 | 25.660 | 27.806 | 1.01x |
| 1728 | global | visible | 25.380 | 27.889 | 25.976 | 28.086 | 0.98x |

Times are medians after frame 0. Ratios use per-seed medians (12 seeds), not independent frame-level significance tests.
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
| 27 | flat_full | 43.5 | 54.5 |
| 27 | flat_incremental | 47.4 | 52.8 |
| 27 | hsg_full | 39.7 | 50.9 |
| 27 | hsg_incremental | 47.8 | 53.6 |
| 216 | flat_full | 256.6 | 342.9 |
| 216 | flat_incremental | 293.1 | 333.8 |
| 216 | hsg_full | 240.1 | 326.6 |
| 216 | hsg_incremental | 292.6 | 333.6 |
| 1728 | flat_full | 2583.3 | 3286.3 |
| 1728 | flat_incremental | 2347.7 | 3116.6 |
| 1728 | hsg_full | 2490.7 | 3193.8 |
| 1728 | hsg_incremental | 2360.5 | 3129.6 |

Full source hashes, per-frame work counts, cold-start times, input-build-plus-B times and seed ranges are in compute.json.gz / summary.json.
Input construction in this synthetic experiment is not RGB-D inference time. No end-to-end robot speedup follows from B-only measurements.
