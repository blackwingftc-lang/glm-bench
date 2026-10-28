# GLM-5.3-Flash CPU-only feasibility (estimates, not measurements)

Model (secondary sources, verify against zai-org/GLM-5.3-Flash): ~320B params, ~18B active/token, MIT, hybrid sparse+linear attention.

| Quantity | Estimate | Basis |
|---|---|---|
| FP8 weights | ~320 GB | 1 byte/param |
| Weight bytes read per decoded token | ~18 GB (batch 1) | active params, 1 byte each |
| Runners needed to hold weights | >=25 (16 GB RAM each, less OS/KV) | RAM only |
| Memory-bound decode, 32 nodes ideal | ~0.6 GB/node/token -> ~20-30 ms | assumes ~20-30 GB/s/node |
| Sync cost | layers x hops x RTT | >=1 ms/hop optimistic, 50+ layers -> >=50-100 ms |
| Realistic decode | <1-2 tok/s | WAN RTT, no inbound links, relay |

Blockers: runner-to-runner connectivity needs an overlay/relay; 6 h lifetime forces re-download of ~10+ GB shard per runner; runners are not permitted for hosting inference (GitHub terms); Oracle free tier (~24 GB RAM) cannot host shards.

Verdict: full-fidelity CPU-only serving on free CI is not feasible. Next measured step: run `cpu-bench` workflow.
