# THU-CHL historical GM result snapshot

This static snapshot includes all 50 epoch metrics from the completed GM run, the Top-1 selection record, and verification of the copied assets and protocol. It contains no weights, raw events, or unfinished-run metrics.

The selected epoch is 10: video Top-1 81.25% (442/544), Top-5 94.85294%, NLL 0.8097813. All 2,392 Test windows are included. Test was used for selection; this is not independent-Test performance.

The original trainer saved NLL-best weights. Subsequent selection over the complete Top-1 curve uniquely chose the same epoch; the preserved Top-1 and NLL model tensors were checked for exact equality. Metrics are archived observations, not a newly reproduced evaluation.

`manifest.json` hashes the exported files. `historical_gm_verified.json` also records hashes of the source assets, including weights that are deliberately not distributed here. Source-file hashes and saved identity support provenance but do not establish bitwise equivalence between three-rank DDP and single-GPU execution.
