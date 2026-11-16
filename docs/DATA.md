# Data provenance

**Primary source:** Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., and Iosifidis, A. *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods*. [Paper](https://arxiv.org/abs/1705.03233) and [Fairdata record](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649). The Fairdata record lists a CC BY 4.0 license.

**Machine-readable derivative used here:** [`seanahmad/fi2010`](https://github.com/seanahmad/fi2010), commit `360fef5a74332c2e2a5bf4ee92ad1a4b2b4fb311`, `data/data.csv`. Its notebook extracts no-auction, decimal-precision FI-2010 order-book features, reconstructs five stock/day segments, and converts official labels from {1, 2, 3} to {+1, 0, -1}. The project pins both the commit and SHA-256 checksum of the 59,130,113-byte CSV.

This repository does not redistribute the full source CSV. It includes `examples/fi2010_sample.csv`, the first 100 ordered observations of each of 50 stock/day groups (5,000 rows) extracted from the cited CC BY 4.0 data, with attribution here. `python -m lobml download` fetches the pinned derivative and verifies its SHA-256 before use. The model and derived evaluation reports are included for inspection. If the mirror becomes unavailable, obtain the official `BenchmarkDatasets.zip` from Fairdata and reproduce the derivative following the cited notebook.

The CSV contains 272,941 snapshots, 40 price/volume features, `STOCK`, `DAY`, five direction labels, and a saved pandas index column that the pipeline ignores. The chosen target, `LABEL_5TICK`, is the five-step horizon in the benchmark's sampled representation. The source paper describes sampling one representation every ten raw events. The code uses the current snapshot and past snapshots only.
