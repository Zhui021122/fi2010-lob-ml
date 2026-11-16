# Limitations and interpretation

- This project predicts the FI-2010 benchmark's future mid-price direction label. It is a classification study, not a trading strategy or executable backtest. No return, Sharpe ratio, or profit claim is made.
- FI-2010 covers five stocks over ten trading days in 2010. Its age and short period limit generalization to current markets.
- The public benchmark files do not expose original instrument names and executable event timestamps. The CSV derivative reconstructs stock and day segments; order within each segment is treated as event order.
- The derived CSV is based on the no-auction, decimal-precision variant of the official benchmark. Its `LABEL_5TICK` maps official class 1 (up), 2 (stationary), 3 (down) to +1, 0, -1. We do not claim equivalence to evaluations using other normalization variants, horizons, or split conventions.
- Features from the current snapshot are assumed available before the prediction. No exchange latency, queue position, partial fills, fees, slippage, market impact, or adverse selection is modeled.
- There may be dependence among adjacent labels because their future windows overlap. We remove the last ten observations of every stock/day segment before evaluation and keep entire days separate across train, validation, and test. This is a conservative boundary guard, but does not make observations independent within a day.
- The model is selected with day 7. Days 8–10 are used only for a final held-out evaluation in the training command. A different dataset or an additional later period is needed to assess robustness before deployment.
- A model artifact is serialized with joblib. Only load the artifact from this trusted repository; untrusted pickle-based artifacts can execute code.
