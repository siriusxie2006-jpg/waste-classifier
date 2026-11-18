# Confusion-Directed Squeeze (CDS)

This folder contains the CDS loss and the integration adapters used by the E2 four-arm implementation. The formal five-seed runs used the implementation whose `integration/cds_loss.py` SHA256 is `86490ccb8b5c86eb86ab894ca46db50e97e5210ce20100c1b485cae1695cac9a`.

## Mechanism

For each normalized 256-D embedding, CDS selects the closest normalized EMA prototype among classes other than the ground-truth class. The selected confuser confidence is taken from the classifier softmax and detached. The corrected CDS implementation detaches EMA centers/prototypes, confuser selection, pair direction, distance, confidence weights, and normalizer; gradients flow through the normalized embedding only. A prototype distance below 0.05 disables that sample's squeeze weight for numerical safety.

Centers store raw EMA means of normalized embeddings. A batch with `n_k` examples for class `k` uses count-adjusted update coefficient `rho_k = 1 - (1 - rho0)^n_k`; absent classes are skipped. Formal parameters: `lambda_cds=0.17614097189991385`, `theta=0.30`, `rho0=0.10`. The training objective is `L = L_CE + lambda_cds * L_CDS`.

## Integration

`integration/cds_loss.py` implements the loss and EMA centers. `integration/e0_branches.py` adapts the frozen E0 model, and `integration/training_branches.py` composes the CDS branch with CE. `integration/cds_config.json` records the frozen formal parameters and provenance. A1/A4/D1 historical source files are not replaced by these adapters.

The loss consumes Train embeddings for center initialization and training. Validation is used for metrics only. Test data is not required by this implementation and was not used in the formal runs.

## Validation

Run the isolated mathematical and engineering tests with the project Python environment:

```powershell
python -m pytest tests/test_cds_math.py tests/test_cds_engineering.py
```

The source copies are verified against the final engineering manifest; the CDS loss SHA above matches the formal-run source manifest.
