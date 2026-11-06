# Supplied minimal Hubble distance-network analysis

This release reproduces a conditional GLS calculation from the supplied minimal
input. It does **not** establish a robust 1% consensus or a new physical measurement
of the Hubble constant. Several requested pathways and total systematic terms are
absent. The original fit is rejected; inferred SN scatter lowers precision further.

From this directory, with Python 3.11+:

```sh
python -m pip install -r requirements.txt
python code/analyze_network.py
python code/repair_consensus.py
```

See `report/report.md` and its six PNG figures for methods, results and limitations.
The unchanged supplied input is in `data/`; derived products, matrices and
profile-likelihood curves are in `outputs/`. The `h` parameter is 5 log10(H0).
The default estimate is 112.41 ± 2.95 km/s/Mpc (formal propagated uncertainty),
with chi2/dof=171.53/17. The estimated-scatter likelihood interval is
106.16–125.34 km/s/Mpc; its asymptotic coverage is not validated for this sample.

`SOURCE_MANIFEST.json` lists SHA-256 of all released files. The release ZIP
packages this same source tree. No third-party literature PDFs are redistributed.
