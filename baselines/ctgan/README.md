# CTGAN Baseline

This directory adapts the standalone CTGAN implementation from `ref/CTGAN` for Medsyn baseline runs.

## Source

- Original project: CTGAN, DataCebo / SDV ecosystem
- Paper: Modeling Tabular data using Conditional GAN, NeurIPS 2019
- Local source snapshot: `baselines/ctgan/ctgan/`
- Original README: `baselines/ctgan/README_original.md`
- License: `baselines/ctgan/LICENSE`

The wrapper keeps the CTGAN algorithm code local to this baseline and exposes a project-compatible entrypoint.

## Adult Smoke Run

The default config uses the prepared CLLM adult split:

- train: `datasets/adult/splits/seed0_n20_train.csv`
- output: `results/adult/ctgan_seed0_n20.csv`
- log: `logs/adult/ctgan_seed0_n20.jsonl`

Run a quick smoke test:

```bash
python baselines/ctgan/main_baseline.py --epochs 1 --num-samples 20 --device cuda:7
```

For CPU only:

```bash
python baselines/ctgan/main_baseline.py --epochs 1 --num-samples 20 --device cpu
```

## Dependencies

This baseline uses the vendored CTGAN source but still needs CTGAN runtime dependencies in the active conda environment. If only `rdt` is missing in the current `cllm` environment, install just that package manually:

```bash
pip install "rdt>=1.14.0"
```

Use `baselines/ctgan/requirements_baseline.txt` as the full dependency reference for a clean baseline environment.

The current project policy is that the agent diagnoses missing packages and gives commands, while the user installs conda/pip packages manually.
