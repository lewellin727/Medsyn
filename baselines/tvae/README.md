# TVAE Baseline

This directory adapts TVAE as a standalone Medsyn baseline.

## Source

- Original project: CTGAN, DataCebo / SDV ecosystem
- Paper family: Modeling Tabular data using Conditional GAN, NeurIPS 2019
- Shared local source: `baselines/ctgan/ctgan/`
- TVAE implementation: `baselines/ctgan/ctgan/synthesizers/tvae.py`

TVAE is exposed as its own baseline because it is a separate synthesizer from CTGAN, even though both are implemented in the same upstream library.

## Adult Smoke Run

The default config uses the prepared CLLM Adult split:

- train: `datasets/adult/splits/seed0_n20_train.csv`
- output: `results/adult/tvae_seed0_n20.csv`
- log: `logs/adult/tvae_seed0_n20.jsonl`

Run a quick smoke test:

```bash
python baselines/tvae/main_baseline.py --epochs 1 --num-samples 20 --device cuda:7
```

For CPU only:

```bash
python baselines/tvae/main_baseline.py --epochs 1 --num-samples 20 --device cpu
```

## Dependencies

TVAE uses the same runtime dependency family as CTGAN. The current `cllm` environment has `rdt` and `torch`; for a clean environment, use:

```bash
pip install -r baselines/tvae/requirements_baseline.txt
```
