# TabDDPM Baseline

This directory adapts the TabDDPM reference implementation for Medsyn
experiments while keeping the original algorithm code under
`tab-ddpm/`.

## Scope

Current supported dataset:

- `adult`

The adapter converts Medsyn CSV splits from `datasets/adult/splits/` into the
NumPy layout expected by TabDDPM, writes a TabDDPM TOML config, and can call the
original `scripts/pipeline.py` for training and sampling.

## Environment

TabDDPM has older and heavier dependencies than the main `cllm` environment.
Use a dedicated environment before running training:

```bash
conda create -n tabddpm python=3.9.7 -y
conda activate tabddpm
pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r baselines/tabddpm/requirements_baseline.txt
```

The current `cllm` environment can run the preparation step, but it does not
include `torch`.

## Commands

Prepare Adult data and config:

```bash
python baselines/tabddpm/main_baseline.py --prepare-only
```

Train and sample after installing the TabDDPM environment:

```bash
conda activate tabddpm
python baselines/tabddpm/main_baseline.py --train --sample
```

By default, outputs are written under:

```text
baselines/tabddpm/tab-ddpm/data/medsyn_adult_seed0_n20/
baselines/tabddpm/tab-ddpm/exp/adult/medsyn_seed0_n20/
results/adult/tabddpm_seed0_n20.csv
```

## Notes

The vendored `tab-ddpm/` code should remain close to the upstream reference.
Project-specific changes should live in `main_baseline.py`,
`config_baseline.yaml`, or `scripts/`.
