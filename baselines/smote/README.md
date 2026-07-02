# SMOTE Baseline

This directory adapts the SMOTE baseline used by the TabDDPM reference code for Medsyn.

## Source

- Original reference: `ref/tab-ddpm/smote/`
- Local reference copy: `baselines/smote/original_tabddpm_smote/`
- Project entrypoint: `baselines/smote/main_baseline.py`

The original TabDDPM SMOTE code targets `imbalanced-learn==0.7.0`. The project wrapper keeps the same main settings but uses the current `imbalanced-learn` API available in the `cllm` environment.

## Adult Smoke Run

```bash
python baselines/smote/main_baseline.py
```

Override sample count:

```bash
python baselines/smote/main_baseline.py --num-samples 20
```

Outputs:

- synthetic samples: `results/adult/smote_seed0_n20.csv`
- live log: `logs/adult/smote_seed0_n20.jsonl`

## Dependency

The current `cllm` environment already has `imbalanced-learn`. For a clean environment, install:

```bash
pip install "imbalanced-learn>=0.14"
```
