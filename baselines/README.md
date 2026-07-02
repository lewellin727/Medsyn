# Baselines

This directory contains the Medsyn baseline adapters. Each baseline has its own
subdirectory with a project-compatible entrypoint:

```text
baselines/<baseline_name>/main_baseline.py
baselines/<baseline_name>/config_baseline.yaml
```

All current baseline configs target the prepared Adult split
`datasets/adult/splits/seed0_n20_train.csv` by default. Synthetic samples are
written to `results/adult/`, and live run logs are written to `logs/adult/` when
the wrapper supports logging.

## Common Pattern

Run a baseline from the project root:

```bash
python baselines/<baseline_name>/main_baseline.py
```

Most training baselines support runtime overrides such as:

```bash
python baselines/<baseline_name>/main_baseline.py --epochs 1 --num-samples 20 --device cuda:7
```

Config files are the source of record for default hyperparameters. Runtime
arguments are intended for dataset/split selection, smoke tests, and controlled
overrides.

## Using Other Dataset Splits

All baseline wrappers support a shared dataset/split CLI:

```text
--dataset adult|drug|compas
--data-seed <0..9>
--size 20|40|100|200
```

Example:

```bash
python baselines/ctgan/main_baseline.py \
  --dataset compas \
  --data-seed 3 \
  --size 100 \
  --epochs 1 \
  --num-samples 20 \
  --device cuda:7
```

This automatically resolves:

```text
datasets/compas/splits/seed3_n100_train.csv
results/compas/ctgan_seed3_n100.csv
logs/compas/ctgan_seed3_n100.jsonl
baselines/ctgan/artifacts/compas_seed3_n100/
```

The dataset schema is resolved from `baselines/common.py`, including label,
numeric columns, categorical columns, and CTGAN/TVAE discrete columns.

Prepared split files follow this pattern:

```text
datasets/<dataset>/splits/seed<seed>_n<size>_train.csv
datasets/<dataset>/splits/seed<seed>_n<size>_oracle.csv
datasets/<dataset>/splits/seed<seed>_n<size>_test.csv
```

Current prepared datasets are `adult`, `drug`, and `compas`, with seeds `0..9`
and total sample sizes `20`, `40`, `100`, and `200`.

Use a separate config file only when you need dataset-specific hyperparameters
or custom schemas beyond the built-in registry:

```bash
cp baselines/ctgan/config_baseline.yaml baselines/ctgan/config_compas_seed3_n100.yaml
```

Then edit the copied config and run:

```bash
python baselines/ctgan/main_baseline.py \
  --config baselines/ctgan/config_compas_seed3_n100.yaml \
  --epochs 1 \
  --num-samples 20 \
  --device cuda:7
```

### Baseline-Specific Path Requirements

- TabDDPM needs `train_csv`, `val_csv`, and `test_csv` because it prepares the
  upstream data directory with train, validation, and test arrays.
- CTGAN, TVAE, SMOTE, and GReaT fit from `train_csv` only and write a synthetic
  CSV to `result_csv`.
- TabDDPM also needs dataset-specific `tabddpm_data_dir` and
  `tabddpm_exp_dir`; make these unique per dataset/split.
- Checkpoint-producing baselines should use dataset/split-specific artifact
  paths to avoid overwriting previous runs.

### Examples

TabDDPM prepare-only on Drug:

```bash
python baselines/tabddpm/main_baseline.py \
  --dataset drug \
  --data-seed 2 \
  --size 40 \
  --prepare-only
```

SMOTE on COMPAS:

```bash
python baselines/smote/main_baseline.py \
  --dataset compas \
  --data-seed 3 \
  --size 100 \
  --num-samples 20
```

GReaT on Adult with a local HuggingFace model:

```bash
python baselines/great/main_baseline.py \
  --dataset adult \
  --data-seed 0 \
  --size 20 \
  --llm /data1/liangzhilin/model/Llama-3.2-1B-Instruct \
  --epochs 1 \
  --num-samples 20 \
  --device cuda:7
```

## Baseline Summary

| Baseline | Idea | Entrypoint | Default Output |
| --- | --- | --- | --- |
| TabDDPM | Diffusion model for mixed tabular data. | `baselines/tabddpm/main_baseline.py` | `results/adult/tabddpm_seed0_n20.csv` |
| CTGAN | Conditional GAN for single-table synthesis. | `baselines/ctgan/main_baseline.py` | `results/adult/ctgan_seed0_n20.csv` |
| TVAE | Variational autoencoder from the CTGAN library. | `baselines/tvae/main_baseline.py` | `results/adult/tvae_seed0_n20.csv` |
| SMOTE | Interpolation-based oversampling baseline. | `baselines/smote/main_baseline.py` | `results/adult/smote_seed0_n20.csv` |
| GReaT | Fine-tunes a causal language model on text-formatted table rows, then samples new rows. | `baselines/great/main_baseline.py` | `results/adult/great_seed0_n20.csv` |

## TabDDPM

TabDDPM models tabular data with a denoising diffusion process. The adapter
converts Medsyn CSV splits into the NumPy layout expected by the upstream
TabDDPM code, writes a TOML config, and calls the original pipeline for training
and sampling.

Prepare data and config:

```bash
python baselines/tabddpm/main_baseline.py --prepare-only
```

Smoke train and sample:

```bash
python baselines/tabddpm/main_baseline.py --train --sample --steps 10 --num-samples 20 --device cuda:7
```

Notes:

- Source snapshot: `baselines/tabddpm/tab-ddpm/`
- Config: `baselines/tabddpm/config_baseline.yaml`
- TabDDPM has older dependency assumptions; use `baselines/tabddpm/requirements_baseline.txt` as the baseline-specific reference.

## CTGAN

CTGAN trains a conditional GAN over transformed continuous and discrete columns.
The wrapper uses the standalone CTGAN implementation and treats Adult categorical
columns plus the label as discrete columns.

Smoke run:

```bash
python baselines/ctgan/main_baseline.py --epochs 1 --num-samples 20 --device cuda:7
```

Notes:

- Source snapshot: `baselines/ctgan/ctgan/`
- Config: `baselines/ctgan/config_baseline.yaml`
- Model artifacts are written under `baselines/ctgan/artifacts/` and should not be committed.

## TVAE

TVAE is a variational autoencoder baseline implemented in the CTGAN package.
It is exposed as a separate baseline because it is a distinct synthesizer from
CTGAN, even though both share the same upstream library.

Smoke run:

```bash
python baselines/tvae/main_baseline.py --epochs 1 --num-samples 20 --device cuda:7
```

Notes:

- Shared source: `baselines/ctgan/ctgan/synthesizers/tvae.py`
- Config: `baselines/tvae/config_baseline.yaml`
- TVAE loss values may become negative because the upstream objective includes a log-sigma likelihood term.

## SMOTE

SMOTE is a non-neural interpolation baseline. It creates synthetic rows by
interpolating between nearest neighbors, using SMOTENC for mixed continuous and
categorical Adult features. The local wrapper preserves the main TabDDPM SMOTE
settings while using the current `imbalanced-learn` API.

Smoke run:

```bash
python baselines/smote/main_baseline.py --num-samples 20
```

Notes:

- Original reference copy: `baselines/smote/original_tabddpm_smote/`
- Config: `baselines/smote/config_baseline.yaml`
- This baseline is fast and does not use GPU.

## GReaT

GReaT is a language-model baseline for tabular synthesis. It formats table rows
as text, fine-tunes a causal language model, samples generated text, and parses
the generated text back into table rows. It does not start a vLLM service; it
loads the model in-process through HuggingFace Transformers.

Smoke run:

```bash
python baselines/great/main_baseline.py --epochs 1 --num-samples 20 --device cuda:7
```

Use a local model path:

```bash
python baselines/great/main_baseline.py \
  --llm /data1/liangzhilin/model/Llama-3.2-1B-Instruct \
  --epochs 1 \
  --num-samples 20 \
  --device cuda:7
```

Notes:

- Source snapshot: `baselines/great/be_great/`
- Config: `baselines/great/config_baseline.yaml`
- Default `great.llm` is `distilgpt2`; local HuggingFace model paths can be used instead.
- Default GReaT is full fine-tuning unless LoRA support is explicitly exposed and enabled later.
- Model checkpoints are written under `baselines/great/artifacts/` and should not be committed.

## Dependency Policy

Baseline dependencies can differ substantially. The project policy is:

- The agent diagnoses missing conda/pip packages and provides commands.
- The user installs packages manually.
- Large model checkpoints and training artifacts should stay out of git.

Baseline-specific dependency references:

```text
baselines/tabddpm/requirements_baseline.txt
baselines/ctgan/requirements_baseline.txt
baselines/tvae/requirements_baseline.txt
baselines/great/requirements_baseline.txt
```

SMOTE currently relies on `imbalanced-learn`, which is already part of the main
project requirements.
