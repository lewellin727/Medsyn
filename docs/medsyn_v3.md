# Medsyn v3: Baseline Suite Construction

## Goal

Build the first complete Medsyn baseline suite for the prepared Adult CLLM split. This version groups the baseline adapters as one experiment-code iteration instead of treating each baseline as a separate project version.

## Baselines

Implemented baseline adapters:

| Baseline | Method Family | Entrypoint | Default Output |
| --- | --- | --- | --- |
| TabDDPM | Diffusion model for mixed tabular data | `baselines/tabddpm/main_baseline.py` | `results/adult/tabddpm_seed0_n20.csv` |
| CTGAN | Conditional GAN | `baselines/ctgan/main_baseline.py` | `results/adult/ctgan_seed0_n20.csv` |
| TVAE | Variational autoencoder | `baselines/tvae/main_baseline.py` | `results/adult/tvae_seed0_n20.csv` |
| SMOTE | Interpolation oversampling | `baselines/smote/main_baseline.py` | `results/adult/smote_seed0_n20.csv` |
| GReaT | Fine-tuned causal language model for tabular rows | `baselines/great/main_baseline.py` | `results/adult/great_seed0_n20.csv` |

The unified baseline guide is `baselines/README.md`.

## Shared Convention

Each baseline is organized as:

```text
baselines/<baseline_name>/
|-- README.md
|-- config_baseline.yaml
`-- main_baseline.py
```

Default dataset:

```text
datasets/adult/splits/seed0_n20_train.csv
```

Default output area:

```text
results/adult/
logs/adult/
```

Baseline-specific source snapshots are kept inside the baseline directory when needed. Large model artifacts and checkpoints are kept under baseline `artifacts/` folders and should not be committed.

All baseline wrappers support command-line dataset split selection:

```text
--dataset adult|drug|compas
--data-seed <0..9>
--size 20|40|100|200
```

The shared schema and path resolver lives in `baselines/common.py`.

## Source Notes

- TabDDPM uses a local source snapshot under `baselines/tabddpm/tab-ddpm/`.
- CTGAN vendors the standalone CTGAN package under `baselines/ctgan/ctgan/`.
- TVAE reuses the CTGAN package source but has a separate wrapper under `baselines/tvae/`.
- SMOTE keeps a reference copy of the TabDDPM SMOTE scripts under `baselines/smote/original_tabddpm_smote/` and uses a modern `imbalanced-learn` wrapper for compatibility.
- GReaT vendors `be_great` under `baselines/great/be_great/` and loads HuggingFace causal language models in-process; it does not start a vLLM service.

## Commands

TabDDPM prepare-only:

```bash
python baselines/tabddpm/main_baseline.py --prepare-only
```

TabDDPM smoke train/sample:

```bash
python baselines/tabddpm/main_baseline.py --train --sample --steps 10 --num-samples 20 --device cuda:7
```

CTGAN smoke:

```bash
python baselines/ctgan/main_baseline.py --dataset compas --data-seed 3 --size 100 --epochs 1 --num-samples 20 --device cuda:7
```

TVAE smoke:

```bash
python baselines/tvae/main_baseline.py --epochs 1 --num-samples 20 --device cuda:7
```

SMOTE smoke:

```bash
python baselines/smote/main_baseline.py --dataset compas --data-seed 3 --size 100 --num-samples 20
```

GReaT smoke:

```bash
python baselines/great/main_baseline.py --epochs 1 --num-samples 20 --device cuda:7
```

GReaT with local HuggingFace model path:

```bash
python baselines/great/main_baseline.py \
  --llm /data1/liangzhilin/model/Llama-3.2-1B-Instruct \
  --epochs 1 \
  --num-samples 20 \
  --device cuda:7
```

## Verification Status

Completed checks:

- TabDDPM `--prepare-only` completed and wrote the TabDDPM data/config layout.
- TabDDPM `--dataset drug --data-seed 2 --size 40 --prepare-only` completed and wrote dataset-specific data/config paths.
- SMOTE smoke run completed and generated `results/adult/smote_seed0_n20.csv` with 20 rows and 13 columns.
- SMOTE `--dataset compas --data-seed 3 --size 100 --num-samples 10 --k-neighbours 3` completed and generated `results/compas/smote_seed3_n100.csv`.
- TVAE CPU smoke run completed and generated `results/adult/tvae_seed0_n20.csv` with 20 rows and 13 columns.
- CTGAN wrapper passed static compile and help checks; full run depends on the CTGAN runtime dependency state.
- GReaT wrapper passed static compile and help checks; full run requires GReaT runtime dependencies and a local or downloadable HuggingFace causal LM.

Environment notes:

- The project policy is that the agent diagnoses missing conda/pip packages and gives commands, while the user installs packages manually.
- GReaT currently needs `transformers`, `accelerate`, and `chugchug` in the active environment.
- GReaT's default mode is full fine-tuning unless LoRA support is explicitly exposed and enabled later.
- TVAE loss values may become negative because the upstream objective includes a log-sigma likelihood term.

## Follow-Up

The next baseline work should focus on running a consistent smoke/evaluation matrix across all baseline outputs, then adding evaluation summaries under `eval/` rather than adding more per-baseline version documents.
