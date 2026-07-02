# GReaT Baseline

This directory adapts GReaT as a standalone Medsyn baseline.

## Source

- Original local reference: `ref/be_great/`
- Local source snapshot: `baselines/great/be_great/`
- Original README: `baselines/great/README_original.md`
- License: `baselines/great/LICENSE`
- CLLM reference call: `ref/CLLM-main/src/cllm/models.py`

GReaT is a language-model baseline and is independent from CTGAN, TVAE, TabDDPM, and SMOTE.

## Adult Run

The default config follows the CLLM reference setting:

- LLM: `distilgpt2`
- epochs: `1000`
- batch size: `64`
- max length: `4000`
- train: `datasets/adult/splits/seed0_n20_train.csv`
- output: `results/adult/great_seed0_n20.csv`
- log: `logs/adult/great_seed0_n20.jsonl`

Quick smoke command:

```bash
python baselines/great/main_baseline.py --epochs 1 --num-samples 20 --device cuda:7
```

For CPU only:

```bash
python baselines/great/main_baseline.py --epochs 1 --num-samples 20 --device cpu
```

## Dependencies

The current project policy is that the agent diagnoses missing packages and gives commands, while the user installs conda/pip packages manually.

Install missing GReaT runtime dependencies manually:

```bash
pip install -r baselines/great/requirements_baseline.txt
```

GReaT loads HuggingFace checkpoints such as `distilgpt2`. If the checkpoint is not already cached locally, the first run may need network access from your shell.
