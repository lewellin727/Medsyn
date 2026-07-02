# Datasets

This directory stores reproducible dataset preparation code and local processed
dataset folders for CLLM-style experiments.

Generated dataset folders are intentionally ignored by git. Keep raw downloads,
processed CSV files, and split files under `datasets/<dataset_name>/`.

## CLLM Public Datasets

The first preparation target is the public/reference data used by CLLM:

- `adult`: copied from `ref/CLLM-main/data/adult.csv` and processed with the
  CLLM reference preprocessing rules.
- `drug`: copied from `ref/CLLM-main/data/Drug_Consumption.csv`; the binary
  label `y` follows the CLLM reference rule for nicotine usage.
- `higgs`: copied from `ref/CLLM-main/data/training.csv` when present.
- `compas`: fetched from OpenML dataset id `42192`.

Prepare them with:

```bash
conda activate cllm
python datasets/prep/prepare_cllm_public.py --datasets adult drug compas
```

Use `--datasets higgs` separately if the large local reference file is needed.
