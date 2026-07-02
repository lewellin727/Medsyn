# Datasets

This directory stores reproducible dataset preparation code and local processed
dataset folders for CLLM-style experiments.

This project intentionally tracks the current lightweight prepared datasets in
git so the repository is self-describing during early development. Large private
datasets, model checkpoints, and cache-heavy artifacts should still use external
storage or Git LFS.

## Principles

- Keep source/reference material separate from project-ready data.
- Keep preparation logic reproducible in `datasets/prep/`.
- Store generated raw, processed, and split files under `datasets/<name>/`.
- Record each dataset source, label column, row/column counts, and split
  settings in `manifest.json`.

## CLLM Public Datasets

The first preparation target is the public/reference data used by CLLM.

| Dataset | Source | Label |
| --- | --- | --- |
| `adult` | local CLLM reference CSV, copied into `datasets/adult/raw/` | `salary` |
| `drug` | local CLLM reference CSV, copied into `datasets/drug/raw/` | `y` |
| `compas` | OpenML dataset id `42192` | `y` |
| `higgs` | local CLLM reference CSV, copied into `datasets/higgs/raw/` when needed | `y` |

Prepare them with:

```bash
conda activate cllm
python datasets/prep/prepare_cllm_public.py --datasets adult drug compas
```

Use `--datasets higgs` separately if the large local reference file is needed.

Each prepared dataset contains:

```text
datasets/<name>/
|-- raw/              # Copied or downloaded source data
|-- processed/        # Processed CLLM-style CSV
|-- splits/           # Low-data train, oracle, and test splits
`-- manifest.json     # Source, label column, row/column counts, split settings
```

Default split settings mirror the CLLM low-data setup:

```text
train sizes: 20, 40, 100, 200 total samples
seeds: 0..9
train labels: balanced per class
```

Private or access-controlled medical datasets from the CLLM paper, such as
Covid, CUTRACT, MAGGIC, and SEER, are not prepared automatically. If local
authorized files are available, add a dataset-specific preparation script under
`datasets/prep/` and record the source and processing notes in `docs/`.
