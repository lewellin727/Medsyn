# medsyn v2 CLLM Public Dataset Preparation

## Goal

Prepare the first public/reference datasets for CLLM-style low-data tabular
augmentation experiments.

## Scope

Added a reusable preparation script and dataset README:

- `datasets/README.md`
- `datasets/prep/prepare_cllm_public.py`
- `requirements.txt`

Prepared local dataset artifacts for:

- `adult`
- `drug`
- `compas`

## Sources

- `adult`: copied from `ref/CLLM-main/data/adult.csv`.
- `drug`: copied from `ref/CLLM-main/data/Drug_Consumption.csv`.
- `compas`: downloaded from OpenML dataset id `42192`.

The reference CLLM paper PDF and code are stored under `ref/`.

## Commands

```bash
conda activate cllm
python datasets/prep/prepare_cllm_public.py --datasets adult drug compas
```

## Outputs

Generated outputs are under:

- `datasets/adult/`
- `datasets/drug/`
- `datasets/compas/`

Each dataset folder contains:

- `raw/`: copied or downloaded source data.
- `processed/<dataset>_cllm.csv`: processed full dataset.
- `splits/`: balanced low-data train splits plus oracle/test splits.
- `manifest.json`: source, label column, row/column counts, split settings.

Default split settings:

- total train sizes: `20`, `40`, `100`, `200`
- seeds: `0` through `9`
- binary labels balanced in each train split

## Verification

Verified with:

```bash
conda run -n cllm python -m py_compile datasets/prep/prepare_cllm_public.py
conda run -n cllm python datasets/prep/prepare_cllm_public.py --datasets adult drug
conda run -n cllm python datasets/prep/prepare_cllm_public.py --datasets compas
```

Checked that `seed0_n20_train.csv` is label-balanced for all three datasets.

## Notes

Dataset artifact directories are ignored by git via the existing
`datasets/*/` rule. The reproducible preparation script and dataset README
remain trackable.
