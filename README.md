# Medsyn

Research experiment repository for the `medsyn` system.

## Status

Initialized as a research scaffold for CLLM-style low-data tabular data
augmentation. Public/reference dataset preparation is available for `adult`,
`drug`, and `compas`; the main Medsyn generation, curation, and evaluation
pipeline is still under development.

## Layout

```text
.
|-- baselines/         # Baseline methods, wrappers, and adaptation notes
|-- datasets/          # Dataset preparation code and processed data folders
|-- docs/              # Iteration notes and reproducibility records
|-- eval/              # Evaluation outputs and metric summaries
|-- logs/              # Live experiment logs
|-- micro_benchmark/   # Self-contained side experiments and ablations
|-- results/           # Prediction and generation outputs
|-- scripts/           # Repeatable helper scripts
|-- src/               # Main medsyn implementation
|-- config.yaml        # Default experiment configuration
|-- eval.py            # Evaluation entrypoint
`-- main.py            # Main experiment entrypoint
```

## Quickstart

```bash
conda activate cllm
pip install -r requirements.txt
python main.py --config config.yaml
python eval.py --config config.yaml
```

The default commands validate the scaffold and write placeholder JSON outputs.
Replace the placeholder implementation in `src/medsyn/` as the method becomes
defined.

## Reference Materials

External papers, reference code, and downloaded supporting materials can be kept
locally in `ref/` during development:

```text
ref/
|-- CLLM-main/    # Optional local reference implementation of Curated LLM
`-- *.pdf         # Optional paper PDF and related local reading material
```

`ref/` is ignored by git and is not required to use the repository. Treat it as
read-mostly reference material. Adapted project code should live in `src/`,
`datasets/prep/`, `baselines/`, or an experiment folder instead of modifying the
reference copy directly.

## Dataset Preparation

Dataset preparation follows four rules:

- Keep source/reference material separate from project-ready data.
- Keep preparation logic reproducible in `datasets/prep/`.
- Store generated raw, processed, and split files under `datasets/<name>/`.
- Track the current lightweight prepared datasets in git for project visibility.

The first supported CLLM public datasets are:

| Dataset | Source | Label |
| --- | --- | --- |
| `adult` | local CLLM reference CSV, copied into `datasets/adult/raw/` | `salary` |
| `drug` | local CLLM reference CSV, copied into `datasets/drug/raw/` | `y` |
| `compas` | OpenML dataset id `42192` | `y` |

Prepare them with:

```bash
conda activate cllm
python datasets/prep/prepare_cllm_public.py --datasets adult drug compas
```

Each prepared dataset contains:

```text
datasets/<name>/
|-- raw/              # Copied or downloaded source data
|-- processed/        # Processed CLLM-style CSV
|-- splits/           # Low-data train, oracle, and test splits
`-- manifest.json     # Source, label column, row/column counts, split settings
```

The split settings mirror the CLLM low-data setup:

```text
train sizes: 20, 40, 100, 200 total samples
seeds: 0..9
train labels: balanced per class
```

Private or access-controlled medical datasets from the CLLM paper, such as
Covid, CUTRACT, MAGGIC, and SEER, are not prepared automatically. If local
authorized files are available, add a dataset-specific preparation script under
`datasets/prep/` and record the source and processing notes in `docs/`.

## Artifact Storage

This project currently tracks lightweight prepared datasets and empty experiment
directories so the GitHub repository is self-describing during early
development. Large checkpoints, model weights, caches, and files that exceed
normal GitHub limits should stay outside the repository or use Git LFS. The
recommended external artifact root is `/data1/liangzhilin`, pending project
confirmation before use.
