# Medsyn

Research experiment repository for the `medsyn` system.

## Status

Initialized as a research scaffold for CLLM-style low-data tabular data
augmentation. The baseline suite and the first CLLM-style Medsyn
generation/evaluation pipeline are now in place. The default config uses
`generation.dry_run: true` so the file flow can be validated without calling an
LLM service.

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
python main.py --config config.yaml --dry-run
python eval.py --config config.yaml
```

Run another prepared split:

```bash
python main.py --config config.yaml --dataset drug --data-seed 2 --size 40 --dry-run
python eval.py --config config.yaml --dataset drug --data-seed 2 --size 40
```

To call an LLM-backed generator, set `generation.dry_run: false` or pass
`--no-dry-run`, then configure `generation.serving`, `generation.model`, and
`generation.api` in `config.yaml`.

## Data

Dataset preparation details live in `datasets/README.md`.

## Baselines

Baseline method details and commands live in `baselines/README.md`.

## Artifact Storage

This project currently tracks lightweight prepared datasets and empty experiment
directories so the GitHub repository is self-describing during early
development. Large checkpoints, model weights, caches, and files that exceed
normal GitHub limits should stay outside the repository or use Git LFS. The
recommended external artifact root is `/data1/liangzhilin`, pending project
confirmation before use.
