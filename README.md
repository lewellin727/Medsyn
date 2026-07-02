# Medsyn

Research experiment repository for the `medsyn` system.

## Status

Initialized as a research scaffold for CLLM-style low-data tabular data
augmentation. The main Medsyn generation, curation, and evaluation pipeline is
still under development.

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

## Data

Dataset preparation details live in `datasets/README.md`.

## Artifact Storage

This project currently tracks lightweight prepared datasets and empty experiment
directories so the GitHub repository is self-describing during early
development. Large checkpoints, model weights, caches, and files that exceed
normal GitHub limits should stay outside the repository or use Git LFS. The
recommended external artifact root is `/data1/liangzhilin`, pending project
confirmation before use.
