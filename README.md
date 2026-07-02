# Medsyn

Research experiment repository for the `medsyn` system.

## Status

Initialized as a lightweight experiment scaffold. The implementation, datasets,
and evaluation metrics are still placeholders.

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
python main.py --config config.yaml
python eval.py --config config.yaml
```

The default commands validate the scaffold and write placeholder JSON outputs.
Replace the placeholder implementation in `src/medsyn/` as the method becomes
defined.

## Artifact Storage

Large datasets, checkpoints, caches, and downloaded files should stay outside
the repository. The recommended artifact root is `/data1/liangzhilin`, pending
project confirmation before use.
