# medsyn v1 Initialization

## Goal

Initialize a reproducible research experiment scaffold for the `medsyn` system.

## Scope

Created the default project layout for code, datasets, results, evaluation
outputs, logs, documentation, helper scripts, micro-benchmarks, and baselines.

## Files and Directories

- `README.md`
- `config.yaml`
- `main.py`
- `eval.py`
- `src/medsyn/`
- `datasets/`
- `results/`
- `eval/`
- `logs/`
- `scripts/`
- `micro_benchmark/`
- `baselines/`

## Artifact Paths

Recommended artifact root: `/data1/liangzhilin`.

No external artifact path was read or written during initialization.

## Commands

```bash
python main.py --config config.yaml
python eval.py --config config.yaml
```

## Outputs

The scaffold commands write placeholder outputs to:

- `results/placeholder/medsyn.json`
- `eval/placeholder/medsyn_metrics.json`
- `logs/placeholder/medsyn_v1_run.jsonl`

## Git

The project root contains a `.git/` mountpoint, but it is read-only and not a
valid git repository. Git was not initialized, and no nonstandard git metadata
directory was created.

## Limitations

The method implementation, dataset definitions, metrics, and baseline
comparisons are not yet defined.
