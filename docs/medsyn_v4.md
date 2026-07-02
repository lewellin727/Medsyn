# Medsyn v4: CLLM Main Pipeline Refactor

## Goal

Move the CLLM reference functionality into the project-standard `src/`, `main.py`, `eval.py`, and `config.yaml` layout while preserving the original high-level behavior:

1. Read low-data train/oracle/test splits.
2. Build a CLLM-style tabular generation prompt.
3. Generate synthetic rows through an LLM serving backend.
4. Clean generated rows into the dataset schema.
5. Optionally run data-centric curation.
6. Evaluate generated rows with downstream classifiers.

## Files

- `main.py`: root generation entrypoint with config loading and CLI overrides.
- `eval.py`: downstream evaluation entrypoint.
- `config.yaml`: default Medsyn run configuration.
- `src/schemas.py`: dataset schema registry for Adult, Drug, and COMPAS.
- `src/data.py`: split resolution, CSV loading, and generated-row cleaning.
- `src/templates.py`: CLLM-style prompt construction.
- `src/llm.py`: OpenAI-compatible, vLLM, Together, and Azure OpenAI serving adapter.
- `src/curation.py`: lightweight data-centric curation adapter based on XGBoost confidence.
- `src/runner.py`: orchestration for generation, logging, result writing, and optional curation.

## CLI

Default dry-run:

```bash
python main.py --config config.yaml --dry-run
python eval.py --config config.yaml
```

Dataset/split override:

```bash
python main.py --config config.yaml --dataset drug --data-seed 2 --size 40 --dry-run --n-samples 10
python eval.py --config config.yaml --dataset drug --data-seed 2 --size 40
```

LLM-backed generation:

```bash
python main.py --config config.yaml --no-dry-run --serving vllm --model /data1/liangzhilin/model/Llama-3.2-1B-Instruct
```

## Notes

- `generation.dry_run: true` writes resampled training rows to validate the file flow without calling an LLM. Disable dry-run for real generation.
- The LLM adapter does not start vLLM; it calls an already running OpenAI-compatible endpoint.
- Curation is disabled by default for faster smoke runs. Set `curation.enabled: true` to produce curation groups.
- Evaluation trains XGBoost, random forest, logistic regression, and decision tree classifiers on synthetic, original, and oracle data, then evaluates against the test split.

## Verification

- `python -m py_compile main.py eval.py src/*.py` passed.
- `conda run -n cllm python main.py --config config.yaml --dry-run --n-samples 10` passed.
- `conda run -n cllm python main.py --config config.yaml --dataset drug --data-seed 2 --size 40 --dry-run --n-samples 10` passed.
- `conda run -n cllm python eval.py --config config.yaml` passed.
- `conda run -n cllm python eval.py --config config.yaml --dataset drug --data-seed 2 --size 40` passed.
