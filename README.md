# SQaLe: A Text-to-SQL Dataset Generation Pipeline grounded in real Schemas

Composable pipeline for curating large-scale text-to-SQL corpora by extending database schemas, synthesising natural-language questions, and validating SQL programs with LLMs.

The dataset can be accessed under [trl-lab/SQaLe-text-to-SQL-dataset/](https://huggingface.co/datasets/trl-lab/SQaLe-text-to-SQL-dataset/) on Hugging Face Datasets.

## Highlights
- Automates schema growth, question generation, SQL synthesis, and quality annotations.
- Runs locally or on Slurm clusters with GPU-backed vLLM inference.
- Ships reusable ReFoRCE text-to-SQL components plus utility scripts for metadata, analysis, and Hugging Face uploads.
- Produces JSONL artefacts ready for downstream fine-tuning or evaluation.

## Repository Contents
- `extend_schemas.py` – grows SQLite schemas with LLM assistance and on-the-fly validation.
- `create_questions.py` – batches vLLM calls to craft semi-synthetic NL questions per schema.
- `create_sql_queries.py` – generates executable SQL via the ReFoRCE voting pipeline.
- `add_metadata.py` / `augment_questions_vllm.py` – enrich or paraphrase dataset entries.
- `ReFoRCE/` – lightweight fork of the ReFoRCE text-to-SQL execution engine.
- `data/` – example schemas, generated triplets, and analysis helpers.
- `jobs/` – Slurm templates for running each phase on H100 class GPUs.
- `util/` – helper scripts (dataset splits, HF pushes, stats).

## Getting Started
1. **Clone & environment**
	```bash
	git clone https://github.com/trl-lab/text-to-sql-dataset.git
	cd text-to-sql-dataset
	conda env create -f conda-env.yml
	conda activate sqale
	```
2. **Provision credentials** (as needed):
	- Hugging Face: `export HF_TOKEN=...` for dataset uploads.

A CUDA enabled GPU is required for the default vLLM models listed in the job scripts.

## End-to-End Pipeline
The pipeline is modular; individual stages can be rerun or swapped. Paths below assume you stay in the repo root.




1. **Convert Schemapile into CREATE TABLE seeds**
	```bash
	python convert_to_create_statement.py
	```
	- Downloads `trl-lab/schemapile` via `datasets` and rewrites each entry into executable SQLite `CREATE TABLE` files under `data/datasets/` and `data/statements/`.
	- Validates every schema by building a temporary SQLite database; failed conversions are skipped with the error logged to stdout.

2. **Extend baseline schemas**
	```bash
	python extend_schemas.py \
	  --folder data/test/statements \
	  --out-dir data/test/extended \
	  --model Qwen/Qwen3-32B-FP8 \
	```
	- Adds 15–25 coherent tables per iteration until a target size is reached.
	- Every batch is executed against SQLite to ensure syntactic and referential correctness.

3. **Seed exemplar questions (optional but recommended)**
	```bash
	python create_examples.py
	```
	- Merges Spider/BIRD prompts with join counts into `data/examples.csv`, guiding later prompting diversity.
	- This can also be replaced with a custom CSV of your own question templates.

4. **Generate NL questions**
	```bash
	python create_questions.py \
	  --schema_folder data/test/extended \
	  --example_questions_file data/examples.csv \
	  --model Qwen/Qwen3-32B-FP8 \
	  --out data/questions.jsonl
	```
	- Uses vLLM to request multiple questions per schema while enforcing a join-count distribution.
	- Automatically retries malformed outputs and records prompt metadata.

5. **Synthesize SQL with ReFoRCE**
	```bash
	python create_sql_queries.py \
	  --questions_file data/questions.jsonl \
	  --model Qwen/Qwen3-32B-FP8 \
	  --batch_size 64 \
	  --out data/semi_synthetic_dataset.jsonl
	```
	- Generates candidates, executes them inside an in-memory SQLite clone, and runs a judge pass for final selection.
	- Output records include `prompt`, `sql_statement`, `schema`, and `cmd_type` fields.

6. **Add structural metadata**
	```bash
	python add_metadata.py \
	  --input data/semi_synthetic_dataset.jsonl \
	  --output data/combined_with_metadata.jsonl
	```
	- Appends token counts, join counts, table/column statistics, and normalises field names.

7. **(Optional) Question paraphrases**
	```bash
	python augment_questions_vllm.py \
	  --input data/combined_with_metadata.jsonl \
	  --output data/combined_with_paraphrases.jsonl \
	  --model Qwen/Qwen3-14B-FP8 \
	  --num-alternatives 3
	```
	- Generates faithful rewrites per entry, updating token counts and tagging augmentation provenance.

8. **Publish to Hugging Face**
	```bash
	python util/push_to_hf.py \
	  --jsonl data/combined_with_paraphrases.jsonl \
	  --repo-id your-org/text-to-sql \
	  --split-name train \
	  --private
	```

## Data Artefacts
- `data/test/statements/` – initial schema seeds (SQLite DDL).
- `data/test/extended/` – schema extensions per growth iteration.
- `data/questions.jsonl` – generated NL questions with schema context.
- `data/semi_synthetic_dataset.jsonl` – question/SQL/schema triplets.
- `data/combined_with_metadata.jsonl` – enriched records ready for release.
- `data/complete/` – sharded outputs from large Slurm runs.

## Running on Slurm
- Templates in `jobs/*.job` load the environment module stack, create the Conda env, and call the relevant script.
- Adjust `--array`, `--start_index`, and `--end_index` to chunk workloads across GPUs.
- Redirected logs live in `slurm_output/`.

## Citation
If you find SQaLe useful in your research, please cite the following paper:
```
@inproceedings{
	wolff2025sqale,
	title={{SQ}aLe: A large text-to-{SQL} corpus grounded in real schemas},
	author={Cornelius Wolff and Daniel Gomm and Madelon Hulsebos},
	booktitle={EurIPS 2025 Workshop: AI for Tabular Data},
	year={2025},
	url={https://openreview.net/forum?id=6PsKDjgoEy}
}
```

## License
This repository is licensed under the GPLv3 License. See `LICENSE` for details.