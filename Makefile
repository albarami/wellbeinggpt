.PHONY: test preflight-eval eval eval-fast eval-deep tune-eligibility eval-stakeholder eval-stakeholder-v2-hard

# Always use python -m pytest on Windows

test:
	python -m pytest -v

preflight-eval:
	python -m scripts.preflight_eval --fail

# Full evaluation: generate datasets, run all 4 datasets, score them.
# Note: this does not run LLM_ONLY_UNGROUNDED by default.

eval:
	python -m scripts.preflight_eval --fail
	python -m eval.datasets.generate
	python -m eval.runner --dataset eval/datasets/gold_qa_ar.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only
	python -m eval.runner --dataset eval/datasets/cross_pillar.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only
	python -m eval.runner --dataset eval/datasets/deep_cross_pillar_gold.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only
	python -m eval.runner --dataset eval/datasets/negative_oos.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only
	python -m eval.runner --dataset eval/datasets/mixed_oos.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only
	python -m eval.runner --dataset eval/datasets/adversarial_injection.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only
	python -m eval.scoring.run --dataset eval/datasets/gold_qa_ar.jsonl --output-dir eval/output
	python -m eval.scoring.run --dataset eval/datasets/cross_pillar.jsonl --output-dir eval/output
	python -m eval.scoring.run --dataset eval/datasets/deep_cross_pillar_gold.jsonl --output-dir eval/output
	python -m eval.scoring.run --dataset eval/datasets/negative_oos.jsonl --output-dir eval/output
	python -m eval.scoring.run --dataset eval/datasets/mixed_oos.jsonl --output-dir eval/output
	python -m eval.scoring.run --dataset eval/datasets/adversarial_injection.jsonl --output-dir eval/output
	python -m eval.reporting.generate_latest

eval-deep:
	python -m scripts.preflight_eval --fail
	python -m eval.runner --dataset eval/datasets/deep_cross_pillar_gold.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only
	python -m eval.scoring.run --dataset eval/datasets/deep_cross_pillar_gold.jsonl --output-dir eval/output
	python -m eval.reporting.generate_latest

eval-stakeholder:
	python -m scripts.preflight_eval --fail
	python -m eval.runner --dataset eval/datasets/stakeholder_acceptance_v1.jsonl --dataset-id stakeholder_acceptance --dataset-version v1 --out-dir eval/output --no-llm-only
	python -m eval.scoring.run --dataset eval/datasets/stakeholder_acceptance_v1.jsonl --output-dir eval/output
	python -m scripts.print_stakeholder_outputs
	python -m eval.reporting.generate_latest

eval-stakeholder-v2-hard:
	python -m scripts.preflight_eval --fail
	python -m eval.runner --dataset eval/datasets/stakeholder_acceptance_v2_hard.jsonl --dataset-id stakeholder_acceptance --dataset-version v2_hard --out-dir eval/output --no-llm-only
	python -m eval.scoring.run --dataset eval/datasets/stakeholder_acceptance_v2_hard.jsonl --output-dir eval/output
	python -m eval.reporting.generate_latest

tune-eligibility:
	python -m scripts.preflight_eval --fail
	python -m eval.tune_eligibility --dataset eval/datasets/mixed_oos.jsonl --out-dir eval/output

# Fast evaluation: golden slice only

eval-fast:
	python -m eval.datasets.generate
	python -m eval.runner --dataset eval/datasets/golden_slice/gold.jsonl --dataset-id wellbeing --dataset-version vfast --out-dir eval/output --no-llm-only
