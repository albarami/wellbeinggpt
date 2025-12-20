@echo off

setlocal enableextensions



rem Windows shim for Makefile targets.

rem Reason: Windows environments may not have GNU make installed.

rem Usage:

rem   make test

rem   make eval

rem   make eval-deep

rem   make tune-eligibility



if "%~1"=="" goto :help



set TARGET=%~1



if /I "%TARGET%"=="test" goto :test

if /I "%TARGET%"=="preflight-eval" goto :preflight

if /I "%TARGET%"=="eval" goto :eval

if /I "%TARGET%"=="eval-deep" goto :eval_deep

if /I "%TARGET%"=="tune-eligibility" goto :tune

if /I "%TARGET%"=="eval-fast" goto :eval_fast

if /I "%TARGET%"=="eval-stakeholder" goto :eval_stakeholder

if /I "%TARGET%"=="eval-muhasibi-ab" goto :eval_muhasibi_ab

if /I "%TARGET%"=="eval-muhasibi-breakthrough" goto :eval_muhasibi_breakthrough

if /I "%TARGET%"=="train-reranker" goto :train_reranker

if /I "%TARGET%"=="eval-bakeoff-depth" goto :eval_bakeoff_depth

if /I "%TARGET%"=="eval-regression" goto :eval_regression

if /I "%TARGET%"=="rescore-bakeoff" goto :rescore_bakeoff

if /I "%TARGET%"=="ci" goto :ci

if /I "%TARGET%"=="reranker-ab" goto :reranker_ab



echo Unknown target: %TARGET%

goto :help



::test

python -m pytest -v

exit /b %ERRORLEVEL%



::preflight

python -m scripts.preflight_eval --fail

exit /b %ERRORLEVEL%



::eval

python -m scripts.preflight_eval --fail || exit /b %ERRORLEVEL%

python -m eval.datasets.generate || exit /b %ERRORLEVEL%

rem Run runner in shards to avoid long single-process runs on Windows/Cursor.
for %%S in (0 50 100 150) do (
  python -m eval.runner --dataset eval/datasets/gold_qa_ar.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only --start %%S --limit 50 || exit /b %ERRORLEVEL%
)
for %%S in (0 40) do (
  python -m eval.runner --dataset eval/datasets/cross_pillar.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only --start %%S --limit 40 || exit /b %ERRORLEVEL%
)
for %%S in (0 30) do (
  python -m eval.runner --dataset eval/datasets/deep_cross_pillar_gold.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only --start %%S --limit 30 || exit /b %ERRORLEVEL%
)
for %%S in (0 30) do (
  python -m eval.runner --dataset eval/datasets/negative_oos.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only --start %%S --limit 30 || exit /b %ERRORLEVEL%
)
for %%S in (0 30) do (
  python -m eval.runner --dataset eval/datasets/mixed_oos.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only --start %%S --limit 30 || exit /b %ERRORLEVEL%
)
for %%S in (0 20) do (
  python -m eval.runner --dataset eval/datasets/adversarial_injection.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only --start %%S --limit 20 || exit /b %ERRORLEVEL%
)

python -m eval.scoring.run --dataset eval/datasets/gold_qa_ar.jsonl --output-dir eval/output || exit /b %ERRORLEVEL%

python -m eval.scoring.run --dataset eval/datasets/cross_pillar.jsonl --output-dir eval/output || exit /b %ERRORLEVEL%

python -m eval.scoring.run --dataset eval/datasets/deep_cross_pillar_gold.jsonl --output-dir eval/output || exit /b %ERRORLEVEL%

python -m eval.scoring.run --dataset eval/datasets/negative_oos.jsonl --output-dir eval/output || exit /b %ERRORLEVEL%

python -m eval.scoring.run --dataset eval/datasets/mixed_oos.jsonl --output-dir eval/output || exit /b %ERRORLEVEL%

python -m eval.scoring.run --dataset eval/datasets/adversarial_injection.jsonl --output-dir eval/output || exit /b %ERRORLEVEL%

python -m eval.reporting.generate_latest

exit /b %ERRORLEVEL%



::eval_deep

python -m scripts.preflight_eval --fail || exit /b %ERRORLEVEL%

for %%S in (0 30) do (
  python -m eval.runner --dataset eval/datasets/deep_cross_pillar_gold.jsonl --dataset-id wellbeing --dataset-version v1 --out-dir eval/output --no-llm-only --start %%S --limit 30 || exit /b %ERRORLEVEL%
)

python -m eval.scoring.run --dataset eval/datasets/deep_cross_pillar_gold.jsonl --output-dir eval/output || exit /b %ERRORLEVEL%

python -m eval.reporting.generate_latest

exit /b %ERRORLEVEL%



::tune

python -m scripts.preflight_eval --fail || exit /b %ERRORLEVEL%

python -m eval.tune_eligibility --dataset eval/datasets/mixed_oos.jsonl --out-dir eval/output

exit /b %ERRORLEVEL%



::eval_fast

python -m eval.datasets.generate || exit /b %ERRORLEVEL%

python -m eval.runner --dataset eval/datasets/golden_slice/gold.jsonl --dataset-id wellbeing --dataset-version vfast --out-dir eval/output --no-llm-only

exit /b %ERRORLEVEL%



:eval_stakeholder

python -m scripts.preflight_eval --fail || exit /b %ERRORLEVEL%

python -m eval.runner --dataset eval/datasets/stakeholder_acceptance_v1.jsonl --dataset-id stakeholder_acceptance --dataset-version v1 --out-dir eval/output --no-llm-only || exit /b %ERRORLEVEL%

python -m eval.scoring.run --dataset eval/datasets/stakeholder_acceptance_v1.jsonl --output-dir eval/output || exit /b %ERRORLEVEL%

python -m scripts.print_stakeholder_outputs || exit /b %ERRORLEVEL%

python -m eval.reporting.generate_latest

exit /b %ERRORLEVEL%



:eval_muhasibi_ab

rem Muḥāsibī A/B evaluation: compare all 6 modes to quantify value-add

rem Runs: LLM_ONLY_UNGROUNDED, RAG_ONLY, RAG_ONLY_INTEGRITY, RAG_PLUS_GRAPH, RAG_PLUS_GRAPH_INTEGRITY, FULL_SYSTEM

python -m scripts.preflight_eval --fail || exit /b %ERRORLEVEL%

python -m eval.datasets.generate || exit /b %ERRORLEVEL%

rem Run eval on gold_qa dataset (runner automatically runs all modes including +INTEGRITY variants)

python -m eval.runner --dataset eval/datasets/gold_qa_ar.jsonl --dataset-id muhasibi_ab --dataset-version v1 --out-dir eval/output/ab || exit /b %ERRORLEVEL%

rem Generate comparative A/B report

python -m eval.reporting.muhasibi_ab_report --outputs-dir eval/output/ab --out eval/reports/muhasibi_ab_report.md

exit /b %ERRORLEVEL%



:eval_muhasibi_breakthrough

rem Muḥāsibī breakthrough evaluation: FULL_SYSTEM with breakthrough mode enabled

set MUHASIBI_ENABLE_BREAKTHROUGH=1

python -m scripts.preflight_eval --fail || exit /b %ERRORLEVEL%

python -m eval.runner --dataset eval/datasets/gold_qa_ar.jsonl --dataset-id muhasibi_breakthrough --dataset-version v1 --out-dir eval/output/breakthrough --no-llm-only || exit /b %ERRORLEVEL%

python -m eval.reporting.muhasibi_ab_report --outputs-dir eval/output/breakthrough --breakthrough --out eval/reports/muhasibi_breakthrough_report.md

set MUHASIBI_ENABLE_BREAKTHROUGH=

exit /b %ERRORLEVEL%



:train_reranker

rem Train CrossEncoder reranker (single GPU)

rem For multi-GPU training use torchrun directly:

rem   torchrun --nproc_per_node=8 scripts/train_reranker_torchrun.py --train data/reranker/train.jsonl --model aubmindlab/bert-base-arabertv2 --out checkpoints/reranker

if not exist "data\reranker\train.jsonl" (

  echo ERROR: Training data not found at data\reranker\train.jsonl

  echo Run: python -m scripts.prepare_reranker_training_data first

  exit /b 1

)

python scripts/train_reranker_torchrun.py --train data\reranker\train.jsonl --model aubmindlab/bert-base-arabertv2 --out checkpoints\reranker --epochs 1 --batch-size 8

exit /b %ERRORLEVEL%



:eval_bakeoff_depth

rem Depth-focused model bakeoff: compare GPT-5/5.1/5.2 on 150+ depth questions

rem Scoring: 45% depth + 35% cross-pillar + 15% naturalness + 5% integrity

python scripts/run_depth_bakeoff.py

exit /b %ERRORLEVEL%



:eval_regression

rem Regression gate: run unexpected failure questions through API and ensure PASS_FULL

rem This should be added to CI after fixing the root causes

if not exist "eval\datasets\regression_unexpected_fails.jsonl" (
  echo No regression dataset found. Run: python scripts/export_unexpected_failures.py
  exit /b 1
)

echo Running regression tests on unexpected failure questions...
python scripts/run_regression_test.py

exit /b %ERRORLEVEL%



:rescore_bakeoff

rem Re-score existing bakeoff output using production signals

python scripts/rescore_bakeoff_final.py

exit /b %ERRORLEVEL%



:ci

rem CI gate: runs unit tests + regression tests
rem Required for all merges/releases

echo ===========================================
echo CI GATE: Running unit tests...
echo ===========================================

python -m pytest tests/ -v --tb=short -x || exit /b %ERRORLEVEL%

echo ===========================================
echo CI GATE: Running regression tests...
echo ===========================================

if not exist "eval\datasets\regression_unexpected_fails.jsonl" (
  echo No regression dataset found. Run: python scripts/export_unexpected_failures.py
  exit /b 1
)

python scripts/run_regression_test.py || exit /b %ERRORLEVEL%

echo ===========================================
echo CI GATE: All checks passed!
echo ===========================================

exit /b 0



:reranker_ab

rem Reranker A/B test: compare performance with reranker ON vs OFF
rem Uses gpt-5.1 on regression dataset

echo Running Reranker A/B Test...
python scripts/run_reranker_ab.py

exit /b %ERRORLEVEL%



:help

echo Usage: make ^<target^>

echo Targets: test preflight-eval eval eval-deep tune-eligibility eval-fast
echo          eval-stakeholder eval-muhasibi-ab eval-muhasibi-breakthrough
echo          train-reranker eval-bakeoff-depth eval-regression rescore-bakeoff
echo          ci reranker-ab

exit /b 2
