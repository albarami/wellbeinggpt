"""Run metadata builder.

All evaluation outputs must be traceable to:
- dataset version/hash
- code commit
- model/prompt identifiers
- deterministic seed

This module does not call external services.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from eval.types import EvalRunMetadata


@dataclass(frozen=True)
class ProviderMeta:
    llm_provider: Optional[str]
    llm_model: Optional[str]
    llm_deployment: Optional[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def try_git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return None
        return (r.stdout or "").strip() or None
    except Exception:
        return None


def provider_meta_from_env() -> ProviderMeta:
    # Keep this broad; different environments set different variables.
    provider = os.getenv("LLM_PROVIDER") or "azure_openai"

    # Common envs in this repo.
    model = os.getenv("AZURE_OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv(
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
    )

    return ProviderMeta(
        llm_provider=provider,
        llm_model=model,
        llm_deployment=deployment,
    )


def build_run_id(
    *,
    dataset_id: str,
    dataset_version: str,
    dataset_sha256: str,
    seed: int,
    prompts_version: str,
) -> str:
    # Short, stable, deterministic run id.
    short = dataset_sha256[:12]
    return f"{dataset_id}__{dataset_version}__{short}__seed{seed}__p{prompts_version}"


def build_run_metadata(
    *,
    repo_root: Path,
    dataset_id: str,
    dataset_version: str,
    dataset_sha256: str,
    seed: int,
    prompts_version: str,
) -> EvalRunMetadata:
    p = provider_meta_from_env()
    run_id = build_run_id(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        seed=seed,
        prompts_version=prompts_version,
    )

    return EvalRunMetadata(
        run_id=run_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        seed=seed,
        llm_provider=p.llm_provider,
        llm_model=p.llm_model,
        llm_deployment=p.llm_deployment,
        prompts_version=prompts_version,
        code_commit=try_git_commit_hash(repo_root),
        generated_at_utc=_utc_now_iso(),
    )
