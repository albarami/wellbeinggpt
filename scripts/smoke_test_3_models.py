"""Quick smoke test for the 3 GPT deployments."""

import os
import sys
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            os.environ[k] = v

from openai import AzureOpenAI

# Config from .env
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

deployments = os.getenv("MODEL_VARIANT_DEPLOYMENTS", "gpt-5-chat,gpt-5.1,gpt-5.2").split(",")
deployments = [d.strip() for d in deployments if d.strip()]

print(f"Endpoint: {endpoint}")
print(f"API Version: {api_version}")
print(f"Deployments to test: {deployments}")
print()

if not endpoint or not api_key:
    print("ERROR: AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set in .env")
    sys.exit(1)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
)

results = []

for dep in deployments:
    print(f"Testing deployment: {dep} ...")
    try:
        response = client.chat.completions.create(
            model=dep,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say OK if you can hear me."},
            ],
            max_completion_tokens=50,
            timeout=30,
        )
        content = (response.choices[0].message.content or "").strip()
        print(f"  -> Response: {content[:100]}")
        results.append((dep, "OK", content[:50]))
    except Exception as e:
        print(f"  -> ERROR: {e}")
        results.append((dep, "FAIL", str(e)[:80]))

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for dep, status, msg in results:
    print(f"  {dep}: {status} - {msg}")

all_ok = all(r[1] == "OK" for r in results)
print()
if all_ok:
    print("All 3 models are WORKING.")
else:
    print("Some models FAILED. Fix before running benchmark.")
    sys.exit(1)
