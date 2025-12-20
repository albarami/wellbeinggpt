"""Debug gpt-5.1 response format."""
import os
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

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview").strip()

print(f"Endpoint: {ENDPOINT}")
print(f"API Version: {API_VERSION}")

client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=API_KEY, api_version=API_VERSION)

# Test gpt-5.1 with full system prompt
SYSTEM = """أنت عالم متخصص في إطار الحياة الطيبة (الركائز الخمس).
- أجب بالعربية الفصحى بأسلوب علمي عميق
- استخدم التنسيق المناسب (عناوين، نقاط، قوائم)
- اربط المفاهيم عبر الركائز المختلفة
- قدم تحليلاً معمقًا واختم بخلاصة تنفيذية
- إذا كان السؤال خارج نطاق الإطار، قل "خارج نطاق الإطار"
- لا تختلق معلومات"""

QUESTION = "كيف يؤدي الإطار إلى ازدهار الإنسان؟"

print("\n=== Testing gpt-5.1 with Arabic question (max_tokens) ===")
try:
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION}
        ],
        max_tokens=2000,
        temperature=0.1,
        seed=1337,
        timeout=120
    )
except Exception as e1:
    print(f"max_tokens failed: {e1}")
    print("\n=== Trying with max_completion_tokens and shorter question ===")
    resp = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "user", "content": "ما هو الإيمان؟"}
        ],
        max_completion_tokens=500,
        temperature=0.1,
        timeout=120
    )

print(f"Type of message.content: {type(resp.choices[0].message.content)}")
print(f"Raw content repr: {repr(resp.choices[0].message.content)}")
print(f"Finish reason: {resp.choices[0].finish_reason}")

# Extract properly
raw = resp.choices[0].message.content
if raw is None:
    content = ""
elif isinstance(raw, str):
    content = raw.strip()
elif isinstance(raw, list):
    parts = []
    for item in raw:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(item.get("text", "") or item.get("content", ""))
    content = "".join(parts).strip()
else:
    content = str(raw).strip()

print(f"\nExtracted content: {content}")
print(f"Word count: {len(content.split())}")


