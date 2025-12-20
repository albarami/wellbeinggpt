"""Debug: Check raw response structure from each model."""

import os
import sys
from pathlib import Path
import json

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
DEPLOYMENTS = ["gpt-5-chat", "gpt-5.1", "gpt-5.2"]

client = AzureOpenAI(
    azure_endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION,
)

TEST_QUESTION = "ما هو الإيمان؟"

for dep in DEPLOYMENTS:
    print(f"\n{'='*60}")
    print(f"MODEL: {dep}")
    print(f"{'='*60}")
    
    try:
        response = client.chat.completions.create(
            model=dep,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": TEST_QUESTION},
            ],
            max_completion_tokens=200,
            timeout=60,
        )
        
        print(f"\nResponse type: {type(response)}")
        print(f"Response model: {response.model}")
        print(f"Choices count: {len(response.choices)}")
        
        if response.choices:
            choice = response.choices[0]
            print(f"\nChoice type: {type(choice)}")
            print(f"Finish reason: {choice.finish_reason}")
            print(f"\nMessage type: {type(choice.message)}")
            print(f"Message role: {choice.message.role}")
            print(f"Message content type: {type(choice.message.content)}")
            print(f"Message content repr: {repr(choice.message.content)[:200]}")
            print(f"Message content length: {len(choice.message.content or '')}")
            
            # Check for other attributes
            msg = choice.message
            print(f"\nMessage attributes: {[a for a in dir(msg) if not a.startswith('_')]}")
            
            # Check if content is in a different field
            if hasattr(msg, 'refusal'):
                print(f"Refusal: {msg.refusal}")
            if hasattr(msg, 'tool_calls'):
                print(f"Tool calls: {msg.tool_calls}")
            if hasattr(msg, 'function_call'):
                print(f"Function call: {msg.function_call}")
                
        print(f"\n--- Content Preview ---")
        content = response.choices[0].message.content or ""
        print(content[:300] if content else "(EMPTY)")
        
    except Exception as e:
        print(f"ERROR: {e}")
