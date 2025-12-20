"""Test normalization of synth-006 question."""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from apps.api.retrieve.normalize_ar import normalize_for_matching
from apps.api.core.muhasibi_listen import _is_global_synthesis_intent

question = "ما المنظور الكلي للإطار؟"
normalized = normalize_for_matching(question)

print(f"Original: {question}")
print(f"Normalized: {normalized}")
print(f"Check marker 'المنظور الكلي': {'المنظور الكلي' in normalized}")

# Test the function
result = _is_global_synthesis_intent(normalized)
print(f"_is_global_synthesis_intent: {result}")

# Check each marker
markers = [
    "المنظور الكلي",
    "الرويه الشامله", 
    "يميز هذا الاطار",
]
for m in markers:
    print(f"  '{m}' in normalized: {m in normalized}")
