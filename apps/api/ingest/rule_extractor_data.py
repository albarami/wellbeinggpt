"""Rule extractor marker constants.

Reason: keep `rule_extractor.py` <500 LOC (project rule).
"""

from __future__ import annotations

import re

# Pillar markers
PILLARS_LIST_MARKERS = [
    "ركائز الحياة الطيبة",
    "الركائز الخمس",
]

PILLAR_MARKERS = [
    re.compile(r"^(أولا|ثانيا|ثالثا|رابعا|خامسا)[.\s:]*الركيزة", re.UNICODE),
    re.compile(r"^الركيزة\s+(الروحية|العاطفية|الفكرية|البدنية|الاجتماعية)", re.UNICODE),
    # Some documents use "الحياة <pillar> الطيبة" as a standalone heading line (must end there)
    re.compile(
        r"^(أولا|ثانيا|ثالثا|رابعا|خامسا)?[.\s:]*الحياة\s+(الروحية|العاطفية|الفكرية|البدنية|الاجتماعية)\s+الطيبة\s*[:：]?\s*$",
        re.UNICODE,
    ),
]

# Core value markers
CORE_VALUES_LIST_MARKERS = [
    "القيم الكلية",
    "القيم الأمهات",
]

# Sub-value markers
SUB_VALUES_LIST_MARKERS = [
    "القيم الجزئية",
    "القيم الأحفاد",
    "الأحفاد",
]

# Definition markers
DEFINITION_MARKERS = [
    "المفهوم",
    "المفهوم:",
    "المفهوم :",
    "التعريف",
    "التعريف:",
    "التعريف :",
    "التعريف الإجرائي",
    "التعريف الإجرائي:",
    "التعريف الإجرائي :",
]

# Evidence markers
EVIDENCE_MARKERS = [
    "التأصيل",
    "التأصيل:",
    "التأصيل :",
    "التفصيل",
    "التفصيل:",
    "التفصيل :",
    "الأدلة",
    "الأدلة:",
    "الأدلة :",
    "الدليل",
    "الدليل:",
    "الدليل :",
    "الشواهد",
    "الشواهد:",
    "الشواهد :",
]

# Section end markers (any of these starts a new section)
SECTION_END_MARKERS = DEFINITION_MARKERS + EVIDENCE_MARKERS + [
    "القيم الكلية",
    "القيم الجزئية",
]

