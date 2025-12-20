## Quranic Human‑Behavior Classification Matrix (Usul al‑Fiqh–Aligned)
### Standalone methodology and implementation specification (enhanced using Bouzidani’s new paper version)

---

## 0) Orientation (why this specification exists)
This specification defines a **Qur’an‑grounded, audit‑grade classification matrix for human behavior** that is **Usul‑aligned** and **fail‑closed**:

- **No evidence → no label** (and no “derived meaning” masquerading as text).
- **Direct vs indirect** is tracked per label.
- **Qur’an‑only signals** are separated from **juristic derivations**.
- **Controlled IDs** prevent drift across annotators and institutions.

This version incorporates key conceptual reinforcement from Ibrahim Bouzidani’s updated study on building a Qur’anic behavior matrix in its Islamic psychology context:
- **فطرة البصيرة** (fitrah‑based moral insight) + **موضوعية الأخلاق** (objective ethics) as a grounding for why a Qur’an‑based behavioral taxonomy is not merely descriptive.
- A **three‑world model** of human experience (sensory / psychic‑cognitive / spiritual) as an explanatory scaffold for “internal vs external” behavior and the role of intention.
- A tighter **عمل/نية/غريزة** framing: behavior as “action” (عمل) can be morally ranked (صالح/غير صالح), and **intention can transform routine/neutral acts into worship‑value** (without collapsing text into fiqh).

It also adopts the paper’s warning about **importing psychological taxonomies without their philosophical premises**. Therefore this specification:
- makes philosophical assumptions explicit (Qur’anic anthropology vs materialist reduction),
- keeps “crosswalks” to modern taxonomies optional and clearly marked as analytic overlays,
- prevents drift by requiring evidence anchors and controlled IDs.

---

## 1) Purpose and scope

### 1.1 Purpose
Produce a structured dataset of **Qur’anic spans** annotated across:
- five major contexts/axes (organic, situational, systemic, spatial, temporal),
- overlays (intention, periodicity, behavioral classification),
- normative textual signals (Qur’an‑only),
- optional juristic derivations (separate, fully sourced).

### 1.2 Intended uses
- **Scholarly**: traceable behavioral and thematic mapping of Qur’anic discourse.
- **Annotation infrastructure**: multi‑institution consistency with measurable IAA.
- **Computational research**: graph + retrieval + analytics without over‑interpretation.

### 1.3 Non‑goals
- Not a fatwa engine.
- Not “Qur’an‑only fiqh.”
- Not replacing tafsir; tafsir is metadata for stabilization and audit.

---

## 2) Epistemic foundations (Usul + Islamic psychology alignment)

### 2.1 Usul boundary: what we claim vs what we store
- **Textual layer**: what the Qur’an indicates in wording/structure.
- **Interpretive layer**: tafsir‑supported disambiguation and controlled indirect inference.
- **Juristic layer**: legal rulings and usul reasoning are optional and must be isolated.

Paper scope alignment note (important):
- The updated Bouzidani paper builds its matrix using **Qur’an + Sunnah + inherited Islamic sciences** as an integrated reference frame.
- This dataset remains **Qur’an‑grounded** and records non‑Qur’anic clarifications (e.g., ritual frequency prescriptions) only in explicitly separated fields (`frequency_prescription` and any optional “external support” blocks). This preserves auditability and prevents Qur’an‑only over‑claims.

### 2.2 Fitrah‑based moral insight and objective ethics
The dataset assumes:
- Humans have **capacity for being affected and affecting** (قابلية التأثر/التأثير),
- Behavior can be evaluated against an **objective moral frame** (موضوعية الأخلاق),
- The Qur’an provides a coherent, comprehensive discourse that can ground a matrix of contexts.

This is not a claim that every label is a ruling; it is a claim that the Qur’an contains **behavioral discourse** that can be structured with evidence discipline.

### 2.3 Three‑world model (sensory / psychic / spiritual)
For operational clarity in annotation:
- **Sensory world**: observable, embodied actions and perceptions.
- **Psychic/cognitive world**: thoughts, emotions, motivations, dispositions.
- **Spiritual world**: worship‑orientation, accountability, and intention‑linked valuation.

This model supports:
- why “internal vs external” is not merely visibility,
- why intention is a separate overlay,
- why “organ” mentions can carry cognitive/spiritual semantics (esp. قلب).

Paper‑anchored note:
- The updated paper cites (as conceptual support) a three‑world framing associated with Ibn Khaldun: sensory perception, cognitive/intellectual apprehension beyond sensory, and a higher spiritual/ghaybi domain (e.g., intentions, orientations, worship‑value). This document uses that framing as **an annotation scaffold**, not as a claim that every span encodes all three.

### 2.4 “Action (عمل)” framing and intention transformation (نية)
Bouzidani emphasizes that Qur’anic discourse often uses **عمل** (actions) rather than the modern term “behavior.” Operationally:
- Store **behavior concepts** as labels.
- Store **textual evaluation** separately.
- Permit an optional “action valuation” axis only when explicitly supported.

Crucially: **intention may elevate a routine act into worship‑value**, but this transformation is not “Qur’an‑only”; it may require Sunnah/fiqh framing. Therefore:
- The dataset stores intention as an overlay.
- Any frequency/ritual prescription derived from Sunnah is stored separately.

### 2.5 Method: inductive qualitative content analysis (paper-aligned)
The updated paper explicitly frames the approach as **Inductive Content Analysis** (often discussed as qualitative content analysis), operationalized here as:
- **Comprehension reading**: understand the texts under study in context.
- **Identify the “big picture”**: define the major contexts/axes that recur.
- **Extract precise meaning-bearing texts**: select spans that directly express the phenomenon.
- **Focus on qualitative points**: encode the recurring qualitative dimensions as controlled labels.
- **Re-compose a knowledge product**: synthesize the extracted categories into a stable matrix + coding manual.

This maps to our workflow as: span segmentation → controlled vocabularies → double-annotation → IAA → manual revision → scale.

---

## 3) Conceptual model

### 3.1 Unit of analysis: span
Primary unit is a **verse‑span**: a contiguous, token‑bounded segment within a single ayah.
- Multi‑ayah discourse is represented via linking (see `linked_spans[]`).

### 3.2 Three label families (do not conflate)
- **Behavior concepts** (`BEH_*`): actions, traits, dispositions, inner states.
- **Thematic constructs** (`THM_*`): frames/themes (accountability, testimony, reward/punishment).
- **Meta‑Qur’anic discourse constructs** (`META_*`, optional): revelation process, recitation mode, rhetoric.

### 3.3 Behavior form taxonomy (controlled)
`behavior.form` captures the structural form of the behavior label:
- `speech_act`
- `physical_act`
- `inner_state`
- `trait_disposition`
- `relational_act`
- `omission`
- `mixed`
- `unknown`

### 3.4 Quranic behavior classification overlay (from paper framing)
To align with the updated paper’s “غريزة/عمل صالح/عمل غير صالح + نية” framing without collapsing into juristic rulings, use an overlay axis:

- **Action class** (`AX_ACTION_CLASS`):
  - `ACT_INSTINCTIVE_OR_AUTOMATIC` (غريزي/لا إرادي)
  - `ACT_VOLITIONAL` (إرادي/مكتسب)
  - `ACT_UNKNOWN`

- **Action moral evaluation (textual)** (`AX_ACTION_TEXTUAL_EVAL`):
  - `EVAL_SALIH` | `EVAL_GHAYR_SALIH` | `EVAL_NEUTRAL` | `EVAL_NOT_APPLICABLE` | `EVAL_UNKNOWN`

Rules:
- Only assign `EVAL_SALIH/GHAYR_SALIH` when the span provides **explicit evaluative cues** (praise/blame, promise/threat, etc.) or when tafsir agreement is high and the evaluation is clearly part of the meaning.
- Do not turn this into juristic status (wajib/haram) unless using the juristic layer.
- If `ACT_INSTINCTIVE_OR_AUTOMATIC`, default `AX_ACTION_TEXTUAL_EVAL` to `EVAL_NOT_APPLICABLE` unless the text explicitly evaluates the act (rare).

### 3.5 Behavior dynamics (stimulus → internal arousal → response) (optional, paper-aligned)
The updated paper emphasizes a behavioral sequence: **stimulus → internal arousal/activation → response**, and identifies three practical pillars for behavior analysis: **stimulus (المثير)**, **goal/purpose (الغاية/الهدف)**, and **intention (النية)**.

To support this without forcing labels where the Qur’an does not specify them, this spec introduces optional assertion axes (often left `unknown`):
- `AX_STIMULUS_MODALITY`: `STM_SENSORY` | `STM_PSYCHIC` | `STM_SPIRITUAL` | `STM_UNKNOWN`
- `AX_GOAL_ORIENTATION`: `GOL_WORLDLY` | `GOL_OTHERWORLDLY` | `GOL_MIXED` | `GOL_UNKNOWN`
- `AX_INTENTION_PRESENCE`: `INT_PRESENT` | `INT_ABSENT` | `INT_UNKNOWN`

These axes are **not required** for the pilot unless your annotation team can justify them with clear anchors and agreement.

**Annotator guidance (to resolve “intention tension”):**
- `AX_INTENTION_PRESENCE` is ONLY for **textually foregrounded intention** (explicit “يريدون/أراد/قصد/ابتغاء/وجه الله/يراءون…”-type signals).
- Do **not** infer “the agent must have intended the act” from the mere existence of an act. Default to `INT_UNKNOWN`.
- The **quality** of intention (sincerity vs showing off vs worldly) is captured in the intention overlay (not by guessing `INT_PRESENT`).

> ⚠️ **WARNING FOR IMPLEMENTERS (research-grade axes):**
>
> These three axes (`AX_STIMULUS_MODALITY`, `AX_GOAL_ORIENTATION`, `AX_INTENTION_PRESENCE`) are **HARD** and will remain `unknown` in the vast majority of spans.
>
> **Recommended for v1 pilot:**
> - Do NOT include them in double-annotation unless:
>   - you have ≥20 calibrated examples per axis,
>   - and you measured IAA ≥ 0.60 for that axis in a micro‑pilot.
>
> **Default stance:** keep them as optional fields but exclude them from v1 “Gold” export profiles unless explicitly enabled.

---

## 4) The matrix (axes)

### 4.1 Organic axis (عضوي/بيولوجي)
Captures organ references when framed as:
- tool of action,
- tool of perception,
- witness/accountability,
- metaphor for cognition/faith/emotion.

Special rule (paper‑motivated): when `ORG_HEART`, annotate `organ_semantic_domains[]`.

Paper‑anchored descriptive statistics (useful for coverage audits, not as a label rule):
- The updated paper reports **62 human organs** referenced across **645 ayat** (as an aggregate count). This can be used as a *sanity target* when building organ vocabularies and coverage audits.

**Coverage audit target (paper‑sourced, non‑normative):**
- Use the 62/645 statistic as a sanity check, not a quota.
- Example heuristic: if a pilot samples ~200 ayat/spans and yields **very low** organ mentions (e.g., <15), investigate selection bias or extraction gaps.

### 4.2 Situational axis (موضعي/حالي)
Captures internal/external:
- **external (ظاهر)**: speech/bodily acts,
- **internal (باطن)**: belief, intention, emotion, cognition.

Optional domain tags (when supported): emotional / spiritual / cognitive / psychological / social.

### 4.3 Systemic axis (نسقي)
Multi‑label frame:
- `SYS_SELF` (النفس)
- `SYS_CREATION` (الخلق)
- `SYS_GOD` (الخالق)
- `SYS_COSMOS` (الكون)
- `SYS_LIFE` (الحياة)

Optional `primary_systemic` with tie‑break rules in the coding manual.

### 4.4 Spatial axis (مكاني/جغرافي)
- `LOC_HOME` | `LOC_WORK` | `LOC_PUBLIC` | `LOC_UNKNOWN`

### 4.5 Temporal axis (زماني)
- `TMP_MORNING` | `TMP_NOON` | `TMP_AFTERNOON` | `TMP_NIGHT` | `TMP_UNKNOWN`

---

## 5) Evidence policy (Usul‑aligned)

### 5.1 Evidence is per assertion, not per span
All labels are stored as **assertions** with their own evidence metadata.

### 5.1.1 Behavior concepts and construct lists are derived caches (schema integrity rule)
For schema integrity and auditability:
- `behavior.concepts[]`, `thematic_constructs[]`, and `meta_discourse_constructs[]` are OPTIONAL **derived caches** for convenience.
- If present, each item MUST have a corresponding assertion in `assertions[]` using:
  - `AX_BEHAVIOR_CONCEPT` → `BEH_*`
  - `AX_THEMATIC_CONSTRUCT` → `THM_*`
  - `AX_META_DISCOURSE_CONSTRUCT` → `META_*`
- Exports MUST be fail‑closed if there is a mismatch between caches and assertions.

### 5.2 Controlled IDs (anti‑drift)
All categorical values MUST be stored as controlled IDs:
- axes (`AX_*`)
- systemic (`SYS_*`)
- spatial (`LOC_*`)
- temporal (`TMP_*`)
- agents (`AGT_*`)
- intentions (`INT_*`)
- periodicity (`PER_*`)
- behavior/construct/meta (`BEH_*`, `THM_*`, `META_*`)

### 5.3 Support type
- `direct`: explicit wording/structure.
- `indirect`: implication, narrative inference, metaphor/metonymy, context reasoning.

### 5.4 Indication tags (starter set)
- `dalalah_mantuq`
- `dalalah_mafhum`
- `narrative_inference`
- `metaphor_metonymy`
- `sabab_nuzul_used`

### 5.5 Polarity and negation
Each assertion includes:
- `negated`: true|false
- `negation_type`: absolute | conditional | exceptionless_affirmation | unknown
- `polarity`: positive|negative|neutral|mixed|unknown

### 5.6 Evidence anchors: token coordinate semantics
- `span.token_start` **inclusive**
- `span.token_end` **exclusive**
- `evidence_anchor` uses the **same ayah‑level token index space**.

`evidence_anchor` SHOULD be a token range:

```json
{"evidence_anchor": {"token_start": 12, "token_end": 15}}
```

If truly no anchor exists (purely contextual inference), `evidence_anchor` may be `null` and must be justified.

### 5.7 Justification (required for indirect)
- `justification_code` (`JST_*`)
- `justification` (short text)

---

## 6) Tafsir and qira’at policies

### 6.1 Tafsir role
Tafsir is metadata to:
- disambiguate meaning,
- stabilize annotation,
- document interpretive variance,
- justify indirect inferences.

It does not convert indirect into direct.

### 6.2 Baseline tafsir set (example)
- Ibn Kathir
- al‑Tabari
- al‑Sa‘di
- al‑Qurtubi

### 6.3 Tafsir agreement levels (operational criteria)
| Level | Criteria |
|---|---|
| `high` | All consulted sources agree on the meaning affecting the current label. |
| `mixed` | Agree on primary meaning; differ on secondary implications not changing label. |
| `disputed` | Differences would lead to different labels or force `unknown`. |

### 6.4 Qira’at policy (v1 default)
- Default text: Hafs ‘an ‘Asim (standard Uthmani).
- Qira’at variants are **out of scope for v1** unless explicitly flagged.
- If a variant plausibly changes a behavioral label, set `qiraat_note.affects_label=true` and record variants.

---

## 7) Normative / evaluative layer

### 7.1 Qur’an‑only textual layer
Store **how the Qur’an speaks**, not juristic rulings:
- `speech_mode`: command | prohibition | informative | narrative | parable | unknown
- `evaluation`: praise | blame | warning | promise | neutral | mixed | unknown
- `quran_deontic_signal`: amr | nahy | targhib | tarhib | khabar

### 7.2 Deontic signal classification rules (coding‑manual minimum)
| Grammatical form | Default signal | Notes |
|---|---|---|
| Explicit imperative (افعل) | `amr` | If quoted within narrative, keep `amr`, set `speech_mode=narrative`, explain in note. |
| Explicit prohibition (لا تفعل) | `nahy` | If exception (لا…إلا), keep `nahy`, document exception. |
| Descriptive praise + reward | `targhib` | Require explicit reward/praise cues. |
| Descriptive blame + threat | `tarhib` | Require explicit threat/punishment cues. |
| Pure report/narrative | `khabar` | Use `evaluation` to capture praise/blame without forcing targhib/tarhib. |

### 7.3 Juristic layer (optional)
If enabled, store separately with provenance:
- `juristic_deontic_status`: wajib|mustahabb|mubah|makruh|haram
- `madhhab`, `usul_rules_invoked[]`, `sources[]`, `disputed`

### 7.4 Abrogation (optional)
Default: `abrogation.status="unknown"` unless assessed and sourced.

---

## 8) Agent modeling

### 8.1 Agent types (controlled)
- AGT_BELIEVER
- AGT_HYPOCRITE
- AGT_DISBELIEVER
- AGT_HUMAN_GENERAL
- AGT_PROPHET_MESSENGER
- AGT_ANGEL
- AGT_JINN
- AGT_ANIMAL
- AGT_COLLECTIVE
- AGT_UNKNOWN

### 8.2 Multi-agent interaction (recommended)
For collective/multi-agent spans:
- `agent.composition`: AGT_*
- `agent.multi_agent_interaction`: true|false
- `agent.interaction_type`: conflict | cooperation | dialogue | unknown

---

## 9) Data model (span record)

### 9.1 Span record (canonical JSON)

```json
{
  "id": "QBM_000001",
  "quran_text_version": "uthmani_hafs_v1",
  "tokenization_id": "tok_v1",

  "reference": {"surah": 24, "ayah": 24},

  "span": {
    "token_start": 0,
    "token_end": 35,
    "raw_text_ar": "...",
    "boundary_confidence": "certain",
    "alternative_boundaries": []
  },

  "linked_spans": [
    {"span_id": "QBM_000002", "relation": "continuation"}
  ],

  "intra_textual_references": [
    {"target_span_id": "QBM_XXXXX", "relation_type": "contrast", "note": "Short rationale"}
  ],

  "behavior": {"concepts": [], "form": "speech_act"},
  "thematic_constructs": ["THM_ACCOUNTABILITY"],
  "meta_discourse_constructs": [],

  "agent": {
    "type": "AGT_HUMAN_GENERAL",
    "group": "unknown",
    "explicit": false,
    "support_type": "indirect",
    "evidence_anchor": null,
    "note": "context inferred"
  },

  "provenance": {
    "extraction_method": "keyword_search",
    "search_query": "",
    "candidate_set_size": 0,
    "selection_rationale": "",
    "excluded_alternatives": [],
    "timestamp": ""
  },

  "tafsir": {
    "sources_used": ["IbnKathir"],
    "agreement_level": "high",
    "consultation_trigger": "baseline",
    "notes": "short summary"
  },

  "assertions": [
    {
      "assertion_id": "QBM_000001_A001",
      "axis": "AX_ORGAN",
      "value": "ORG_TONGUE",
      "organ_role": "tool",
      "organ_semantic_domains": [],
      "primary_organ_semantic_domain": "unknown",
      "support_type": "direct",
      "indication_tags": ["dalalah_mantuq"],
      "evidence_anchor": {"token_start": 5, "token_end": 6},
      "justification_code": "JST_EXPLICIT_MENTION",
      "justification": "Explicit organ mention.",
      "confidence": 0.95,
      "polarity": "neutral",
      "negated": false,
      "negation_type": "unknown",
      "ess": {"directness": 1.0, "clarity": 0.9, "specificity": 0.9, "tafsir_agreement": 1.0, "final": 0.81}
    },
    {
      "assertion_id": "QBM_000001_A002",
      "axis": "AX_THEMATIC_CONSTRUCT",
      "value": "THM_ACCOUNTABILITY",
      "support_type": "direct",
      "indication_tags": ["dalalah_mantuq"],
      "evidence_anchor": null,
      "justification_code": "JST_NARRATIVE_CONTEXT",
      "justification": "Span-level accountability framing.",
      "confidence": 0.80,
      "polarity": "neutral",
      "negated": false,
      "negation_type": "unknown",
      "ess": {"directness": 0.7, "clarity": 0.8, "specificity": 0.6, "tafsir_agreement": 1.0, "final": 0.34}
    }
  ],

  "periodicity": {
    "category": "PER_UNKNOWN",
    "grammatical_indicator": "GRM_NONE",
    "support_type": "unknown",
    "indication_tags": [],
    "evidence_anchor": null,
    "justification_code": null,
    "justification": "",
    "confidence": null,
    "ess": null
  },

  "frequency_prescription": {"source": "unknown", "allow": false, "note": ""},

  "normative_textual": {
    "speech_mode": "informative",
    "evaluation": "neutral",
    "quran_deontic_signal": "khabar",
    "support_type": "direct",
    "note": ""
  },

  "normative_juristic": null,

  "agent_intention_interaction": null,

  "abrogation": {"status": "unknown", "related_spans": [], "sources": [], "note": ""},

  "qiraat_note": {"variant_exists": false, "affects_label": false, "variants": [], "policy": "out_of_scope_v1"},

  "review": {
    "status": "draft",
    "annotator_id": "A1",
    "reviewer_id": "",
    "created_at": "",
    "manual_version": "v1.0"
  }
}
```

### 9.2 Periodicity + frequency (paper alignment)
The updated paper distinguishes routine/instinctive patterns and emphasizes intention.

Rules:
- Periodicity is only labeled when supported by explicit grammar/lexemes; otherwise `PER_UNKNOWN`.
- If a frequency is known from Sunnah/ijma (e.g., prayer counts), store in `frequency_prescription` and keep Qur’anic periodicity unknown unless the Qur’an encodes it.

Paper‑aligned periodicity interpretation notes:
- When periodicity is applicable, map to: `PER_AUTOMATIC` (تلقائي/غريزي/لا إرادي), `PER_HABIT` (عادة متكررة), and optionally `PER_ROUTINE_DAILY` as a **project operationalization** for highly regular daily practices (not a Qur’anic grammar claim unless explicitly encoded).
- Application guidance (not textual labeling): the paper references research indicating habit formation ranges widely (e.g., **~3 weeks** up to **~9 months**) and also suggests applying the matrix over an application window (e.g., **~3–36 weeks** with a minimum not less than 3 weeks). Do **not** backfill Qur’anic periodicity labels from these timelines.

---

## 10) Measurement, scoring, and reliability

### 10.1 Measurement instruments (behavioral science) mapped to Qur’anic spans
The updated paper cites common measurement dimensions (frequency, duration, topography, intensity, latency, location). For this dataset:
- **Frequency** → `periodicity` (only if text supports) and/or `frequency_prescription` (external source).
- **Duration / persistence** → may be represented via grammatical indicators (e.g., continuity forms) or thematic constructs; do not invent.
- **Topography (form)** → `behavior.form`.
- **Intensity/strength** → may be approximated by evaluation/ESS, but do not claim quantitative intensity unless text indicates.
- **Latency** → generally out of scope unless text encodes “immediate/afterward” explicitly.
- **Location** → `spatial` axis.

Paper mapping to common Arabic measurement terms (for annotator training, not as schema fields):
- Frequency rate: **معدل تكرار السلوك**
- Duration: **مدة حدوث السلوك**
- Topography: **طبوغرافيا السلوك**
- Intensity: **شدة/قوة السلوك**
- Latency: **كمون السلوك**
- Location: **مكان حدوثه**

### 10.2 Evidence Strength Score (ESS) per assertion
Store ESS components per assertion:
- directness
- clarity
- specificity
- tafsir_agreement

### 10.3 Confidence calibration (starter caps)
- If `support_type="indirect"` and `ess.clarity < 0.7` then `confidence <= 0.75`.
- If `metaphor_metonymy` then `confidence <= 0.80` unless tafsir resolves it with `high` agreement.
- If tafsir agreement is `disputed` then `confidence <= 0.70` or set label to `unknown`.

### 10.4 IAA targets (pilot)
| Axis | Metric | Pilot target |
|---|---|---:|
| Organ (direct) | Krippendorff’s α | ≥ 0.80 |
| Behavior concepts | Jaccard (multi‑label) | ≥ 0.70 |
| Situational | Cohen’s κ | ≥ 0.75 |
| Systemic | F1 (macro) | ≥ 0.60 |
| Periodicity | Krippendorff’s α | ≥ 0.50 |

---

## 10.5 Optional crosswalks to modern behavioral typologies (paper context)
The updated paper reviews modern typologies (e.g., instinctive vs acquired, respondent vs operant, overt vs covert) and warns against importing them uncritically.

If you decide to use a crosswalk for research analytics, treat it as **optional** and never as a replacement for Qur’anic evidence discipline:
- **Instinctive vs acquired** → approximate via `AX_ACTION_CLASS` (`ACT_INSTINCTIVE_OR_AUTOMATIC` vs `ACT_VOLITIONAL`), but only when text supports it.
- **Overt vs covert (ظاهر/باطن)** → approximate via `AX_SITUATIONAL` (external/internal) and domain tags.
- **Respondent vs operant** → usually not directly encoded in Qur’anic text; keep out of the core schema unless a specialized research track is created.

## 10.6 Application-only guidance: “life-trajectory” positioning (paper context)
The updated paper frames the matrix as a **diagnostic / self‑development instrument** oriented toward *الحياة الطيبة* (good life) and behavior refinement. If you build an application layer, you may model user-facing states separately from Qur’anic annotation, e.g.:

- **Development / promotion** (إنماء/إنشاء)
- **Prevention** (وقاية)
- **Therapeutic** (علاجي)

And optionally map to Qur’anic “servants inheriting the Book” categories as **user‑state metaphors** (not labels assigned to Qur’anic spans):

- ظالم لنفسه
- مقتصد
- سابق بالخيرات

These are **application-layer constructs** and MUST NOT be stored as Qur’anic labels unless a separate, sourced interpretive project is defined.

## 11) Optional graph representation (relation triples)

```json
{
  "relation_triples": [
    {
      "subject": {"type": "agent", "value": "AGT_BELIEVER"},
      "predicate": {"id": "PRED_QWL", "value_ar": "يقول", "root": "ق-و-ل"},
      "object": {"type": "systemic", "value": "SYS_CREATION"},
      "modifiers": [{"type": "manner", "value": "kindly"}],
      "support_type": "direct",
      "evidence_anchor": {"token_start": 10, "token_end": 13},
      "confidence": 0.90,
      "negated": false
    }
  ]
}
```

---

## 12) Governance and export policy

### 12.1 Versioning
Version:
- vocabularies
- tokenization
- manual
- exports

### 12.2 Review states
- draft
- disputed
- approved

### 12.3 Fail‑closed export
Disputed/draft records are excluded from Gold by default.

### 12.4 Export tiers
- **Gold**: reviewer‑approved only
- **Silver**: meets ESS threshold + no dispute + audit passed (recommended `ess.final >= 0.75` on critical assertions)
- **Research**: includes flagged/disputed items

Rationale (Silver threshold, v1 default):
- `0.75` is intended to require either (a) high directness + high clarity, or (b) a strongly justified indirect assertion with high tafsir agreement.
- Threshold MUST be recalibrated after pilot calibration and IAA review.

---

## 13) Implementation workflow (pilot-ready)

### Phase 0: Setup
- freeze text version + tokenization
- freeze v1 vocabularies
- define tafsir baseline

### Phase 1: Coding manual v1
Must include:
- span segmentation rules
- deontic decision rules
- heart semantic domains playbook
- periodicity indicator rules
- negation type examples

### Phase 2: Micro‑pilot
- 50–100 spans
- double/triple annotate
- calibrate confidence
- revise manual

### Phase 3: Pilot
- 200–500 spans
- measure IAA against targets
- publish a small Gold slice (optional)

---

## Appendix A) Controlled vocabularies (starter)

### A.1 Axes
- AX_ORGAN
- AX_SITUATIONAL
- AX_SYSTEMIC
- AX_SPATIAL
- AX_TEMPORAL
- AX_BEHAVIOR_CONCEPT
- AX_THEMATIC_CONSTRUCT
- AX_META_DISCOURSE_CONSTRUCT
- AX_ACTION_CLASS
- AX_ACTION_TEXTUAL_EVAL
- AX_STIMULUS_MODALITY
- AX_GOAL_ORIENTATION
- AX_INTENTION_PRESENCE

### A.2 Systemic
- SYS_SELF
- SYS_CREATION
- SYS_GOD
- SYS_COSMOS
- SYS_LIFE

### A.3 Spatial
- LOC_HOME
- LOC_WORK
- LOC_PUBLIC
- LOC_UNKNOWN

### A.4 Temporal
- TMP_MORNING
- TMP_NOON
- TMP_AFTERNOON
- TMP_NIGHT
- TMP_UNKNOWN

### A.5 Organ labels
- ORG_HEART
- ORG_TONGUE
- ORG_EYE
- ORG_EAR
- ORG_HAND
- ORG_FOOT
- ORG_SKIN

### A.6 Organ semantic domains (for ORG_HEART)
- physiological
- cognitive
- spiritual
- emotional

### A.7 Organ roles
- tool
- perception
- accountability_witness
- metaphor
- unknown

### A.8 Justification codes (starter)
- JST_EXPLICIT_MENTION
- JST_GRAMMATICAL_FORM
- JST_TAFSIR_CONSENSUS
- JST_NARRATIVE_CONTEXT
- JST_PRONOUN_RESOLVED
- JST_CONTEXT_ONLY
- JST_METAPHOR_RESOLVED

### A.9 Meta‑discourse constructs (optional starter)
- META_REVELATION_PROCESS
- META_RECITATION_MODE
- META_RHETORICAL_DEVICE
- META_INTERTEXTUAL_REFERENCE

### A.10 Agent types
- AGT_BELIEVER
- AGT_HYPOCRITE
- AGT_DISBELIEVER
- AGT_HUMAN_GENERAL
- AGT_PROPHET_MESSENGER
- AGT_ANGEL
- AGT_JINN
- AGT_ANIMAL
- AGT_COLLECTIVE
- AGT_UNKNOWN

### A.11 Periodicity
- PER_HABIT
- PER_ROUTINE_DAILY  
  *Project operationalization for highly regular daily practices (e.g., explicit “كل يوم” / daily-regularity cues). Not a Qur’anic grammar claim unless explicitly encoded.*
- PER_AUTOMATIC
- PER_UNKNOWN

### A.14 Behavior concept taxonomy (starter for pilot)
A comprehensive hierarchical taxonomy (speech/financial/emotional/spiritual/social/cognitive) is part of the coding manual. For pilot, start with a minimal set and expand carefully:

- BEH_SPEECH_TRUTHFULNESS
- BEH_SPEECH_LYING
- BEH_EMO_GRATITUDE
- BEH_SPI_SINCERITY
- BEH_SPI_SHOWING_OFF
- BEH_SOC_OPPRESSION

### A.11.1 Action class (paper-aligned)
- ACT_INSTINCTIVE_OR_AUTOMATIC
- ACT_VOLITIONAL
- ACT_UNKNOWN

### A.11.2 Action textual evaluation (paper-aligned)
- EVAL_SALIH
- EVAL_GHAYR_SALIH
- EVAL_NEUTRAL
- EVAL_NOT_APPLICABLE
- EVAL_UNKNOWN

### A.11.3 Stimulus modality (optional)
- STM_SENSORY
- STM_PSYCHIC
- STM_SPIRITUAL
- STM_UNKNOWN

### A.11.4 Goal orientation (optional)
- GOL_WORLDLY
- GOL_OTHERWORLDLY
- GOL_MIXED
- GOL_UNKNOWN

### A.11.5 Intention presence (optional)
- INT_PRESENT
- INT_ABSENT
- INT_UNKNOWN

### A.12 Grammatical indicators
- GRM_KANA_IMPERFECT
- GRM_NOMINAL_SENTENCE_STABILITY
- GRM_EXPLICIT_FREQUENCY_TERM
- GRM_CONDITIONAL_CONTINGENT
- GRM_REPEATED_IMPERATIVE
- GRM_CONTEXT_ONLY
- GRM_NONE

### A.13 Predicate vocabulary (starter)
- PRED_QWL (ق-و-ل)
- PRED_MSHY (م-ش-ي)
- PRED_NFQ (ن-ف-ق)

---

## Appendix B) Worked example (schema-consistent)

```json
{
  "id": "QBM_EXAMPLE_0001",
  "quran_text_version": "uthmani_hafs_v1",
  "tokenization_id": "tok_v1",
  "reference": {"surah": 24, "ayah": 24},
  "span": {"token_start": 0, "token_end": 40, "raw_text_ar": "...", "boundary_confidence": "certain", "alternative_boundaries": []},

  "behavior": {"concepts": [], "form": "mixed"},
  "thematic_constructs": ["THM_ACCOUNTABILITY", "THM_TESTIMONY"],
  "meta_discourse_constructs": [],

  "agent": {"type": "AGT_HUMAN_GENERAL", "group": "unknown", "explicit": false, "support_type": "indirect", "evidence_anchor": null, "note": "context inferred"},

  "assertions": [
    {"assertion_id": "QBM_EXAMPLE_0001_A001", "axis": "AX_ORGAN", "value": "ORG_TONGUE", "organ_role": "accountability_witness", "organ_semantic_domains": [], "primary_organ_semantic_domain": "unknown", "support_type": "direct", "indication_tags": ["dalalah_mantuq"], "evidence_anchor": {"token_start": 5, "token_end": 6}, "justification_code": "JST_EXPLICIT_MENTION", "justification": "Explicit organ mention.", "confidence": 0.95, "polarity": "neutral", "negated": false, "negation_type": "unknown", "ess": {"directness": 1.0, "clarity": 0.9, "specificity": 0.9, "tafsir_agreement": 1.0, "final": 0.81}},
    {"assertion_id": "QBM_EXAMPLE_0001_A002", "axis": "AX_SYSTEMIC", "value": "SYS_SELF", "support_type": "indirect", "indication_tags": ["narrative_inference"], "evidence_anchor": null, "justification_code": "JST_NARRATIVE_CONTEXT", "justification": "Organs testifying implies self-accountability frame.", "confidence": 0.65, "polarity": "neutral", "negated": false, "negation_type": "unknown", "ess": {"directness": 0.4, "clarity": 0.7, "specificity": 0.6, "tafsir_agreement": 1.0, "final": 0.17}},
    {"assertion_id": "QBM_EXAMPLE_0001_A003", "axis": "AX_THEMATIC_CONSTRUCT", "value": "THM_ACCOUNTABILITY", "support_type": "direct", "indication_tags": ["dalalah_mantuq"], "evidence_anchor": null, "justification_code": "JST_NARRATIVE_CONTEXT", "justification": "Span-level accountability framing.", "confidence": 0.80, "polarity": "neutral", "negated": false, "negation_type": "unknown", "ess": {"directness": 0.7, "clarity": 0.8, "specificity": 0.6, "tafsir_agreement": 1.0, "final": 0.34}}
  ],

  "periodicity": {"category": "PER_UNKNOWN", "grammatical_indicator": "GRM_NONE", "support_type": "unknown", "indication_tags": [], "evidence_anchor": null, "justification_code": null, "justification": "", "confidence": null, "ess": null},

  "normative_textual": {"speech_mode": "informative", "evaluation": "neutral", "quran_deontic_signal": "khabar", "support_type": "direct", "note": ""},
  "normative_juristic": null,

  "abrogation": {"status": "unknown", "related_spans": [], "sources": [], "note": ""},
  "qiraat_note": {"variant_exists": false, "affects_label": false, "variants": [], "policy": "out_of_scope_v1"},

  "review": {"status": "approved", "annotator_id": "A1", "reviewer_id": "R1", "created_at": "2025-12-20", "manual_version": "v1.0"}
}
```

### Appendix B.1) Worked example: ACT_INSTINCTIVE_OR_AUTOMATIC + EVAL_NOT_APPLICABLE (paper-aligned)
This is a **rare** case and is provided only as a calibrated template for annotators.

```json
{
  "id": "QBM_EXAMPLE_INSTINCTIVE_0001",
  "quran_text_version": "uthmani_hafs_v1",
  "tokenization_id": "tok_v1",
  "reference": {"surah": 30, "ayah": 54},
  "span": {"token_start": 0, "token_end": 12, "raw_text_ar": "اللَّهُ الَّذِي خَلَقَكُم مِّن ضَعْفٍ...", "boundary_confidence": "certain", "alternative_boundaries": []},

  "behavior": {"concepts": [], "form": "inner_state"},
  "thematic_constructs": [],
  "meta_discourse_constructs": [],

  "agent": {"type": "AGT_HUMAN_GENERAL", "group": "unknown", "explicit": false, "support_type": "direct", "evidence_anchor": {"token_start": 0, "token_end": 1}, "note": "General خطاب/description."},

  "assertions": [
    {
      "assertion_id": "QBM_EXAMPLE_INSTINCTIVE_0001_A001",
      "axis": "AX_ACTION_CLASS",
      "value": "ACT_INSTINCTIVE_OR_AUTOMATIC",
      "support_type": "indirect",
      "indication_tags": ["dalalah_mafhum"],
      "evidence_anchor": {"token_start": 5, "token_end": 7},
      "justification_code": "JST_TAFSIR_CONSENSUS",
      "justification": "Weakness/infancy-type states are involuntary biological conditions; use only with high tafsir agreement for the meaning.",
      "confidence": 0.70,
      "polarity": "neutral",
      "negated": false,
      "negation_type": "unknown",
      "ess": {"directness": 0.4, "clarity": 0.7, "specificity": 0.6, "tafsir_agreement": 1.0, "final": 0.17}
    },
    {
      "assertion_id": "QBM_EXAMPLE_INSTINCTIVE_0001_A002",
      "axis": "AX_ACTION_TEXTUAL_EVAL",
      "value": "EVAL_NOT_APPLICABLE",
      "support_type": "direct",
      "indication_tags": ["dalalah_mantuq"],
      "evidence_anchor": null,
      "justification_code": "JST_GRAMMATICAL_FORM",
      "justification": "Automatic/involuntary conditions are not evaluated as صالح/غير صالح unless the text explicitly does so.",
      "confidence": 0.90,
      "polarity": "neutral",
      "negated": false,
      "negation_type": "unknown",
      "ess": {"directness": 0.8, "clarity": 0.9, "specificity": 0.7, "tafsir_agreement": 1.0, "final": 0.50}
    }
  ],

  "periodicity": {"category": "PER_UNKNOWN", "grammatical_indicator": "GRM_NONE", "support_type": "unknown", "indication_tags": [], "evidence_anchor": null, "justification_code": null, "justification": "", "confidence": null, "ess": null},
  "normative_textual": {"speech_mode": "informative", "evaluation": "neutral", "quran_deontic_signal": "khabar", "support_type": "direct", "note": "Descriptive; not a prescriptive حكم."},
  "normative_juristic": null,
  "abrogation": {"status": "unknown", "related_spans": [], "sources": [], "note": ""},
  "qiraat_note": {"variant_exists": false, "affects_label": false, "variants": [], "policy": "out_of_scope_v1"},
  "review": {"status": "draft", "annotator_id": "A1", "reviewer_id": "", "created_at": "2025-12-20", "manual_version": "v1.0"}
}
```

### Appendix B.2) Worked example: “لا…ولكن/لكنهم” negation–contrast (schema pattern)
This is a **structure template** showing how to encode the pattern using two assertions and an intra-textual contrast link.

```json
{
  "id": "QBM_EXAMPLE_NEGATION_CONTRAST_0001",
  "quran_text_version": "uthmani_hafs_v1",
  "tokenization_id": "tok_v1",
  "reference": {"surah": 0, "ayah": 0},
  "span": {"token_start": 0, "token_end": 20, "raw_text_ar": "لا X ... ولكن Y ...", "boundary_confidence": "negotiated", "alternative_boundaries": []},

  "behavior": {"concepts": [], "form": "mixed"},
  "thematic_constructs": [],
  "meta_discourse_constructs": [],

  "assertions": [
    {
      "assertion_id": "QBM_EXAMPLE_NEGATION_CONTRAST_0001_A001",
      "axis": "AX_BEHAVIOR_CONCEPT",
      "value": "BEH_COG_KNOWLEDGE",
      "support_type": "direct",
      "indication_tags": ["dalalah_mantuq"],
      "evidence_anchor": {"token_start": 0, "token_end": 4},
      "justification_code": "JST_GRAMMATICAL_FORM",
      "justification": "Negated claim X is explicitly marked by لا/ليس/ما.",
      "confidence": 0.90,
      "polarity": "neutral",
      "negated": true,
      "negation_type": "absolute",
      "ess": {"directness": 1.0, "clarity": 0.9, "specificity": 0.6, "tafsir_agreement": 0.0, "final": 0.54}
    },
    {
      "assertion_id": "QBM_EXAMPLE_NEGATION_CONTRAST_0001_A002",
      "axis": "AX_BEHAVIOR_CONCEPT",
      "value": "BEH_COG_SUPERFICIAL_KNOWING",
      "support_type": "direct",
      "indication_tags": ["dalalah_mantuq"],
      "evidence_anchor": {"token_start": 8, "token_end": 12},
      "justification_code": "JST_GRAMMATICAL_FORM",
      "justification": "Affirmed corrective clause introduced by لكن/ولكن.",
      "confidence": 0.90,
      "polarity": "neutral",
      "negated": false,
      "negation_type": "unknown",
      "ess": {"directness": 1.0, "clarity": 0.9, "specificity": 0.6, "tafsir_agreement": 0.0, "final": 0.54}
    }
  ],

  "intra_textual_references": [
    {"target_span_id": "QBM_EXAMPLE_NEGATION_CONTRAST_0001", "relation_type": "contrast", "note": "Within-span X vs Y contrast; assertions A001↔A002."}
  ],

  "review": {"status": "draft", "annotator_id": "A1", "reviewer_id": "", "created_at": "2025-12-20", "manual_version": "v1.0"}
}
```

---

## Appendix C) Analytic-only crosswalks to modern behavior typologies (optional)
This appendix is **for annotator training and research notes only**. It must not override Qur’anic evidence discipline, and it must not introduce “hidden premises” into the dataset.

Rules:
- Do not assign crosswalk labels unless the Qur’anic text (or high-agreement tafsir for meaning) supports the mapping.
- Prefer leaving these as **unknown** rather than guessing.
- If you need them for a research track, implement them as separate *analytic overlays* outside the core export tiers.

| Modern typology | Quranic approximation (if any) | Rule |
|---|---|---|
| Overt vs covert | `AX_SITUATIONAL` external/internal | Only when the text frames the act as outward (قول/فعل) vs inward (اعتقاد/نية/خاطر). |
| Instinctive vs acquired | `AX_ACTION_CLASS` | Use only when the text clearly indicates involuntary/automatic vs deliberate volition. |
| Respondent vs operant | Usually not encoded | Keep out of the core schema; treat as research-only and default to unknown. |

---

## Appendix D) Negation–contrast patterns (“لا…ولكن / لكنهم …”) (coding-manual addendum)
The updated paper highlights dense Qur’anic discourse patterns that contrast **negated cognition/claim** with **affirmed behavior/reality** (often using “لكن/ولكن/لكنهم”). These are high-value for consistent `negated`, `negation_type`, and `polarity` annotation.

### D.1 Pattern family
Common surface forms include:
- **لا X ولكن Y** / **ليس X ولكن Y** / **ما X ولكن Y**
- **X … لكنهم Y** (often: stated claim vs corrective reality)

### D.2 Operational rule (single-span contrast)
When the contrast occurs within the same span:
- Create **two assertions**:
  1) **Negated proposition** (X): set `negated=true` and choose `negation_type`:
     - `absolute` for direct negation/prohibition patterns.
     - `conditional` when the negation is structured with conditions/exceptions.
  2) **Affirmed proposition** (Y): set `negated=false`.

- Prefer `AX_BEHAVIOR_CONCEPT` when X/Y are behaviors; otherwise use `AX_SITUATIONAL`, `AX_SYSTEMIC`, or other relevant axes.
- Use `justification_code=JST_GRAMMATICAL_FORM` (or `JST_EXPLICIT_MENTION`) and anchor tokens to the negation and contrast clause when possible.
- Optional: in `justification`, reference the paired `assertion_id` (e.g., “Contrasts with QBM_…_A00X”).

### D.3 Operational rule (cross-span contrast)
When the contrast is across spans/verses:
- Use `intra_textual_references[]` with `relation_type="contrast"`.
- Annotate the contrasted claims as assertions in each span.

### D.4 Polarity guidance
- If X is a condemned cognitive failure (e.g., “لا يعلمون”), set X’s `polarity="negative"` only if the wider context marks blame; otherwise keep `polarity="neutral"` and let `normative_textual.evaluation` carry the evaluative tone.
- Y may be positive/neutral depending on context.

### D.5 Examples (pattern examples; do not pre-assign labels)
- “لا يعلمون … ولكن … يعلمون ظاهراً من الحياة الدنيا …”
- “لا يضرهم … ولكن … يتعلمون …”
- “يصلون … ولكن … هم عن صلاتهم ساهون …”
- “يؤمنون … ولكن … بالجبت والطاغوت …”
