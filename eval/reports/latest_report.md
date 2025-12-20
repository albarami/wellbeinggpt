## Evaluation Report (latest)

- **Date (UTC)**: 2025-12-16
- **Repo commit**: `9cab9fa0e7f72bcc786d9ed0d270a33ac76a9ca0`
- **Seed**: 1337
- **Prompts version**: v1

### Datasets (versions + run ids)

| Dataset | Rows | SHA256 | Run ID |
|---|---:|---|---|
| Gold QA | 200 | `f14d813f4519daac99f581eed4b6a14ffa9776ed0d3ce258a2f312d3e343292a` | `wellbeing__v1__f14d813f4519__seed1337__pv1` |
| Cross-pillar | 80 | `5104718c1d552a7576d43eb99ea47f4ab712cc362b7ea3075db016890ebe1a3a` | `wellbeing__v1__5104718c1d55__seed1337__pv1` |
| Deep Cross-pillar | 60 | `cc73f490514e07f09db227018c3fc0a379ca286ec65831ff32344e744e134804` | `wellbeing__v1__cc73f490514e__seed1337__pv1` |
| Negative/OOS | 60 | `464e083154d8257d58bf224e260f0061c5e4bfcafdf085f0f44d248b64492935` | `wellbeing__v1__464e083154d8__seed1337__pv1` |
| Mixed (30 in-scope + 30 OOS) | 60 | `709779785ce9942c2ba64c909c635cc6fe79093e78ea2335265d7ffa718c64ad` | `wellbeing__v1__709779785ce9__seed1337__pv1` |
| Injection | 40 | `95716b50f424900576fc46358d0b4f295f7b7cb69c9a79b70190497c8186c949` | `wellbeing__v1__95716b50f424__seed1337__pv1` |
| Stakeholder acceptance | 5 | `690753d4d3fb6a53961ed91a63f1e0d682e6c6eba15746b859d6cb560aa4892f` | `wellbeing__v1__690753d4d3fb__seed1337__pv1` |

### Hard gates status

- **Citation validity**: PASS (0 validity errors)
- **Claim policy audit (Gold QA)**: PASS (0 violations)
- **Injection override rate**: PASS (0.00)
- **Injection false_answer_rate**: PASS (0.00)
- **Injection citation validity**: PASS (0 validity errors)
- **Injection unsupported_claim_rate (must_cite)**: PASS (0.0000)
- **Zero hallucination claim allowed?**: **YES**

### Key metrics (FULL_SYSTEM)

#### Gold QA
- **unsupported_claim_rate (must_cite only)**: 0.0000
- **policy_audit_violation_rate**: 0.0000
- **rubric_average_score (/10)**: 6.72
- **median_sentence_count_post_prune**: 23.0
- **median_claim_count_post_prune**: 23.0
- **median_must_cite_claims_post_prune**: 17.0
- **contract_pass_rate**: 0.30
- **section_nonempty_rate**: 0.47
- **required_entities_coverage_rate**: 0.82
- **graph_required_satisfaction_rate**: 1.00

#### Negative/OOS
- **abstention_precision**: 1.00
- **abstention_recall**: 1.00
- **false_answer_rate**: 0.00
- **false_abstention_rate**: 0.00

**Confusion matrix (expect_abstain vs abstained)**
- TP_abstain: 60
- FN_answered: 0
- FP_abstained: 0
- TN_answered: 0

#### Mixed (30 in-scope + 30 OOS)
- **abstention_precision**: 1.00
- **abstention_recall**: 1.00
- **false_answer_rate**: 0.00
- **false_abstention_rate**: 0.00
- **contract_pass_rate**: 0.40
- **section_nonempty_rate**: 0.52
- **required_entities_coverage_rate**: 0.88
- **graph_required_satisfaction_rate**: 1.00

**Confusion matrix (expect_abstain vs abstained)**
- TP_abstain: 30
- FN_answered: 0
- FP_abstained: 0
- TN_answered: 30

#### Cross-pillar
- **path_valid_rate**: 1.00
- **cross_pillar_hit_rate**: 1.00
- **explanation_grounded_rate (justification-bound)**: 1.00
- **contract_pass_rate**: 0.00
- **section_nonempty_rate**: 1.00
- **required_entities_coverage_rate**: 1.00
- **graph_required_satisfaction_rate**: 0.00

#### Deep Cross-pillar
- **rubric_average_score (/10)**: 10.00
- **median_sentence_count_post_prune**: 23.0
- **median_claim_count_post_prune**: 23.0
- **median_must_cite_claims_post_prune**: 17.0
- **contract_pass_rate**: 0.00
- **section_nonempty_rate**: 0.83
- **required_entities_coverage_rate**: 1.00
- **graph_required_satisfaction_rate**: 0.00

#### Injection
- **injection_override_rate**: 0.00
- **false_answer_rate**: 0.00
- **citation_validity_errors**: 0
- **unsupported_claim_rate (must_cite only)**: 0.0000
- **contract_pass_rate**: N/A

#### Stakeholder acceptance
- **status**: (not run)

### Gold QA exemplars

- **Unsupported MUST_CITE failures**: none ✅

#### Gold QA borderline depth (lowest rubric rows, first 20)
- id=`gold-0132` score=4: ما الفرق بين التخلي وبين مفهوم قريب منه داخل الإطار؟ اذكر التعريف مع الدليل.
- id=`gold-0133` score=4: قدّم مثالًا تطبيقيًا لكيفية ممارسة التخلي في موقف واقعي، مع ربطه بتعريفه في الإطار.
- id=`gold-0174` score=4: ما الفرق بين التخلي وبين مفهوم قريب منه داخل الإطار؟ اذكر التعريف مع الدليل.
- id=`gold-0175` score=4: قدّم مثالًا تطبيقيًا لكيفية ممارسة التخلي في موقف واقعي، مع ربطه بتعريفه في الإطار.
- id=`gold-0001` score=6: عرّف الحياة الروحية كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
- id=`gold-0002` score=6: قدّم مثالًا تطبيقيًا لكيفية ممارسة الحياة الروحية في موقف واقعي، مع ربطه بتعريفه في الإطار.
- id=`gold-0005` score=6: عرّف الحياة الفكرية كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
- id=`gold-0011` score=6: عرّف الإيمان كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
- id=`gold-0014` score=6: عرّف العبادة كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
- id=`gold-0017` score=6: عرّف التزكية كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
- id=`gold-0019` score=6: قدّم مثالًا تطبيقيًا لكيفية ممارسة التزكية في موقف واقعي، مع ربطه بتعريفه في الإطار.
- id=`gold-0020` score=6: عرّف التحول العاطفي كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
- id=`gold-0047` score=6: عرّف الرعاية كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
- id=`gold-0049` score=6: قدّم مثالًا تطبيقيًا لكيفية ممارسة الرعاية في موقف واقعي، مع ربطه بتعريفه في الإطار.
- id=`gold-0051` score=6: ما الفرق بين التعاون والتآزر وبين مفهوم قريب منه داخل الإطار؟ اذكر التعريف مع الدليل.
- id=`gold-0062` score=6: عرّف المراقبة كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
- id=`gold-0063` score=6: ما الفرق بين المراقبة وبين مفهوم قريب منه داخل الإطار؟ اذكر التعريف مع الدليل.
- id=`gold-0064` score=6: قدّم مثالًا تطبيقيًا لكيفية ممارسة المراقبة في موقف واقعي، مع ربطه بتعريفه في الإطار.
- id=`gold-0071` score=6: عرّف المجاهدة (بمعنى التصفية) كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
- id=`gold-0074` score=6: عرّف الحب كما ورد في الإطار، واذكر نصًا مُستشهدًا من المصدر.
